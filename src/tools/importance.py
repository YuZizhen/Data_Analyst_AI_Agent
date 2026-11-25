"""Held-out feature importance reports for numeric prediction targets."""

import numpy as np
import pandas as pd
from langchain.tools import ToolRuntime, tool
from langchain_core.tools import ToolException
from matplotlib.ticker import PercentFormatter
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from src.reports import render_report

from .dataset import AnalysisContext, CHART_COLOR, column, finite_numbers, numeric_column, working_frame, working_source

MIN_VALID_ROWS = 30
VALIDATION_FOLDS = 3
PERMUTATION_REPEATS = 5
RANDOM_SEED = 42


def prepare_features(frame: pd.DataFrame, fields: dict) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    features = {}
    numeric, categorical, skipped = [], [], []
    for field in fields:
        series = column(frame, field)
        if is_datetime64_any_dtype(series):
            skipped.append(f"{field} (date column)")
        elif is_numeric_dtype(series) and not is_bool_dtype(series):
            series = finite_numbers(series).astype(float)
            if series.nunique() < 2:
                skipped.append(f"{field} (fewer than two usable numeric values)")
            else:
                features[field] = series
                numeric.append(field)
        else:
            labels = series.astype("string").str.strip()
            labels = labels.mask(labels.eq(""))
            counts = labels.value_counts()
            encoded = ("value:" + labels).fillna("missing").astype(object)
            if encoded.nunique() < 2 or counts.empty or counts.max() < 2:
                skipped.append(f"{field} (constant, blank, or all categories unique)")
            else:
                features[field] = encoded
                categorical.append(field)
    return pd.DataFrame(features, index=frame.index), numeric, categorical, skipped


def validation_importance(
    features: pd.DataFrame, target: pd.Series, numeric: list[str], categorical: list[str],
) -> tuple[pd.Series, float, float]:
    preprocess = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=2), categorical),
    ])
    pipeline = make_pipeline(preprocess, RandomForestRegressor(
        n_estimators=64, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=1,
    ))
    folds = KFold(n_splits=VALIDATION_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    validation = []
    for train, test in folds.split(features):
        y_train, y_test = target.iloc[train], target.iloc[test]
        x_test = features.iloc[test]
        fitted = clone(pipeline).fit(features.iloc[train], y_train)
        model_mae = mean_absolute_error(y_test, fitted.predict(x_test))
        baseline_mae = mean_absolute_error(y_test, np.full(len(test), y_train.median()))
        validation.append((x_test, y_test, model_mae, baseline_mae, fitted))

    weights = [len(y) for _, y, _, _, _ in validation]
    model_mae = float(np.average([mae for _, _, mae, _, _ in validation], weights=weights))
    baseline_mae = float(np.average([mae for _, _, _, mae, _ in validation], weights=weights))
    if not model_mae < baseline_mae:
        raise ToolException(
            f"The model did not beat the median baseline on validation (MAE {model_mae:.4g} vs {baseline_mae:.4g}). "
            "A feature importance percentage would be misleading for this model."
        )
    increases = [
        permutation_importance(
            fitted, x, y, scoring="neg_mean_absolute_error", n_repeats=PERMUTATION_REPEATS,
            random_state=RANDOM_SEED, n_jobs=1,
        ).importances_mean
        for x, y, _, _, fitted in validation
    ]
    values = pd.Series(np.average(increases, axis=0, weights=weights), index=features.columns)
    return values, model_mae, baseline_mae


@tool(response_format="content_and_artifact", parse_docstring=True)
def feature_importance_report(
    target_column: str, runtime: ToolRuntime[AnalysisContext], top_n: int = 5,
    candidate_columns: list[str] | None = None,
) -> tuple[str, dict[str, str]]:
    """Make a Top N predictive feature importance chart with percentage shares for a numeric target.

    Uses a validation-based regression model and permutation importance. Use for requests like
    top contributing factors, importance percentages, or a feature importance chart.

    Args:
        target_column: Exact name of the numeric column to predict.
        top_n: Number of individually displayed factors from 1 to 25. Remaining positive factors are combined.
        candidate_columns: Exact input columns requested by the user. Omit for all eligible columns except the target.
    """
    if not 1 <= top_n <= 25:
        raise ToolException("top_n must be between 1 and 25.")
    context = runtime.context
    frame = working_frame(context, runtime.state)
    target = numeric_column(frame, target_column).dropna()
    if len(target) < MIN_VALID_ROWS:
        raise ToolException(
            f"Predictive importance needs at least {MIN_VALID_ROWS} usable target rows for {VALIDATION_FOLDS}-fold validation. "
            f"The working data have {len(target)}. Descriptive associations are available through rank_relationships."
        )
    if target.nunique() < 2:
        raise ToolException("The numeric target must vary across rows.")
    fields = dict.fromkeys(frame.columns if candidate_columns is None else candidate_columns)
    fields.pop(target_column, None)
    features, numeric, categorical, skipped = prepare_features(frame.loc[target.index], fields)
    if not len(features.columns):
        raise ToolException("No eligible candidate factors in the working data. Check the candidate columns and their values.")

    raw, model_mae, baseline_mae = validation_importance(features, target, numeric, categorical)
    positive = raw[raw > 0].sort_values(ascending=False, kind="stable")
    if positive.empty:
        raise ToolException("No factor increased mean absolute error when shuffled, so there is no positive total to divide.")
    shares = positive / positive.sum()
    shown = shares.head(top_n)
    other = shares.iloc[len(shown):]
    labels, amounts = list(shown.index), list(shown)
    colors = [CHART_COLOR] * len(shown)
    if not other.empty:
        labels.append(f"Other factors ({len(other)})")
        amounts.append(float(other.sum()))
        colors.append("#8398ac")

    title = f"Predictive feature importance for {target_column}"
    subtitle = f"{VALIDATION_FOLDS}-fold validation | {len(target):,} rows | MAE {model_mae:,.4g} vs. median baseline {baseline_mae:,.4g}"

    def draw(axes):
        bars = axes.barh(range(len(labels)), amounts, color=colors, height=0.63)
        axes.set_yticks(range(len(labels)), labels)
        axes.invert_yaxis()
        axes.set_xlim(0, 1.13)
        axes.set_xticks(np.linspace(0, 1, 6))
        axes.xaxis.set_major_formatter(PercentFormatter(xmax=1))
        axes.set_xlabel("Share of total positive validation MAE increase")
        axes.grid(False, axis="y")
        axes.bar_label(bars, labels=[f"{share:.1%}" for share in amounts], color="#243b53", padding=6)

    artifact = render_report(
        context.output_dir, "importance", title, subtitle, working_source(context, runtime.state), draw,
        height=max(5.2, 2.8 + 0.31 * len(labels)),
    )
    lines = [f"PNG report created. {title}", subtitle]
    lines.extend(f"- {field}: {share:.1%} | validation MAE increase {positive[field]:+.5g}" for field, share in shown.items())
    if not other.empty:
        lines.append(f"- Other factors ({len(other)}): {other.sum():.1%}")
    lines.append(f"Evaluated {len(raw)} factors. {len(raw) - len(positive)} had zero or negative mean MAE increase.")
    if skipped:
        lines.append("Skipped: " + ", ".join(skipped) + ".")
    lines.append("Shares divide each positive increase by the sum across all positive factors. This measures model reliance during validation.")
    return "\n".join(lines), artifact

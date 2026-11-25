"""Held-out feature importance reports for numeric prediction targets."""

import numpy as np
import pandas as pd
from langchain.tools import ToolRuntime, tool
from langchain_core.tools import ToolException
from matplotlib.ticker import PercentFormatter
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

from .candidates import candidates_for_tool
from .dataset import AnalysisContext, CHART_COLOR, numeric_column, working_frame, working_source

MIN_VALID_ROWS = 30
VALIDATION_FOLDS = 3
PERMUTATION_REPEATS = 5
RANDOM_SEED = 42


def validation_importance(
    features: pd.DataFrame, target: pd.Series, numeric: list[str], categorical: list[str],
) -> tuple[pd.Series, pd.DataFrame, float, float, int]:
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
    fold_scores = pd.DataFrame(increases, columns=features.columns)
    baseline_wins = sum(mae < baseline for _, _, mae, baseline, _ in validation)
    return values, fold_scores, model_mae, baseline_mae, baseline_wins


@tool(response_format="content_and_artifact", parse_docstring=True)
def feature_importance_report(
    target_column: str, runtime: ToolRuntime[AnalysisContext], top_n: int = 5,
    candidate_columns: list[str] | None = None,
) -> tuple[str, dict]:
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
    review = candidates_for_tool(runtime, frame, target_column, target, candidate_columns, "feature_importance_report")
    numeric = [row["field"] for row in review.evidence["fields"]
               if row["status"] == "included" and row["kind"] == "numeric"]
    categorical = [name for name in review.values if name not in numeric]
    if not review.values:
        raise ToolException("No eligible candidate factors in the working data. See the candidate review for details.")
    features = pd.DataFrame({
        **{name: review.values[name].astype(float) for name in numeric},
        **{name: ("value:" + review.values[name].astype("string")).fillna("missing").astype(object) for name in categorical},
    }, index=target.index)

    runtime.stream_writer({"event": "candidate_calculation", "tool": "feature_importance_report", "args": {
        "target_column": target_column, "candidate_columns": list(review.values), "top_n": top_n,
    }})
    raw, fold_scores, model_mae, baseline_mae, baseline_wins = validation_importance(
        features, target, numeric, categorical,
    )
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
    fold_maxima = fold_scores.max(axis=1)
    factors = [
        {
            "field": field,
            "share": float(share),
            "mae_increase": float(positive[field]),
            "positive_folds": int((fold_scores[field] > 0).sum()),
            "leading_folds": int(((fold_scores[field] == fold_maxima) & (fold_scores[field] > 0)).sum()),
        }
        for field, share in shown.items()
    ]
    artifact["feature_importance"] = {
        "target": target_column, "rows": len(target), "folds": VALIDATION_FOLDS,
        "validation_mae": model_mae, "baseline_mae": baseline_mae, "baseline_wins": baseline_wins,
        "factors": factors,
    }
    lines = [f"PNG report created. {title}", subtitle]
    lines.append(f"The model beat the median baseline in {baseline_wins}/{VALIDATION_FOLDS} validation folds.")
    for factor in factors:
        lines.append(
            f"- {factor['field']}: {factor['share']:.1%} | validation MAE increase {factor['mae_increase']:+.5g}"
            f" | positive in {factor['positive_folds']}/{VALIDATION_FOLDS} folds"
            f" | tied for or ranked first in {factor['leading_folds']}/{VALIDATION_FOLDS} folds"
        )
    if not other.empty:
        lines.append(f"- Other factors ({len(other)}): {other.sum():.1%}")
    lines.append(f"Evaluated {len(raw)} factors. {len(raw) - len(positive)} had zero or negative mean MAE increase.")
    lines.append(
        "Shares divide each positive increase by the sum across all positive factors. "
        "Fold counts are descriptive stability checks. This measures model reliance during validation. "
        "Correlated factors can share importance. Candidate screening only flags some possible leakage. "
        "Generalization to other datasets was not tested."
    )
    return "\n".join(lines), artifact

"""Held-out permutation importance reports for numeric prediction targets."""

import numpy as np
import pandas as pd
from langchain.tools import ToolRuntime, tool
from langchain_core.tools import ToolException
from matplotlib.ticker import PercentFormatter
from sklearn.inspection import permutation_importance

from src.reports import render_report

from .dataset import AnalysisContext, CHART_COLOR, working_source
from .predictions import RANDOM_SEED, assess_model, evaluation_summary

PERMUTATION_REPEATS = 5


def score_importance(validation, columns, progress):
    increases = []
    for fold, (x, y, _, _, fitted) in enumerate(validation, start=1):
        progress("importance", fold)
        increases.append(permutation_importance(
            fitted, x, y, scoring="neg_mean_absolute_error", n_repeats=PERMUTATION_REPEATS,
            random_state=RANDOM_SEED, n_jobs=1,
        ).importances_mean)
    weights = [len(y) for _, y, _, _, _ in validation]
    return pd.Series(np.average(increases, axis=0, weights=weights), index=columns), pd.DataFrame(increases, columns=columns)


@tool(response_format="content_and_artifact", parse_docstring=True)
def feature_importance_report(
    target_column: str, runtime: ToolRuntime[AnalysisContext], top_n: int = 5,
    candidate_columns: list[str] | None = None,
) -> tuple[str, dict]:
    """Make a Top N predictive feature importance chart with percentage shares for a numeric target.

    Use when the user asks for a factor ranking, importance percentages, or a feature importance chart.
    For prediction feasibility or error estimates, use evaluate_prediction.

    Args:
        target_column: Exact name of the numeric column to predict.
        top_n: Number of individually displayed factors from 1 to 25. Remaining positive factors are combined.
        candidate_columns: Exact input columns requested by the user. Omit for all eligible columns except the target.
    """
    if not 1 <= top_n <= 25:
        raise ToolException("top_n must be between 1 and 25.")
    context = runtime.context
    validation, evidence, progress = assess_model(runtime, target_column, candidate_columns, "feature_importance_report")
    model_mae, baseline_mae = evidence["validation_mae"], evidence["baseline_mae"]
    if not model_mae < baseline_mae:
        raise ToolException(
            f"{evaluation_summary(evidence)}. The model did not beat the median baseline on validation. "
            "A feature importance percentage would be misleading for this model."
        )
    raw, fold_scores = score_importance(validation, evidence["fields"], progress)
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
    subtitle = evaluation_summary(evidence)

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

    progress("chart")
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
        **evidence, "factors": factors,
    }
    lines = [f"PNG report created. {title}", subtitle]
    lines.append(f"The model beat the median baseline in {evidence['baseline_wins']}/{evidence['folds']} validation folds.")
    lines.append(f"Showing {len(shown)} individual factors out of {len(positive)} with positive MAE increases.")
    for factor in factors:
        lines.append(
            f"- {factor['field']}: {factor['share']:.1%} | validation MAE increase {factor['mae_increase']:+.5g}"
            f" | positive in {factor['positive_folds']}/{evidence['folds']} folds"
            f" | tied for or ranked first in {factor['leading_folds']}/{evidence['folds']} folds"
        )
    if not other.empty:
        lines.append(f"- Other factors ({len(other)}): {other.sum():.1%}")
    lines.append(f"Evaluated {len(raw)} factors. {len(raw) - len(positive)} had zero or negative mean MAE increase.")
    lines.append(
        "Shares divide each positive increase by the sum across all positive increases. They are neither shares of total error nor price. "
        "Fold counts are descriptive stability checks. This measures model reliance during validation. "
        "Correlated factors can share importance. Candidate screening only flags some possible leakage. "
        "Generalization to other datasets was not tested."
    )
    return "\n".join(lines), artifact

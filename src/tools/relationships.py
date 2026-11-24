"""Tools for pairwise scatter reports and field association rankings."""

from math import isclose, isfinite, sqrt

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import ToolException
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from src.reports import render_report

from .dataset import AnalysisContext, CHART_COLOR, column, finite_numbers, numeric_column, working_frame, working_source

@tool(response_format="content_and_artifact", parse_docstring=True)
def relationship_report(
    x_column: str, y_column: str, runtime: ToolRuntime[AnalysisContext]
) -> tuple[str, dict[str, str]]:
    """Plot two specified numeric columns as a PNG scatter chart and calculate Pearson r.

    For the strongest relationship with a target across fields, use rank_relationships first.

    Args:
        x_column: Numeric CSV column for the horizontal axis.
        y_column: Numeric CSV column for the vertical axis.
    """
    if x_column == y_column:
        raise ToolException("Choose two different numeric columns.")
    context = runtime.context
    df = working_frame(context, runtime.state)
    x = numeric_column(df, x_column)
    y = numeric_column(df, y_column)
    valid = x.notna() & y.notna()
    x, y = x[valid], y[valid]
    if x.empty:
        raise ToolException("No rows have valid values in both columns.")
    correlation = "undefined"
    if x.nunique() > 1 and y.nunique() > 1:
        correlation = f"{x.corr(y):.3f}"
    title = f"{y_column} vs. {x_column}"
    subtitle = f"{len(x):,} of {len(df):,} paired rows used | Pearson r = {correlation}"

    def draw(axes):
        axes.scatter(x, y, s=32, color=CHART_COLOR, alpha=0.68, edgecolors="none")
        axes.set(xlabel=x_column, ylabel=y_column)

    artifact = render_report(context.output_dir, "relationship", title, subtitle, working_source(context, runtime.state), draw)
    return f"PNG report created. {title}\n{subtitle}", artifact


@tool(parse_docstring=True)
def rank_relationships(
    target_column: str, runtime: ToolRuntime[AnalysisContext],
    candidate_columns: list[str] | None = None, top_n: int = 10,
) -> str:
    """Find which fields are most related to a numeric target by comparing them all.

    Ranks numeric fields by absolute Pearson r and categorical fields separately by eta.
    Use for questions like what is most correlated with price, or which field matters most.

    Args:
        target_column: Exact CSV column name of the numeric outcome to compare against.
        candidate_columns: Exact names of candidates explicitly requested by the user. Leave null for all other fields.
        top_n: Maximum results to display per field type, between 1 and 25. All candidates are evaluated.
    """
    if not 1 <= top_n <= 25:
        raise ToolException("top_n must be between 1 and 25.")
    context = runtime.context
    df = working_frame(context, runtime.state)
    target = numeric_column(df, target_column)
    if target.nunique() < 2:
        raise ToolException("The numeric target must vary across rows.")
    names = dict.fromkeys(df.columns if candidate_columns is None else candidate_columns)
    names.pop(target_column, None)
    numeric = []
    categorical = []
    skipped = []
    for name in names:
        candidate = column(df, name)
        if is_numeric_dtype(candidate) and not is_bool_dtype(candidate):
            values = finite_numbers(candidate)
            valid = values.notna() & target.notna()
            x, y = values[valid], target[valid]
            if len(x) >= 3 and x.nunique() >= 2 and y.nunique() >= 2:
                r = x.corr(y)
                if isfinite(r):
                    numeric.append((name, r, len(x)))
                    continue
            skipped.append(f"{name}: Pearson r requires at least 3 usable pairs and variation in both fields.")
            continue

        valid = candidate.notna() & target.notna()
        values = target[valid]
        groups = values.groupby(candidate[valid], observed=True).agg(["size", "mean"])
        if len(groups) < 2 or groups["size"].min() < 2 or values.nunique() < 2:
            skipped.append(f"{name}: eta requires at least two groups, two usable rows per group and target variation.")
            continue
        between = (groups["size"] * (groups["mean"] - values.mean()).pow(2)).sum()
        total = (values - values.mean()).pow(2).sum()
        if total > 0 and isfinite(total) and isfinite(between):
            categorical.append((name, sqrt(min(between / total, 1.0)), len(values), len(groups)))
        else:
            skipped.append(f"{name}: group variation could not be computed.")

    numeric.sort(key=lambda item: abs(item[1]), reverse=True)
    categorical.sort(key=lambda item: item[1], reverse=True)
    lines = [
        f"Dataset: {working_source(context, runtime.state)}. Target: {target_column}. "
        f"Evaluated {len(names)} candidate fields across {len(df)} rows."
    ]
    if numeric:
        best_name, best_r, best_n = numeric[0]
        leaders = [name for name, r, _ in numeric if isclose(abs(r), abs(best_r), abs_tol=1e-12)]
        if len(leaders) == 1:
            lines.append(f"Among numeric fields only, {best_name} ranks first (Pearson r = {best_r:+.3f}, n = {best_n}).")
        else:
            lines.append(f"Among numeric fields only, {len(leaders)} fields tie for first with |r| = {abs(best_r):.3f}.")
        lines.append(f"Numeric ranking (Pearson r, sorted by |r|; showing {min(top_n, len(numeric))} of {len(numeric)}):")
        lines.extend(f"- {name}: Pearson r = {r:+.3f}, n = {size}" for name, r, size in numeric[:top_n])
    else:
        lines.append("No eligible numeric candidates.")
    if categorical:
        lines.append(f"Categorical ranking (eta, shown separately; showing {min(top_n, len(categorical))} of {len(categorical)}):")
        lines.extend(
            f"- {name}: eta = {eta:.3f}, n = {size}, groups = {group_count}"
            for name, eta, size, group_count in categorical[:top_n]
        )
        if len(categorical) == 1:
            lines.append("Only one categorical field was eligible, so it has no categorical comparator.")
    if skipped:
        lines.append("Skipped fields:\n- " + "\n- ".join(skipped))
    lines.append("These are descriptive associations. Compare scores within each ranking. They do not identify causal drivers.")
    return "\n".join(lines)

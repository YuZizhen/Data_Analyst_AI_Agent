"""Group statistics, comparisons, trends, and distributions."""

from typing import Literal

import pandas as pd
from langchain.tools import ToolRuntime, tool
from langchain_core.tools import ToolException
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

from src.reports import render_report

from .dataset import AnalysisContext, CHART_COLOR, column, load_dataset, numeric_column, working_frame, working_source
from .formulas import formula_values

Aggregation = Literal["mean", "median", "sum", "count"]
AGGREGATION_LABELS = {"mean": "Average", "median": "Median", "sum": "Total"}
GROUP_STATISTICS = {
    "mean": ("平均值", "mean"), "median": ("中位数", "median"), "sum": ("总和", "total"), "count": ("记录数", "row count"),
}


def summarize_groups(
    groups: pd.Series, values: pd.Series, aggregation: Aggregation
) -> pd.Series:
    grouped = values.groupby(groups, dropna=False, observed=True)
    if aggregation == "sum":
        return grouped.sum(min_count=1).dropna()
    return grouped.agg(aggregation).dropna()


def group_results(table, groups, values):
    counts = values.groupby(groups, dropna=False, observed=True).count().reindex(table.index)
    displayed = pd.DataFrame({"statistic": table, "valid_rows": counts})
    summary = [{"group": "(Missing)" if pd.isna(name) else str(name), "value": float(value), "valid_rows": int(count)}
               for name, value, count in zip(table.index, table, counts)]
    return displayed.to_string(), summary


def grouped_answer(result: dict, *, chinese: bool) -> str:
    group = result["group_column"]
    field = result["value_column"]
    zh, en = GROUP_STATISTICS[result["aggregation"]]
    if not result["groups"]:
        return "当前范围没有可显示的分组结果。" if chinese else "There are no group results for the current scope."
    first = result["groups"][0]
    leading = [row for row in result["groups"] if row["value"] == first["value"]]
    value = f"{first['value']:,.12g}"
    used, total = result["rows_used"], result["working_rows"]
    shown, all_groups = len(result["groups"]), result["group_count"]
    if chinese:
        metric = f"「{field}」的{zh}" if field else zh
        line = f"按「{group}」计算{metric}。"
        if all_groups == 1:
            return f"{line} 当前范围只有「{first['group']}」一个有效分组，统计值为 **{value}**，使用了 {used:,} 条有效记录。"
        line += f"最高值为 **{value}**（{'、'.join(row['group'] for row in leading)}）。"
        return f"{line} 结果展示 {all_groups:,} 组中的 {shown:,} 组。本次分组统计使用了 {total:,} 条记录中的 **{used:,} 条**有效记录。"
    labels = ", ".join(row["group"] for row in leading)
    metric = f"{en} of {field}" if field else en
    if all_groups == 1:
        return f"The {metric} for {first['group']} is **{value}**, using {used:,} valid rows. The current scope has one usable group."
    return (f"Grouped the {metric} by {group}. The highest value is **{value}** ({labels}). "
            f"Showing {shown:,} of {all_groups:,} groups, using {used:,} of {total:,} rows.")


def metric(
    df: pd.DataFrame, aggregation: Aggregation, value_column: str | None
) -> tuple[pd.Series, str]:
    if aggregation == "count":
        if value_column is not None:
            raise ToolException("Omit value_column when counting rows.")
        return pd.Series(1, index=df.index), "Row count"
    if not value_column:
        raise ToolException(f"value_column is required for {aggregation}.")
    return numeric_column(df, value_column), f"{AGGREGATION_LABELS[aggregation]} of {value_column}"


@tool(response_format="content_and_artifact", parse_docstring=True)
def group_distribution_report(
    group_column: str, value_column: str, runtime: ToolRuntime[AnalysisContext], top_n: int = 12,
) -> tuple[str, dict]:
    """Draw box plots comparing the distribution of a numeric value across categories.

    Args:
        group_column: Exact categorical column name.
        value_column: Exact numeric column name.
        top_n: Maximum categories to show, ranked by count (1 to 25).
    """
    if not 1 <= top_n <= 25:
        raise ToolException("top_n must be between 1 and 25.")
    context = runtime.context
    frame = working_frame(context, runtime.state)
    values = numeric_column(frame, value_column)
    labels = column(frame, group_column).astype("string")
    valid = labels.notna() & labels.str.strip().ne("") & values.notna()
    groups = values[valid].groupby(labels[valid], observed=True)
    counts = groups.size().nlargest(top_n)
    if counts.empty:
        raise ToolException("No usable groups for the requested distribution.")
    shown = [(str(name), groups.get_group(name)) for name in counts.index]
    summary = [{"group": name, "rows": len(group), "median": float(group.median())} for name, group in shown]
    title = f"Distribution of {value_column} by {group_column}"
    subtitle = f"{len(shown)} categories shown | {sum(row['rows'] for row in summary):,} paired rows shown"

    def draw(axes):
        axes.boxplot([group.to_numpy(dtype=float) for _, group in shown], vert=False, showfliers=False)
        axes.set_yticks(range(1, len(shown) + 1), [name for name, _ in shown])
        axes.invert_yaxis()
        axes.set_xlabel(value_column)

    artifact = render_report(context.output_dir, "group-distribution", title, subtitle,
                             working_source(context, runtime.state), draw, height=max(5.2, 2.8 + len(shown) * 0.35))
    artifact["group_distribution"] = {"groups": summary, "rows": int(valid.sum())}
    return f"PNG report created. {title}. {subtitle}. Categories selected by frequency. Outlier markers are hidden.", artifact


@tool(response_format="content_and_artifact", parse_docstring=True)
def average_by_group(
    group_column: str, value_column: str | None = None, top_n: int = 10,
    runtime: ToolRuntime[AnalysisContext] = None, aggregation: Aggregation = "mean",
) -> tuple[str, dict]:
    """Calculate group statistics without a chart: mean, median, total, or row count.

    Args:
        group_column: Name of the categorical column (such as town).
        value_column: Name of the numeric column. Omit when counting all rows.
        top_n: Number of groups to return, between 1 and 25.
        aggregation: Statistic per group. Pass sum for totals or count for row counts.
    """
    df = working_frame(runtime.context, runtime.state) if runtime else load_dataset()
    try:
        if not 1 <= top_n <= 25:
            raise ToolException("top_n must be between 1 and 25.")
        groups = column(df, group_column)
        values, label = metric(df, aggregation, value_column)
    except ToolException as error:
        if runtime is not None:
            raise
        return str(error), {}

    ranked = summarize_groups(groups, values, aggregation)
    table = ranked.nlargest(top_n)
    displayed, summary = group_results(table, groups, values)
    return f"{label} by {group_column} (descending):\n{displayed}", {"group_summary": {
        "aggregation": aggregation, "group_column": group_column, "value_column": value_column,
        "rows_used": int(values.count()), "working_rows": len(df), "group_count": len(ranked),
        "groups": summary,
    }}


@tool(response_format="content_and_artifact", parse_docstring=True)
def formula_group_totals(
    group_column: str, value_column: str, expression: str, runtime: ToolRuntime[AnalysisContext],
) -> tuple[str, dict]:
    """Evaluate an arithmetic formula on every row and sum its values by group.

    Args:
        group_column: Exact CSV column used for grouping.
        value_column: Display name of the derived measure.
        expression: User-provided formula using numeric columns and arithmetic.
    """
    frame = working_frame(runtime.context, runtime.state)
    groups = column(frame, group_column)
    values = formula_values(frame, expression)
    ranked = summarize_groups(groups, values, "sum")
    displayed, summary = group_results(ranked.nlargest(25), groups, values)
    evidence = {"aggregation": "sum", "group_column": group_column, "value_column": value_column,
                "expression": expression, "rows_used": int(values.count()), "working_rows": len(frame),
                "group_count": len(ranked), "groups": summary}
    return f"Calculated {value_column} = {expression} per row, then summed by {group_column}.\n{displayed}", {"group_summary": evidence}


@tool(response_format="content_and_artifact", parse_docstring=True)
def group_report(
    group_column: str,
    runtime: ToolRuntime[AnalysisContext],
    value_column: str | None = None,
    aggregation: Aggregation = "mean",
    top_n: int = 10,
) -> tuple[str, dict]:
    """Compare categories or rank groups as a horizontal bar chart in a PNG report.

    Args:
        group_column: Category column used to group CSV rows.
        value_column: Numeric CSV column to aggregate. Omit for row counts.
        aggregation: Statistic to compute in each group: mean, median, sum, or row count.
        top_n: Number of highest-ranking groups to show, between 1 and 25.
    """
    if not 1 <= top_n <= 25:
        raise ToolException("top_n must be between 1 and 25.")
    context = runtime.context
    df = working_frame(context, runtime.state)
    groups = column(df, group_column)
    values, label = metric(df, aggregation, value_column)
    ranked = summarize_groups(groups, values, aggregation)
    if ranked.empty:
        raise ToolException("There are no rows to group.")
    table = ranked.nlargest(top_n)
    display_labels = ["(Missing)" if pd.isna(group) else str(group) for group in table.index]
    subtitle = f"Top {len(table):,} of {len(ranked):,} groups | {values.count():,} of {len(df):,} rows used"
    title = f"{label} by {group_column}"

    def draw(axes):
        bars = axes.barh(range(len(table)), table.to_numpy(dtype=float), color=CHART_COLOR, height=0.62)
        axes.set_yticks(range(len(table)), display_labels)
        axes.invert_yaxis()
        axes.set_xlabel(label)
        axes.grid(False, axis="y")
        axes.bar_label(bars, labels=[f"{value:,.6g}" for value in table], color="#243b53", padding=5)
        axes.margins(x=0.18)

    artifact = render_report(
        context.output_dir, "group", title, subtitle, working_source(context, runtime.state), draw,
        height=max(5.2, 2.6 + len(table) * 0.29),
    )
    displayed, summary = group_results(table, groups, values)
    artifact["group_summary"] = {
        "aggregation": aggregation, "group_column": group_column, "value_column": value_column,
        "rows_used": int(values.count()), "working_rows": len(df), "group_count": len(ranked),
        "groups": summary,
    }
    return f"PNG report created. {title}\n{subtitle}\n{displayed}", artifact


@tool(response_format="content_and_artifact", parse_docstring=True)
def trend_report(
    date_column: str,
    runtime: ToolRuntime[AnalysisContext],
    value_column: str | None = None,
    aggregation: Aggregation = "mean",
    frequency: Literal["day", "week", "month", "quarter", "year"] = "month",
) -> tuple[str, dict]:
    """Plot a time trend as a PNG line chart using an actual date column.

    Args:
        date_column: CSV column containing dates, for example YYYY-MM-DD or YYYY-MM.
        value_column: Numeric CSV column to aggregate. Omit for row counts.
        aggregation: Statistic to compute in each period: mean, median, sum, or row count.
        frequency: Time period to group by, such as month. Weeks start Monday.
    """
    context = runtime.context
    df = working_frame(context, runtime.state)
    raw_dates = column(df, date_column).astype("string").str.strip()
    dates = pd.to_datetime(raw_dates, format="mixed", errors="coerce", utc=True).dt.tz_localize(None)
    values, label = metric(df, aggregation, value_column)
    valid = dates.notna() & values.notna()
    if not valid.any():
        raise ToolException("No rows have a valid date and metric value.")
    period_code = {"day": "D", "week": "W-SUN", "month": "M", "quarter": "Q", "year": "Y"}
    periods = dates[valid].dt.to_period(period_code[frequency])
    table = summarize_groups(periods, values[valid], aggregation).sort_index()
    title = f"{label} by {frequency}"
    subtitle = f"{len(table):,} observed periods | {valid.sum():,} of {len(df):,} rows used"

    def draw(axes):
        # Break the line across missing periods without filling gaps with zeroes.
        x, y = [], []
        previous = None
        for period, value in table.items():
            if previous is not None and period != previous + 1:
                x.append((previous + 1).start_time)
                y.append(float("nan"))
            x.append(period.start_time)
            y.append(value)
            previous = period
        axes.plot(x, y, color=CHART_COLOR, marker="o", markersize=3.5, linewidth=1.8)
        locator = AutoDateLocator(minticks=3, maxticks=8)
        axes.xaxis.set_major_locator(locator)
        axes.xaxis.set_major_formatter(ConciseDateFormatter(locator))
        axes.set(xlabel=date_column, ylabel=label)

    artifact = render_report(context.output_dir, "trend", title, subtitle, working_source(context, runtime.state), draw)
    artifact["trend_summary"] = {
        "periods": len(table), "rows_used": int(valid.sum()),
        "start_period": str(table.index[0]), "end_period": str(table.index[-1]),
    }
    first = f"First {table.index[0]} = {table.iloc[0]:,.8g}"
    last = f"Last {table.index[-1]} = {table.iloc[-1]:,.8g}"
    low = f"Minimum {table.min():,.8g} in {table.idxmin()}"
    high = f"Maximum {table.max():,.8g} in {table.idxmax()}"
    return f"PNG report created. {title}\n{subtitle}\n{first}. {last}. {low}. {high}.", artifact


@tool(response_format="content_and_artifact", parse_docstring=True)
def distribution_report(
    value_column: str, runtime: ToolRuntime[AnalysisContext], bins: int = 10
) -> tuple[str, dict]:
    """Plot the frequency distribution of a numeric column as a PNG histogram.

    Args:
        value_column: Numeric CSV column whose distribution should be plotted.
        bins: Number of equal-width histogram bins, between 1 and 50.
    """
    if not 1 <= bins <= 50:
        raise ToolException("bins must be between 1 and 50.")
    context = runtime.context
    df = working_frame(context, runtime.state)
    values = numeric_column(df, value_column).dropna()
    title = f"Distribution of {value_column}"
    subtitle = (
        f"{len(values):,} of {len(df):,} rows used | "
        f"Median {values.median():,.8g} | Range {values.min():,.8g} to {values.max():,.8g}"
    )

    def draw(axes):
        axes.hist(values, bins=bins, color=CHART_COLOR, edgecolor="white")
        axes.set(xlabel=value_column, ylabel="Row count")
        axes.grid(False, axis="x")

    artifact = render_report(context.output_dir, "distribution", title, subtitle, working_source(context, runtime.state), draw)
    artifact["distribution_summary"] = {
        "rows_used": len(values), "working_rows": len(df),
        "median": float(values.median()), "min": float(values.min()), "max": float(values.max()),
    }
    return f"PNG report created. {title}\n{subtitle}", artifact

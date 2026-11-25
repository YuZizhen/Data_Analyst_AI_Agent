"""Group statistics, comparisons, trends, and distributions."""

from typing import Literal

import pandas as pd
from langchain.tools import ToolRuntime, tool
from langchain_core.tools import ToolException
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

from src.reports import render_report

from .dataset import AnalysisContext, CHART_COLOR, column, load_dataset, numeric_column, working_frame, working_source

Aggregation = Literal["mean", "median", "sum", "count"]
AGGREGATION_LABELS = {"mean": "Average", "median": "Median", "sum": "Total"}


def summarize_groups(
    groups: pd.Series, values: pd.Series, aggregation: Aggregation
) -> pd.Series:
    grouped = values.groupby(groups, dropna=False, observed=True)
    if aggregation == "sum":
        return grouped.sum(min_count=1).dropna()
    return grouped.agg(aggregation).dropna()


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
def average_by_group(
    group_column: str, value_column: str, top_n: int = 10,
    runtime: ToolRuntime[AnalysisContext] = None,
) -> tuple[str, dict]:
    """Calculate text-only group averages. For a chart, use group_report instead.

    Args:
        group_column: Name of the categorical column (such as town).
        value_column: Name of the numeric column (such as resale_price).
        top_n: Number of groups to return, between 1 and 25.
    """
    df = working_frame(runtime.context, runtime.state) if runtime else load_dataset()
    try:
        if not 1 <= top_n <= 25:
            raise ToolException("top_n must be between 1 and 25.")
        groups = column(df, group_column)
        values = numeric_column(df, value_column)
    except ToolException as error:
        if runtime is not None:
            raise
        return str(error), {}

    ranked = summarize_groups(groups, values, "mean")
    table = ranked.nlargest(top_n)
    return "Average by group (descending):\n" + table.to_string(), {"group_summary": {
        "rows_used": int(values.count()), "working_rows": len(df), "group_count": len(ranked),
        "groups": [{"group": "(Missing)" if pd.isna(name) else str(name), "value": float(value)} for name, value in table.items()],
    }}


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
    artifact["group_summary"] = {
        "rows_used": int(values.count()), "working_rows": len(df), "group_count": len(ranked),
        "groups": [{"group": name, "value": float(value)} for name, value in zip(display_labels, table)],
    }
    return f"PNG report created. {title}\n{subtitle}\n{table.to_string()}", artifact


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

"""CSV analysis functions and PNG report tools exposed to the agent."""

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal

import pandas as pd
from langchain.tools import ToolRuntime, tool
from langchain_core.tools import ToolException
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

from src.reports import render_report

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "hdb_resale.csv"
Aggregation = Literal["mean", "median", "sum", "count"]
AGGREGATION_LABELS = {"mean": "Average", "median": "Median", "sum": "Total"}
CHART_COLOR = "#147d83"
DEFAULT_PREVIEW_ROWS = 10


@dataclass(frozen=True)
class AnalysisContext:
    dataset: pd.DataFrame
    source: str
    output_dir: Path


def load_dataset(path: Path | BinaryIO = DATA_PATH) -> pd.DataFrame:
    """Load a CSV, defaulting to the repository's demonstration dataset."""
    return pd.read_csv(path)


def describe_dataset(df: pd.DataFrame) -> str:
    columns = ", ".join(f"{column} ({dtype})" for column, dtype in df.dtypes.items())
    return f"Rows: {len(df)}. Columns: {columns}"


def column(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        raise ToolException(f"Invalid columns. Available columns: {', '.join(df.columns)}")
    return df[name]


def numeric_column(df: pd.DataFrame, name: str) -> pd.Series:
    values = pd.to_numeric(column(df, name), errors="coerce")
    values = values.replace([float("inf"), -float("inf")], float("nan"))
    if not values.notna().any():
        raise ToolException(f"Column {name!r} contains no valid numeric values.")
    return values


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


@tool
def dataset_overview(runtime: ToolRuntime[AnalysisContext] = None) -> str:
    """Show the available dataset columns, their types, and the row count."""
    df = runtime.context.dataset if runtime else load_dataset()
    return describe_dataset(df)


@tool(response_format="content_and_artifact", parse_docstring=True)
def dataset_preview(
    runtime: ToolRuntime[AnalysisContext], limit: int = DEFAULT_PREVIEW_ROWS
) -> tuple[str, dict]:
    """Display actual rows when a user wants to look at or browse the current dataset.

    Args:
        limit: Number of rows to display from the start of the dataset, between 1 and 50.
    """
    if not 1 <= limit <= 50:
        raise ToolException("limit must be between 1 and 50.")
    dataset = runtime.context.dataset
    preview = dataset.head(limit)
    return (
        f"Dataset preview ready for display. Showing {len(preview):,} of {len(dataset):,} rows.",
        {"table": preview.to_dict(orient="split", index=False), "total_rows": len(dataset)},
    )


@tool
def average_by_group(
    group_column: str, value_column: str, top_n: int = 10,
    runtime: ToolRuntime[AnalysisContext] = None,
) -> str:
    """Calculate the largest group averages for a numeric column in the CSV.

    Args:
        group_column: Name of the categorical column (such as town).
        value_column: Name of the numeric column (such as resale_price).
        top_n: Number of groups to return, between 1 and 25.
    """
    df = runtime.context.dataset if runtime else load_dataset()
    if not 1 <= top_n <= 25:
        return "top_n must be between 1 and 25."
    try:
        groups = column(df, group_column)
        values = numeric_column(df, value_column)
    except ToolException as error:
        return str(error)

    table = summarize_groups(groups, values, "mean").nlargest(top_n)
    return "Average by group (descending):\n" + table.to_string()


@tool(response_format="content_and_artifact", parse_docstring=True)
def group_report(
    group_column: str,
    runtime: ToolRuntime[AnalysisContext],
    value_column: str | None = None,
    aggregation: Aggregation = "mean",
    top_n: int = 10,
) -> tuple[str, dict[str, str]]:
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
    df = context.dataset
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
        context.output_dir, "group", title, subtitle, context.source, draw,
        height=max(5.2, 2.6 + len(table) * 0.29),
    )
    return f"PNG report created. {title}\n{subtitle}\n{table.to_string()}", artifact


@tool(response_format="content_and_artifact", parse_docstring=True)
def trend_report(
    date_column: str,
    runtime: ToolRuntime[AnalysisContext],
    value_column: str | None = None,
    aggregation: Aggregation = "mean",
    frequency: Literal["day", "week", "month", "quarter", "year"] = "month",
) -> tuple[str, dict[str, str]]:
    """Plot a time trend as a PNG line chart using an actual date column.

    Args:
        date_column: CSV column containing dates, for example YYYY-MM-DD or YYYY-MM.
        value_column: Numeric CSV column to aggregate. Omit for row counts.
        aggregation: Statistic to compute in each period: mean, median, sum, or row count.
        frequency: Time period to group by, such as month. Weeks start Monday.
    """
    context = runtime.context
    df = context.dataset
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

    artifact = render_report(context.output_dir, "trend", title, subtitle, context.source, draw)
    first = f"First {table.index[0]} = {table.iloc[0]:,.8g}"
    last = f"Last {table.index[-1]} = {table.iloc[-1]:,.8g}"
    low = f"Minimum {table.min():,.8g} in {table.idxmin()}"
    high = f"Maximum {table.max():,.8g} in {table.idxmax()}"
    return f"PNG report created. {title}\n{subtitle}\n{first}. {last}. {low}. {high}.", artifact


@tool(response_format="content_and_artifact", parse_docstring=True)
def distribution_report(
    value_column: str, runtime: ToolRuntime[AnalysisContext], bins: int = 10
) -> tuple[str, dict[str, str]]:
    """Plot the frequency distribution of a numeric column as a PNG histogram.

    Args:
        value_column: Numeric CSV column whose distribution should be plotted.
        bins: Number of equal-width histogram bins, between 1 and 50.
    """
    if not 1 <= bins <= 50:
        raise ToolException("bins must be between 1 and 50.")
    context = runtime.context
    values = numeric_column(context.dataset, value_column).dropna()
    title = f"Distribution of {value_column}"
    subtitle = (
        f"{len(values):,} of {len(context.dataset):,} rows used | "
        f"Median {values.median():,.8g} | Range {values.min():,.8g} to {values.max():,.8g}"
    )

    def draw(axes):
        axes.hist(values, bins=bins, color=CHART_COLOR, edgecolor="white")
        axes.set(xlabel=value_column, ylabel="Row count")
        axes.grid(False, axis="x")

    artifact = render_report(context.output_dir, "distribution", title, subtitle, context.source, draw)
    return f"PNG report created. {title}\n{subtitle}", artifact


@tool(response_format="content_and_artifact", parse_docstring=True)
def relationship_report(
    x_column: str, y_column: str, runtime: ToolRuntime[AnalysisContext]
) -> tuple[str, dict[str, str]]:
    """Plot two numeric columns as a PNG scatter chart and calculate Pearson correlation.

    Args:
        x_column: Numeric CSV column for the horizontal axis.
        y_column: Numeric CSV column for the vertical axis.
    """
    if x_column == y_column:
        raise ToolException("Choose two different numeric columns.")
    context = runtime.context
    df = context.dataset
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

    artifact = render_report(context.output_dir, "relationship", title, subtitle, context.source, draw)
    return f"PNG report created. {title}\n{subtitle}", artifact

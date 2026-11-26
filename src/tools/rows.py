"""Return facts about original rows on the current working table."""

import json
from math import isfinite
from typing import Literal

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import ToolException

from .dataset import AnalysisContext, column, finite_numbers, working_frame

Statistic = Literal["sum", "mean", "median", "min", "max"]
STATISTIC_LABELS = {
    "sum": ("总和", "sum"), "mean": ("平均值", "mean"), "median": ("中位数", "median"),
    "min": ("最小值", "minimum"), "max": ("最大值", "maximum"),
}


def statistic_answer(result: dict, *, chinese: bool) -> str:
    zh, en = STATISTIC_LABELS[result["statistic"]]
    field = result["value_column"]
    used, total, value = result["valid_rows"], result["working_rows"], result["value"]
    if value is None:
        return (f"当前范围的 {total:,} 行中，「{field}」没有有效数值可计算。"
                if chinese else f"No usable numeric values for {field} in the {total:,} selected rows.")
    displayed = f"{value:,}" if isinstance(value, int) else f"{value:,.12g}"
    if chinese:
        return f"「{field}」的{zh}是 **{displayed}**。共核对 {total:,} 行，其中 {used:,} 行有有效数值。"
    return f"The {en} of {field} is **{displayed}**. Used {used:,} numeric values across {total:,} records."


@tool(response_format="content_and_artifact", parse_docstring=True)
def count_rows(runtime: ToolRuntime[AnalysisContext]) -> tuple[str, dict]:
    """Count all rows in the working table, including rows with missing data."""
    total = len(working_frame(runtime.context, runtime.state))
    return f"Working table row count: {total:,}. Missing values are included.", {"row_count": {"rows": total}}


@tool(response_format="content_and_artifact", parse_docstring=True)
def aggregate_column(
    value_column: str, statistic: Statistic, runtime: ToolRuntime[AnalysisContext],
) -> tuple[str, dict]:
    """Compute one overall statistic for one numeric column over all rows in the current scope.

    Use this for an overall total, average, median, minimum, or maximum without grouping.

    Args:
        value_column: Exact field containing the numeric values.
        statistic: The requested statistic, one of sum, mean, median, min, max.
    """
    frame = working_frame(runtime.context, runtime.state)
    values = finite_numbers(column(frame, value_column)).dropna()
    value = None
    if not values.empty:
        if statistic == "sum" and values.dtype.kind in "iu":
            value = sum(map(int, values))
        else:
            value = getattr(values, statistic)().item()
        if isinstance(value, float) and not isfinite(value):
            raise ToolException("The calculated result exceeds the supported numeric range.")
    evidence = {"value_column": value_column, "statistic": statistic, "value": value,
                "valid_rows": len(values), "working_rows": len(frame)}
    return statistic_answer(evidence, chinese=False), {"column_statistic": evidence}


@tool(response_format="content_and_artifact", parse_docstring=True)
def rank_rows(
    sort_column: str, columns: list[str], runtime: ToolRuntime[AnalysisContext], limit: int = 10, ascending: bool = False,
    include_sort_value: bool = True,
) -> tuple[str, dict]:
    """List original records ordered by a numeric field; use for top or bottom individual records.

    Args:
        sort_column: Exact column name with the numeric sort values.
        columns: Exact field names to display from each record.
        limit: Maximum number of rows to return, between 1 and 50.
        ascending: Sort from smallest to largest when true.
        include_sort_value: Also show the sort column if it was not requested in columns.
    """
    if not 1 <= limit <= 50 or not columns:
        raise ToolException("Specify the fields to display and a limit between 1 and 50.")
    frame = working_frame(runtime.context, runtime.state)
    fields = list(dict.fromkeys([*columns, sort_column] if include_sort_value else columns))
    for field in fields:
        column(frame, field)
    values = finite_numbers(column(frame, sort_column)).dropna()
    order = values.sort_values(ascending=ascending, kind="stable").head(limit)
    rows = frame.loc[order.index, fields].to_json(orient="records", date_format="iso", double_precision=15)
    artifact = {"columns": fields, "rows": json.loads(rows), "row_ids": order.index.tolist(),
                "working_rows": len(frame), "rows_with_sort_value": len(values), "sort_column": sort_column,
                "ascending": ascending}
    summary = f"Returned {len(order)} of {len(values)} rows with a numeric {sort_column} (working table has {len(frame)} rows)."
    return f"{summary}\n{rows}", {"ranked_rows": artifact}

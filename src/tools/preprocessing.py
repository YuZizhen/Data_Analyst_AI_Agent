"""Inspect and prepare a working table for the current analysis."""

from operator import ge, gt, le, lt
from typing import Literal

import pandas as pd
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.tools import ToolException
from langgraph.types import Command
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from .dataset import AnalysisContext, PreparedDataset, column, finite_numbers, working_frame

Scalar = str | int | float | bool
FilterOperator = Literal["equals", "not_equals", "in", "contains", "greater_than", "at_least", "less_than", "at_most"]
ORDER_OPERATORS = {"greater_than": gt, "at_least": ge, "less_than": lt, "at_most": le}


def missing_mask(values: pd.Series) -> pd.Series:
    missing = values.isna()
    if not (is_numeric_dtype(values) or is_datetime64_any_dtype(values)):
        missing |= values.astype("string").str.strip().eq("").fillna(False)
    return missing


def selected_columns(frame: pd.DataFrame, columns: list[str] | None) -> list[str]:
    names = list(frame.columns if columns is None else dict.fromkeys(columns))
    if not names:
        raise ToolException("Choose at least one column.")
    for name in names:
        column(frame, name)
    return names


def save_preparation(
    runtime: ToolRuntime[AnalysisContext], frame: pd.DataFrame, name: str, parameters: dict, detail: str,
) -> Command:
    previous = runtime.state.get("prepared")
    before = len(working_frame(runtime.context, runtime.state))
    steps = () if previous is None else previous.steps
    record = {"tool": name, "parameters": parameters, "rows_before": before, "rows_after": len(frame), "detail": detail}
    message = ToolMessage(
        content=f"{detail} Working rows: {before:,} → {len(frame):,} of {len(runtime.context.dataset):,} original rows.",
        name=name, tool_call_id=runtime.tool_call_id,
    )
    return Command(update={"prepared": PreparedDataset(frame, (*steps, record)), "messages": [message]})


@tool(parse_docstring=True)
def profile_dataset(runtime: ToolRuntime[AnalysisContext], columns: list[str] | None = None) -> str:
    """Check data types, missing values and distinct values before cleaning the working dataset.

    Args:
        columns: Exact column names to inspect. Omit to profile the first 25 columns.
    """
    frame = working_frame(runtime.context, runtime.state)
    fields = list(frame.columns[:25]) if columns is None else selected_columns(frame, columns)
    lines = [f"Working dataset: {len(frame):,} rows, {len(frame.columns):,} columns. Profiling {len(fields)} columns."]
    for name in fields:
        values = frame[name]
        missing = missing_mask(values)
        lines.append(f"- {name}: {values.dtype}, {missing.sum():,} missing or blank, {values[~missing].nunique():,} distinct")
    return "\n".join(lines)


@tool(parse_docstring=True)
def inspect_column_values(column_name: str, runtime: ToolRuntime[AnalysisContext], limit: int = 10) -> str:
    """List frequent values in a column to find exact labels such as countries or product categories.

    Args:
        column_name: Exact column name whose values to list.
        limit: Number of distinct values to show, from 1 through 25.
    """
    if not 1 <= limit <= 25:
        raise ToolException("limit must be between 1 and 25.")
    frame = working_frame(runtime.context, runtime.state)
    values = column(frame, column_name)
    missing = missing_mask(values)
    counts = values[~missing].value_counts()
    lines = [f"{column_name}: {len(counts):,} distinct nonblank values; {missing.sum():,} missing or blank."]
    lines.extend(f"- {value!r}: {count:,} rows" for value, count in counts.head(limit).items())
    return "\n".join(lines)


@tool(parse_docstring=True)
def filter_rows(column_name: str, operator: FilterOperator, value: Scalar | list[Scalar],
                runtime: ToolRuntime[AnalysisContext]) -> Command:
    """Keep working rows that match a user-requested condition, before later analysis calls.

    Text matching ignores case and surrounding whitespace. Ordered comparisons use numbers
    or a datetime column. Missing values are excluded. Repeated filters narrow the current table.

    Args:
        column_name: Exact CSV column to filter, for example a country or category column.
        operator: equals, not_equals, in (one of a list), contains (literal text), or an ordered comparison.
        value: A single comparison value, or a nonempty list for the in operator.
    """
    frame = working_frame(runtime.context, runtime.state)
    values = column(frame, column_name)
    if operator == "in" and (not isinstance(value, list) or not value):
        raise ToolException("The in operator needs a nonempty list of values.")
    if operator != "in" and isinstance(value, list):
        raise ToolException("Use a single value with this operator, or use in for a list.")
    requested = value if isinstance(value, list) else [value]
    valid = ~missing_mask(values)
    if operator == "contains":
        if not isinstance(value, str) or not value.strip():
            raise ToolException("The contains operator needs nonempty text.")
        matched = values.astype("string").str.casefold().str.contains(value.strip().casefold(), regex=False, na=False)
    else:
        if is_datetime64_any_dtype(values):
            operands = pd.to_datetime(pd.Series(requested, dtype="string"), format="ISO8601", utc=True, errors="coerce")
            values = pd.to_datetime(values, utc=True)
        elif operator in ORDER_OPERATORS or (is_numeric_dtype(values) and not is_bool_dtype(values)):
            operands = finite_numbers(pd.Series(requested))
            values = finite_numbers(values)
            if valid.any() and not values.notna().any():
                raise ToolException("This column has no comparable numbers. Convert a date column before an ordered date filter.")
        else:
            operands = pd.Series(requested, dtype="string").str.strip().str.casefold()
            values = values.astype("string").str.strip().str.casefold()
        if operands.isna().any():
            raise ToolException("The requested value is incompatible with the filter column type.")
        valid &= values.notna()
        if operator in ORDER_OPERATORS:
            matched = ORDER_OPERATORS[operator](values, operands.iloc[0])
        else:
            matched = values.isin(operands)
            if operator == "not_equals":
                matched = ~matched
    result = frame.loc[(valid & matched).fillna(False)].copy()
    detail = f"Applied filter {column_name} {operator} {value!r}."
    if result.empty and not frame.empty:
        labels = column(frame, column_name)
        available = ", ".join(repr(item) for item in labels[~missing_mask(labels)].value_counts().head(8).index)
        detail += (
            f" No rows matched. Common values before this filter: {available or '(none)'}."
            " If the intended label is clear, reset, reapply any required earlier preparation, and retry with that exact label."
        )
    parameters = {"column_name": column_name, "operator": operator, "value": value}
    return save_preparation(runtime, result, "filter_rows", parameters, detail)


@tool(parse_docstring=True)
def drop_missing(columns: list[str], runtime: ToolRuntime[AnalysisContext]) -> Command:
    """Remove working rows with null or blank values in any of the specified columns.

    Args:
        columns: Exact columns that must be present for this analysis.
    """
    frame = working_frame(runtime.context, runtime.state)
    fields = selected_columns(frame, columns)
    usable = pd.Series(True, index=frame.index)
    for name in fields:
        usable &= ~missing_mask(frame[name])
    result = frame.loc[usable].copy()
    detail = f"Removed {len(frame) - len(result):,} rows missing a value in {', '.join(fields)}."
    return save_preparation(runtime, result, "drop_missing", {"columns": fields}, detail)


@tool(parse_docstring=True)
def drop_duplicates(runtime: ToolRuntime[AnalysisContext], columns: list[str] | None = None) -> Command:
    """Remove duplicate working rows, keeping the first occurrence; use only if requested.

    Args:
        columns: Exact columns that define a duplicate. Omit to compare complete rows.
    """
    frame = working_frame(runtime.context, runtime.state)
    fields = None if columns is None else selected_columns(frame, columns)
    result = frame.drop_duplicates(subset=fields)
    detail = f"Removed {len(frame) - len(result):,} duplicate rows using {', '.join(fields) if fields else 'all columns'}."
    return save_preparation(runtime, result, "drop_duplicates", {"columns": fields}, detail)


@tool(parse_docstring=True)
def convert_column(column_name: str, data_type: Literal["number", "date", "text"],
                   runtime: ToolRuntime[AnalysisContext], date_format: str | None = None) -> Command:
    """Convert a working column for analysis. Unparseable numbers or dates become missing and are counted.

    Args:
        column_name: Exact column name to convert.
        data_type: number, date, or text (text conversion trims whitespace).
        date_format: Optional strptime date format such as %d/%m/%Y. Default is ISO-8601.
    """
    frame = working_frame(runtime.context, runtime.state)
    original = column(frame, column_name)
    missing = missing_mask(original)
    if data_type == "number":
        converted = finite_numbers(original)
    elif data_type == "date":
        converted = pd.to_datetime(original.astype("string"), format=date_format or "ISO8601", errors="coerce", utc=True)
    else:
        converted = original.astype("string").str.strip()
    converted = converted.mask(missing)
    newly_missing = int((converted.isna() & ~missing).sum())
    if data_type != "text" and (~missing).any() and not converted.notna().any():
        raise ToolException("No values could be converted. Check the requested type or date format.")
    detail = f"Converted {column_name} to {data_type}; {newly_missing:,} nonblank values could not be parsed."
    params = {"column_name": column_name, "data_type": data_type, "date_format": date_format}
    return save_preparation(runtime, frame.assign(**{column_name: converted}), "convert_column", params, detail)


@tool
def reset_dataset(runtime: ToolRuntime[AnalysisContext]) -> Command:
    """Restore the original data when the user wants to undo all preparation or start from all rows."""
    message = ToolMessage(
        content=f"Original dataset restored: {len(runtime.context.dataset):,} rows. All working transformations cleared.",
        name="reset_dataset", tool_call_id=runtime.tool_call_id,
    )
    return Command(update={"prepared": None, "messages": [message]})


PREPARATION_TOOLS = (filter_rows, drop_missing, drop_duplicates, convert_column, reset_dataset)

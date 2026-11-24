"""Shared dataset loading, types and column helpers."""

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import pandas as pd
from langchain_core.tools import ToolException

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "hdb_resale.csv"
CHART_COLOR = "#147d83"
DEFAULT_PREVIEW_ROWS = 10


@dataclass(frozen=True)
class AnalysisContext:
    dataset: pd.DataFrame
    source: str
    output_dir: Path


@dataclass(frozen=True)
class PreparedDataset:
    frame: pd.DataFrame
    steps: tuple[dict, ...]


def working_frame(context: AnalysisContext, state: dict) -> pd.DataFrame:
    prepared = state.get("prepared")
    return context.dataset if prepared is None else prepared.frame


def working_source(context: AnalysisContext, state: dict) -> str:
    prepared = state.get("prepared")
    if prepared is None:
        return context.source
    return f"{context.source} | Prepared data: {len(prepared.frame):,} of {len(context.dataset):,} original rows"


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


def finite_numbers(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    return values.replace([float("inf"), -float("inf")], float("nan"))


def numeric_column(df: pd.DataFrame, name: str) -> pd.Series:
    values = finite_numbers(column(df, name))
    if not values.notna().any():
        raise ToolException(f"Column {name!r} contains no valid numeric values.")
    return values

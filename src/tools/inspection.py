"""Tools for browsing and inspecting the active dataset."""

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import ToolException

from .dataset import AnalysisContext, DEFAULT_PREVIEW_ROWS, describe_dataset, load_dataset, working_frame

@tool
def dataset_overview(runtime: ToolRuntime[AnalysisContext] = None) -> str:
    """Show the available dataset columns, their types, and the row count."""
    df = working_frame(runtime.context, runtime.state) if runtime else load_dataset()
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
    dataset = working_frame(runtime.context, runtime.state)
    preview = dataset.head(limit)
    return (
        f"Dataset preview ready for display. Showing {len(preview):,} of {len(dataset):,} rows.",
        {"table": preview.to_dict(orient="split", index=False), "total_rows": len(dataset)},
    )

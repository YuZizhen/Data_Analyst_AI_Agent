"""Public tools for the data analyst agent."""

from .charts import average_by_group, distribution_report, formula_group_totals, group_distribution_report, group_report, trend_report
from .dataset import (
    DATA_PATH, DEFAULT_PREVIEW_ROWS, AnalysisContext, PreparedDataset, describe_dataset, load_dataset, working_frame,
)
from .importance import feature_importance_report
from .inspection import dataset_overview, dataset_preview
from .predictions import evaluate_prediction, predict_value
from .preprocessing import PREPARATION_TOOLS, inspect_column_values, profile_dataset
from .relationships import rank_relationships, relationship_report
from .rows import aggregate_column, count_rows, rank_rows

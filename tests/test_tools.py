"""Unit tests for data tools; do not require an API key."""

from src.tools import average_by_group, dataset_overview


def test_overview():
    result = dataset_overview.invoke({})
    assert "Rows: 6" in result
    assert "resale_price" in result


def test_average_by_group():
    result = average_by_group.invoke(
        {"group_column": "town", "value_column": "resale_price", "top_n": 3}
    )
    assert "BISHAN" in result
    assert "820000" in result


def test_invalid_column():
    result = average_by_group.invoke(
        {"group_column": "not_a_column", "value_column": "resale_price"}
    )
    assert "Invalid columns" in result

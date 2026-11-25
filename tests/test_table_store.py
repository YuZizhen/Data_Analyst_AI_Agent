from pathlib import Path

import pandas as pd

from src.table_store import TableStore
from src.tools import PreparedDataset
from src.tools.preprocessing import convert_frame_column, filter_frame


def test_saved_tables_and_replay(tmp_path, monkeypatch):
    store = TableStore(tmp_path / "t")
    original = store.save_original("events.csv", b"day,region\n2025-01-01, East \n2025-01-02,West\n")
    frame = store.read(original)
    params = {"column_name": "region", "operator": "equals", "value": "east"}
    clean = filter_frame(frame, **params)
    conversion = {"column_name": "day", "data_type": "date", "date_format": None}
    clean = convert_frame_column(clean, **conversion)
    steps = (
        {"tool": "filter_rows", "parameters": params, "rows_before": 2, "rows_after": 1, "detail": "Filtered."},
        {"tool": "convert_column", "parameters": conversion, "rows_before": 1, "rows_after": 1, "detail": "Converted."},
    )
    saved = store.save_prepared(original, PreparedDataset(clean, steps))
    reopened = TableStore(tmp_path / "t")
    catalog = reopened.list()
    assert set(catalog) == {original.key, saved.key}
    pd.testing.assert_frame_equal(reopened.read(saved), clean)
    assert len(reopened.read(original)) == 2
    assert reopened.save_prepared(original, PreparedDataset(clean, steps)).key == saved.key

    script, implementation = reopened.preparation_code(saved)
    assert "str.casefold" in implementation and "pd.to_datetime" in implementation
    monkeypatch.chdir(tmp_path)
    exec(compile(script, "replay.py", "exec"), {})
    output = Path(f"cleaned-{saved.key[:10]}.csv")
    assert pd.read_csv(output)["region"].tolist() == [" East "]

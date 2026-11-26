"""Local, immutable versions of uploaded and prepared datasets."""

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from inspect import getsource
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.tools.dataset import PreparedDataset, load_dataset
from src.tools.preprocessing import PREPARATION_FUNCTIONS

DATASET_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "private" / "datasets"
METADATA_KEY = b"data_analyst_table"


@dataclass(frozen=True)
class SavedTable:
    key: str
    name: str
    source_key: str
    checksum: str
    rows: int
    is_demo: bool = False
    steps: tuple[dict, ...] = ()

    @property
    def filter_only(self) -> bool:
        return bool(self.steps) and all(step["tool"] == "filter_rows" for step in self.steps)

    @property
    def export_name(self) -> str:
        prefix = "filtered" if self.filter_only else "cleaned"
        return f"{prefix}-{self.key[:10]}.csv"

    @property
    def label(self) -> str:
        name = "HDB 转售示例" if self.is_demo else self.name
        if self.steps:
            kind = "筛选版" if self.filter_only else "清洗版"
            return f"{name} · {kind} · {self.rows:,} 行 · {self.key[:6]}"
        return f"{name} · 原表 · {self.rows:,} 行"


class TableStore:
    def __init__(self, directory: Path):
        self.directory = directory

    def path(self, key: str) -> Path:
        return self.directory / f"{key}.parquet"

    def _metadata(self, path: Path) -> SavedTable:
        metadata = pq.read_schema(path).metadata or {}
        information = json.loads(metadata[METADATA_KEY])
        information["steps"] = tuple(information["steps"])
        entry = SavedTable(**information)
        if entry.key != path.stem:
            raise ValueError("Saved table metadata does not match its file name.")
        return entry

    def list(self) -> dict[str, SavedTable]:
        entries = {}
        for path in self.directory.glob("*.parquet"):
            try:
                entry = self._metadata(path)
            except (OSError, ValueError, KeyError, TypeError, pa.ArrowException):
                continue
            entries[entry.key] = entry
        return {key: entry for key, entry in entries.items() if entry.source_key in entries}

    def read(self, entry: SavedTable) -> pd.DataFrame:
        return pq.read_table(self.path(entry.key)).to_pandas()

    def _save(self, entry: SavedTable, frame: pd.DataFrame) -> SavedTable:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path(entry.key)
        if path.exists():
            return self._metadata(path)
        table = pa.Table.from_pandas(frame)
        metadata = dict(table.schema.metadata or {})
        metadata[METADATA_KEY] = json.dumps(asdict(entry), ensure_ascii=False).encode("utf-8")
        with NamedTemporaryFile(dir=self.directory, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            pq.write_table(table.replace_schema_metadata(metadata), temporary)
            try:
                temporary.replace(path)
            except PermissionError:
                if not path.exists():
                    raise
                return self._metadata(path)
        finally:
            temporary.unlink(missing_ok=True)
        return entry

    def save_original(self, name: str, data: bytes, *, is_demo: bool = False) -> SavedTable:
        checksum = sha256(data).hexdigest()
        key = sha256(f"original:{is_demo}:{name}:{checksum}".encode("utf-8")).hexdigest()[:24]
        path = self.path(key)
        if path.exists():
            return self._metadata(path)
        frame = load_dataset(BytesIO(data))
        if frame.empty:
            raise ValueError("The CSV has no data rows.")
        return self._save(SavedTable(key, name, key, checksum, len(frame), is_demo), frame)

    def save_prepared(self, original: SavedTable, prepared: PreparedDataset) -> SavedTable:
        recipe = [(step["tool"], step["parameters"]) for step in prepared.steps]
        key = sha256(json.dumps([original.key, recipe], sort_keys=True).encode("utf-8")).hexdigest()[:24]
        entry = SavedTable(
            key, original.name, original.key, original.checksum, len(prepared.frame), original.is_demo, prepared.steps,
        )
        return self._save(entry, prepared.frame)

    def preparation_code(self, entry: SavedTable) -> tuple[str, str]:
        actions = [PREPARATION_FUNCTIONS[step["tool"]] for step in entry.steps]
        functions = list(dict.fromkeys(actions))
        imports = "\n".join(f"    {function.__name__}," for function in functions)
        lines = [
            "from pathlib import Path",
            "import pandas as pd",
            "from src.tools.preprocessing import (",
            imports,
            ")",
            "",
            f"frame = pd.read_parquet(Path({str(self.path(entry.source_key))!r}))",
        ]
        for function, step in zip(actions, entry.steps):
            lines.append(f"frame = {function.__name__}(frame, **{step['parameters']!r})")
        lines.extend([
            "", f'frame.to_csv({entry.export_name!r}, index=False, encoding="utf-8-sig", mode="x")', "",
        ])
        code = "\n\n".join(getsource(function) for function in functions)
        return "\n".join(lines), code

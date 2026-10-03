"""Freeze release input, reference and code inventories with SHA-256 hashes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]


def file_record(path: Path, root: Path) -> dict:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    result = {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": digest}
    if path.suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        result.update(rows=parquet.metadata.num_rows, columns=parquet.schema_arrow.names)
    return result


def inventory(root: Path, directories: list[str], suffixes: set[str] | None = None) -> list[dict]:
    files = []
    for directory in directories:
        for path in (root / directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and (suffixes is None or path.suffix in suffixes):
                files.append(path)
    return [file_record(path, root) for path in sorted(files)]


def build(root: Path) -> dict:
    target = root / "manifests"
    target.mkdir(exist_ok=True)
    definitions = {
        "input_manifest.json": ("frozen_inputs_v1", ["data", "configs"], None),
        "reference_manifest.json": ("frozen_reference_v1", ["results/reference"], None),
        "code_manifest.json": ("replication_code_v1", ["code"], {".py", ".json", ".md"}),
    }
    counts = {}
    for name, (version, directories, suffixes) in definitions.items():
        files = inventory(root, directories, suffixes)
        (target / name).write_text(json.dumps({"schema_version": version, "algorithm": "sha256", "files": files},
                                             ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        counts[name] = len(files)
    return counts


if __name__ == "__main__":
    print(json.dumps(build(ROOT), indent=2))

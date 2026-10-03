"""Export the frozen candidate-retrieval rule snapshot into compact files."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath

import pandas as pd


SOURCE_DIR = Path(
    "data/annotation_gold/annotation_sampling/random_3000_2017_2025/rule_data/active"
)
SNAPSHOT = {
    "candidate_recall_terms.csv": (
        32706, "29b2ffc993451bd6b74dff6f47f282502d20eec71eda2e74f3fa84874ecda51c"
    ),
    "candidate_recall_patterns.csv": (
        35, "46f047dabe6b3f9c0260b381c60aa8c1015a9bfae6a6b1b468e21b076ad73bfd"
    ),
    "negative_patterns.csv": (
        10, "eda28c75576f3307aeb7213b91a9e446aae971b34b08cd16f4f79a15793cca08"
    ),
    "label_rules.csv": (
        9, "e133a36ffe4a9bede08a793eb1f5db4d4c2825c4445c1459318d87179b0f78f2"
    ),
    "label_definitions.csv": (
        7, "018ff93aee2774c2def1f43e58824fd1dc975b7de8799f08100ba3ae8bac86d2"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_source_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Represent logical source tables as a group and filename, without paths."""
    if "source_table" not in frame:
        return frame.copy()
    frame = frame.copy()
    position = frame.columns.get_loc("source_table")
    values = frame.pop("source_table")
    frame.insert(position, "source_group", values.map(lambda x: PurePosixPath(x).parent.name))
    frame.insert(position + 1, "source_table_name", values.map(lambda x: PurePosixPath(x).name))
    return frame


def sorted_counts(values: pd.Series) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def prepare_retrieval(source_root: Path, output_dir: Path) -> dict:
    """Verify the five source files, preserve rule rows, and export the snapshot."""
    loaded: dict[str, tuple[Path, pd.DataFrame]] = {}
    for filename, (expected_rows, expected_sha) in SNAPSHOT.items():
        source = source_root / SOURCE_DIR / filename
        actual_sha = sha256(source)
        if actual_sha != expected_sha:
            raise ValueError(f"Source snapshot hash differs: {SOURCE_DIR.as_posix()}/{filename}")
        frame = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        if len(frame) != expected_rows:
            raise ValueError(f"Expected {expected_rows} rows in {filename}, found {len(frame)}")
        key = "rule_id" if "rule_id" in frame else "pattern_id" if "pattern_id" in frame else "label_id"
        if not frame[key].is_unique or frame[key].eq("").any():
            raise ValueError(f"Rule identifier is empty or duplicated in {filename}")
        loaded[filename] = source, frame

    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for filename, (source, original) in loaded.items():
        frame = project_source_table(original)
        destination_name = (
            "candidate_recall_terms.parquet" if filename == "candidate_recall_terms.csv" else filename
        )
        destination = output_dir / destination_name
        if destination.suffix == ".parquet":
            frame.to_parquet(destination, compression="zstd", index=False)
            readback = pd.read_parquet(destination)
        else:
            frame.to_csv(destination, index=False, encoding="utf-8", lineterminator="\n")
            readback = pd.read_csv(destination, dtype=str, keep_default_na=False)
        pd.testing.assert_frame_equal(frame, readback, check_dtype=False)
        counts = {}
        for field in ("candidate_labels", "recall_tier", "confidence", "match_type", "rule_type"):
            if field in original:
                counts[field] = sorted_counts(original[field])
        record = {
            "source_relative_path": (SOURCE_DIR / filename).as_posix(),
            "source_sha256": sha256(source),
            "source_bytes": source.stat().st_size,
            "source_columns": list(original.columns),
            "destination_file": destination_name,
            "destination_sha256": sha256(destination),
            "destination_bytes": destination.stat().st_size,
            "destination_columns": list(frame.columns),
            "rows": len(frame),
            "counts": counts,
        }
        if "source_table" in original:
            record["source_table_mapping"] = [
                {
                    "source_table": value,
                    "source_group": PurePosixPath(value).parent.name,
                    "source_table_name": PurePosixPath(value).name,
                }
                for value in sorted(original["source_table"].unique())
            ]
        records.append(record)

    terms = loaded["candidate_recall_terms.csv"][1]
    patterns = loaded["candidate_recall_patterns.csv"][1]
    manifest = {
        "schema": "retrieval_rule_snapshot_v1",
        "source_snapshot": "random_3000_2017_2025/rule_data/active",
        "package_role": "supplementary measurement-method rules",
        "source_files": records,
        "term_counts": {
            "rule_rows": len(terms),
            "distinct_terms": int(terms["term"].nunique()),
            "distinct_normalized_terms": int(terms["normalized_term"].nunique()),
        },
        "supplementary_l5_context_rule_ids": patterns.loc[
            patterns["rule_id"].str.startswith("RECALL_L5_CONTEXT_"), "rule_id"
        ].tolist(),
        "projection": {
            "source_table": "source_group plus source_table_name; mapping retained per source file",
            "scientific_values": "All remaining values, rule rows, notes, term text and regex strings preserved.",
        },
        "round_trip_verified": True,
    }
    (output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Root of the frozen upstream source tree")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/measurement/retrieval",
        help="Directory receiving the compact rule files and source manifest",
    )
    args = parser.parse_args()
    manifest = prepare_retrieval(args.source_root, args.output_dir)
    print(json.dumps({
        "files": len(manifest["source_files"]),
        "rows": {entry["destination_file"]: entry["rows"] for entry in manifest["source_files"]},
        "round_trip_verified": manifest["round_trip_verified"],
    }, indent=2))


if __name__ == "__main__":
    main()

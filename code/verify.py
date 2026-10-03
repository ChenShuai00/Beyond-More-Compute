"""Verify frozen release files, samples and regenerated scientific results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FLOW = {
    "initial_records": 114618, "collected_corpus": 113822, "metadata_text_eligible": 107872,
    "l1_positive": 54948, "hardware_eligible": 49090, "normalized_gpu": 40956,
    "complete_hardware": 34483, "performance_imputed_complete_hardware": 3524,
    "analysis_frame": 29591, "baseline_api_regression": 29487, "ksc_regression": 29239,
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_inputs(repo_root: Path) -> dict:
    manifest = repo_root / "manifests/input_manifest.json"
    checked = 0
    definitions = [(manifest, ["data", "configs"], None),
                   (repo_root / "manifests/reference_manifest.json", ["results/reference"], None),
                   (repo_root / "manifests/code_manifest.json", ["code"], {".py", ".json", ".md"})]
    for manifest, directories, suffixes in definitions:
        records = json.loads(manifest.read_text(encoding="utf-8"))["files"]
        expected = {item["path"] for item in records}
        actual = {path.relative_to(repo_root).as_posix() for directory in directories
                  for path in (repo_root / directory).rglob("*") if path.is_file()
                  and "__pycache__" not in path.parts and (suffixes is None or path.suffix in suffixes)}
        if expected != actual or len(expected) != len(records):
            raise AssertionError(f"Frozen inventory changed: {manifest.name}")
        for item in records:
            path = repo_root / item["path"]
            if not path.is_file() or sha256(path) != item["sha256"]:
                raise AssertionError(f"Input SHA-256 changed: {item['path']}")
            checked += 1
    corpus = pd.read_parquet(repo_root / "data/corpus/paper_index.parquet")
    frame = pd.read_parquet(repo_root / "data/analysis_ready/resource_descriptive.parquet")
    membership = pd.read_parquet(repo_root / "data/corpus/sample_membership.parquet")
    if len(corpus) != 113822 or len(frame) != 107872 or len(membership) != 114618:
        raise AssertionError("Corpus sample sizes changed")
    if any(x.paper_id.duplicated().any() for x in (corpus, frame, membership)):
        raise AssertionError("Duplicate corpus paper IDs")
    if set(corpus.paper_id) != set(membership.loc[membership.corpus_included.eq(True), "paper_id"]):
        raise AssertionError("Collected-corpus ID membership changed")
    if set(frame.paper_id) != set(membership.loc[membership.scope_metadata_text_eligible.eq(True), "paper_id"]):
        raise AssertionError("Metadata/text eligible ID membership changed")
    if not set(frame.paper_id).issubset(set(corpus.paper_id)):
        raise AssertionError("Resource sample is outside the corpus")
    return {"passed": True, "frozen_files_checked": checked,
            "corpus_rows": len(corpus), "resource_rows": len(frame), "membership_rows": len(membership)}


def reproduce_corpus_tables(repo_root: Path, output_dir: Path) -> dict:
    index = pd.read_parquet(repo_root / "data/corpus/paper_index.parquet")
    frame = pd.read_parquet(repo_root / "data/analysis_ready/resource_descriptive.parquet")
    membership = pd.read_parquet(repo_root / "data/corpus/sample_membership.parquet")
    baseline = pd.read_parquet(repo_root / "data/analysis_ready/baseline_membership.parquet")
    ksc = pd.read_parquet(repo_root / "data/analysis_ready/ksc_membership.parquet")
    hardware = frame.resource_L1.eq(1) & frame.resource_L2.eq(0) & frame.resource_L3.eq(0)
    complete = hardware & frame.compute_gpu_scale_strict_main_sample.eq(True)
    counts = {
        "initial_records": len(membership), "collected_corpus": len(index), "metadata_text_eligible": len(frame),
        "l1_positive": int(frame.resource_L1.eq(1).sum()), "hardware_eligible": int(hardware.sum()),
        "normalized_gpu": int((membership.hardware_eligible.eq(True) & membership.has_normalized_gpu.eq(True)).sum()),
        "complete_hardware": int(complete.sum()),
        "performance_imputed_complete_hardware": int((complete & frame.compute_performance_imputed.eq(True)).sum()),
        "analysis_frame": len(baseline), "baseline_api_regression": int(baseline.included_impact.sum()),
        "ksc_regression": int(ksc.included_venue_year_topic_impact.sum()),
    }
    if counts != EXPECTED_FLOW:
        raise AssertionError(f"Sample flow changed: {counts}")
    directory = output_dir / "tables"
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(counts.items(), columns=["stage", "papers"]).to_csv(directory / "sample_flow.csv", index=False)
    paper_counts = (index.groupby(["venue", "year"]).size().unstack(fill_value=0)
                    .reindex(columns=range(2020, 2026), fill_value=0).sort_index())
    paper_counts["total"] = paper_counts.sum(axis=1)
    paper_counts.loc["Total"] = paper_counts.sum(axis=0)
    paper_counts.to_csv(directory / "conference_year_counts.csv", index_label="venue")
    labels = pd.DataFrame([{"label": label, "papers": int(frame[f"resource_{label}"].eq(1).sum()),
                            "denominator": len(frame)} for label in ["L1", "L2", "L3", "L4"]])
    labels.to_csv(directory / "resource_access_modes.csv", index=False)
    return {"passed": True, "sample_counts": counts, "findings_papers": int(index.scope_is_findings.eq(True).sum())}


def _compare_csv(actual: Path, reference: Path, keys: list[str], tolerance: float = 1e-10) -> dict:
    left = pd.read_csv(actual).sort_values(keys).reset_index(drop=True)
    right = pd.read_csv(reference).sort_values(keys).reset_index(drop=True)
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        raise AssertionError(f"Table schema changed: {actual.name}")
    for key in keys:
        if left[key].astype(str).tolist() != right[key].astype(str).tolist():
            raise AssertionError(f"Table keys changed: {actual.name}")
    for column in left:
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(right[column]):
            if not np.allclose(left[column], right[column], atol=tolerance, rtol=1e-9, equal_nan=True):
                raise AssertionError(f"Table values changed: {actual.name}/{column}")
        elif left[column].fillna("").astype(str).tolist() != right[column].fillna("").astype(str).tolist():
            raise AssertionError(f"Table values changed: {actual.name}/{column}")
    return {"table": actual.name, "rows": len(left), "passed": True}


def verify_outputs(repo_root: Path, output_dir: Path) -> dict:
    from analysis.check_reference import check_reference
    from citations import verify_citations

    regression = check_reference(output_dir, repo_root / "results/reference")
    tables = []
    for name, keys in [("conference_year_counts.csv", ["venue"]), ("resource_access_modes.csv", ["label"]),
                       ("sample_flow.csv", ["stage"])]:
        tables.append(_compare_csv(output_dir / "tables" / name, repo_root / "results/reference/tables" / name, keys))
    for name, keys in [("annual_hardware_api_configurations.csv", ["year", "configuration"]),
                       ("annual_gpu_hardware_configuration_quantiles.csv", ["year", "variable"])]:
        tables.append(_compare_csv(output_dir / "figure_data" / name,
                                   repo_root / "results/reference/figure_data" / name, keys))
    measurement_path = output_dir / "measurement/validation.json"
    measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
    if measurement.get("status") != "pass":
        raise AssertionError("Measurement validation failed")
    figures_path = output_dir / "figures/qa/figure_validation.json"
    figures = json.loads(figures_path.read_text(encoding="utf-8"))
    if not figures.get("passed", False):
        raise AssertionError("Figure validation failed")
    citations = verify_citations(repo_root)
    if not citations.get("passed", False):
        raise AssertionError("Citation measurement validation failed")
    report = {"passed": True, "regression": regression, "data_summary_tables": tables,
              "measurement": measurement, "citations": citations, "figures": figures}
    (output_dir / "verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/reproduced")
    args = parser.parse_args()
    report = {"inputs": verify_inputs(ROOT), "outputs": verify_outputs(ROOT, args.output_dir.resolve())}
    print(json.dumps({"passed": True, "frozen_files_checked": report["inputs"]["frozen_files_checked"],
                      "reference_comparisons": len(report["outputs"]["regression"]["reference_comparisons"])}, indent=2))

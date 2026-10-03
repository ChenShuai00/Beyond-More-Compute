"""Collect the fixed citation-impact reference tables from the source snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "outputs/analysis/gpu_api_scientific_creativity_v14_20260914/current"
KSC = "outputs/analysis/gpu_api_scientific_impact_ksc_v14_20260917/current"
TABLES = {
    "baseline_regression_impact_table.csv": BASELINE,
    "baseline_coefficients.csv": BASELINE,
    "baseline_main_effects.csv": BASELINE,
    "model_metrics.csv": BASELINE,
    "model_comparisons.csv": BASELINE,
    "interaction_coefficients.csv": BASELINE,
    "appendix_interaction_impact_table.csv": BASELINE,
    "appendix_interaction_multiple_testing_table.csv": BASELINE,
    "table4_impact_ksc_interactions.csv": KSC,
    "table4_interaction_source.csv": KSC,
    "linear_coefficients.csv": KSC,
    "linear_model_metrics.csv": KSC,
    "linear_model_comparisons.csv": KSC,
    "interaction_results.csv": KSC,
    "appendix_full_model_details.csv": KSC,
    "conditional_associations.csv": KSC,
    "key_marginal_associations.csv": KSC,
    "self_exclusion_audit.csv": KSC,
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export(source_root: Path, repo_root: Path = ROOT) -> dict:
    result = []
    directory = repo_root / "results/reference/tables"
    directory.mkdir(parents=True, exist_ok=True)
    for name, base in TABLES.items():
        relative = f"{base}/tables/{name}"
        source = source_root / relative
        frame = pd.read_csv(source)
        if "family" in frame:
            frame = frame.loc[frame.family.eq("impact")].copy()
        destination = directory / name
        frame.to_csv(destination, index=False)
        result.append({"source_root_alias": "compute_snapshot", "source": relative,
                       "source_sha256": digest(source), "destination": destination.relative_to(repo_root).as_posix(),
                       "sha256": digest(destination), "rows": len(frame), "columns": list(frame),
                       "operation": "citation_impact_rows"})
    path = repo_root / "manifests/reference_sources.json"
    path.write_text(json.dumps({"schema_version": "reference_sources_v1", "files": result}, indent=2) + "\n", encoding="utf-8")
    return {"tables": len(result)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.source_root.resolve())))

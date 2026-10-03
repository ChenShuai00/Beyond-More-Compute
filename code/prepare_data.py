"""Export the frozen corpus and impact-analysis inputs from a source snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
INTEGRATED = "data/derived/integrated_papers/conference_papers_2020_2025_20260918/current"
BASELINE = "outputs/analysis/gpu_api_scientific_creativity_v14_20260914/current"
KSC = "outputs/analysis/gpu_api_scientific_impact_ksc_v14_20260917/current"
SCOPE = "outputs/analysis/corpus_main_findings_2020_2025_v1_20260906/current"

DESCRIPTIVE_COLUMNS = [
    "paper_id", "year", "venue", "resource_L1", "resource_L2", "resource_L3", "resource_L4",
    "compute_gpu_scale_strict_main_sample", "compute_strict_gpu", "compute_performance_imputed",
    "compute_selected_gpu_count", "compute_selected_per_gpu_capability", "compute_selected_vram_gb",
    "compute_benchmark_gpu_name", "compute_selected_config_id", "compute_selected_config_total_compute_capability",
    "compute_has_count_conflict", "compute_has_missing_count_mention",
    "compute_performance_original_flops", "compute_performance_donor_model_count",
]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def merge_analysis_definitions(definitions: dict, repo_root: Path) -> dict:
    """Add the published core-variable definitions while retaining source metadata."""
    config = json.loads((repo_root / "configs/analysis_variables.json").read_text(encoding="utf-8"))
    for field, variable in config["fields"].items():
        entry = definitions.setdefault(field, {"field": field})
        entry.setdefault("dtype", variable["dtype"])
        entry.update({key: variable[key] for key in ("description", "unit", "transform")})
    return definitions


def export(source_root: Path, repo_root: Path = ROOT) -> dict:
    records: list[dict] = []

    def write(source: str, target: str, columns: list[str] | None = None,
              exclude: list[str] | None = None) -> pd.DataFrame:
        source_path = source_root / source
        frame = pd.read_parquet(source_path, columns=columns)
        if exclude:
            frame = frame.drop(columns=[x for x in exclude if x in frame])
        target_path = repo_root / target
        target_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target_path, index=False, compression="zstd")
        records.append({"source_root_alias": "compute_snapshot", "source": source,
                        "source_sha256": sha256(source_path), "destination": target,
                        "sha256": sha256(target_path), "rows": len(frame), "columns": list(frame),
                        "operation": "column_projection_and_lossless_parquet_export"})
        return frame

    write(BASELINE + "/data/analysis_dataset.parquet", "data/analysis_ready/baseline_impact.parquet",
          exclude=["y_semantic", "y_new_word", "y_new_phrase", "y_new_word_comb", "y_new_phrase_comb"])
    baseline_members = write(BASELINE + "/data/sample_membership.parquet",
                             "data/analysis_ready/baseline_membership.parquet",
                             exclude=["eligible_novelty", "singleton_iteration_novelty", "included_novelty"])
    write(KSC + "/data/analysis_dataset.parquet", "data/analysis_ready/ksc_impact.parquet")
    ksc_members = write(KSC + "/data/sample_membership.parquet", "data/analysis_ready/ksc_membership.parquet")
    index = write(INTEGRATED + "/data/paper_index.parquet", "data/corpus/paper_index.parquet", columns=[
        "paper_id", "title", "year", "venue", "openalex_work_id", "work_id_norm", "openalex_matched",
        "focal_publication_date", "scope_is_findings", "scope_source_track", "scope_track_missing",
        "resource_classification_status", "resource_text_observable", "scope_openalex_matched",
        "scope_metadata_text_eligible", "scope_version"])
    scope = write(INTEGRATED + "/data/paper_scope.parquet", "data/corpus/paper_scope.parquet")
    write(INTEGRATED + "/audit/paper_analysis_sample_membership.parquet",
          "data/corpus/analysis_frame_membership.parquet")
    descriptive = write(INTEGRATED + "/data/paper_metadata_text_eligible_wide.parquet",
                        "data/analysis_ready/resource_descriptive.parquet", columns=DESCRIPTIVE_COLUMNS)
    initial = write(SCOPE + "/paper_scope_audit.parquet", "data/corpus/initial_scope_audit.parquet", columns=[
        "paper_id", "venue", "year", "source_track", "findings_evidence", "source_track_missing",
        "exclude_explicit_nonmain_track", "exclude_extended_abstract", "exclude_naacl_industry",
        "exclude_other_nonresearch", "exclude_duplicate_artifact", "cohort_retained", "exclusion_reason",
        "metadata_and_text_eligible", "openalex_matched", "has_normalized_gpu"])
    if not initial.paper_id.is_unique:
        raise ValueError("Initial paper IDs must be unique")
    if not index.paper_id.is_unique or not descriptive.paper_id.is_unique:
        raise ValueError("Corpus paper IDs must be unique")

    membership = initial[["paper_id", "venue", "year", "cohort_retained", "exclusion_reason",
                          "has_normalized_gpu"]].rename(columns={"cohort_retained": "corpus_included"})
    membership = membership.merge(scope[["paper_id", "scope_metadata_text_eligible", "scope_is_findings"]],
                                  on="paper_id", how="left", validate="one_to_one")
    labels = descriptive[["paper_id", "resource_L1", "resource_L2", "resource_L3", "resource_L4",
                          "compute_gpu_scale_strict_main_sample", "compute_performance_imputed"]].copy()
    labels["hardware_eligible"] = (labels.resource_L1.eq(1) & labels.resource_L2.eq(0) & labels.resource_L3.eq(0))
    labels["complete_hardware"] = labels.hardware_eligible & labels.compute_gpu_scale_strict_main_sample.eq(True)
    membership = membership.merge(labels, on="paper_id", how="left", validate="one_to_one")
    membership["analysis_frame_included"] = membership.paper_id.isin(baseline_members.paper_id)
    membership["baseline_included"] = membership.paper_id.isin(
        baseline_members.loc[baseline_members.included_impact, "paper_id"])
    membership["ksc_included"] = membership.paper_id.isin(
        ksc_members.loc[ksc_members.included_venue_year_topic_impact, "paper_id"])
    target = repo_root / "data/corpus/sample_membership.parquet"
    membership.to_parquet(target, index=False, compression="zstd")
    records.append({"destination": "data/corpus/sample_membership.parquet", "sha256": sha256(target),
                    "source_root_alias": "compute_snapshot", "operation": "id_level_membership_join",
                    "rows": len(membership), "columns": list(membership),
                    "derived_from": ["data/corpus/initial_scope_audit.parquet", "data/corpus/paper_scope.parquet",
                                     "data/analysis_ready/resource_descriptive.parquet",
                                     "data/analysis_ready/baseline_membership.parquet",
                                     "data/analysis_ready/ksc_membership.parquet"]})

    paper_counts = (index.groupby(["venue", "year"]).size().unstack(fill_value=0)
                    .reindex(columns=range(2020, 2026), fill_value=0).sort_index())
    paper_counts["total"] = paper_counts.sum(axis=1)
    paper_counts.loc["Total"] = paper_counts.sum(axis=0)
    reference = repo_root / "results/reference/tables"
    reference.mkdir(parents=True, exist_ok=True)
    paper_counts.to_csv(reference / "conference_year_counts.csv", index_label="venue")
    resource_counts = pd.DataFrame([{"label": label, "papers": int(descriptive[f"resource_{label}"].eq(1).sum()),
                                    "denominator": len(descriptive)} for label in ["L1", "L2", "L3", "L4"]])
    resource_counts.to_csv(reference / "resource_access_modes.csv", index=False)
    flow = pd.DataFrame([
        ("initial_records", len(initial)), ("collected_corpus", len(index)),
        ("metadata_text_eligible", len(descriptive)), ("l1_positive", int(descriptive.resource_L1.eq(1).sum())),
        ("hardware_eligible", int(labels.hardware_eligible.sum())),
        ("normalized_gpu", int((membership.hardware_eligible.eq(True) & membership.has_normalized_gpu.eq(True)).sum())),
        ("complete_hardware", int(labels.complete_hardware.sum())),
        ("performance_imputed_complete_hardware", int((labels.complete_hardware & labels.compute_performance_imputed.eq(True)).sum())),
        ("analysis_frame", len(baseline_members)), ("baseline_api_regression", int(baseline_members.included_impact.sum())),
        ("ksc_regression", int(ksc_members.included_venue_year_topic_impact.sum())),
    ], columns=["stage", "papers"])
    flow.to_csv(reference / "sample_flow.csv", index=False)
    expected = [114618, 113822, 107872, 54948, 49090, 40956, 34483, 3524, 29591, 29487, 29239]
    if flow.papers.tolist() != expected:
        raise ValueError(f"Frozen sample counts changed: {flow.to_dict('records')}")

    registry = json.loads((source_root / INTEGRATED / "field_registry.json").read_text(encoding="utf-8-sig"))
    if isinstance(registry, dict):
        entries = registry.get("fields", registry.get("columns", {}))
    else:
        entries = registry
    # Select definitions by published field name; the dictionary stores scientific descriptions.
    selected = set(DESCRIPTIVE_COLUMNS) | set(index.columns) | set(scope.columns)
    if isinstance(entries, list):
        entries = {str(x.get("name", x.get("field", x.get("column", "")))): x for x in entries}
    definitions = {}
    if isinstance(entries, dict):
        for key in selected:
            item = entries.get(key)
            if isinstance(item, dict):
                definitions[key] = {k: v for k, v in item.items()
                                    if k in {"name", "field", "column", "dtype", "type", "description",
                                             "definition", "unit", "domain", "nullable"}}
    merge_analysis_definitions(definitions, repo_root)
    catalog = {"schema_version": "published_field_dictionary_v1", "fields": definitions,
               "datasets": {x["destination"]: {"rows": x["rows"], "columns": x["columns"]} for x in records}}
    (repo_root / "data/data_dictionary.json").write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_dir = repo_root / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    result = {"schema_version": "data_projection_sources_v1", "source_root_alias": "compute_snapshot", "files": records,
              "sample_counts": dict(flow.itertuples(index=False, name=None))}
    (manifest_dir / "data_sources.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--repo-root", default=ROOT, type=Path)
    args = parser.parse_args()
    result = export(args.source_root.resolve(), args.repo_root.resolve())
    print(json.dumps({"exported_datasets": len(result["files"]), "sample_counts": result["sample_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

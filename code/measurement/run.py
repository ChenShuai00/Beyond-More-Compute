"""Reproduce measurement results from compact frozen labels and predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .gpu_metrics import evaluate_mode, f1_from_counts


RESOURCE_LABELS = ["L1_INTERNAL", "L2_CLOUD", "L3_PUBLIC_HPC", "L4_EXTERNAL_API"]
GPU_METRIC_KEYS = ["rows", "successful_requests", "failed_requests", "valid_json_and_schema", "schema_valid_rate", "record_detection", "hardware_model_normalized", "hardware_model_strict_surface", "quantity", "joint_normalized", "joint_strict_surface"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def assert_matching(observed: Any, reference: Any, context: str, tolerance: float = 1e-12) -> None:
    if isinstance(reference, dict):
        for key, value in reference.items():
            if key not in observed:
                raise ValueError(f"{context}: missing field {key}")
            assert_matching(observed[key], value, f"{context}.{key}", tolerance)
    elif isinstance(reference, (int, float)) and not isinstance(reference, bool):
        if not math.isclose(float(observed), float(reference), rel_tol=tolerance, abs_tol=tolerance):
            raise ValueError(f"{context}: observed {observed}, reference {reference}")
    elif observed != reference:
        raise ValueError(f"{context}: observed {observed}, reference {reference}")


def unique_index(rows: list[dict[str, Any]], key: str, context: str) -> dict[str, dict[str, Any]]:
    index = {str(row[key]): row for row in rows}
    if len(index) != len(rows) or "None" in index or "" in index:
        raise ValueError(f"{context}: IDs must be unique and nonempty")
    return index


def score_classification(gold_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold = unique_index(gold_rows, "gold_id", "classification gold")
    predictions = unique_index(prediction_rows, "gold_id", "classification predictions")
    if set(gold) != set(predictions):
        raise ValueError("Classification gold and prediction ID sets differ")
    per_label = {}
    exact = 0
    for label in RESOURCE_LABELS:
        tp = fp = fn = tn = 0
        for window_id, row in gold.items():
            predicted = predictions[window_id]
            if "paper_id" in predicted and predicted["paper_id"] != row["paper_id"]:
                raise ValueError(f"Paper identity differs for {window_id}")
            actual_positive = label in row["labels"]
            predicted_positive = label in predicted["labels"]
            if "scores" in predicted and "thresholds" in predicted:
                threshold_positive = predicted["scores"][label] >= predicted["thresholds"][label]
                if threshold_positive != predicted_positive:
                    raise ValueError(f"Saved label and probability threshold differ for {window_id}, {label}")
            tp += int(actual_positive and predicted_positive)
            fp += int(not actual_positive and predicted_positive)
            fn += int(actual_positive and not predicted_positive)
            tn += int(not actual_positive and not predicted_positive)
        per_label[label] = {**f1_from_counts(tp, fp, fn), "tn": tn, "support": tp + fn, "predicted_positive": tp + fp}
    for window_id, row in gold.items():
        actual = set(row["labels"]) & set(RESOURCE_LABELS)
        predicted = set(predictions[window_id]["labels"]) & set(RESOURCE_LABELS)
        exact += int(actual == predicted)
    totals = {key: sum(row[key] for row in per_label.values()) for key in ["tp", "fp", "fn"]}
    return {"label_scope": RESOURCE_LABELS, "rows": len(gold), "papers": len({row["paper_id"] for row in gold.values()}), "exact_match_l1_l4": exact / len(gold), "micro": f1_from_counts(**totals), "macro_f1": sum(row["f1"] for row in per_label.values()) / len(RESOURCE_LABELS), "per_label": per_label}


def validate_ksc(data_dir: Path, contract: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    mapping = pd.read_parquet(data_dir / "focal_paper_ksc_mapping.parquet")
    topic = pd.read_parquet(data_dir / "topic_year_ksc.parquet")
    robust = pd.read_parquet(data_dir / "topic_year_ksc_robustness.parquet")
    keys = ["primary_topic_id", "focal_year"]
    if not mapping.paper_id.is_unique or topic.duplicated(keys).any() or robust.duplicated(keys + ["minimum_historical_papers"]).any():
        raise ValueError("KSC mapping or topic-year keys are not unique")
    primary = int(contract["quality"]["primary_min_historical_papers"])
    thresholds = sorted([primary, *contract["quality"]["robustness_min_historical_papers"]])
    if set(robust.minimum_historical_papers) != set(thresholds):
        raise ValueError("KSC robustness thresholds differ from the contract")
    if len(robust) != len(topic) * len(thresholds):
        raise ValueError("KSC robustness table does not cover every threshold and topic-year")
    if not (topic.window_start_year == topic.focal_year - 5).all() or not (topic.window_end_year == topic.focal_year - 1).all():
        raise ValueError("KSC topic windows do not match the five-year prior window")
    joined = mapping.merge(topic, on=keys, validate="many_to_one", suffixes=("_mapping", "_topic"), indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("KSC focal records lack a matching topic-year")
    expected_count = joined.eligible_historical_paper_count - joined.own_h_p_subtracted.astype(int)
    if not expected_count.eq(joined.eligible_historical_paper_count_after_self_exclusion).all():
        raise ValueError("KSC self-exclusion counts differ")
    if not joined.base_eligible_history_count.eq(joined.eligible_historical_paper_count).all():
        raise ValueError("KSC base history counts differ")
    own_h = joined.own_h_p.where(joined.own_h_p_subtracted, 0.0)
    expected_sum = joined.h_p_sum - own_h
    expected_mean = expected_sum / expected_count
    checks = []
    for threshold in thresholds:
        col = "ksc_mapping" if threshold == primary else f"ksc_min_{threshold}"
        expected = expected_mean.where(expected_count >= threshold)
        values = joined[col]
        if not values.isna().eq(expected.isna()).all() or not np.allclose(values.fillna(0), expected.fillna(0), rtol=1e-10, atol=1e-10):
            raise ValueError(f"KSC focal mapping arithmetic differs at threshold {threshold}")
        subset = robust.loc[robust.minimum_historical_papers.eq(threshold)].merge(topic, on=keys, suffixes=("_robust", "_topic"), validate="one_to_one")
        available = subset.eligible_historical_paper_count_topic >= threshold
        expected_topic = (subset.h_p_sum / subset.eligible_historical_paper_count_topic).where(available)
        if not subset.available.eq(available).all() or not subset.ksc_robust.isna().eq(expected_topic.isna()).all() or not np.allclose(subset.ksc_robust.fillna(0), expected_topic.fillna(0), rtol=1e-10, atol=1e-10):
            raise ValueError(f"KSC topic threshold arithmetic differs at threshold {threshold}")
        checks.append({"minimum_historical_papers": threshold, "topic_year_cells": len(subset), "available_cells": int(available.sum()), "focal_papers": len(joined), "available_focal_papers": int(values.notna().sum())})
    expected_topic_primary = (topic.h_p_sum / topic.eligible_historical_paper_count).where(topic.eligible_historical_paper_count >= primary)
    if not np.allclose(topic.ksc.fillna(0), expected_topic_primary.fillna(0), rtol=1e-10, atol=1e-10) or not topic.ksc.isna().eq(expected_topic_primary.isna()).all():
        raise ValueError("KSC primary topic-year arithmetic differs")
    if not mapping.ksc.dropna().between(contract["quality"]["h_lower_bound"], contract["quality"]["h_upper_bound"]).all():
        raise ValueError("KSC values fall outside the contract bounds")
    write_csv(output_dir / "threshold_availability.csv", checks)
    mapping.groupby("focal_year").agg(papers=("paper_id", "size"), available=("ksc", "count"), mean=("ksc", "mean"), sd=("ksc", "std"), minimum=("ksc", "min"), maximum=("ksc", "max")).reset_index().to_csv(output_dir / "focal_summary_by_year.csv", index=False, lineterminator="\n")
    return {"focal_papers": len(mapping), "topic_year_cells": len(topic), "primary_topics": int(topic.primary_topic_id.nunique()), "self_conflicts": int(mapping.focal_self_history_conflict.sum()), "own_h_p_subtracted": int(mapping.own_h_p_subtracted.sum()), "primary_available_focal_papers": int(mapping.ksc.notna().sum()), "thresholds": thresholds, "checks": "focal and topic-year arithmetic, missingness, threshold coverage, unique keys and bounds"}


def validate_performance_supplement(data_dir: Path, reference: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    models = pd.read_parquet(data_dir / "model_performance_supplement.parquet")
    csv_models = pd.read_csv(data_dir / "model_performance_supplement.csv")
    pd.testing.assert_frame_equal(models.sort_values("benchmark_gpu_name").reset_index(drop=True), csv_models.sort_values("benchmark_gpu_name").reset_index(drop=True), check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    donors = pd.read_csv(data_dir / "generation_donors.csv")
    if not models.benchmark_gpu_name.is_unique or not donors.benchmark_generation.is_unique:
        raise ValueError("GPU performance supplement has duplicate model or generation IDs")
    assert_matching(len(models), reference["models"], "GPU supplement model count")
    observed = models.loc[models.original_performance_flops.gt(0)].groupby("benchmark_generation").agg(generation_mean_flops=("original_performance_flops", "mean"), donor_model_count=("benchmark_gpu_name", "nunique")).reset_index()
    pd.testing.assert_frame_equal(observed.sort_values("benchmark_generation").reset_index(drop=True), donors.sort_values("benchmark_generation").reset_index(drop=True), check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    joined = models.merge(donors, on="benchmark_generation", how="left", validate="many_to_one", suffixes=("_stored", "_computed"))
    for field in ["generation_mean_flops", "donor_model_count"]:
        stored, computed = joined[f"{field}_stored"], joined[f"{field}_computed"]
        if not stored.isna().eq(computed.isna()).all() or not np.allclose(stored.fillna(0), computed.fillna(0), rtol=1e-12, atol=1e-12):
            raise ValueError(f"GPU supplement {field} differs from generation donors")
    is_original = models.original_performance_flops.gt(0)
    is_imputed = ~is_original & models.generation_mean_flops.notna()
    expected = models.original_performance_flops.where(is_original, models.generation_mean_flops)
    if not models.imputed.eq(is_imputed).all() or not models.supplemented_performance_flops.isna().eq(expected.isna()).all() or not np.allclose(models.supplemented_performance_flops.fillna(0), expected.fillna(0), rtol=1e-12, atol=1e-12):
        raise ValueError("GPU performance imputation arithmetic differs")
    status_counts = {str(key): int(value) for key, value in models.status.value_counts().items()}
    assert_matching(status_counts, reference["status_counts"], "GPU performance supplement status")
    observed.to_csv(output_dir / "generation_donors_recomputed.csv", index=False, lineterminator="\n")
    return {"models": len(models), "generations_with_donors": len(donors), "status_counts": status_counts, "donor_rule": "unique originally observed models; exact generation; unweighted arithmetic mean before log transformation"}


def run_measurement(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    repo_root, output_dir = Path(repo_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_manifest = read_json(repo_root / "code/measurement/asset_sources.json")
    for asset in asset_manifest["assets"]:
        path = repo_root / asset["destination"]
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != asset["destination_sha256"]:
            raise ValueError(f"Measurement asset hash differs: {asset['destination']}")
    data = repo_root / "data/measurement"
    references = repo_root / "results/reference/measurement"
    classification = {}
    aggregate_rows, label_rows = [], []
    student_gold = read_jsonl(data / "classification/student_gold.jsonl")
    teacher_gold = read_jsonl(data / "classification/teacher_gold.jsonl")
    for model in ["teacher", "bert_base", "roberta_base", "scibert"]:
        gold = teacher_gold if model == "teacher" else student_gold
        metrics = score_classification(gold, read_jsonl(data / f"classification/{model}_predictions.jsonl"))
        reference = read_json(references / f"classification/{model}_metrics.json")
        if metrics["rows"] != reference["row_counts"]["gold"]:
            raise ValueError(f"{model}: evaluation row count differs")
        for label in RESOURCE_LABELS:
            retained = {key: reference["per_label"][label][key] for key in ["tp", "fp", "fn", "precision", "recall", "f1", "support", "predicted_positive"]}
            assert_matching(metrics["per_label"][label], retained, f"{model}.{label}")
            label_rows.append({"model": model, "label": label, "rows": metrics["rows"], **metrics["per_label"][label]})
        classification[model] = metrics
        aggregate_rows.append({"model": model, "labels": "L1-L4", "windows": metrics["rows"], "papers": metrics["papers"], **{f"micro_{key}": value for key, value in metrics["micro"].items()}, "macro_f1": metrics["macro_f1"], "exact_match_l1_l4": metrics["exact_match_l1_l4"]})
    split_ref = read_json(references / "classification/split_audit.json")
    paper_sets, overlap_rows, split_rows = {}, [], []
    for split in ["trusted_train", "candidate_train", "gold_calib", "gold_test"]:
        frame = pd.read_csv(data / f"classification/{split}_membership.csv", dtype=str)
        if frame.window_id.isna().any() or frame.paper_id.isna().any() or not frame.window_id.is_unique:
            raise ValueError(f"{split}: invalid membership IDs")
        paper_sets[split] = set(frame.paper_id)
        assert_matching({"rows": len(frame), "papers": len(paper_sets[split])}, {key: split_ref["splits"][split][key] for key in ["rows", "papers"]}, split)
        split_rows.append({"split": split, "windows": len(frame), "papers": len(paper_sets[split])})
    for left, right in combinations(paper_sets, 2):
        overlap = len(paper_sets[left] & paper_sets[right])
        assert_matching(overlap, split_ref["paper_overlap_matrix"][left][right], f"{left}/{right} overlap")
        if overlap:
            raise ValueError(f"Paper-level split overlap: {left}/{right}")
        overlap_rows.append({"left_split": left, "right_split": right, "shared_papers": overlap})
    write_csv(output_dir / "classification/model_metrics.csv", aggregate_rows)
    write_csv(output_dir / "classification/per_label_metrics.csv", label_rows)
    write_csv(output_dir / "classification/split_overlap.csv", overlap_rows)
    write_csv(output_dir / "classification/split_summary.csv", split_rows)
    write_json(output_dir / "classification/metrics.json", classification)
    gpu_gold = unique_index(read_jsonl(data / "gpu/gold.jsonl"), "sample_id", "GPU gold")
    gpu_results, gpu_rows = {}, []
    for mode in ["disabled", "enabled"]:
        predictions = read_jsonl(data / f"gpu/thinking_{mode}_predictions.jsonl")
        indexed = unique_index(predictions, "sample_id", "GPU predictions")
        if set(indexed) != set(gpu_gold):
            raise ValueError(f"GPU {mode}: gold/prediction ID sets differ")
        full_metrics = evaluate_mode(predictions, gpu_gold)
        metrics = {key: full_metrics[key] for key in GPU_METRIC_KEYS}
        assert_matching(metrics, read_json(references / f"gpu/thinking_{mode}_metrics.json"), f"GPU {mode}")
        gpu_results[mode] = metrics
        for metric, values in metrics.items():
            if isinstance(values, dict):
                gpu_rows.append({"thinking": mode, "metric": metric, "scored_rows": values.get("scored_rows", metrics["rows"]), "tp": values["tp"], "fp": values["fp"], "fn": values["fn"], "precision": values["precision"], "recall": values["recall"], "f1": values["f1"]})
    write_csv(output_dir / "gpu/extraction_metrics.csv", gpu_rows)
    write_json(output_dir / "gpu/metrics.json", gpu_results)
    mapping = pd.read_parquet(data / "gpu/gpu_model_normalization_mapping.parquet")
    specifications = pd.read_csv(data / "gpu/hardware_specifications.csv")
    normalization = {"mapping_rows": len(mapping), "specification_rows": len(specifications), "catalog_gpu_rows": int(specifications["Type"].astype(str).str.lower().eq("gpu").sum()), "mapping_columns": list(mapping.columns)}
    write_json(output_dir / "gpu/normalization_inventory.json", normalization)
    performance_supplement = validate_performance_supplement(data / "gpu", read_json(repo_root / "configs/measurement/gpu/performance_supplement.json"), output_dir / "gpu")
    write_json(output_dir / "gpu/performance_supplement_validation.json", performance_supplement)
    ksc_result = validate_ksc(data / "ksc", read_json(repo_root / "configs/measurement/ksc/run_contract.json"), output_dir / "ksc")
    write_json(output_dir / "ksc/validation.json", ksc_result)
    report = {"status": "pass", "asset_hashes_verified": len(asset_manifest["assets"]), "classification": {name: {"rows": result["rows"], "papers": result["papers"], "macro_f1_l1_l4": result["macro_f1"], "micro_f1_l1_l4": result["micro"]["f1"]} for name, result in classification.items()}, "split_pairs_checked": len(overlap_rows), "gpu_gold_rows": len(gpu_gold), "gpu_modes_verified": list(gpu_results), "normalization": normalization, "performance_supplement": performance_supplement, "ksc": ksc_result}
    write_json(output_dir / "validation.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_measurement(args.repo_root, args.output_dir), indent=2))

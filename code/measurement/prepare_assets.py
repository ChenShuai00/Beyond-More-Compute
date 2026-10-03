"""Create the compact measurement release from read-only frozen sources."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


CLASSIFICATION_BUNDLE = Path("data/derived/compute_resources/compute_resource_extraction_validation_v1_20260812/current/compute_resource_extraction_validation")
GPU_BUNDLE = Path("data/derived/compute_resources/gpu_extraction_validation_v1_20260805/current/gpu_extraction_validation")
INTEGRATED_MEASUREMENT = Path("data/derived/compute_resources/compute_resource_integrated_measurement_v1_20260906/current/compute_resource_integrated_measurement")
ENCODERS = {"bert_base": "trial_00008", "roberta_base": "trial_00000", "scibert": "trial_00001"}
GPU_METRICS = ["rows", "successful_requests", "failed_requests", "valid_json_and_schema", "schema_valid_rate", "record_detection", "hardware_model_normalized", "hardware_model_strict_surface", "quantity", "joint_normalized", "joint_strict_surface"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(portable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def portable(value: Any) -> Any:
    """Replace machine-specific paths with source-relative identifiers."""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items() if key not in {"db", "api_base", "proxy_bypass", "key_label", "key_labels"}}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, str) and re.match(r"^[A-Za-z]:[\\/]", value):
        normalized = value.replace("\\", "/")
        for anchor, alias in [("/code/compute/", "compute/"), ("/openalex_cache/", "external_cache/")]:
            if anchor in normalized:
                return alias + normalized.split(anchor, 1)[1]
        return "external_source/" + Path(normalized).name
    return value


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--ksc-release", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    repo = args.repo_root.resolve()
    ksc = args.ksc_release.resolve()
    sources: list[dict[str, Any]] = []

    def register(src: Path, dest: Path, transform: str, source_group: str = "compute", columns: list[str] | None = None) -> None:
        source_root = source if source_group == "compute" else ksc
        record = {"source_group": source_group, "source_path": src.relative_to(source_root).as_posix(), "source_sha256": sha256(src), "source_bytes": src.stat().st_size, "destination": dest.relative_to(repo).as_posix(), "destination_sha256": sha256(dest), "destination_bytes": dest.stat().st_size, "transform": transform}
        if columns is not None:
            record["selected_fields"] = columns
        sources.append(record)

    def project_jsonl(src: Path, dest: Path, columns: list[str]) -> None:
        rows = [{key: row[key] for key in columns if key in row} for row in read_jsonl(src)]
        write_jsonl(dest, rows)
        register(src, dest, "field projection; source row order preserved", columns=columns)

    def project_json(src: Path, dest: Path, columns: list[str] | None = None) -> None:
        obj = json.loads(src.read_text(encoding="utf-8"))
        if columns is not None:
            obj = {key: obj[key] for key in columns if key in obj}
        write_json(dest, portable(obj))
        register(src, dest, "scientific fields retained; machine paths replaced with relative source identifiers", columns=columns)

    def copy_text(src: Path, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        register(src, dest, "UTF-8 text copy")

    cb = source / CLASSIFICATION_BUNDLE
    class_dir = repo / "data/measurement/classification"
    ref_dir = repo / "results/reference/measurement/classification"
    gold_cols = ["gold_id", "paper_id", "split", "labels", "L1_INTERNAL", "L2_CLOUD", "L3_PUBLIC_HPC", "L4_EXTERNAL_API", "L5_OPEN_MODEL", "L0_NO_SPECIFIC_RESOURCE"]
    project_jsonl(cb / "gold/window_level_gold_test.jsonl", class_dir / "student_gold.jsonl", gold_cols)
    project_jsonl(cb / "gold/deepseek_paper_task_gold_test.jsonl", class_dir / "teacher_gold.jsonl", gold_cols)
    pred_cols = ["gold_id", "paper_id", "labels", "scores", "thresholds", "validation_issues", "model"]
    project_jsonl(cb / "results/deepseek_fewshot_v2/predictions.jsonl", class_dir / "teacher_predictions.jsonl", pred_cols)
    metric_cols = ["label_set", "row_counts", "per_label", "thresholds", "primary_l1_l5", "evaluated_labels", "target_labels", "exact_match"]
    project_json(cb / "results/deepseek_fewshot_v2/metrics_summary.json", ref_dir / "teacher_metrics.json", metric_cols)
    config_cols = ["model", "backend", "max_tokens", "module", "thinking", "window_input_format", "few_shot_format", "few_shot_count", "rows_selected", "prediction_failures_this_run", "invalid_predictions_this_run"]
    project_json(cb / "results/deepseek_fewshot_v2/run_summary.json", repo / "configs/measurement/classification/teacher_evaluation.json", config_cols)
    for encoder, trial in ENCODERS.items():
        trial_root = source / "outputs/resource_architecture_classify_model_training/candist_backbone_optuna_20260707" / encoder / "trials" / trial
        project_jsonl(trial_root / "test_eval/predictions.jsonl", class_dir / f"{encoder}_predictions.jsonl", pred_cols)
        project_json(cb / f"results/encoder_comparison/reps/{encoder}/test_eval/metrics_summary.json", ref_dir / f"{encoder}_metrics.json", metric_cols)
        project_json(cb / f"results/encoder_comparison/reps/{encoder}/train_summary.json", repo / f"configs/measurement/classification/{encoder}_training.json", ["model_name", "rows", "resource_heads", "derived_l0", "pos_weight", "selection_policy", "candist_policy", "boundary_aux_policy", "best_thresholds"])
        project_json(trial_root / "trial_request.json", repo / f"configs/measurement/classification/{encoder}_parameters.json", ["trial_number", "params", "objective", "selection_objective", "run_test_each_trial"])
    membership: list[dict[str, Any]] = []
    split_root = source / "data/resource_architecture_train_test/candist_primary_l1_l5_20260704"
    for split in ["trusted_train", "candidate_train", "gold_calib", "gold_test"]:
        src = split_root / f"{split}.jsonl"
        rows = read_jsonl(src)
        dest = class_dir / f"{split}_membership.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["window_id", "paper_id", "split"])
            writer.writeheader()
            for row in rows:
                record = {"window_id": row.get("gold_id") or row.get("candidate_id") or row.get("annotation_id"), "paper_id": row["paper_id"], "split": split}
                writer.writerow(record)
                membership.append(record)
        register(src, dest, "window and paper IDs only; texts and annotation identities omitted", columns=["gold_id", "candidate_id", "annotation_id", "paper_id"])
    project_json(split_root / "split_audit.json", ref_dir / "split_audit.json", ["label_view", "policy", "splits", "candidate_excluded_gold_paper_rows", "candidate_rejected_rows", "paper_overlap_matrix"])
    model_manifest = source / "outputs/resource_architecture_classify_full_corpus/candist_backbone_best_2015_2025_20260712_v1/run_manifest.json"
    manifest = json.loads(model_manifest.read_text(encoding="utf-8"))
    model_card = {key: manifest["config"][key] for key in ["model_hashes", "thresholds", "resource_labels", "derived_l0_label", "max_length", "overlap_tokens", "precision", "excluded_section_kinds", "excluded_path_pattern"]}
    model_card.update({"encoder": "allenai/scibert_scivocab_uncased", "selected_trial": ENCODERS["scibert"], "weights_bytes": 439712772})
    dest = repo / "configs/measurement/classification/production_model.json"
    write_json(dest, model_card)
    register(model_manifest, dest, "frozen inference configuration and recorded weight hash; weights not copied")
    for filename in ["label_schema_compute_resource_compact.md", "instruction_compute_resource_compact.md", "few_shot/few-shot-v2.md"]:
        copy_text(source / "configs/classification/compute_resource_prompts" / filename, repo / "configs/measurement/classification" / filename)

    gb = source / GPU_BUNDLE
    original_evaluator = gb / "process/evaluate_deepseek_gpu_extraction.py"
    evaluator = load_module(original_evaluator, "source_gpu_evaluation")
    gold_records = evaluator.build_gold_records(read_jsonl(gb / "gold/gold.json"))
    gold_export = [{"sample_id": row["sample_id"], "source_row_index": row["source_row_index"], "source_id": row["source_id"], "gold": row["gold"]} for row in gold_records]
    dest = repo / "data/measurement/gpu/gold.jsonl"
    write_jsonl(dest, gold_export)
    register(gb / "gold/gold.json", dest, "original span labels converted with frozen evaluator; retain model/count entity spans and IDs, omit sentences and comments")
    for mode in ["disabled", "enabled"]:
        project_jsonl(gb / f"results/thinking_{mode}/predictions.jsonl", repo / f"data/measurement/gpu/thinking_{mode}_predictions.jsonl", ["sample_id", "source_row_index", "source_id", "parsed", "validation_issues", "error", "thinking", "model"])
        project_json(gb / f"results/thinking_{mode}/metrics.json", repo / f"results/reference/measurement/gpu/thinking_{mode}_metrics.json", GPU_METRICS)
        project_json(gb / f"results/thinking_{mode}/run_summary.json", repo / f"configs/measurement/gpu/thinking_{mode}.json", ["model", "thinking", "max_tokens", "temperature", "reasoning_effort", "seed", "rows_selected", "gold_sha256", "prompt_sha256"])
    copy_text(gb / "process/gpu_extraction_prompt_fewshot.md", repo / "configs/measurement/gpu/extraction_prompt.md")
    pure_functions = {"normalize_spaces", "strict_surface", "canonical_gpu_model", "parse_prediction", "number_from_usage", "nested_number_from_usage", "f1_from_counts", "counter_score", "prediction_items", "gold_model_sets", "predicted_model_sets", "evaluate_mode", "percentile"}
    original_text = original_evaluator.read_text(encoding="utf-8")
    tree = ast.parse(original_text)
    segments = [ast.get_source_segment(original_text, node) for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in pure_functions]
    dest = repo / "code/measurement/gpu_metrics.py"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text('"""Pure offline GPU scoring functions from the frozen evaluator."""\n\nfrom __future__ import annotations\n\nimport math\nimport re\nimport unicodedata\nfrom collections import Counter\nfrom typing import Any\n\n\n' + "\n\n\n".join(segments) + "\n", encoding="utf-8")
    register(original_evaluator, dest, "extract pure scoring functions only; API collection and credential handling omitted", columns=sorted(pure_functions))
    for filename in ["gpu_model_normalization_mapping.parquet", "gpu_normalization_review.parquet"]:
        src = gb / "normalization_qa" / filename
        frame = pd.read_parquet(src)
        for col in frame.select_dtypes(include=["object", "string"]).columns:
            frame[col] = frame[col].map(portable)
        dest = repo / "data/measurement/gpu" / filename
        frame.to_parquet(dest, index=False)
        register(src, dest, "all normalization fields; machine paths converted to relative identifiers", columns=list(frame.columns))
    for filename in ["normalization_summary.json", "normalization_finalization.json"]:
        project_json(gb / "normalization_qa" / filename, repo / "results/reference/measurement/gpu" / filename)
    for src in (source / "configs/gpu/normalization").glob("*.yaml"):
        dest = repo / "configs/measurement/gpu/normalization" / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(yaml.safe_dump(portable(yaml.safe_load(src.read_text(encoding="utf-8"))), allow_unicode=True, sort_keys=False), encoding="utf-8")
        register(src, dest, "YAML scientific rules retained; machine paths replaced with relative identifiers")
    project_json(source / "configs/gpu/normalization/gpu_reviewed_mapping_import_exceptions.json", repo / "configs/measurement/gpu/normalization/gpu_reviewed_mapping_import_exceptions.json")
    copy_text(source / "configs/analysis/novelty/gpu_vram_overrides_v1.csv", repo / "configs/measurement/gpu/vram_overrides.csv")
    spec_source = source / "configs/gpu/ml_hardware/ml_hardware.xlsx"
    sys.path.insert(0, str(source))
    from scripts.gpu.normalization.gpu_catalog import load_hardware_catalog
    catalog = load_hardware_catalog(spec_source)
    frame = pd.DataFrame(catalog.rows)
    dest = repo / "data/measurement/gpu/hardware_specifications.csv"
    frame.to_csv(dest, index=False, encoding="utf-8", lineterminator="\n")
    register(spec_source, dest, "hardware catalog worksheet exported to CSV using frozen catalog reader; workbook metadata omitted", columns=list(frame.columns))
    project_json(source / INTEGRATED_MEASUREMENT / "tables/gpu_normalization_policy.json", repo / "configs/measurement/gpu/normalization_policy.json")
    src = source / "configs/analysis/novelty/gpu_scale_config_aligned_v2.yaml"
    configuration = yaml.safe_load(src.read_text(encoding="utf-8"))
    dest = repo / "configs/measurement/gpu/configuration_selection.json"
    write_json(dest, {key: configuration[key] for key in ["schema_version", "hardware_generation", "vram_policy", "selection"]})
    register(src, dest, "configuration selection, tie rules and VRAM policy only")
    supplement_root = source / "outputs/analysis/gpu_generation_mean_supplement_side_20260906"
    for filename in ["generation_donors.csv", "model_performance_supplement.csv", "model_performance_supplement.parquet"]:
        src = supplement_root / filename
        dest = repo / "data/measurement/gpu" / filename
        dest.write_bytes(src.read_bytes())
        register(src, dest, "byte-identical generation donors or model-level performance supplement")
    project_json(supplement_root / "manifest.json", repo / "configs/measurement/gpu/performance_supplement.json", ["models", "status_counts", "method"])
    schema = {"type": "object", "required": ["gpus"], "additionalProperties": False, "properties": {"gpus": {"type": "array", "items": {"type": "object", "required": ["gpu_model", "count"], "additionalProperties": False, "properties": {"gpu_model": {"type": "string", "minLength": 1}, "count": {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}]}}}}}}
    dest = repo / "configs/measurement/gpu/extraction_schema.json"
    write_json(dest, schema)
    register(original_evaluator, dest, "JSON schema transcribed from the frozen parse_prediction function")

    for filename in ["focal_paper_ksc_mapping.parquet", "topic_year_ksc.parquet", "topic_year_ksc_robustness.parquet"]:
        src = ksc / "data" / filename
        dest = repo / "data/measurement/ksc" / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
        register(src, dest, "byte-identical frozen lightweight data", source_group="ksc_release")
    for filename in ["run_contract.json", "release_status.json", "qa/validation_report.json", "qa/aggregate_summary.json"]:
        src = ksc / filename
        dest = repo / ("configs/measurement/ksc" if filename == "run_contract.json" else "results/reference/measurement/ksc") / Path(filename).name
        write_json(dest, portable(json.loads(src.read_text(encoding="utf-8"))))
        register(src, dest, "scientific contract or validation summary; machine paths replaced with relative source identifiers", source_group="ksc_release")
    write_json(repo / "code/measurement/asset_sources.json", {"schema_version": 1, "source_groups": {"compute": "source research repository", "ksc_release": "formal ksc_v1_20260912_v14r1 release"}, "assets": sources})
    print(json.dumps({"assets": len(sources), "exported_bytes": sum(row["destination_bytes"] for row in sources), "classification_encoders": list(ENCODERS), "gpu_modes": ["disabled", "enabled"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

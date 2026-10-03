"""Pure offline GPU scoring functions from the frozen evaluator."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any


def normalize_spaces(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).strip()


def strict_surface(value: str) -> str:
    return normalize_spaces(value).casefold()


def canonical_gpu_model(value: str) -> str:
    text = normalize_spaces(value).casefold()
    text = text.replace("_", " ").replace("–", "-").replace("—", "-")
    text = re.sub(r"\b(?:gpu|gpus|graphics\s+card|graphics\s+cards|card|cards)\b", " ", text)
    text = re.sub(r"\b(?:nvidia|navidia|amd|ati|tesla|geforce|quadro)\b", " ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s*(?:gb|g)\b", " ", text)
    text = re.sub(r"\b(?:of|the|on|using|with|and)\b", " ", text)
    text = re.sub(r"\brtx\s*-?\s*([a-z]*\d+[a-z]*)\b", r"rtx \1", text)
    text = re.sub(r"\bgtx\s*-?\s*([a-z]*\d+[a-z]*)\b", r"gtx \1", text)
    text = re.sub(r"\ba\s*-\s*(\d+)\b", r"a\1", text)
    text = re.sub(r"\bv\s*-\s*(\d+)\b", r"v\1", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return normalize_spaces(text)


def parse_prediction(content: str) -> tuple[dict[str, Any] | None, list[str]]:
    issues: list[str] = []
    try:
        value = json.loads(content)
    except Exception as exc:  # noqa: BLE001 - response validation must not stop a batch.
        return None, ["invalid_json", repr(exc)]
    if not isinstance(value, dict) or set(value) != {"gpus"} or not isinstance(value["gpus"], list):
        return None, ["invalid_top_level_schema"]
    items: list[dict[str, Any]] = []
    for index, item in enumerate(value["gpus"]):
        if not isinstance(item, dict) or set(item) != {"gpu_model", "count"}:
            issues.append(f"invalid_item_schema:{index}")
            continue
        if not isinstance(item["gpu_model"], str) or not item["gpu_model"].strip():
            issues.append(f"invalid_gpu_model:{index}")
            continue
        count = item["count"]
        if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 1):
            issues.append(f"invalid_count:{index}")
            continue
        items.append({"gpu_model": item["gpu_model"], "count": count})
    if issues:
        return None, issues
    return {"gpus": items}, []


def number_from_usage(usage: dict[str, Any], key: str) -> int:
    value = usage.get(key, 0)
    return int(value or 0) if isinstance(value, (int, float)) else 0


def nested_number_from_usage(usage: dict[str, Any], parent: str, key: str) -> int:
    value = usage.get(parent)
    if not isinstance(value, dict):
        return 0
    nested = value.get(key, 0)
    return int(nested or 0) if isinstance(nested, (int, float)) else 0


def f1_from_counts(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def counter_score(gold: Counter[Any], predicted: Counter[Any]) -> dict[str, float | int]:
    tp = sum((gold & predicted).values())
    return f1_from_counts(tp, sum(predicted.values()) - tp, sum(gold.values()) - tp)


def prediction_items(result: dict[str, Any]) -> list[dict[str, Any]]:
    parsed = result.get("parsed")
    if not isinstance(parsed, dict) or not isinstance(parsed.get("gpus"), list):
        return []
    return [item for item in parsed["gpus"] if isinstance(item, dict)]


def gold_model_sets(gold: dict[str, Any], strict: bool = False) -> set[str]:
    key = "strict_model" if strict else "canonical_model"
    return {str(entity[key]) for entity in gold["hardware_entities"] if entity[key]}


def predicted_model_sets(result: dict[str, Any], strict: bool = False) -> set[str]:
    return {
        (strict_surface(item["gpu_model"]) if strict else canonical_gpu_model(item["gpu_model"]))
        for item in prediction_items(result)
        if item.get("gpu_model")
    }


def evaluate_mode(results: list[dict[str, Any]], records_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result_by_id = {str(result["sample_id"]): result for result in results}
    detection_tp = detection_fp = detection_fn = detection_tn = 0
    hardware_norm = Counter()
    hardware_strict = Counter()
    quantity_score = Counter()
    joint_norm = Counter()
    joint_strict = Counter()
    valid_json = schema_valid = successful_requests = 0
    latencies: list[float] = []
    usage_total: Counter[str] = Counter()
    cost_totals = {
        "usd": Counter(),
        "cny": Counter(),
    }
    quantity_scored_rows = 0
    joint_scored_rows = 0

    for sample_id, record in records_by_id.items():
        result = result_by_id.get(sample_id)
        gold = record["gold"]
        if result is None:
            predicted_positive = False
        else:
            predicted_positive = bool(prediction_items(result))
            if result.get("error") is None:
                successful_requests += 1
                if not result.get("validation_issues"):
                    valid_json += 1
                    schema_valid += 1
                latencies.append(float(result.get("latency_seconds") or 0.0))
                usage = result.get("usage") or {}
                for key in ("prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
                    usage_total[key] += number_from_usage(usage, key)
                usage_total["reasoning_tokens"] += nested_number_from_usage(
                    usage, "completion_tokens_details", "reasoning_tokens"
                )
                usage_total["cached_tokens_detail"] += nested_number_from_usage(
                    usage, "prompt_tokens_details", "cached_tokens"
                )
                usage_cost = result.get("usage_cost") or {}
                for currency in ("usd", "cny"):
                    for key in ("prompt_cache_hit", "prompt_cache_miss", "completion", "total", "standardized_all_cache_miss"):
                        cost_totals[currency][key] += float((usage_cost.get(currency) or {}).get(key) or 0.0)
        gold_positive = bool(gold["positive"])
        if gold_positive and predicted_positive:
            detection_tp += 1
        elif not gold_positive and predicted_positive:
            detection_fp += 1
        elif gold_positive:
            detection_fn += 1
        else:
            detection_tn += 1

        pred_norm = predicted_model_sets(result or {}, strict=False)
        pred_strict = predicted_model_sets(result or {}, strict=True)
        hardware_norm["tp"] += len(gold_model_sets(gold) & pred_norm)
        hardware_norm["fp"] += len(pred_norm - gold_model_sets(gold))
        hardware_norm["fn"] += len(gold_model_sets(gold) - pred_norm)
        hardware_strict["tp"] += len(gold_model_sets(gold, strict=True) & pred_strict)
        hardware_strict["fp"] += len(pred_strict - gold_model_sets(gold, strict=True))
        hardware_strict["fn"] += len(gold_model_sets(gold, strict=True) - pred_strict)

        gold_counts = [item["value"] for item in gold["count_entities"] if item["value"] is not None]
        if gold_counts:
            quantity_scored_rows += 1
            predicted_counts = [item["count"] for item in prediction_items(result or {}) if item.get("count") is not None]
            quantity_score["tp"] += sum((Counter(gold_counts) & Counter(predicted_counts)).values())
            quantity_score["fp"] += max(0, len(predicted_counts) - sum((Counter(gold_counts) & Counter(predicted_counts)).values()))
            quantity_score["fn"] += max(0, len(gold_counts) - sum((Counter(gold_counts) & Counter(predicted_counts)).values()))

        if gold["joint_scorable"]:
            joint_scored_rows += 1
            gold_entities = sorted(gold["hardware_entities"], key=lambda item: item["start"])
            gold_count_entities = sorted(
                [item for item in gold["count_entities"] if item["value"] is not None],
                key=lambda item: item["start"],
            )
            gold_pairs_norm = Counter(
                (gold_entities[index]["canonical_model"], gold_count_entities[index]["value"])
                for index in range(len(gold_entities))
            )
            gold_pairs_strict = Counter(
                (gold_entities[index]["strict_model"], gold_count_entities[index]["value"])
                for index in range(len(gold_entities))
            )
            pred_pairs_norm = Counter(
                (canonical_gpu_model(item["gpu_model"]), item["count"])
                for item in prediction_items(result or {})
                if item.get("count") is not None
            )
            pred_pairs_strict = Counter(
                (strict_surface(item["gpu_model"]), item["count"])
                for item in prediction_items(result or {})
                if item.get("count") is not None
            )
            joint_norm["tp"] += sum((gold_pairs_norm & pred_pairs_norm).values())
            joint_norm["fp"] += sum((pred_pairs_norm - gold_pairs_norm).values())
            joint_norm["fn"] += sum((gold_pairs_norm - pred_pairs_norm).values())
            joint_strict["tp"] += sum((gold_pairs_strict & pred_pairs_strict).values())
            joint_strict["fp"] += sum((pred_pairs_strict - gold_pairs_strict).values())
            joint_strict["fn"] += sum((gold_pairs_strict - pred_pairs_strict).values())

    def metric_from_counter(counter: Counter[str]) -> dict[str, Any]:
        return f1_from_counts(int(counter["tp"]), int(counter["fp"]), int(counter["fn"]))

    record_metric = f1_from_counts(detection_tp, detection_fp, detection_fn)
    record_metric["tn"] = detection_tn
    record_metric["accuracy"] = (detection_tp + detection_tn) / len(records_by_id) if records_by_id else 0.0
    usage_summary_dict = dict(usage_total)
    cache_denominator = usage_summary_dict.get("prompt_cache_hit_tokens", 0) + usage_summary_dict.get(
        "prompt_cache_miss_tokens", 0
    )
    usage_summary_dict["cache_hit_rate"] = (
        usage_summary_dict.get("prompt_cache_hit_tokens", 0) / cache_denominator
        if cache_denominator
        else None
    )
    return {
        "rows": len(records_by_id),
        "successful_requests": successful_requests,
        "failed_requests": len(records_by_id) - successful_requests,
        "valid_json_and_schema": valid_json,
        "schema_valid_rate": valid_json / len(records_by_id) if records_by_id else 0.0,
        "record_detection": record_metric,
        "hardware_model_normalized": metric_from_counter(hardware_norm),
        "hardware_model_strict_surface": metric_from_counter(hardware_strict),
        "quantity": {
            "scored_rows": quantity_scored_rows,
            **metric_from_counter(quantity_score),
        },
        "joint_normalized": {
            "scored_rows": joint_scored_rows,
            **metric_from_counter(joint_norm),
        },
        "joint_strict_surface": {
            "scored_rows": joint_scored_rows,
            **metric_from_counter(joint_strict),
        },
        "latency_seconds": {
            "mean": sum(latencies) / len(latencies) if latencies else None,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
        "usage_summary": usage_summary_dict,
        "usage_cost": {currency: dict(values) for currency, values in cost_totals.items()},
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

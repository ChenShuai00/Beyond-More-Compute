"""Verify frozen citation measurements, ranks and model outcomes offline."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


TOP_SHARES = {"10": 0.10, "5": 0.05, "1": 0.01}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def same_values(actual: pd.Series, expected: pd.Series, message: str) -> float:
    actual_values = pd.to_numeric(actual, errors="raise").to_numpy(dtype=float, na_value=np.nan)
    expected_values = pd.to_numeric(expected, errors="raise").to_numpy(dtype=float, na_value=np.nan)
    require(np.array_equal(np.isnan(actual_values), np.isnan(expected_values)), f"{message}: missingness differs")
    require(np.allclose(actual_values, expected_values, rtol=0, atol=1e-12, equal_nan=True), message)
    valid = np.isfinite(actual_values) & np.isfinite(expected_values)
    return float(np.max(np.abs(actual_values[valid] - expected_values[valid]))) if valid.any() else 0.0


def thresholds_from_histogram(histogram: Mapping[int, int], reference_n: int) -> dict:
    """Match the frozen source's inclusive nearest-rank rule, including its prefix histogram."""
    clean = {int(key): int(count) for key, count in histogram.items() if int(key) >= 0}
    require(all(count >= 0 for count in clean.values()), "Negative histogram work count")
    n = int(reference_n)
    require(n > 0 and sum(clean.values()) <= n, "Invalid reference population size")
    result = {"reference_n": n}
    for label, share in TOP_SHARES.items():
        target = max(1, math.ceil(share * n))
        rank = n - target + 1
        cumulative, below, cutoff = 0, 0, None
        for citations, count in sorted(clean.items()):
            previous = cumulative
            cumulative += count
            if cumulative >= rank:
                cutoff, below = citations, previous
                break
        require(cutoff is not None, f"Histogram does not reach top-{label} rank")
        qualified = n - below
        result.update({f"top_cited_{label}_target_count": target,
                       f"top_cited_{label}_cutoff": cutoff,
                       f"top_cited_{label}_reference_count": qualified,
                       f"top_cited_{label}_realized_share": qualified / n})
    return result


def verify_citations(repo_root: Path) -> dict:
    repo_root = Path(repo_root).resolve()
    contract = json.loads((repo_root / "configs/citation_contract.json").read_text(encoding="utf-8"))
    manifest_path = repo_root / contract["asset_manifest"]
    require(sha256(manifest_path) == contract["asset_manifest_sha256"], "Citation asset manifest SHA-256 differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = {}
    for asset in manifest["assets"]:
        path = repo_root / asset["path"]
        require(sha256(path) == asset["sha256"], f"Citation asset SHA-256 differs: {asset['path']}")
        frame = pd.read_parquet(path)
        require(len(frame) == asset["rows"] and list(frame.columns) == asset["columns"], "Citation table shape/schema differs")
        require(len(frame) == contract["expected_rows"][path.name], "Citation expected row count differs")
        frames[path.stem] = frame
    metrics = frames["paper_citation_metrics"]
    annual = frames["paper_citation_counts_by_year"]
    outside = frames["counts_by_year_outside_window"]
    reference = frames["reference_thresholds"]
    hist = frames["reference_histograms"]
    require(metrics.paper_id.notna().all() and metrics.paper_id.is_unique, "Citation paper IDs are missing/duplicated")
    id_set = set(metrics.paper_id)
    observed_id_hash = hashlib.sha256("".join(f"{value}\n" for value in sorted(id_set)).encode("utf-8")).hexdigest()
    require(observed_id_hash == contract["paper_id_set_sha256"], "Citation paper-ID set differs")
    require(metrics.citation_metric_status.value_counts().to_dict() == contract["expected_metric_status_counts"], "Citation status counts differ")
    require(metrics.citation_reconciliation_status.value_counts().to_dict() == contract["expected_reconciliation_status_counts"], "Citation reconciliation counts differ")
    require(set(annual.paper_id) == id_set, "Annual citation paper IDs differ")
    require(not annual.duplicated(["paper_id", "citation_year"]).any(), "Annual citation keys duplicated")
    require(annual.groupby("paper_id").size().eq(contract["annual_rows_per_paper"]).all(), "Annual rows per paper differ")
    require(set(annual.citation_year) == set(range(contract["annual_start_year"], contract["annual_end_year"] + 1)), "Annual citation years differ")
    require(not outside.duplicated(["paper_id", "citation_year"]).any() and set(outside.paper_id).issubset(id_set), "Outside-window annual keys differ")
    require((outside.citation_year.lt(contract["annual_start_year"]) | outside.citation_year.gt(contract["annual_end_year"])).all(), "Outside-window record falls inside annual window")
    for frame, column in [(annual, "citation_count"), (outside, "citation_count"), (hist, "work_count"), (hist, "cited_by_count")]:
        values = frame[column].dropna()
        require((values.ge(0) & values.mod(1).eq(0)).all(), f"Invalid nonnegative integer: {column}")
    indexed = metrics.set_index("paper_id")
    annual = annual.join(indexed[["citation_openalex_publication_year", "citation_counts_by_year_status"]], on="paper_id")
    same_values(annual.focal_publication_year_current, annual.citation_openalex_publication_year, "Annual publication year disagrees")
    expected_in = annual.citation_openalex_publication_year.notna() & annual.citation_year.ge(annual.citation_openalex_publication_year) & annual.citation_year.le(contract["snapshot_year"])
    expected_pre = annual.citation_openalex_publication_year.notna() & annual.citation_year.lt(annual.citation_openalex_publication_year)
    require(annual.in_cumulative_window.eq(expected_in.fillna(False)).all(), "Annual cumulative-window flags disagree")
    require(annual.is_prepublication_year.eq(expected_pre.fillna(False)).all(), "Annual prepublication flags disagree")
    require(annual.citation_count.notna().eq(annual.citation_counts_by_year_status.eq("ok")).all(), "Annual citation availability disagrees")
    complete = pd.concat([annual[["paper_id", "citation_year", "citation_count"]], outside], ignore_index=True)
    complete = complete.join(indexed[["citation_openalex_publication_year"]], on="paper_id")
    within = complete.citation_year.ge(complete.citation_openalex_publication_year) & complete.citation_year.le(contract["snapshot_year"])
    before = complete.citation_year.lt(complete.citation_openalex_publication_year)
    after = complete.citation_year.gt(contract["snapshot_year"])
    count_ok = indexed.citation_counts_by_year_status.eq("ok") & indexed.citation_openalex_publication_year.notna()

    def aggregate(mask: pd.Series) -> pd.Series:
        sums = complete.loc[mask.fillna(False)].groupby("paper_id").citation_count.sum(min_count=1)
        result = sums.reindex(indexed.index).fillna(0).astype(float)
        return result.where(count_ok, np.nan)

    same_values(indexed.citation_cumulative_from_publication_year_current, aggregate(within), "Cumulative citations disagree with annual window")
    same_values(indexed.citation_prepublication_count, aggregate(before), "Prepublication citations disagree with annual counts")
    same_values(indexed.citation_postsnapshot_count, aggregate(after), "Post-collection citations disagree with annual counts")
    same_values(indexed.citation_counts_by_year_sum_all, aggregate(pd.Series(True, index=complete.index)), "All annual counts disagree")
    difference = indexed.citation_current_total - indexed.citation_counts_by_year_sum_all
    same_values(indexed.citation_total_minus_counts_by_year_sum_all, difference, "Current-total reconciliation arithmetic disagrees")
    reconciliation = pd.Series("unavailable", index=indexed.index)
    reconciliation.loc[difference.eq(0).fillna(False)] = "exact"
    reconciliation.loc[difference.gt(0).fillna(False)] = "annual_sum_below_total"
    reconciliation.loc[difference.lt(0).fillna(False)] = "annual_sum_above_total"
    require(indexed.citation_reconciliation_status.eq(reconciliation).all(), "Current-total reconciliation label disagrees")

    cell_keys = ["reference_publication_year", "reference_primary_topic_short_id"]
    require(not reference.duplicated(cell_keys).any(), "Reference cell keys duplicated")
    require(not hist.duplicated(cell_keys + ["cited_by_count"]).any(), "Histogram bins duplicated")
    reference_indexed = reference.set_index(cell_keys)
    grouped = hist.groupby(cell_keys, sort=True)
    require(set(grouped.groups) == set(reference_indexed.index), "Histogram and threshold cells differ")
    require(reference.reference_corpus.eq(contract["reference"]["corpus"]).all(), "Reference corpus differs")
    require(reference.reference_work_types.eq("|".join(contract["reference"]["work_types"])).all(), "Reference work types differ")
    for key, bins in grouped:
        row = reference_indexed.loc[key]
        require(bins.source_page.nunique() == int(row.reference_histogram_pages), "Reference histogram page count differs")
        require(int(bins.cited_by_count.max()) == int(row.reference_histogram_max_observed), "Reference histogram maximum differs")
        result = thresholds_from_histogram(dict(zip(bins.cited_by_count, bins.work_count)), int(row.reference_n))
        for column, value in result.items():
            require(math.isclose(float(row[column]), value, rel_tol=0, abs_tol=1e-12), f"Nearest-rank threshold differs: {key}, {column}")
    scored = metrics.citation_metric_status.eq("ok")
    scored_frame = metrics.loc[scored].copy()
    scored_frame["reference_primary_topic_short_id"] = scored_frame.citation_openalex_primary_topic_id.str.rsplit("/", n=1).str[-1]
    attached = scored_frame.merge(reference, how="left", left_on=["citation_openalex_publication_year", "reference_primary_topic_short_id"], right_on=cell_keys, validate="many_to_one", suffixes=("", "_reference"))
    require(len(attached) == int(scored.sum()) and attached.reference_n_reference.notna().all(), "Scored paper lacks reference cell")
    for column in ["reference_n"] + [f"top_cited_{label}_{suffix}" for label in TOP_SHARES for suffix in ["cutoff", "reference_count", "realized_share"]]:
        same_values(attached[column], attached[column + "_reference"], f"Attached threshold differs: {column}")
    for label in TOP_SHARES:
        column = f"top_cited_{label}_current"
        require(metrics.loc[~scored, column].isna().all(), "Ineligible top-cited outcome is nonmissing")
        same_values(attached[column], (attached.citation_current_total >= attached[f"top_cited_{label}_cutoff"]).astype(int), f"Top-{label} indicator disagrees with inclusive cutoff")
    require((attached.top_cited_1_current.le(attached.top_cited_5_current) & attached.top_cited_5_current.le(attached.top_cited_10_current)).all(), "Top-cited indicators are not nested")

    baseline = pd.read_parquet(repo_root / "data/analysis_ready/baseline_impact.parquet").set_index("paper_id")
    require(baseline.index.is_unique and set(baseline.index) == id_set, "Baseline citation paper-ID set differs")
    baseline = baseline.reindex(indexed.index)
    outcome_checks = {}
    cumulative = pd.to_numeric(indexed.citation_cumulative_from_publication_year_current).astype(float)
    outcome_checks["y_cumulative_citations_max_abs_error"] = same_values(baseline.y_cumulative_citations, np.log1p(cumulative), "Baseline log cumulative outcome differs")
    for label in TOP_SHARES:
        outcome_checks[f"y_top{label}_max_abs_error"] = same_values(baseline[f"y_top{label}"], indexed[f"top_cited_{label}_current"], f"Baseline top-{label} outcome differs")
    for column in ["citation_current_total", "citation_cumulative_from_publication_year_current", "citation_openalex_publication_year"]:
        same_values(baseline[column], indexed[column], f"Baseline measurement differs: {column}")
    return {"passed": True, "assets_checked": len(manifest["assets"]), "paper_rows": len(metrics),
            "annual_rows": len(annual), "outside_window_annual_rows": len(outside), "reference_cells_recomputed": len(reference),
            "reference_histogram_bins": len(hist), "scored_papers": int(scored.sum()),
            "metric_status_counts": metrics.citation_metric_status.value_counts().to_dict(),
            "outcome_checks": outcome_checks, "paper_id_set_sha256": observed_id_hash}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(verify_citations(args.repo_root), indent=2))


if __name__ == "__main__":
    main()

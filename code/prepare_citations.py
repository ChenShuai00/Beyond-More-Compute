"""Export frozen citation measurements and scientific cache projections."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa


CURRENT = Path("outputs/analysis/focal_citation_metrics_v1_20260912/current")
TABLES = ("paper_citation_metrics", "paper_citation_counts_by_year", "reference_thresholds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_zst(path: Path) -> dict:
    source = pa.OSFile(str(path), "rb")
    stream = pa.CompressedInputStream(source, "zstd")
    try:
        return json.loads(stream.read().decode("utf-8"))
    finally:
        stream.close()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export_citations(source_root: Path, repo_root: Path) -> dict:
    source_root, repo_root = source_root.resolve(), repo_root.resolve()
    current = source_root / CURRENT
    destination = repo_root / "data/citations"
    destination.mkdir(parents=True, exist_ok=True)
    summary = json.loads((current / "summary.json").read_text(encoding="utf-8"))
    snapshot_year = int(summary["collection_window_utc"]["started_at"][:4])
    annual_start = snapshot_year - 9
    assets, sources = [], []

    def record_source(path: Path, projection: str, target: str) -> None:
        sources.append({
            "source_group": "compute", "source_path": path.relative_to(source_root).as_posix(),
            "source_sha256": sha256(path), "source_bytes": path.stat().st_size,
            "projection": projection, "destination": target,
        })

    def record_asset(path: Path, columns: list[str], rows: int) -> None:
        assets.append({"path": path.relative_to(repo_root).as_posix(), "sha256": sha256(path),
                       "bytes": path.stat().st_size, "rows": rows, "columns": columns})

    frames = {}
    for name in TABLES:
        source = current / f"{name}.parquet"
        target = destination / source.name
        frame = pd.read_parquet(source)
        # Every column in these three frozen tables describes a scientific measurement.
        shutil.copyfile(source, target)
        frames[name] = frame
        record_source(source, "all scientific columns; original frozen Parquet bytes", target.relative_to(repo_root).as_posix())
        record_asset(target, list(frame.columns), len(frame))

    thresholds = frames["reference_thresholds"]
    histogram_rows = []
    raw = current / "raw" / summary["run_id"]
    for row in thresholds.itertuples(index=False):
        year, topic = int(row.reference_publication_year), row.reference_primary_topic_short_id
        pages = raw / "reference_cells" / f"year={year}" / f"topic={topic}"
        keys = set()
        previous_max = None
        for page_index in range(int(row.reference_histogram_pages)):
            path = pages / f"page-{page_index:04d}.json.zst"
            envelope = read_json_zst(path)
            response = envelope["response"]
            if int(response["meta"]["count"]) != int(row.reference_n):
                raise ValueError("Reference population changed across cached pages")
            params = envelope["request"]["params"]
            expected_filter = f"publication_year:{year},type:{row.reference_work_types},primary_topic.id:{topic}"
            if params["filter"] != expected_filter or params["corpus"] != row.reference_corpus:
                raise ValueError("Reference cache does not match its scientific cell")
            page_keys = []
            for group in response.get("group_by", []):
                key, count = int(group["key"]), int(group["count"])
                if key < 0:
                    if count:
                        raise ValueError("Nonzero unknown citation-count group")
                    continue
                if key in keys or count < 0:
                    raise ValueError("Duplicate or negative reference histogram bin")
                keys.add(key)
                page_keys.append(key)
                histogram_rows.append({
                    "reference_publication_year": year, "reference_primary_topic_short_id": topic,
                    "cited_by_count": key, "work_count": count, "source_page": page_index,
                })
            if page_keys and previous_max is not None and min(page_keys) <= previous_max:
                raise ValueError("Cached reference pages are not increasing")
            if page_keys:
                previous_max = max(page_keys)
            record_source(path, "response.meta.count checked; response.group_by nonnegative cited_by_count/work_count only",
                          "data/citations/reference_histograms.parquet")
    histogram = pd.DataFrame(histogram_rows).sort_values(
        ["reference_publication_year", "reference_primary_topic_short_id", "cited_by_count"]
    ).reset_index(drop=True)
    histogram_path = destination / "reference_histograms.parquet"
    histogram.to_parquet(histogram_path, index=False)
    record_asset(histogram_path, list(histogram.columns), len(histogram))

    # The annual release covers 2017--2026; retain the small preceding tail separately.
    work_papers = frames["paper_citation_metrics"].set_index("citation_openalex_work_id_current")["paper_id"].to_dict()
    outside_rows = []
    for path in sorted((raw / "focal_batches").glob("batch-*.json.zst")):
        envelope = read_json_zst(path)
        for work in envelope["response"].get("results", []):
            paper_id = work_papers.get(work.get("id"))
            if paper_id is None:
                continue
            for annual in work.get("counts_by_year") or []:
                year, count = int(annual["year"]), int(annual["cited_by_count"])
                if year < annual_start or year > snapshot_year:
                    outside_rows.append({"paper_id": paper_id, "citation_year": year, "citation_count": count})
        record_source(path, "response.results.id used for join; only counts_by_year outside released annual window",
                      "data/citations/counts_by_year_outside_window.parquet")
    outside = pd.DataFrame(outside_rows, columns=["paper_id", "citation_year", "citation_count"])
    outside = outside.sort_values(["paper_id", "citation_year"]).reset_index(drop=True)
    outside_path = destination / "counts_by_year_outside_window.parquet"
    outside.to_parquet(outside_path, index=False)
    record_asset(outside_path, list(outside.columns), len(outside))

    method_sources = []
    for relative in ["scripts/analysis/citations/build_focal_citation_metrics_v1.py",
                     "scripts/analysis/citations/run_gpu_api_citation_association_v14.py",
                     "configs/analysis/citations/focal_citation_metrics_v1_20260912.yaml",
                     (CURRENT / "summary.json").as_posix()]:
        path = source_root / relative
        method_sources.append({"source_group": "compute", "source_path": relative,
                               "source_sha256": sha256(path), "source_bytes": path.stat().st_size})
    manifest = {"schema_version": "citation_assets_v1", "assets": assets, "sources": sources,
                "method_sources": method_sources}
    manifest_path = destination / "asset_sources.json"
    write_json(manifest_path, manifest)
    contract = {
        "schema_version": "citation_contract_v1", "analysis_id": summary["analysis_id"],
        "run_id": summary["run_id"], "collection_window_utc": summary["collection_window_utc"],
        "snapshot_year": snapshot_year, "annual_start_year": annual_start,
        "annual_end_year": snapshot_year, "annual_rows_per_paper": 10,
        "paper_id_set_sha256": summary["input"]["paper_id_set_sha256"],
        "expected_rows": {asset["path"].rsplit("/", 1)[-1]: asset["rows"] for asset in assets},
        "reference": {"corpus": "core", "work_types": ["article", "conference-paper", "preprint"],
                      "cell": ["publication_year", "primary_topic"], "shares": {"10": 0.10, "5": 0.05, "1": 0.01},
                      "target_tail_count": "max(1, ceil(share * reference_n))",
                      "ascending_rank": "reference_n - target_tail_count + 1",
                      "cutoff": "first citation-count bin whose cumulative work count reaches ascending_rank",
                      "qualifying_reference_count": "reference_n minus number strictly below cutoff",
                      "indicator": "current cited_by_count >= cutoff", "ties": "inclusive_nearest_rank",
                      "histogram_coverage": "ascending prefix sufficient for all three cutoff ranks"},
        "cumulative": {"start": "current OpenAlex publication_year", "end": snapshot_year,
                       "include_same_publication_year": True, "exclude_self_citations": False,
                       "missing_annual_bins": "zero when counts_by_year status is ok"},
        "outcomes": {"y_cumulative_citations": "log1p(citation_cumulative_from_publication_year_current)",
                     "y_top10": "top_cited_10_current", "y_top5": "top_cited_5_current", "y_top1": "top_cited_1_current"},
        "expected_metric_status_counts": summary["coverage"]["metric_status_counts"],
        "expected_reconciliation_status_counts": summary["citation_reconciliation"]["status_counts"],
        "asset_manifest": "data/citations/asset_sources.json", "asset_manifest_sha256": sha256(manifest_path),
        "source_methods": method_sources,
    }
    write_json(repo_root / "configs/citation_contract.json", contract)
    (destination / "README.md").write_text(
        "# Citation measurements\n\n"
        "The frozen OpenAlex collection was made on 12 September 2026, 07:32:17--08:10:48 UTC. "
        "The release contains 29,591 paper measurements, 295,910 annual records for 2017--2026, "
        "and 2,239 publication-year by primary-topic reference cells.\n\n"
        "Cumulative citations sum counts from the current OpenAlex publication year through 2026, "
        "including the publication year. Current cited_by_count determines the three top-cited indicators. "
        "The reference population is the OpenAlex core corpus of articles, conference papers and preprints; "
        "ties at the nearest-rank cutoff qualify. Self-citations are included.\n\n"
        "reference_histograms.parquet contains the citation-count bins needed to recompute all cutoffs. "
        "counts_by_year_outside_window.parquet retains the preceding annual tail used in reconciliation. "
        "asset_sources.json records relative source paths and SHA-256 hashes.\n\n"
        "Run `python code/citations.py` to verify the annual cumulative values, reference thresholds, "
        "top-cited indicators and baseline outcomes without network access.\n",
        encoding="utf-8",
    )
    return {"assets": len(assets), "source_files": len(sources),
            "rows": contract["expected_rows"], "bytes": sum(asset["bytes"] for asset in assets)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(export_citations(args.source_root, args.repo_root), indent=2))


if __name__ == "__main__":
    main()

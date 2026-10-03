"""Reproduce the released data summaries, impact models, measurements and figures."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tmp/matplotlib"))


def run(repo_root: Path, output_dir: Path) -> dict:
    from analysis import run_analysis
    from figures.run_figures import run_figures
    from measurement import run_measurement
    from verify import reproduce_corpus_tables, sha256, verify_inputs, verify_outputs

    for relative in ["data", "configs", "code", "manifests", "results/reference"]:
        protected = (repo_root / relative).resolve()
        if output_dir == protected or protected in output_dir.parents or output_dir in protected.parents:
            raise ValueError("Choose an output directory separate from the frozen release files")
    if (output_dir / "model_statistics/coefficients.csv").exists():
        raise FileExistsError("This run already contains fitted models; choose a fresh --output-dir")
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Checking frozen input files", flush=True)
    inputs = verify_inputs(repo_root)
    frozen_manifest_sha256 = {f"manifests/{name}": sha256(repo_root / "manifests" / name)
                              for name in ["input_manifest.json", "reference_manifest.json", "code_manifest.json"]}
    print("Reproducing corpus and resource summaries", flush=True)
    corpus = reproduce_corpus_tables(repo_root, output_dir)
    print("Fitting 32 citation-impact models", flush=True)
    analysis = run_analysis(repo_root, output_dir)
    print("Scoring classification, GPU and KSC measurements", flush=True)
    measurement = run_measurement(repo_root, output_dir / "measurement")
    print("Rendering three empirical figures", flush=True)
    figures = run_figures(repo_root, output_dir)
    print("Comparing regenerated statistics with reference outputs", flush=True)
    verification = verify_outputs(repo_root, output_dir)
    report = {"schema_version": "replication_run_v1", "passed": True,
              "frozen_manifest_sha256": frozen_manifest_sha256,
              "inputs": inputs, "corpus": corpus, "analysis": analysis,
              "measurement": measurement, "figures": figures, "verification": verification,
              "runtime_seconds": round(time.perf_counter() - started, 3),
              "package_versions": {p: importlib.metadata.version(p) for p in
                                   ["numpy", "pandas", "pyarrow", "scipy", "matplotlib", "PyYAML", "Pillow", "PyMuPDF"]}}
    (output_dir / "run_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "primary_models": 32, "figures": 3,
                      "runtime_seconds": report["runtime_seconds"]}, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/reproduced")
    args = parser.parse_args()
    run(ROOT, args.output_dir.resolve())

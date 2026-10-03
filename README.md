# Beyond More Compute

Replication data and code for **Beyond More Compute: How Capabilities and Constraints Shape Scientific Value**.

The repository reproduces citation-impact analyses from frozen paper-level data: corpus summaries, four baseline models, twelve hardware-by-API models, sixteen resource-by-complexity models, classification and GPU extraction scores, citation measurements, and three empirical figures.

## Reproduce

Use Python 3.11 and run these commands from the repository root:

```powershell
uv sync --frozen
uv run --no-sync python code/run_all.py
uv run --no-sync python code/verify.py
```

The analysis runs on a CPU using the released data. After dependency installation, reproduction runs offline. Outputs are written to `results/reproduced/`. A complete run produces `run_summary.json` and `verification.json`, full-precision tables, model covariance matrices, measurement scores, figure coordinates, and PNG/PDF/SVG figures.

An alternative installation uses `python -m pip install -r requirements.txt`, followed by `python code/run_all.py` and `python code/verify.py`.

## Contents

| Directory | Contents |
|---|---|
| `data/corpus/` | Paper index, scope rules and ID-level sample membership |
| `data/analysis_ready/` | Frozen model inputs and resource-description data |
| `data/measurement/` | Classification labels/predictions, GPU evidence and KSC measurements |
| `data/citations/` | Citation counts, reference thresholds and count histograms |
| `code/` | Data exporters, measurement scoring, regression estimation, figures and verification |
| `configs/` | Classification, hardware, citation and figure specifications |
| `results/reference/` | Frozen tables, coordinates, figures and measurement summaries |
| `manifests/` | File hashes, shapes, source lineage and result mapping |
| `docs/` | Reproduction instructions, variable definitions and methods |

The collected corpus contains 113,822 papers from 2020–2025. The eligible resource sample contains 107,872 papers. Citation models use 29,487 papers; the complexity-interaction models use 29,239 papers. Both model samples contain 69 venue-by-year clusters. Four auxiliary context models supply the complexity-model fit comparisons.

See [reproduction instructions](docs/reproduction.md), [sample construction](docs/sample_construction.md), [analysis methods](docs/analysis_methods.md), [variable definitions](docs/data_dictionary.md), [measurement methods](docs/measurement_methods.md), [figure methods](docs/figure_methods.md), and [result mapping](manifests/artifact_map.csv).

Code licensing and data attribution are described in [LICENSE](LICENSE) and [DATA_LICENSES.md](DATA_LICENSES.md).

Repository contents and maintenance are described in [repository maintenance](docs/github_repository.md).

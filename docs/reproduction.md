# Reproduction

Run the commands in the root README using Python 3.11. `uv.lock` freezes direct and transitive packages; `requirements.txt` supplies the same versions for pip. Estimation and drawing use a CPU.

`code/run_all.py` runs these steps:

1. Check SHA-256 hashes of the frozen data, configurations, reference files and implementation.
2. Aggregate conference-by-year counts, access-mode counts and the eleven sample stages.
3. Fit 32 primary models and four auxiliary context models, retaining coefficients, clustered uncertainty and full covariance matrices.
4. Score four classification evaluations and two GPU extraction modes; check hardware supplementation, KSC arithmetic and citation measurements.
5. Recompute annual resource summaries and draw the three empirical figures from regenerated coordinates.
6. Compare model statistics, tables and coordinates with the reference values; check figure alignment, fonts and collisions.

The program stops when a check fails. `results/reproduced/run_summary.json` records the result, runtime and installed scientific package versions. `results/reproduced/verification.json` records comparisons and their maximum numerical errors.

## Outputs

| Path under `results/reproduced/` | Result |
|---|---|
| `tables/` | Corpus summaries, primary estimates, interactions, Holm adjustments and conditional associations |
| `model_statistics/coefficients.csv` | All primary and auxiliary coefficient estimates |
| `model_statistics/model_metrics.csv` | Model role, sample size, rank, clusters and fit statistics |
| `model_statistics/covariance_full.json` | Ordered terms, estimates, covariance matrices and sample hashes |
| `figure_data/` | Annual summaries and model-derived plot coordinates |
| `measurement/classification/` | Model-level and label-level precision, recall, F1 and split membership checks |
| `measurement/gpu/` | Extraction scores, normalization inventory and generation-mean checks |
| `measurement/ksc/` | Historical-window arithmetic and threshold coverage |
| `figures/` | Three figures in PNG, PDF and SVG |
| `figures/qa/` | Alignment, font, collision and rendering checks |

To verify an existing run, execute `uv run --no-sync python code/verify.py`. To save a separate run, append `--output-dir results/reproduced_second` to both commands.

## Source export

The repository data are already frozen. The exporter scripts record how that snapshot was assembled from the research workspace. `code/prepare_data.py`, `code/prepare_reference.py`, `code/prepare_citations.py` and `code/prepare_retrieval.py` accept `--source-root`. Measurement export uses `code/measurement/prepare_assets.py --source-root SOURCE --ksc-release KSC_RELEASE --repo-root DESTINATION`. The source aliases and hashes in the manifests identify the matching frozen research products; each script lists its arguments through `--help`.

After an intentional snapshot update, run `python code/build_manifests.py`, then the complete reproduction in a fresh output directory. Save that run's `run_summary.json` as `manifests/validated_run.json` to record the verified repository state. The summary stores the three frozen manifest hashes and the package versions used for reproduction.

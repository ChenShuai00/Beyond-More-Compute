The `run_analysis(repo_root, output_dir)` entry reads four frozen Parquet files in
`data/analysis_ready` and writes the citation-impact estimates and figure data.
Import it with `from analysis import run_analysis` after adding `code/` to the
Python module path.

The model set contains four baseline models, twelve hardware-by-API models and
sixteen resource-by-KSC models. Four context models support the incremental R²
comparisons for KSC and have `role=auxiliary_context` in the model metrics.

`statistics.py` preserves the original two-way fixed-effect absorption,
Venue-by-Year clustered covariance, full-design-rank finite-cluster correction,
Student t inference and Holm adjustment. The fixed samples contain 29,487 and
29,239 papers, each with 69 clusters. Hardware centers and the KSC mean and SD
are taken from the frozen input populations.

Outputs include full-precision CSV tables, model statistics, a long covariance
table, `model_statistics/covariance_full.json`, 606 hardware/API plot coordinates,
404 KSC plot coordinates, and 48 conditional associations. The JSON covariance
records include the ordered terms, coefficients, sample hashes and correction.
The API figure uses differences in log(1+citations), divided by the outcome SD,
relative to no API at median hardware. KSC curves use exponentiated percentage
changes for cumulative citations and percentage points for binary outcomes.

Required packages are NumPy, pandas, PyArrow and SciPy on Python 3.11.
Run `check_reference.py --output-dir PATH --reference-dir results/reference` to
compare regenerated coefficients, clustered SEs, p-values, model fit, confidence
intervals and plot coordinates with the frozen reference outputs.

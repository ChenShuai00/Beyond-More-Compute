# Empirical figure methods

The figure runner generates three scientific figures from the released paper-level data and fitted-model coordinates. Run the complete workflow with:

```powershell
python code/run_all.py --output-dir results/reproduced
```

The analysis step writes model coordinates to `results/reproduced/figure_data/`. The figure step writes PNG, PDF and SVG exports to `results/reproduced/figures/` and measured export checks to `results/reproduced/figures/qa/`.

## Annual resource distributions and configurations

`combined_computational_resource_evolution` reads `data/analysis_ready/resource_descriptive.parquet`. The 107,872 eligible papers supply each publication year's configuration denominator. Hardware is identified by `resource_L1 = 1`, `resource_L2 = 0` and `resource_L3 = 0`; API access is `resource_L4 = 1`. These indicators assign each paper to Hardware only, Hardware + API, API only or Neither identified.

The three hardware panels use 34,483 complete, positive, configuration-aligned records from the 49,090-paper hardware cohort. Their variables are `compute_selected_gpu_count`, `compute_selected_per_gpu_capability` and `compute_selected_vram_gb`. Performance is divided by 10¹² for display in TFLOP/s. The selected 3,524 imputed performance configurations remain in the summaries. The complete flag is checked against the joint positive-field predicate.

For each year and hardware variable, pandas linear quantiles supply the median, 25th percentile, 75th percentile and 90th percentile. Solid lines show medians, shaded bands show the interquartile range and dashed lines show the 90th percentile. The configuration panel shows annual shares; its auxiliary zoom expands the two API configurations.

The runner saves the 24 configuration rows and 18 hardware-summary rows as `annual_hardware_api_configurations.csv` and `annual_gpu_hardware_configuration_quantiles.csv` in `figure_data/`.

## Hardware associations by API access

`interaction_slopes_cumulative_citations` reads `cumulative_citation_interaction_predictions.csv` and `cumulative_citation_interaction_contrasts.csv` from `figure_data/`. The 606 coordinate rows form six curves: 101 hardware values for each hardware dimension and API group. The three contrast rows supply the displayed interaction p-values.

The x-coordinate is `doublings_from_full_v14_median`. `adjusted_change`, `ci_low` and `ci_high` describe the fitted change in `log(1 + citations)`, divided by the outcome standard deviation, relative to API = no at median hardware. The 95% intervals use venue-by-year-clustered covariance, including covariance between model terms. The figure preserves this display scale and distinguishes the API groups with color and line style.

## Associations across knowledge-space complexity

`ksc_marginal_associations` reads `marginal_figure_source.csv`. Its 404 rows form four panels of 101 standardized-KSC coordinates: GPU quantity for cumulative citations, Top-Cited 10% and Top-Cited 5%, and API access for Top-Cited 1%.

Each curve follows the main resource coefficient plus the resource-by-KSC coefficient times `ksc_z`. Coordinates span the fitted sample's 5th through 95th KSC percentiles. `marginal_association`, `marginal_ci_low` and `marginal_ci_high` supply the displayed curve and its 95% interval. Cumulative-citation contrasts use `100 × [exp(estimate) − 1]`; LPM contrasts use percentage points. The intervals use the full covariance of the resource main effect and interaction. The dashed vertical line marks mean KSC and the horizontal line marks zero association.

## Export and review

The Matplotlib styles preserve the reference figures' panel arrangement, colors, labels, line styles and interval bands. PNG previews use 600 dpi. PDF and SVG exports retain editable text.

Each run measures final panel geometry with a 1.5 pt alignment tolerance, scans the exported PDF for a 5 pt text floor and checks text, stroke and page-boundary collisions. The annual figure's auxiliary zoom is recorded separately because its extent and scale differ from the main panels. Coordinate checks require finite estimates, ordered confidence limits and distinct increasing x-values for every curve.

`figures/qa/figure_validation.json` combines the three figures' actual alignment, font and collision results, export inventory and recorded visual review. The panel-by-panel review is stored in `configs/figures/qa_review.json` and bound to a SHA-256 fingerprint of the drawing styles, export settings and annual-figure configuration. Each run checks that fingerprint and records the current PNG's dimensions and pixel comparison with the reviewed reference. Numeric model-coordinate and annual-summary comparisons are handled by the repository verifier.

To inspect the drawing code using saved coordinates, run:

```powershell
python code/figures/run_figures.py --output-dir .tmp/figure-review --from-reference
```

Reference exports and their CSV/Parquet coordinates are in `results/reference/figures/` and `results/reference/figure_data/`. Source-to-release mappings are recorded in `code/figures/asset_sources.json`.

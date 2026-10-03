# Citation-impact replication methods

The analysis reads the frozen Parquet files in `data/analysis_ready/`. It estimates
32 primary models and four supporting context models, then writes full-precision
tables, covariance matrices and figure coordinates. The statistical implementation
is in [replication.py](../code/analysis/replication.py) and
[statistics.py](../code/analysis/statistics.py).

## Run and inspect

From the repository root, install the locked environment and run the workflow:

```powershell
uv sync --frozen
uv run --no-sync python code/run_all.py
```

The default output directory is `results/reproduced/`. For a subsequent run,
choose a fresh directory:

```powershell
uv run --no-sync python code/run_all.py --output-dir results/reproduced_second_run
```

The analysis module also exposes `run_analysis(repo_root, output_dir)` through
`from analysis import run_analysis` when `code/` is on the Python module path.
The module reads its inputs and writes results under the chosen output directory.

## Samples and variables

Each of the four model input and membership files has 29,591 rows, keyed by
`paper_id`. The baseline and hardware-by-API models use 29,487 papers:
29,547 eligible papers followed by removal of 60 recursive fixed-effect
singletons. The KSC models use 29,239 papers: 29,322 with available KSC,
29,279 jointly eligible papers and removal of 40 recursive singletons. The
included flags in the frozen membership files select these samples.

Every model contains Venue-by-Year and OpenAlex Primary Topic fixed effects.
The baseline sample has 671 topic levels; the KSC sample has 599. Inference
clusters observations into the same 69 Venue-by-Year groups. The runner checks
the included paper IDs, sample hashes, group counts and singleton condition.

The four outcomes are `y_cumulative_citations`, `y_top10`, `y_top5` and `y_top1`.
Cumulative citations use the natural logarithm `log(1 + citations)` and OLS.
The three binary outcomes use linear probability models. Citation counts were
collected during 2026-09-12 07:32:17–08:10:48 UTC. The cumulative count sums
OpenAlex annual counts from the current OpenAlex publication year through the
collection year, including the publication year. Top-Cited indicators compare
the current total citation count with the external publication-year-by-primary-topic
reference distribution. The reference contains OpenAlex core works of type
article, conference-paper or preprint. The descending rank `ceil(p*N)` defines
each top-share cutoff; all works tied at that cutoff qualify.

Hardware variables are log2 GPU count, log2 per-GPU theoretical peak FLOP/s and
log2 per-GPU VRAM in GB. Subtracting their medians across the full 29,591-paper
input gives the centered variables. The stored centers are:

| Variable | Median on its log2 scale |
| --- | ---: |
| `quantity` | 0.0 |
| `performance` | 46.7591667416251 |
| `memory` | 4.584962500721156 |

A one-unit hardware change represents a doubling. The hardware measures derive
from the selected normalized configuration with the highest measurable
GPU-count-by-per-GPU theoretical performance. Its count, performance and VRAM
remain aligned to that configuration. Generation-based performance supplementation
uses the arithmetic mean across unique GPU models with positive original values
in the same generation, followed by configuration reselection.

`api` codes hosted-model/API evidence as 1 or 0. Controls are natural
`log(1 + author count)`, natural `log(1 + organization count)`, binary company
participation and prior team productivity. Prior team productivity is the mean
across team authors of `log(1 + eligible publications during [t-5,t-1])`.
This saved team-level measure enters the regression directly.

## Knowledge-space complexity

Historical knowledge-space complexity uses the frozen OpenAlex AI Topic
knowledge space and SPECTER reference embeddings. The embeddings use 768
dimensions, 512-token inputs, title-plus-abstract text, mean pooling of the final
hidden states including padding, and L2 normalization.

For an eligible historical work, its semantic span is the mean pairwise cosine
distance among usable direct-reference vectors. With `m` unit vectors `z_r`,
the implementation calculates:

```text
H_p = 1 - (||sum_r z_r||^2 - m) / (m*(m-1))
```

Reference IDs are deduplicated. Same-year references are eligible; future-year
references are excluded. A historical work qualifies with at least five usable
reference vectors and at least 70% vector coverage of its parseable direct-reference
IDs. For focal year `t`, KSC is the equal-weight mean of eligible historical
works in the same primary topic over `[t-5,t-1]`, requiring at least 100 works.
The focal-work mapping applies work-specific self-exclusion before evaluating
that minimum. The frozen mapping has 285 focal works with their own span
subtracted and 113 topic-year cells with focal-specific KSC values.

The base topic-year table contains 2,167 cells. The companion threshold table
has 6,501 rows for minimum-history thresholds of 50, 100 and 200. The model
inputs use the focal mapping at the 100-work threshold. KSC is standardized over
all 29,322 available focal values using population mean `0.2212022498317855`
and population SD `0.014462611340978518` (`ddof=0`). The resulting `ksc_z`
is retained when the final model sample is selected.

## Model specifications

| Analysis | Model names | Terms beyond controls and fixed effects | Number of fits |
| --- | --- | --- | ---: |
| `baseline_api` | `M2` | Three centered hardware variables and API | 4 |
| `baseline_api` | `M3_quantity`, `M3_performance`, `M3_memory` | M2 plus one hardware-by-API interaction per model | 12 |
| `ksc` | `M3_quantity`, `M3_performance`, `M3_memory`, `M3_api` | All four resources, `ksc_z`, and one resource-by-KSC interaction per model | 16 |
| `ksc` | `M2_context` | All four resources and `ksc_z` | 4 |

Each specification is fitted separately for each outcome. The four
`M2_context` fits supply the restricted models for the KSC increment comparisons
and carry `role=auxiliary_context`. The other 32 fits carry `role=primary`.
The stable `model_id` combines `analysis/outcome/model`.

## Estimation and inference

Alternating group demeaning absorbs the two fixed effects with tolerance
`1e-11`. OLS is then fitted to the residualized outcome and regressors.
The fixed-effect rank is calculated from the connected components of the
two-way group graph. With `R` equal to this rank plus the regressor count,
the clustered covariance uses the factor:

```text
(G / (G-1)) * ((N-1) / (N-R)), where G = 69
```

Coefficient p-values and 95% confidence intervals use Student t with 68 degrees
of freedom. A linear contrast with weight vector `w` uses estimate `w'beta`
and variance `w'Vw`, retaining main-effect/interaction covariance. Nested-model
Wald tests use the clustered covariance and an F distribution with the number
of tested restrictions and 68 denominator degrees of freedom.

Holm adjustment applies across the four outcomes separately for each of the
four baseline resource coefficients, each of the three hardware-by-API terms,
and each of the four resource-by-KSC terms. Conditional-association p-values
form separate four-outcome families for each resource at each of the three KSC
levels. The saved tables retain raw and adjusted p-values. Baseline and API
coefficient stars use raw p-values; KSC interaction stars use Holm-adjusted
p-values. The star thresholds are 0.01, 0.05 and 0.10. Confidence bands use the
pointwise 95% t intervals.

## Model fit and displayed associations

Let `SSE` denote residual sum of squares, `TSS` the mean-centered raw outcome
sum of squares and `within_TSS` its sum of squares after fixed-effect absorption.
The output definitions are:

```text
r2                = 1 - SSE/TSS
within_r2         = 1 - SSE/within_TSS
adjusted_r2       = 1 - (1-r2)*(N-1)/(N-R)
delta_r2          = (SSE_restricted-SSE_full)/TSS
delta_within_r2   = (SSE_restricted-SSE_full)/within_TSS
partial_r2        = (SSE_restricted-SSE_full)/SSE_restricted
```

API increments compare each M3 hardware-by-API model with M2 on the same sample.
KSC increments compare each resource-by-KSC model with `M2_context`.

The hardware-by-API figure uses 101 values between the hardware variable's 5th
and 95th sample percentiles for each hardware dimension and API group, giving
606 coordinates. At centered hardware `x` and API state `a`, its plotted
difference is:

```text
[beta_h*x + a*(beta_api + theta_h*x)] / SD(log(1+citations))
```

The baseline for this difference is API=0 at median hardware. The outcome SD
is calculated over the 29,487-paper estimation sample with `ddof=0` and equals
`1.5527016896753016`. Confidence limits receive the same SD scaling.

For KSC value `c`, the conditional resource association is `beta_r + phi_r*c`.
The output includes all resource/outcome associations at `c=-1,0,+1`, giving
48 rows. Four displayed resource/outcome pairs use 101 KSC values between the
5th and 95th sample percentiles, giving 404 coordinates. Cumulative-citation
effects and interval endpoints use `100*(exp(b)-1)`, expressing percentage change
in one plus citations. Binary-outcome coefficients and intervals are multiplied
by 100 to express percentage points. Coefficients describe conditional
resource–impact associations.

## Key outputs and comparison

| Output under the run directory | Contents |
| --- | --- |
| `model_statistics/coefficients.csv` | 336 coefficient rows across 32 primary and four context fits |
| `model_statistics/model_metrics.csv` | Sample, role, rank, correction, and fit statistics for 36 models |
| `model_statistics/covariance.csv` | Long covariance table keyed by model and ordered term pair |
| `model_statistics/covariance_full.json` | Ordered terms, coefficient vectors and complete covariance matrices |
| `model_statistics/model_comparisons.csv` | 12 API and 16 KSC nested-model comparisons |
| `tables/` | Resource main effects, baseline and interaction tables, KSC details and model statistics |
| `figure_data/cumulative_citation_interaction_predictions.csv` | 606 API figure coordinates and confidence limits |
| `figure_data/cumulative_citation_interaction_contrasts.csv` | Three hardware slope contrasts for cumulative citations |
| `figure_data/marginal_figure_source.csv` | 404 KSC figure coordinates on model and display scales |
| `figure_data/conditional_associations.csv` | 48 conditional associations with raw and Holm p-values |

The top-level workflow compares regenerated results with `results/reference/`.
[check_reference.py](../code/analysis/check_reference.py) also supports a standalone
check using `--output-dir` and `--reference-dir`. It joins stable result keys,
compares coefficients, SEs, p-values, R² and plot coordinates, and checks each
covariance matrix against the corresponding coefficient SEs. The numerical
comparison uses `rtol=1e-9` and `atol=1e-11`.

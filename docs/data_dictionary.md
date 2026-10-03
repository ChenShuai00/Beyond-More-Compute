# Core data dictionary

The Parquet files contain one row per `paper_id`. The machine-readable dataset
and column inventory is [data_dictionary.json](../data/data_dictionary.json).
This document supplies the units, construction rules and analysis roles for the
core fields. See [analysis_methods.md](analysis_methods.md) for the estimators.

## Files and keys

| File in `data/analysis_ready/` | Rows | Columns | Purpose |
| --- | ---: | ---: | --- |
| `resource_descriptive.parquet` | 107,872 | 20 | Resource access counts and annual hardware/configuration summaries |
| `baseline_impact.parquet` | 29,591 | 30 | Frozen outcomes, resources and controls for baseline/API models |
| `baseline_membership.parquet` | 29,591 | 11 | Eligibility, singleton removal and inclusion for 29,487 model rows |
| `ksc_impact.parquet` | 29,591 | 45 | Resources/outcomes plus historical KSC mapping and interaction variables |
| `ksc_membership.parquet` | 29,591 | 14 | KSC mapping and eligibility/inclusion for 29,239 model rows |

`paper_id` is the join key across these files. OpenAlex Work IDs identify the
linked bibliographic record. `year` is the canonical conference/corpus year;
`citation_openalex_publication_year` is the year returned by the citation
snapshot. Each year field retains its own role in the frozen construction.

## Identity, fixed effects and citation outcomes

| Field | Meaning and use |
| --- | --- |
| `paper_id` | Unique focal-record identifier; unit of analysis and membership joins |
| `year`, `venue` | Canonical corpus publication year and venue |
| `venue_year_v14` | Frozen combined venue/year label used for the first fixed effect and clustering |
| `base_primary_topic_id` | Frozen integrated-data OpenAlex Primary Topic ID, retained in the baseline input |
| `citation_openalex_primary_topic_id`, `citation_openalex_primary_topic_name` | Primary Topic recorded in the citation snapshot; the ID supplies the regression topic fixed effect |
| `citation_openalex_publication_year` | Publication year recorded in the citation snapshot; used in citation accumulation and the top-cited reference cell |
| `citation_openalex_type` | OpenAlex work type recorded in the citation snapshot |
| `citation_metric_status` | `ok`, `ranking_ineligible_work_type`, or `not_found`; the input has 29,547, 42 and 2 rows, respectively |
| `citation_cumulative_from_publication_year_current` | Sum of annual citation counts from the current OpenAlex publication year through 2026, including same-year citations |
| `citation_current_total` | OpenAlex `cited_by_count` preserved at collection time; the quantity used for top-cited comparisons |
| `y_cumulative_citations` | Natural `log(1 + citation_cumulative_from_publication_year_current)`; OLS dependent variable |
| `y_top10`, `y_top5`, `y_top1` | Binary top-cited indicators; 1 when the current total reaches the corresponding external year/topic threshold, with cutoff ties included |
| `fe_venue_year`, `cluster_group` | Saved aliases of `venue_year_v14` in the KSC input; the runner derives the same aliases for each selected sample |
| `fe_topic` | Saved alias of `citation_openalex_primary_topic_id` in the KSC input |

Citation collection took place on 2026-09-12, 07:32:17–08:10:48 UTC. Annual
citation accumulation includes self-citations as represented in OpenAlex.
Top-cited reference cells use OpenAlex core article, conference-paper and
preprint records in the same publication year and primary topic. The eligible
binary outcomes obey `y_top1 <= y_top5 <= y_top10`.

## Regression resources and controls

| Field | Scale and definition |
| --- | --- |
| `quantity` | `log2(selected GPU count)` |
| `performance` | `log2(selected per-GPU theoretical peak FLOP/s)`; the underlying performance unit is FLOP/s |
| `memory` | `log2(selected per-GPU VRAM in GB)` |
| `quantity_centered` | `quantity - 0.0` |
| `performance_centered` | `performance - 46.7591667416251` |
| `memory_centered` | `memory - 4.584962500721156` |
| `api` | 1 for reported hosted-model/API use and 0 for no reported API use in the frozen classification |
| `api_x_quantity`, `api_x_performance`, `api_x_memory` | API multiplied by the corresponding centered hardware variable |
| `log_team` | Natural `log(1 + author count)` |
| `log_org` | Natural `log(1 + organization count)` |
| `company` | 1 for company participation and 0 for no company participation |
| `prior_team_productivity_5y` | Team-author mean of `log(1 + prior_publications_5y)`; entered directly in the regression |

The hardware centers are medians over the full 29,591-row input. A unit change
in a hardware log2 measure is a doubling. The selected GPU count, performance
and VRAM describe the same normalized configuration.

For prior team productivity, each author's count includes distinct eligible
OpenAlex works in `[year-5,year-1]`. Eligible team measurements have resolved
author identities and complete histories for all listed authors. The stored
team measure averages the authors' log-transformed counts.

## Resource-description and GPU audit fields

| Field | Meaning and units |
| --- | --- |
| `resource_L1` | Binary physical-hardware access label |
| `resource_L2` | Binary commercial-cloud access label |
| `resource_L3` | Binary public/shared-HPC access label |
| `resource_L4` | Binary hosted-model/API-service access label |
| `compute_gpu_scale_strict_main_sample` | Current complete, positive, configuration-aligned hardware flag, including eligible supplemented performance values |
| `compute_strict_gpu` | Preserved historical strict-GPU flag |
| `compute_selected_gpu_count` | GPU count in the selected normalized configuration |
| `compute_selected_per_gpu_capability` | Per-GPU theoretical Tensor FP16/BF16 peak performance in FLOP/s; divide by `10^12` for TFLOP/s |
| `compute_selected_vram_gb` | Per-GPU VRAM capacity in GB |
| `compute_benchmark_gpu_name` | Normalized GPU benchmark-model name used for specification matching |
| `compute_selected_config_id` | Identifier of the selected normalized configuration row |
| `compute_selected_config_total_compute_capability` | Selected GPU count multiplied by its per-GPU theoretical peak performance, in FLOP/s |
| `compute_performance_imputed` | Whether the final selected configuration uses generation-mean performance supplementation |
| `compute_performance_original_flops` | Original per-GPU performance value retained for the selected configuration |
| `compute_performance_donor_model_count` | Number of unique GPU models contributing positive original values to the same-generation supplementation mean |
| `compute_has_count_conflict` | Paper-level indicator of conflicting GPU-count mentions in normalized extraction records |
| `compute_has_missing_count_mention` | Paper-level indicator that a normalized GPU mention lacks a count |

Access labels are multi-label. The hardware-provenance cohort satisfies
`resource_L1=1`, `resource_L2=0`, `resource_L3=0` and contains 49,090 rows.
The annual hardware quantiles additionally select the complete hardware flag,
giving 34,483 rows, including 3,524 supplemented-performance configurations.
The four configuration categories use the hardware-provenance predicate and
`resource_L4`: hardware only, hardware plus API, API only, and neither identified.
Their annual denominator is all eligible resource-description rows in that year.

## KSC fields

| Field | Meaning |
| --- | --- |
| `focal_work_id` | OpenAlex Work ID used for the focal KSC mapping |
| `focal_year` | Canonical focal year that anchors the prior five-year history window |
| `ksc_primary_topic_id`, `ksc_primary_topic_name` | Focal Primary Topic used for historical KSC mapping |
| `base_eligible_history_count` | Eligible historical works in the topic/window before focal-work self-exclusion |
| `focal_self_history_conflict` | Whether the focal work's bibliographic year places it in its own prior-window history |
| `own_h_p_subtracted` | Whether its own eligible semantic span was subtracted from the focal mapping's historical sum and count |
| `own_h_p` | Focal work's historical-reference semantic span when available for the self-exclusion audit |
| `eligible_historical_paper_count_after_self_exclusion` | Topic/window historical support after the focal adjustment |
| `ksc` | Mean historical reference-semantic span in the same topic during `[focal_year-5,focal_year-1]`, using at least 100 eligible works after self-exclusion |
| `mapping_status` | `available` for 29,322 focal rows or `insufficient_history` for 269 rows |
| `ksc_z` | `(ksc - 0.2212022498317855) / 0.014462611340978518`, using the available-input population |
| `quantity_x_ksc`, `performance_x_ksc`, `memory_x_ksc`, `api_x_ksc` | Corresponding resource main-effect variable multiplied by `ksc_z` |

The focal mapping preserves paper-specific self-exclusion. The measurement
files in `data/measurement/ksc/` include the 29,591-row focal mapping, the
2,167-row base topic-year table and the 6,501-row threshold table for 50, 100
and 200 minimum historical works. `ksc` and `ksc_z` are missing for an
insufficient-history mapping.

## Membership fields

| File and field | Meaning |
| --- | --- |
| Baseline: `eligible_impact` | Joint outcome and fixed-effect eligibility before recursive singleton removal |
| Baseline: `singleton_iteration_impact` | Removal iteration; zero indicates the row was not removed as a singleton |
| Baseline: `included_impact` | Final inclusion in the 29,487-row baseline/API sample |
| KSC: `eligible_venue_year_topic_impact` | Joint KSC, outcome and fixed-effect eligibility before singleton removal |
| KSC: `singleton_iteration_venue_year_topic_impact` | Removal iteration for the KSC specification |
| KSC: `included_venue_year_topic_impact` | Final inclusion in the 29,239-row KSC sample |

The wider ID-level sequence is in `data/corpus/sample_membership.parquet`.
Its `corpus_included`, `scope_metadata_text_eligible`, `hardware_eligible`,
`complete_hardware`, `analysis_frame_included`, `baseline_included` and
`ksc_included` fields identify successive analysis populations. The baseline
and KSC membership files contain the final statistical-sample flags.

## Result keys and units

`model_statistics/coefficients.csv` has one row per `(model_id, term)`.
`model_id` is `analysis/outcome/model`; use the full identifier when joining
baseline/API and KSC results. `coefficient`, `standard_error`, `ci_low` and
`ci_high` use the fitted outcome scale. `p_value` is the clustered t-test value;
`p_value_holm` stores the applicable four-outcome family adjustment.

Model metrics have one row per `model_id`. `n`, `clusters`, `fe_rank` and
`design_rank` describe the sample and full design. `r2`, `within_r2` and
`adjusted_r2` retain their definitions in the methods document. Covariance
CSV rows are keyed by `(model_id, term, other_term)`; the JSON records preserve
the same ordered term list and matrix.

In API figure data, `doublings_from_full_v14_median` is centered hardware on its
log2 scale, and `adjusted_change` plus its confidence limits use the SD scale
of `log(1+citations)`. In KSC figure and conditional-association data,
`estimate` uses the fitted outcome scale; `marginal_association` and
`marginal_ci_low/high` use percentage change in one plus citations or
percentage points, as identified by `marginal_unit`.

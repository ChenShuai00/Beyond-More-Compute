"""Reproduce the baseline, hardware/API, and KSC citation-impact models."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from .statistics import Fit, absorb, compare_nested, fit_absorbed, fixed_effect_rank, holm

OUTCOMES = ("cumulative_citations", "top10", "top5", "top1")
CONTROLS = ("log_team", "log_org", "company", "prior_team_productivity_5y")
RESOURCES = {"quantity": "quantity_centered", "performance": "performance_centered", "memory": "memory_centered", "api": "api"}
MAIN = list(RESOURCES.values())
API_TERMS = {name: f"api_x_{name}" for name in ("quantity", "performance", "memory")}
KSC_TERMS = {name: f"{name}_x_ksc" for name in RESOURCES}
OUTCOME_LABELS = {"cumulative_citations": "Cumulative citations, log(1+y)", "top10": "Top 10% cited", "top5": "Top 5% cited", "top1": "Top 1% cited"}
RESOURCE_LABELS = {"quantity": "GPU quantity", "performance": "Per-GPU performance", "memory": "Per-GPU VRAM", "api": "API access"}
EXPECTED_SAMPLES = {"baseline_api": (29487, "d6807e4979873472ffa7565585fbeba4d5951f0ac229c12a9e95f5a10513ffb7"), "ksc": (29239, "1b6b84deb2b845abca82df6d98abdae1d2e8d682005decb2759a2810247007cc")}
KSC_MEAN = 0.2212022498317855
KSC_SD = 0.014462611340978518


def sample_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(("\n".join(sorted(frame.paper_id.astype(str))) + "\n").encode()).hexdigest()


def _sample(repo_root: Path, analysis: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    stem = "baseline" if analysis == "baseline_api" else "ksc"
    root = repo_root / "data/analysis_ready"
    data = pd.read_parquet(root / f"{stem}_impact.parquet")
    membership = pd.read_parquet(root / f"{stem}_membership.parquet")
    if len(data) != 29591 or len(membership) != 29591:
        raise ValueError(f"{analysis}: expected 29,591 input and membership rows")
    if data.paper_id.duplicated().any() or membership.paper_id.duplicated().any():
        raise ValueError(f"{analysis}: duplicate paper IDs")
    if set(data.paper_id) != set(membership.paper_id):
        raise ValueError(f"{analysis}: input/membership paper IDs differ")
    for resource in API_TERMS:
        centered = f"{resource}_centered"
        if not np.allclose(data[centered], data[resource] - data[resource].median(), atol=1e-12, rtol=0):
            raise ValueError(f"{analysis}: frozen {resource} centering changed")
    flag = "included_impact" if analysis == "baseline_api" else "included_venue_year_topic_impact"
    if membership[flag].isna().any() or not membership[flag].isin([True, False]).all():
        raise ValueError(f"{analysis}: invalid frozen membership flag")
    sample = data.loc[data.paper_id.isin(membership.loc[membership[flag], "paper_id"])].copy()
    n, expected_hash = EXPECTED_SAMPLES[analysis]
    if len(sample) != n or sample_sha256(sample) != expected_hash:
        raise ValueError(f"{analysis}: frozen regression sample changed")
    sample["fe_venue_year"] = sample.venue_year_v14
    sample["fe_topic"] = sample.citation_openalex_primary_topic_id
    sample["cluster_group"] = sample.venue_year_v14
    if sample.cluster_group.nunique() != 69:
        raise ValueError(f"{analysis}: expected 69 venue-year clusters")
    if any(sample.groupby(column).paper_id.transform("size").eq(1).any() for column in ("fe_venue_year", "fe_topic")):
        raise ValueError(f"{analysis}: frozen sample contains fixed-effect singletons")
    if not data.api.isin([0, 1]).all():
        raise ValueError("API must be binary")
    if analysis == "baseline_api":
        for resource, term in API_TERMS.items():
            sample[term] = sample[RESOURCES[resource]] * sample.api
    else:
        available = data.ksc.notna()
        if int(available.sum()) != 29322:
            raise ValueError("Expected 29,322 KSC-available input papers")
        if not math.isclose(float(data.loc[available, "ksc"].mean()), KSC_MEAN, abs_tol=1e-13, rel_tol=0) or not math.isclose(float(data.loc[available, "ksc"].std(ddof=0)), KSC_SD, abs_tol=1e-13, rel_tol=0):
            raise ValueError("Frozen KSC standardization population changed")
        if not np.allclose(data.loc[available, "ksc_z"], (data.loc[available, "ksc"] - KSC_MEAN) / KSC_SD, atol=1e-11, rtol=0):
            raise ValueError("Frozen KSC z values changed")
        for resource, term in KSC_TERMS.items():
            sample[term] = sample[RESOURCES[resource]] * sample.ksc_z
    required = [*MAIN, *CONTROLS, *[f"y_{outcome}" for outcome in OUTCOMES]]
    if not np.isfinite(sample[required].to_numpy(dtype=float)).all():
        raise ValueError(f"{analysis}: nonfinite model inputs")
    return sample, membership


def _contrast(fit: Fit, weights: np.ndarray) -> dict:
    estimate = float(weights @ fit.beta)
    se = math.sqrt(max(0.0, float(weights @ fit.covariance @ weights)))
    critical = float(t.ppf(0.975, fit.clusters - 1))
    p_value = float(2 * t.sf(abs(estimate / se), fit.clusters - 1)) if se else (1.0 if estimate == 0 else 0.0)
    return {"estimate": estimate, "standard_error": se, "ci_low": estimate - critical * se, "ci_high": estimate + critical * se, "p_value": p_value}


def _scaled(outcome: str, item: dict) -> dict:
    if outcome == "cumulative_citations":
        estimate = 100 * math.expm1(item["estimate"])
        se = 100 * math.exp(item["estimate"]) * item["standard_error"]
        low, high = (100 * math.expm1(item[name]) for name in ("ci_low", "ci_high"))
        unit = "percent_change_in_1_plus_outcome"
    else:
        estimate, se, low, high = (100 * item[name] for name in ("estimate", "standard_error", "ci_low", "ci_high"))
        unit = "percentage_points"
    return dict(marginal_association=estimate, marginal_standard_error=se, marginal_ci_low=low, marginal_ci_high=high, marginal_unit=unit)


def _adjust(frame: pd.DataFrame, groups: list[str], raw: str = "p_value", target: str = "p_value_holm") -> pd.DataFrame:
    result = frame.copy()
    for _, indices in result.groupby(groups, sort=False).groups.items():
        result.loc[indices, target] = holm(result.loc[indices, raw].tolist())
    return result


def _write(frame: pd.DataFrame, root: Path, relative: str) -> None:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, encoding="utf-8")


def _stars(value: float) -> str:
    return "***" if value < 0.01 else "**" if value < 0.05 else "*" if value < 0.1 else ""


def write_paper_tables(coefficients: pd.DataFrame, metrics: pd.DataFrame, interactions: pd.DataFrame, output_dir: Path) -> None:
    """Format paper tables from the same full-precision model output."""
    baseline = coefficients.loc[(coefficients.analysis == "baseline_api") & (coefficients.model == "M2")].copy()
    baseline = baseline.merge(metrics[["model_id", "r2", "design_rank", "adjusted_r2"]], on="model_id", validate="many_to_one")
    baseline["coefficient_scale"] = "raw_model"
    baseline["estimator"] = baseline.outcome.map(lambda value: "OLS" if value == "cumulative_citations" else "LPM")
    baseline["stars"] = baseline.p_value.map(_stars)
    baseline["coefficient_display"] = baseline.coefficient.map(lambda value: f"{0.0 if abs(value) < 0.0005 else value:.3f}")
    baseline["standard_error_display"] = baseline.standard_error.map(lambda value: f"{value:.3f}")
    baseline["venue_year_fixed_effects"] = True
    baseline["primary_topic_fixed_effects"] = True
    _write(baseline, output_dir, "tables/baseline_regression_impact_table.csv")
    main = []
    for outcome in OUTCOMES:
        rows = baseline.loc[baseline.outcome == outcome].set_index("term")
        item = {"family": "impact", "outcome": outcome, "n": int(rows.n.iloc[0]), "clusters": 69, "adjusted_r2": float(rows.adjusted_r2.iloc[0])}
        for resource, term in RESOURCES.items():
            source = rows.loc[term]
            for field in ("coefficient", "standard_error", "p_value", "p_value_holm", "stars"):
                item[f"{resource}_{field}"] = source[field]
        main.append(item)
    _write(pd.DataFrame(main), output_dir, "tables/main_effects.csv")
    appendix = coefficients.loc[(coefficients.analysis == "baseline_api") & coefficients.model.str.startswith("M3_") & (coefficients.term.isin(MAIN) | coefficients.term.isin(API_TERMS.values()))].copy()
    appendix = appendix.merge(metrics[["model_id", "r2", "design_rank", "adjusted_r2"]], on="model_id", validate="many_to_one")
    appendix["hardware"] = appendix.model.str.removeprefix("M3_")
    appendix["panel"] = appendix.hardware.map({"quantity": "A", "performance": "B", "memory": "C"})
    appendix["coefficient_scale"] = "raw_model"
    appendix["stars"] = appendix.p_value.map(_stars)
    _write(appendix, output_dir, "tables/appendix_interaction_impact_table.csv")
    source = interactions.copy()
    source["stars"] = source.p_value_holm.map(_stars)
    source["coefficient_display"] = source.apply(lambda row: f"{0.0 if abs(row.coefficient) < 0.00005 else row.coefficient:.4f}" + row.stars, axis=1)
    _write(source, output_dir, "tables/table4_interaction_source.csv")
    wide = []
    for outcome in OUTCOMES:
        rows = source.loc[source.outcome == outcome].set_index("resource")
        item = {"family": "impact", "outcome": outcome, "outcome_label": OUTCOME_LABELS[outcome]}
        for resource in RESOURCES:
            row = rows.loc[resource]
            for field in ("coefficient", "p_value", "p_value_holm", "reject_holm_05"):
                item[f"{resource}_{field}"] = row[field]
            item[f"{resource}_display"] = row.coefficient_display
        wide.append(item)
    _write(pd.DataFrame(wide), output_dir, "tables/table4_impact_ksc_interactions.csv")


def run_analysis(repo_root: Path, output_dir: Path) -> dict:
    """Write 32 primary models, four context fits, tables, and plot coordinates."""
    repo_root, output_dir = Path(repo_root).resolve(), Path(output_dir).resolve()
    if output_dir == repo_root or output_dir == repo_root / "data" or (repo_root / "data") in output_dir.parents or output_dir == repo_root / "results/reference" or (repo_root / "results/reference") in output_dir.parents:
        raise ValueError("Choose an output directory outside frozen data and reference results")
    if (output_dir / "model_statistics/coefficients.csv").exists():
        raise FileExistsError("Analysis output already exists; choose a fresh output directory")
    samples = {name: _sample(repo_root, name)[0] for name in EXPECTED_SAMPLES}
    coefficients, metrics, covariance, comparisons = [], [], [], []
    fits: dict[tuple[str, str, str], Fit] = {}
    cov_full = {}
    for analysis, sample in samples.items():
        groups = [sample.fe_venue_year, sample.fe_topic]
        rank = fixed_effect_rank(groups)
        sample_hash = sample_sha256(sample)
        if analysis == "baseline_api":
            all_design = [*MAIN, *API_TERMS.values(), *CONTROLS]
            models = {"M2": [*MAIN, *CONTROLS], **{f"M3_{name}": [*MAIN, term, *CONTROLS] for name, term in API_TERMS.items()}}
        else:
            base = [*MAIN, "ksc_z", *CONTROLS]
            all_design = [*base, *KSC_TERMS.values()]
            models = {"M2_context": base, **{f"M3_{name}": [*base, term] for name, term in KSC_TERMS.items()}}
        for outcome in OUTCOMES:
            y_raw = sample[f"y_{outcome}"].to_numpy(dtype=float)
            residual = absorb(np.column_stack([y_raw, sample[all_design]]), groups, tolerance=1e-11)
            design = pd.DataFrame(residual[:, 1:], columns=all_design, index=sample.index)
            for model, terms in models.items():
                fit = fit_absorbed(y_raw, residual[:, 0], design[terms], sample.cluster_group, rank, terms)
                if fit.dropped:
                    raise ValueError(f"Dropped model terms: {analysis}/{outcome}/{model}")
                fits[analysis, outcome, model] = fit
                model_id = f"{analysis}/{outcome}/{model}"
                common = dict(analysis=analysis, specification="venue_year_topic", family="impact", outcome=outcome, model=model, model_id=model_id, n=fit.n, clusters=fit.clusters, sample_sha256=sample_hash)
                r2 = 1 - fit.sse / fit.tss
                metrics.append({**common, "role": "auxiliary_context" if model == "M2_context" else "primary", "fe_rank": fit.fe_rank, "design_rank": fit.full_rank, "r2": r2, "within_r2": 1 - fit.sse / fit.within_tss, "adjusted_r2": 1 - (1 - r2) * (fit.n - 1) / (fit.n - fit.full_rank), "sse": fit.sse, "tss": fit.tss, "within_tss": fit.within_tss, "cluster_correction": fit.correction, "topic_levels": sample.fe_topic.nunique(), "venue_year_levels": sample.fe_venue_year.nunique()})
                sd = float(sample[f"y_{outcome}"].std(ddof=0))
                for row in fit.coefficients():
                    coefficients.append({**common, **row, "outcome_sd": sd, **{f"{key}_outcome_sd": row[key] / sd for key in ("coefficient", "standard_error", "ci_low", "ci_high")}})
                cov_full[model_id] = {"terms": fit.names, "coefficients": fit.beta.tolist(), "covariance": fit.covariance.tolist(), "n": fit.n, "clusters": fit.clusters, "sample_sha256": sample_hash, "cluster_correction": fit.correction}
                for i, term in enumerate(fit.names):
                    for j, other in enumerate(fit.names):
                        covariance.append({**common, "term": term, "other_term": other, "covariance": float(fit.covariance[i, j])})
            for resource, term in (API_TERMS if analysis == "baseline_api" else KSC_TERMS).items():
                restricted = "M2" if analysis == "baseline_api" else "M2_context"
                full = f"M3_{resource}"
                comparisons.append(dict(analysis=analysis, specification="venue_year_topic", family="impact", outcome=outcome, resource=resource, restricted_model=restricted, full_model=full, **compare_nested(fits[analysis, outcome, restricted], fits[analysis, outcome, full], [term]), n=len(sample), clusters=69, sample_sha256=sample_hash))
    coefficient_table = pd.DataFrame(coefficients)
    metric_table = pd.DataFrame(metrics)
    comparison_table = pd.DataFrame(comparisons)
    focal = coefficient_table.loc[((coefficient_table.analysis == "baseline_api") & (coefficient_table.model == "M2") & coefficient_table.term.isin(MAIN)) | ((coefficient_table.analysis == "baseline_api") & coefficient_table.term.isin(API_TERMS.values())) | ((coefficient_table.analysis == "ksc") & coefficient_table.term.isin(KSC_TERMS.values()))].copy()
    focal = _adjust(focal, ["analysis", "term"])
    focal["holm_family"] = focal.analysis + ":" + focal.term + "_across_four_impact_outcomes"
    focal["holm_family_size"] = 4
    focal["reject_holm_05"] = focal.p_value_holm.lt(0.05)
    coefficient_table = coefficient_table.merge(focal[["model_id", "term", "p_value_holm", "holm_family", "holm_family_size", "reject_holm_05"]], on=["model_id", "term"], how="left", validate="one_to_one")
    _write(coefficient_table, output_dir, "model_statistics/coefficients.csv")
    _write(metric_table, output_dir, "model_statistics/model_metrics.csv")
    _write(pd.DataFrame(covariance), output_dir, "model_statistics/covariance.csv")
    _write(comparison_table, output_dir, "model_statistics/model_comparisons.csv")
    (output_dir / "model_statistics/covariance_full.json").write_text(json.dumps(cov_full, indent=2) + "\n", encoding="utf-8")
    _write(coefficient_table.loc[(coefficient_table.analysis == "baseline_api") & (coefficient_table.model == "M2")], output_dir, "tables/baseline_regression_impact_table.csv")
    main_effects = focal.loc[(focal.analysis == "baseline_api") & (focal.model == "M2")].copy()
    main_effects["reported_effect"] = [100 * math.expm1(row.coefficient) if row.outcome == "cumulative_citations" else 100 * row.coefficient for row in main_effects.itertuples()]
    main_effects["reported_effect_unit"] = main_effects.outcome.map(lambda value: "percent_change_in_1_plus_outcome" if value == "cumulative_citations" else "percentage_points")
    _write(main_effects, output_dir, "tables/baseline_main_effects.csv")
    _write(coefficient_table.loc[(coefficient_table.analysis == "baseline_api") & (coefficient_table.model == "M2")], output_dir, "tables/baseline_coefficients.csv")
    _write(coefficient_table.loc[(coefficient_table.analysis == "baseline_api") & coefficient_table.model.str.startswith("M3_")], output_dir, "tables/interaction_coefficients.csv")
    _write(focal.loc[(focal.analysis == "baseline_api") & focal.term.isin(API_TERMS.values())], output_dir, "tables/appendix_interaction_multiple_testing_table.csv")
    api_contrasts, api_predictions, conditionals, ksc_grid = [], [], [], []
    for outcome in OUTCOMES:
        sample = samples["baseline_api"]
        sd = float(sample[f"y_{outcome}"].std(ddof=0))
        for hardware, interaction in API_TERMS.items():
            fit = fits["baseline_api", outcome, f"M3_{hardware}"]
            indices = {name: i for i, name in enumerate(fit.names)}
            no_weights = np.zeros(len(fit.names))
            no_weights[indices[RESOURCES[hardware]]] = 1
            yes_weights = no_weights.copy()
            yes_weights[indices[interaction]] = 1
            no, yes = _contrast(fit, no_weights), _contrast(fit, yes_weights)
            row = focal.loc[(focal.analysis == "baseline_api") & (focal.outcome == outcome) & (focal.term == interaction)].iloc[0]
            api_contrasts.append(dict(family="impact", outcome=outcome, model=f"M3_{hardware}", hardware=hardware, hardware_main_term=RESOURCES[hardware], interaction_term=interaction, slope_api_no=no["estimate"], slope_api_no_se=no["standard_error"], slope_api_no_ci_low=no["ci_low"], slope_api_no_ci_high=no["ci_high"], slope_api_yes=yes["estimate"], slope_api_yes_se=yes["standard_error"], slope_api_yes_ci_low=yes["ci_low"], slope_api_yes_ci_high=yes["ci_high"], interaction_difference=row.coefficient, interaction_se=row.standard_error, interaction_ci_low=row.ci_low, interaction_ci_high=row.ci_high, interaction_p_value=row.p_value, interaction_p_value_holm=row.p_value_holm, slope_api_no_outcome_sd=no["estimate"] / sd, slope_api_yes_outcome_sd=yes["estimate"] / sd, interaction_outcome_sd=row.coefficient / sd, n=len(sample), clusters=69, sample_sha256=sample_sha256(sample)))
            if outcome == "cumulative_citations":
                grid = np.linspace(*sample[RESOURCES[hardware]].quantile([0.05, 0.95]).to_numpy(), 101)
                for api in (0, 1):
                    for value in grid:
                        weights = no_weights * value
                        if api:
                            weights[indices["api"]] = 1
                            weights[indices[interaction]] = value
                        contrast = _contrast(fit, weights)
                        api_predictions.append(dict(family="impact", outcome=outcome, model=f"M3_{hardware}", hardware=hardware, api=api, api_label="API reported" if api else "No API reported", doublings_from_full_v14_median=float(value), adjusted_change=contrast["estimate"] / sd, ci_low=contrast["ci_low"] / sd, ci_high=contrast["ci_high"] / sd, display_unit="outcome-SD change from API=no at median hardware", n=len(sample), clusters=69))
        sample = samples["ksc"]
        low, high = sample.ksc_z.quantile([0.05, 0.95])
        for resource, interaction in KSC_TERMS.items():
            fit = fits["ksc", outcome, f"M3_{resource}"]
            indices = {name: i for i, name in enumerate(fit.names)}
            common = dict(specification="venue_year_topic", family="impact", outcome=outcome, model=f"M3_{resource}", resource=resource, n=len(sample), clusters=69, sample_sha256=sample_sha256(sample))
            def conditional(level: float) -> dict:
                weights = np.zeros(len(fit.names))
                weights[indices[RESOURCES[resource]]] = 1
                weights[indices[interaction]] = level
                contrast = _contrast(fit, weights)
                return {**contrast, **_scaled(outcome, contrast)}
            for level in (-1.0, 0.0, 1.0):
                conditionals.append({**common, "ksc_z_level": level, **conditional(level), "effect_type": "api_discrete_difference_lpm" if resource == "api" else "conditional_linear_slope"})
            panel = {("cumulative_citations", "quantity"): "a", ("top10", "quantity"): "b", ("top5", "quantity"): "c", ("top1", "api"): "d"}.get((outcome, resource))
            if panel:
                for level in np.linspace(low, high, 101):
                    ksc_grid.append({**common, "panel": panel, "outcome_label": OUTCOME_LABELS[outcome], "resource_label": RESOURCE_LABELS[resource], "ksc_z": float(level), **conditional(level), "grid_quantile_low": 0.05, "grid_quantile_high": 0.95, "grid_value_low": float(low), "grid_value_high": float(high)})
    interactions = focal.loc[focal.analysis == "ksc"].rename(columns={"term": "interaction_term"}).copy()
    interactions["resource"] = interactions.interaction_term.map({value: key for key, value in KSC_TERMS.items()})
    interactions = interactions.merge(metric_table[["model_id", "within_r2", "adjusted_r2"]], on="model_id", validate="one_to_one")
    interactions = interactions.merge(comparison_table.loc[comparison_table.analysis == "ksc", ["outcome", "resource", "delta_r2", "delta_within_r2", "partial_r2"]], on=["outcome", "resource"], validate="one_to_one")
    _write(interactions, output_dir, "tables/interaction_results.csv")
    _write(interactions, output_dir, "tables/table4_interaction_source.csv")
    _write(coefficient_table.loc[(coefficient_table.analysis == "ksc") & coefficient_table.model.str.startswith("M3_")], output_dir, "tables/appendix_full_model_details.csv")
    _write(coefficient_table.loc[coefficient_table.analysis == "ksc"], output_dir, "tables/linear_coefficients.csv")
    _write(metric_table.loc[metric_table.analysis == "ksc"], output_dir, "tables/linear_model_metrics.csv")
    _write(comparison_table.loc[comparison_table.analysis == "ksc"], output_dir, "tables/linear_model_comparisons.csv")
    _write(metric_table.loc[metric_table.analysis == "baseline_api"], output_dir, "tables/model_metrics.csv")
    _write(comparison_table.loc[comparison_table.analysis == "baseline_api"], output_dir, "tables/model_comparisons.csv")
    _write(pd.DataFrame(api_contrasts), output_dir, "tables/interaction_contrasts.csv")
    _write(pd.DataFrame(api_contrasts).loc[lambda frame: frame.outcome == "cumulative_citations"], output_dir, "figure_data/cumulative_citation_interaction_contrasts.csv")
    _write(pd.DataFrame(api_predictions), output_dir, "figure_data/cumulative_citation_interaction_predictions.csv")
    conditionals_table = _adjust(pd.DataFrame(conditionals), ["resource", "ksc_z_level"])
    conditionals_table["holm_family_size"] = 4
    conditionals_table["holm_family"] = conditionals_table.apply(lambda row: f"venue_year_topic:impact:{row.resource}:ksc_{row.ksc_z_level:+.0f}", axis=1)
    conditionals_table["reject_holm_05"] = conditionals_table.p_value_holm.lt(0.05)
    _write(conditionals_table, output_dir, "figure_data/conditional_associations.csv")
    _write(conditionals_table, output_dir, "tables/conditional_associations.csv")
    _write(pd.DataFrame(ksc_grid), output_dir, "figure_data/marginal_figure_source.csv")
    write_paper_tables(coefficient_table, metric_table, interactions, output_dir)
    summary = dict(primary_models=int(metric_table.role.eq("primary").sum()), auxiliary_context_models=4, samples={name: dict(n=len(sample), clusters=69, sample_sha256=sample_sha256(sample)) for name, sample in samples.items()}, api_figure_rows=len(api_predictions), ksc_figure_rows=len(ksc_grid), conditional_rows=len(conditionals), multiple_testing="Holm across four outcomes within each focal resource/test family")
    (output_dir / "model_statistics/analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary

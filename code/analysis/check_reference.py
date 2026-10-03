"""Compare regenerated models and plot coordinates with frozen reference CSVs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def check_reference(output_dir: Path, reference_dir: Path) -> dict:
    checks = []

    def compare(relative: str, keys: list[str], columns: list[str], models: list[str] | None = None, curve: str | None = None) -> None:
        actual = pd.read_csv(output_dir / relative)
        expected = pd.read_csv(reference_dir / relative)
        if models is not None:
            expected = expected.loc[expected.model.isin(models)]
        if curve is not None:
            actual = actual.sort_values([*keys, curve]).copy()
            expected = expected.sort_values([*keys, curve]).copy()
            actual["grid_index"] = actual.groupby(keys, sort=False).cumcount()
            expected["grid_index"] = expected.groupby(keys, sort=False).cumcount()
            keys = [*keys, "grid_index"]
        if actual.duplicated(keys).any() or expected.duplicated(keys).any():
            raise AssertionError(f"Duplicate comparison keys: {relative}")
        joined = actual.merge(expected, on=keys, suffixes=("_actual", "_reference"), how="outer", indicator=True, validate="one_to_one")
        if not joined._merge.eq("both").all():
            raise AssertionError(f"Different result keys: {relative}")
        errors = {}
        for column in columns:
            left = joined[f"{column}_actual"].to_numpy(dtype=float)
            right = joined[f"{column}_reference"].to_numpy(dtype=float)
            if not np.allclose(left, right, rtol=1e-9, atol=1e-11, equal_nan=True):
                raise AssertionError(f"Numerical mismatch: {relative}/{column}: {np.nanmax(np.abs(left-right))}")
            errors[column] = float(np.nanmax(np.abs(left - right)))
        checks.append({"file": relative, "rows": len(joined), "maximum_absolute_errors": errors})

    numeric = ["coefficient", "standard_error", "ci_low", "ci_high", "p_value"]
    compare("tables/baseline_regression_impact_table.csv", ["outcome", "model", "term"], [*numeric, "p_value_holm", "r2", "adjusted_r2"])
    compare("tables/interaction_coefficients.csv", ["outcome", "model", "term"], numeric)
    compare("tables/appendix_interaction_impact_table.csv", ["outcome", "model", "term"], [*numeric, "r2", "adjusted_r2"])
    compare("tables/baseline_main_effects.csv", ["outcome", "term"], [*numeric, "p_value_holm", "reported_effect"])
    compare("tables/linear_coefficients.csv", ["outcome", "model", "term"], numeric)
    compare("tables/model_metrics.csv", ["outcome", "model"], ["r2", "within_r2", "sse", "design_rank"], ["M2", "M3_quantity", "M3_performance", "M3_memory"])
    compare("tables/linear_model_metrics.csv", ["outcome", "model"], ["r2", "within_r2", "adjusted_r2", "design_rank"])
    compare("tables/interaction_results.csv", ["outcome", "resource"], [*numeric, "p_value_holm", "delta_r2", "partial_r2"])
    compare("tables/table4_impact_ksc_interactions.csv", ["outcome"], [f"{resource}_{field}" for resource in ("quantity", "performance", "memory", "api") for field in ("coefficient", "p_value", "p_value_holm")])
    compare("figure_data/cumulative_citation_interaction_predictions.csv", ["hardware", "api"], ["doublings_from_full_v14_median", "adjusted_change", "ci_low", "ci_high"], curve="doublings_from_full_v14_median")
    compare("figure_data/cumulative_citation_interaction_contrasts.csv", ["outcome", "hardware"], ["slope_api_no", "slope_api_yes", "slope_api_no_se", "slope_api_yes_se", "interaction_difference", "interaction_se", "interaction_p_value", "interaction_p_value_holm"])
    compare("figure_data/marginal_figure_source.csv", ["panel"], ["ksc_z", "estimate", "standard_error", "ci_low", "ci_high", "marginal_association", "marginal_ci_low", "marginal_ci_high"], curve="ksc_z")
    compare("figure_data/conditional_associations.csv", ["outcome", "resource", "ksc_z_level"], ["estimate", "standard_error", "ci_low", "ci_high", "p_value", "p_value_holm", "marginal_association", "marginal_ci_low", "marginal_ci_high"])
    coefficients = pd.read_csv(output_dir / "model_statistics/coefficients.csv")
    covariance = json.loads((output_dir / "model_statistics/covariance_full.json").read_text(encoding="utf-8"))
    for model_id, item in covariance.items():
        matrix = np.asarray(item["covariance"])
        rows = coefficients.loc[coefficients.model_id == model_id].set_index("term").loc[item["terms"]]
        np.testing.assert_allclose(matrix, matrix.T, rtol=0, atol=1e-14)
        np.testing.assert_allclose(np.diag(matrix), rows.standard_error.to_numpy() ** 2, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(item["coefficients"], rows.coefficient, rtol=1e-12, atol=1e-14)
    return {"passed": True, "reference_comparisons": checks, "covariance_models_checked": len(covariance)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, default=Path("results/reference"))
    args = parser.parse_args()
    print(json.dumps(check_reference(args.output_dir, args.reference_dir), indent=2))

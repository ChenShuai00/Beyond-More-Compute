"""Checks for covariance propagation and protected output locations."""
from pathlib import Path

import numpy as np
import pytest

from analysis.replication import _contrast, run_analysis
from analysis.statistics import Fit


def test_contrast_uses_main_interaction_covariance():
    fit = Fit(["main", "interaction"], np.array([0.04, -0.02]), np.array([[0.01, -0.004], [-0.004, 0.0025]]), np.zeros(10), [], 10, 69, 2, 4, 1.0, 2.0, 1.5, 1.0)
    result = _contrast(fit, np.array([1.0, 1.0]))
    assert result["estimate"] == pytest.approx(0.02)
    assert result["standard_error"] == pytest.approx(np.sqrt(0.0045))
    assert _contrast(fit, np.zeros(2))["standard_error"] == 0


def test_frozen_inputs_and_reference_outputs_are_protected():
    repo_root = Path(__file__).resolve().parents[2]
    for relative in ("data", "data/analysis_ready", "results/reference", "results/reference/tables"):
        with pytest.raises(ValueError, match="outside frozen"):
            run_analysis(repo_root, repo_root / relative)

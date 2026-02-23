"""Tests for experiments/monte_carlo.py."""

import numpy as np
import pytest

from virtualfermlab.experiments.doe import ExperimentCondition
from virtualfermlab.experiments.monte_carlo import run_monte_carlo, MonteCarloResult
from virtualfermlab.models.ode_systems import ModelConfig
from virtualfermlab.parameters.library import load_strain_profile


class TestMonteCarlo:
    def test_basic_run(self):
        profile = load_strain_profile("F_venenatum_A35")
        config = ModelConfig(
            n_substrates=2, enzyme_mode="direct",
            use_cardinal_pH=True, pH=6.0, pH_min=3.5, pH_opt=6.0, pH_max=7.5,
        )
        condition = ExperimentCondition(
            strain="F_venenatum_A35",
            substrate_A="glucose",
            substrate_B="xylose",
            ratio=0.5,
            pH=6.0,
            total_concentration=30.0,
        )
        times = np.linspace(0, 100, 200)

        result = run_monte_carlo(
            condition, profile, config, times,
            n_samples=5, n_jobs=1, seed=42,
        )

        assert isinstance(result, MonteCarloResult)
        assert len(result.yields) == 5
        assert result.mean_yield > 0
        assert np.isfinite(result.std_yield)

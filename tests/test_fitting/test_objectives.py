"""Tests for fitting/objectives.py."""

import numpy as np
import pandas as pd
import pytest

from virtualfermlab.fitting.objectives import calc_all_errors, calc_total_error, calc_BIC


@pytest.fixture
def simple_params():
    return {
        "n_substrates": 2,
        "n_feeds": 1,
        "growth_model": "Monod",
        "enzyme_induction": False,
        "mu_max1": 0.10,
        "mu_max2": 0.045,
        "K_s1": 0.20,
        "K_s2": 0.15,
        "Yx1": 0.29,
        "Yx2": 0.35,
        "K_I": 0.50,
        "S1": 15.0,
        "S2": 15.0,
        "S_in1": 15.0,
        "S_in2": 15.0,
        "X0": 0.03,
        "y0": 0.0,
        "dilutionRate": 0.0,
    }


@pytest.fixture
def simple_exp_data():
    """Synthetic experimental data for testing."""
    times = [0, 24, 48, 72, 96]
    return pd.DataFrame({
        "Time (h)": times,
        "[Biomass] (g/L)": [0.03, 0.1, 0.5, 1.5, 2.0],
        "[Glucose] (g/L)": [15.0, 14.0, 10.0, 3.0, 0.5],
        "[Xylose] (g/L)": [15.0, 15.0, 14.5, 12.0, 5.0],
    })


class TestCalcAllErrors:
    def test_returns_dict(self, simple_params, simple_exp_data):
        errors = calc_all_errors(simple_params, simple_exp_data["Time (h)"], simple_exp_data)
        assert "biomass_error" in errors
        assert "S1_error" in errors
        assert "S2_error" in errors

    def test_error_lengths(self, simple_params, simple_exp_data):
        errors = calc_all_errors(simple_params, simple_exp_data["Time (h)"], simple_exp_data)
        assert len(errors["biomass_error"]) == len(simple_exp_data["Time (h)"].unique())


class TestCalcTotalError:
    def test_mae(self, simple_params, simple_exp_data):
        err = calc_total_error(simple_params, simple_exp_data["Time (h)"], simple_exp_data, "MAE")
        assert err >= 0
        assert np.isfinite(err)

    def test_likelihood(self, simple_params, simple_exp_data):
        err = calc_total_error(simple_params, simple_exp_data["Time (h)"], simple_exp_data, "likelihood")
        assert err >= 0
        assert np.isfinite(err)


class TestCalcBIC:
    def test_returns_finite(self, simple_params, simple_exp_data):
        bic = calc_BIC(simple_params, simple_exp_data["Time (h)"], simple_exp_data, N_params=8)
        assert np.isfinite(bic)

    def test_more_params_higher_bic(self, simple_params, simple_exp_data):
        bic8 = calc_BIC(simple_params, simple_exp_data["Time (h)"], simple_exp_data, N_params=8)
        bic12 = calc_BIC(simple_params, simple_exp_data["Time (h)"], simple_exp_data, N_params=12)
        assert bic12 > bic8

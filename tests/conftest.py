"""Shared fixtures for VirtualFermLab tests."""

import numpy as np
import pytest

from virtualfermlab.models.ode_systems import ModelConfig


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def batch_params_direct():
    """Params for 2-substrate batch, direct inhibition (Monod)."""
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
def config_direct():
    """ModelConfig for Monod + direct inhibition, batch."""
    return ModelConfig(
        n_substrates=2,
        n_feeds=1,
        growth_model="Monod",
        enzyme_mode="direct",
    )


@pytest.fixture
def config_enzyme():
    """ModelConfig for Monod + enzyme induction, batch."""
    return ModelConfig(
        n_substrates=2,
        n_feeds=1,
        growth_model="Monod",
        enzyme_mode="enzyme",
    )


@pytest.fixture
def config_kompala():
    """ModelConfig for Monod + Kompala cybernetic, batch."""
    return ModelConfig(
        n_substrates=2,
        n_feeds=1,
        growth_model="Monod",
        enzyme_mode="kompala",
    )


@pytest.fixture
def batch_params_enzyme():
    """Params for 2-substrate batch with enzyme induction."""
    return {
        "n_substrates": 2,
        "n_feeds": 1,
        "growth_model": "Monod",
        "enzyme_induction": True,
        "mu_max1": 0.10,
        "mu_max2": 0.045,
        "K_s1": 0.20,
        "K_s2": 0.15,
        "Yx1": 0.29,
        "Yx2": 0.35,
        "K_Z_c": 0.30,
        "K_Z_S": 0.20,
        "K_Z_d": 0.05,
        "K_I": 0.50,
        "S1": 15.0,
        "S2": 15.0,
        "S_in1": 15.0,
        "S_in2": 15.0,
        "X0": 0.03,
        "Z0": 0.01,
        "y0": 0.0,
        "dilutionRate": 0.0,
    }


@pytest.fixture
def batch_params_kompala():
    """Params for 2-substrate batch with Kompala model."""
    return {
        "n_substrates": 2,
        "n_feeds": 1,
        "growth_model": "Monod",
        "enzyme_induction": "Kompala",
        "mu_max1": 0.10,
        "mu_max2": 0.045,
        "K_s1": 0.20,
        "K_s2": 0.15,
        "Yx1": 0.29,
        "Yx2": 0.35,
        "K_Z_c": 0.30,
        "K_Z_S": 0.20,
        "K_Z_d": 0.05,
        "S1": 15.0,
        "S2": 15.0,
        "S_in1": 15.0,
        "S_in2": 15.0,
        "X0": 0.03,
        "Z1": 0.01,
        "Z2": 0.01,
        "y0": 0.0,
        "dilutionRate": 0.0,
    }

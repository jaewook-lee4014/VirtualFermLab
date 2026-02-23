"""Tests for simulator/integrator.py."""

import numpy as np
import pytest

from virtualfermlab.models.ode_systems import ModelConfig
from virtualfermlab.simulator.integrator import SimulationResult, simulate


class TestSimulate:
    def test_batch_direct(self, config_direct, batch_params_direct):
        times = np.linspace(0, 200, 500)
        result = simulate(config_direct, batch_params_direct, times)
        assert isinstance(result, SimulationResult)
        assert result.X[-1] > result.X[0]
        assert "S1" in result.substrates
        assert "S2" in result.substrates
        assert len(result.enzymes) == 0

    def test_batch_enzyme(self, config_enzyme, batch_params_enzyme):
        times = np.linspace(0, 200, 500)
        result = simulate(config_enzyme, batch_params_enzyme, times)
        assert result.X[-1] > result.X[0]
        assert "Z" in result.enzymes

    def test_batch_kompala(self, config_kompala, batch_params_kompala):
        times = np.linspace(0, 200, 500)
        result = simulate(config_kompala, batch_params_kompala, times)
        assert result.X[-1] > result.X[0]
        assert "Z1" in result.enzymes
        assert "Z2" in result.enzymes

    def test_solve_ivp_method(self, config_direct, batch_params_direct):
        times = np.linspace(0, 200, 500)
        result = simulate(config_direct, batch_params_direct, times, method="RK45")
        assert result.X[-1] > result.X[0]

    def test_1_substrate(self):
        config = ModelConfig(n_substrates=1, growth_model="Monod")
        params = {
            "mu_max": 0.3, "K_s": 1.0, "Yx": 0.5,
            "S_in": 10.0, "X0": 0.1, "y0": 0.0, "dilutionRate": 0.0,
        }
        times = np.linspace(0, 50, 500)
        result = simulate(config, params, times)
        assert result.X[-1] > 4.0
        assert "S" in result.substrates


class TestSimulationResult:
    def test_yield_biomass(self, config_direct, batch_params_direct):
        times = np.linspace(0, 200, 500)
        result = simulate(config_direct, batch_params_direct, times)
        assert result.yield_biomass > 0

    def test_mu_max_effective(self, config_direct, batch_params_direct):
        times = np.linspace(0, 200, 500)
        result = simulate(config_direct, batch_params_direct, times)
        assert result.mu_max_effective > 0

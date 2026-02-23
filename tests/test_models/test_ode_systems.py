"""Tests for models/ode_systems.py."""

import numpy as np
import pytest
from scipy.integrate import odeint

from virtualfermlab.models.ode_systems import FermentationODE, ModelConfig


class TestFermentationODE1Substrate:
    def test_batch_growth(self):
        config = ModelConfig(n_substrates=1, growth_model="Monod")
        params = {
            "mu_max": 0.3, "K_s": 1.0, "Yx": 0.5,
            "S_in": 10.0, "dilutionRate": 0.0,
        }
        ode = FermentationODE(config, params)
        y0 = [0.1, 10.0, 0.0]
        times = np.linspace(0, 50, 500)
        yobs = odeint(ode, y0, times)
        X = yobs[:, 0]
        # Biomass should increase
        assert X[-1] > X[0]
        # Should reach near X0 + Yx*S0 = 0.1 + 0.5*10 = 5.1
        assert X[-1] == pytest.approx(5.1, abs=0.2)

    def test_state_names(self):
        config = ModelConfig(n_substrates=1)
        ode = FermentationODE(config, {})
        assert ode.state_names() == ["X", "S", "totalOutput"]


class TestFermentationODE2SubstrateDirect:
    def test_batch_growth(self, config_direct, batch_params_direct):
        ode = FermentationODE(config_direct, batch_params_direct)
        y0 = [0.03, 15.0, 15.0, 0.0]
        times = np.linspace(0, 200, 2000)
        yobs = odeint(ode, y0, times)
        X, S1, S2 = yobs[:, 0], yobs[:, 1], yobs[:, 2]
        # Biomass should grow
        assert X[-1] > X[0]
        # Glucose should be consumed first (diauxic)
        # Find time when S1 ≈ 0
        s1_depleted = np.argmax(S1 < 0.01)
        if s1_depleted > 0:
            # Xylose should still have some left at that point
            assert S2[s1_depleted] > 1.0

    def test_state_names(self, config_direct):
        ode = FermentationODE(config_direct, {})
        assert ode.state_names() == ["X", "S1", "S2", "totalOutput"]


class TestFermentationODE2SubstrateEnzyme:
    def test_batch_growth(self, config_enzyme, batch_params_enzyme):
        ode = FermentationODE(config_enzyme, batch_params_enzyme)
        y0 = [0.03, 15.0, 15.0, 0.01, 0.0]
        times = np.linspace(0, 200, 2000)
        yobs = odeint(ode, y0, times)
        X = yobs[:, 0]
        assert X[-1] > X[0]

    def test_state_names(self, config_enzyme):
        ode = FermentationODE(config_enzyme, {})
        assert ode.state_names() == ["X", "S1", "S2", "Z", "totalOutput"]


class TestFermentationODE2SubstrateKompala:
    def test_batch_growth(self, config_kompala, batch_params_kompala):
        ode = FermentationODE(config_kompala, batch_params_kompala)
        y0 = [0.03, 15.0, 15.0, 0.01, 0.01, 0.0]
        times = np.linspace(0, 200, 2000)
        yobs = odeint(ode, y0, times)
        X = yobs[:, 0]
        assert X[-1] > X[0]

    def test_state_names(self, config_kompala):
        ode = FermentationODE(config_kompala, {})
        assert ode.state_names() == ["X", "S1", "S2", "Z1", "Z2", "totalOutput"]


class TestMassConservation:
    """Check approximate mass conservation: d(X/Y + S)/dt ≈ 0 in batch."""

    def test_1substrate_mass_balance(self):
        config = ModelConfig(n_substrates=1, growth_model="Monod")
        params = {
            "mu_max": 0.3, "K_s": 1.0, "Yx": 0.5,
            "S_in": 10.0, "dilutionRate": 0.0,
        }
        ode = FermentationODE(config, params)
        y0 = [0.1, 10.0, 0.0]
        times = np.linspace(0, 50, 500)
        yobs = odeint(ode, y0, times)
        X, S = yobs[:, 0], yobs[:, 1]
        # Mass balance: X/Yx + S should be constant
        total = X / params["Yx"] + S
        assert total[0] == pytest.approx(total[-1], rel=0.01)


class TestCardinalPHIntegration:
    def test_pH_reduces_growth(self, batch_params_direct):
        """Low pH should reduce final biomass."""
        config_no_pH = ModelConfig(n_substrates=2, enzyme_mode="direct")
        config_low_pH = ModelConfig(
            n_substrates=2, enzyme_mode="direct",
            use_cardinal_pH=True, pH=4.5, pH_min=3.5, pH_opt=6.0, pH_max=7.5,
        )
        times = np.linspace(0, 200, 2000)

        ode_no = FermentationODE(config_no_pH, batch_params_direct)
        ode_low = FermentationODE(config_low_pH, batch_params_direct)
        y0 = [0.03, 15.0, 15.0, 0.0]

        X_no = odeint(ode_no, y0, times)[:, 0]
        X_low = odeint(ode_low, y0, times)[:, 0]

        # At low pH, growth should be slower
        assert X_low[-1] < X_no[-1]

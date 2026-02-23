"""Composable ODE right-hand side for fermentation models.

Refactors the monolithic ``plantODE()`` from the ESCAPE25 notebook into a
configurable :class:`FermentationODE` class.  Each branch of the original
if/elif tree is now an explicit private method, and kinetics × enzyme ×
pH factors are composed independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from virtualfermlab.models.kinetics import substrate_factor
from virtualfermlab.models.enzyme_regulation import (
    direct_inhibition,
    enzyme_induction_factor,
    enzyme_production_rate,
    kompala_matching_law,
    kompala_proportional_law,
)
from virtualfermlab.models.ph_model import cardinal_pH_factor, lag_switch


@dataclass
class ModelConfig:
    """Declarative configuration for an ODE model variant.

    Attributes
    ----------
    n_substrates : int
        Number of substrates (1 or 2).
    n_feeds : int
        Number of feed streams (1 or 2).
    growth_model : str
        ``"Monod"`` or ``"Contois"``.
    enzyme_mode : str
        ``"direct"`` (competitive inhibition), ``"enzyme"`` (enzyme pool *Z*),
        or ``"kompala"`` (cybernetic with *Z1*, *Z2*).
    use_cardinal_pH : bool
        Whether to multiply *mu_max* by the Cardinal pH factor.
    use_lag : bool
        Whether to apply a lag-phase switch.
    pH : float | None
        Current pH (required when *use_cardinal_pH* is True).
    pH_min, pH_opt, pH_max : float | None
        Cardinal pH parameters.
    lag : float | None
        Lag phase duration (h).
    """

    n_substrates: int = 2
    n_feeds: int = 1
    growth_model: str = "Monod"
    enzyme_mode: str = "direct"
    use_cardinal_pH: bool = False
    use_lag: bool = False
    pH: float | None = None
    pH_min: float | None = None
    pH_opt: float | None = None
    pH_max: float | None = None
    lag: float | None = None


class FermentationODE:
    """Composable ODE RHS for fermentation models.

    Supports ``odeint`` signature ``(y, t)`` via :meth:`__call__` and
    ``solve_ivp`` signature ``(t, y)`` via :meth:`solve_ivp_rhs`.

    Parameters
    ----------
    config : ModelConfig
        Model configuration.
    params : dict
        Kinetic / stoichiometric parameters.
    """

    def __init__(self, config: ModelConfig, params: dict) -> None:
        self.config = config
        self.params = params

    # ---- Public interface ----

    def __call__(self, y: Sequence[float], t: float) -> list[float]:
        """RHS compatible with :func:`scipy.integrate.odeint`."""
        return self._rhs(y, t)

    def solve_ivp_rhs(self, t: float, y: Sequence[float]) -> list[float]:
        """RHS compatible with :func:`scipy.integrate.solve_ivp`."""
        return self._rhs(y, t)

    def state_names(self) -> list[str]:
        """Ordered list of state variable names."""
        cfg = self.config
        if cfg.n_substrates == 1:
            return ["X", "S", "totalOutput"]
        # n_substrates == 2
        base = ["X", "S1", "S2"]
        if cfg.enzyme_mode == "enzyme":
            base.append("Z")
        elif cfg.enzyme_mode == "kompala":
            base.extend(["Z1", "Z2"])
        base.append("totalOutput")
        return base

    # ---- Internal dispatch ----

    def _rhs(self, y: Sequence[float], t: float) -> list[float]:
        cfg = self.config
        if cfg.n_substrates == 1:
            return self._rhs_1s(y, t)
        # n_substrates == 2
        if cfg.enzyme_mode == "direct":
            return self._rhs_2s_direct(y, t)
        elif cfg.enzyme_mode == "enzyme":
            return self._rhs_2s_enzyme(y, t)
        elif cfg.enzyme_mode == "kompala":
            return self._rhs_2s_kompala(y, t)
        else:
            raise ValueError(f"Unknown enzyme_mode: {cfg.enzyme_mode!r}")

    # ---- pH / lag modulation helpers ----

    def _mu_modifier(self, t: float) -> float:
        """Multiplicative modifier for mu_max (pH and/or lag)."""
        mod = 1.0
        cfg = self.config
        if cfg.use_cardinal_pH and cfg.pH is not None:
            mod *= cardinal_pH_factor(cfg.pH, cfg.pH_min, cfg.pH_opt, cfg.pH_max)
        if cfg.use_lag and cfg.lag is not None:
            mod *= lag_switch(t, cfg.lag)
        return mod

    # ---- Dilution rate helpers ----

    def _get_dilution(self) -> float:
        cfg = self.config
        if cfg.n_feeds == 1:
            return self.params["dilutionRate"]
        else:
            return self.params["dilutionRate1"] + self.params["dilutionRate2"]

    # ---- 1-substrate RHS ----

    def _rhs_1s(self, y: Sequence[float], t: float) -> list[float]:
        p = self.params
        cfg = self.config
        X, S, totalOutput = y[0], y[1], y[2]

        mu_max = p["mu_max"] * self._mu_modifier(t)
        sf = substrate_factor(S, p["K_s"], cfg.growth_model, X)
        gr = mu_max * sf
        D = self._get_dilution()

        dX = gr * X - D * X
        dS = D * (p["S_in"] - S) - gr * (1.0 / p["Yx"]) * X
        dO = D * X
        return [dX, dS, dO]

    # ---- 2-substrate, direct inhibition ----

    def _rhs_2s_direct(self, y: Sequence[float], t: float) -> list[float]:
        p = self.params
        cfg = self.config
        X, S1, S2, totalOutput = y[0], y[1], y[2], y[3]
        mod = self._mu_modifier(t)

        sf1 = substrate_factor(S1, p["K_s1"], cfg.growth_model, X)
        sf2 = substrate_factor(S2, p["K_s2"], cfg.growth_model, X) * direct_inhibition(S1, p["K_I"])

        gr1 = p["mu_max1"] * mod * sf1
        gr2 = p["mu_max2"] * mod * sf2

        if cfg.n_feeds == 1:
            D = p["dilutionRate"]
            dX = (gr1 + gr2) * X - D * X
            dS1 = D * (p["S_in1"] - S1) - gr1 * (1.0 / p["Yx1"]) * X
            dS2 = D * (p["S_in2"] - S2) - gr2 * (1.0 / p["Yx2"]) * X
            dO = D * X
        else:
            D1, D2 = p["dilutionRate1"], p["dilutionRate2"]
            D = D1 + D2
            dX = (gr1 + gr2) * X - D * X
            dS1 = D1 * p["S_in1"] - D * S1 - gr1 * (1.0 / p["Yx1"]) * X
            dS2 = D2 * p["S_in2"] - D * S2 - gr2 * (1.0 / p["Yx2"]) * X
            dO = D * X
        return [dX, dS1, dS2, dO]

    # ---- 2-substrate, enzyme induction ----

    def _rhs_2s_enzyme(self, y: Sequence[float], t: float) -> list[float]:
        p = self.params
        cfg = self.config
        X, S1, S2, Z, totalOutput = y[0], y[1], y[2], y[3], y[4]
        mod = self._mu_modifier(t)

        ef = enzyme_induction_factor(Z, p["K_Z_S"])
        sf1 = substrate_factor(S1, p["K_s1"], cfg.growth_model, X)
        sf2 = substrate_factor(S2, p["K_s2"], cfg.growth_model, X) * ef

        gr1 = p["mu_max1"] * mod * sf1
        gr2 = p["mu_max2"] * mod * sf2
        total_gr = gr1 + gr2

        if cfg.n_feeds == 1:
            D = p["dilutionRate"]
            dX = total_gr * X - D * X
            dS1 = D * (p["S_in1"] - S1) - gr1 * (1.0 / p["Yx1"]) * X
            dS2 = D * (p["S_in2"] - S2) - gr2 * (1.0 / p["Yx2"]) * X
            dZ = (
                -(p["K_Z_d"] + D) * Z
                + enzyme_production_rate(total_gr, X, S2, p["K_s2"], S1, p["K_Z_S"], p["K_Z_c"])
            )
            dO = D * X
        else:
            D1, D2 = p["dilutionRate1"], p["dilutionRate2"]
            D = D1 + D2
            dX = total_gr * X - D * X
            dS1 = D1 * p["S_in1"] - D * S1 - gr1 * (1.0 / p["Yx1"]) * X
            dS2 = D2 * p["S_in2"] - D * S2 - gr2 * (1.0 / p["Yx2"]) * X
            dZ = (
                -(p["K_Z_d"] + D - p["K_Z_d"] * D) * Z
                + enzyme_production_rate(total_gr, 1.0, S2, p["K_s2"], S1, p["K_Z_S"], p["K_Z_c"]) * X
            )
            dO = D * X
        return [dX, dS1, dS2, dZ, dO]

    # ---- 2-substrate, Kompala cybernetic ----

    def _rhs_2s_kompala(self, y: Sequence[float], t: float) -> list[float]:
        p = self.params
        cfg = self.config
        X, S1, S2, Z1, Z2, totalOutput = y[0], y[1], y[2], y[3], y[4], y[5]
        mod = self._mu_modifier(t)

        Z1 = max(Z1, 0.0)
        Z2 = max(Z2, 0.0)
        S1 = max(S1, 0.01)
        S2 = max(S2, 0.01)

        gr1 = p["mu_max1"] * mod * Z1 * S1 / (p["K_s1"] + S1)
        gr2 = p["mu_max2"] * mod * Z2 * S2 / (p["K_s2"] + S2)

        v1, v2 = kompala_matching_law(gr1, gr2)
        u1, u2 = kompala_proportional_law(gr1, gr2)

        r_Z1 = p["K_Z_c"] * S1 / (p["K_s1"] + S1)
        r_Z2 = p["K_Z_c"] * S2 / (p["K_s2"] + S2)

        total_specific_gr = (gr1 * v1 + gr2 * v2)

        D = p.get("dilutionRate", 0.0)
        dX = total_specific_gr * X - D * X
        dS1 = D * (p["S_in1"] - S1) - gr1 * v1 * (1.0 / p["Yx1"]) * X
        dS2 = D * (p["S_in2"] - S2) - gr2 * v2 * (1.0 / p["Yx2"]) * X

        # Enzyme dynamics (specific, per-biomass)
        biomass_dilution = total_specific_gr / X if X > 0 else 0.0
        dZ1 = r_Z1 * u1 - p["K_Z_d"] * Z1 - biomass_dilution * Z1
        dZ2 = r_Z2 * u2 - p["K_Z_d"] * Z2 - biomass_dilution * Z2

        dO = D * X
        return [dX, dS1, dS2, dZ1, dZ2, dO]

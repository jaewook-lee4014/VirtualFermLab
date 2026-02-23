"""Steady-state analysis utilities for continuous fermentation.

Extracted from ESCAPE25 cells 15–17 (optimal dilution rate scanning).
"""

from __future__ import annotations

import numpy as np

from virtualfermlab.models.ode_systems import ModelConfig
from virtualfermlab.simulator.integrator import simulate


def optimal_dilution_rate(
    config: ModelConfig,
    params: dict,
    D_range: np.ndarray,
    S_in1: float,
    S_in2: float,
    duration: float = 500.0,
    n_steps: int = 5000,
) -> dict:
    """Scan dilution rates and find the one maximising biomass productivity.

    Parameters
    ----------
    config : ModelConfig
        Model config (should have ``n_feeds=1``).
    params : dict
        Base kinetic parameters. ``dilutionRate`` will be overwritten.
    D_range : array_like
        Dilution rates to scan (1/h).
    S_in1, S_in2 : float
        Feed substrate concentrations (g/L).
    duration : float
        Simulation length to reach steady-state (h).
    n_steps : int
        Number of time points for integration.

    Returns
    -------
    dict
        Keys: ``D_opt``, ``max_productivity``, ``production_rates``.
    """
    D_range = np.asarray(D_range, dtype=float)
    times = np.linspace(0, duration, n_steps)
    prod_rates = np.zeros_like(D_range)

    p = params.copy()
    p["S_in1"] = S_in1
    p["S_in2"] = S_in2

    for i, D in enumerate(D_range):
        p["dilutionRate"] = D
        result = simulate(config, p, times)
        prod_rates[i] = result.X[-1] * D

    opt_idx = int(np.nanargmax(prod_rates))
    return {
        "D_opt": float(D_range[opt_idx]),
        "max_productivity": float(prod_rates[opt_idx]),
        "production_rates": prod_rates,
    }


def substrate_conversion_efficiency(
    S_final: float,
    S_in: float,
    Yx: float,
) -> float:
    """Effective substrate yield: (1 - S_final/S_in) * Yx.

    Parameters
    ----------
    S_final : float
        Final (or steady-state) substrate concentration (g/L).
    S_in : float
        Feed substrate concentration (g/L).
    Yx : float
        Biomass yield coefficient (g/g).

    Returns
    -------
    float
        Effective substrate conversion efficiency.
    """
    if S_in == 0:
        return 0.0
    return (1.0 - S_final / S_in) * Yx

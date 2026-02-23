"""ODE integration wrapper producing :class:`SimulationResult`.

Combines ``get_traj()`` from ESCAPE25 and ``simulate_monod_lag()`` from the
MPR notebook into a single :func:`simulate` entry-point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.integrate import odeint, solve_ivp

from virtualfermlab.models.ode_systems import FermentationODE, ModelConfig


@dataclass
class SimulationResult:
    """Container for a single simulation trajectory.

    Attributes
    ----------
    times : np.ndarray
        Time points.
    X : np.ndarray
        Biomass concentration (g/L).
    substrates : dict[str, np.ndarray]
        Named substrate time-series (e.g. ``{"S1": ..., "S2": ...}``).
    enzymes : dict[str, np.ndarray]
        Named enzyme time-series (empty if no enzyme model).
    total_output : np.ndarray
        Cumulative diluted biomass output.
    config : ModelConfig
        Model configuration used for this run.
    params : dict
        Parameters used for this run.
    """

    times: np.ndarray
    X: np.ndarray
    substrates: dict[str, np.ndarray]
    enzymes: dict[str, np.ndarray]
    total_output: np.ndarray
    config: ModelConfig
    params: dict

    @property
    def yield_biomass(self) -> float:
        """Biomass yield on substrate Y_X/S (g biomass / g substrate consumed).

        Calculated as (X_final - X0) / total substrate consumed.
        Falls back to 0.0 when no substrate has been consumed.
        """
        X0 = float(self.X[0])
        delta_X = float(self.X[-1]) - X0

        S_consumed = 0.0
        for name, series in self.substrates.items():
            S_consumed += max(0.0, float(series[0]) - float(series[-1]))

        if S_consumed <= 0.0:
            return 0.0
        return delta_X / S_consumed

    @property
    def final_biomass(self) -> float:
        """Final biomass concentration X(t_end) (g/L)."""
        return float(self.X[-1])

    @property
    def mu_max_effective(self) -> float:
        """Maximum instantaneous specific growth rate observed (1/h).

        Computed as max of d(ln X)/dt using a 5-point moving average
        to reduce numerical noise from stiff solver artifacts.
        """
        X = self.X
        t = self.times
        # Guard against non-positive biomass
        mask = X > 0
        if mask.sum() < 2:
            return 0.0
        lnX = np.log(X[mask])
        dt = np.diff(t[mask])
        dlnX = np.diff(lnX)
        valid = dt > 0
        if not valid.any():
            return 0.0
        rates = dlnX[valid] / dt[valid]
        # Smooth with a moving average to dampen stiff-solver spikes
        window = min(5, len(rates))
        if window > 1:
            kernel = np.ones(window) / window
            rates = np.convolve(rates, kernel, mode="valid")
        return float(np.max(rates))


def _build_initial_state(config: ModelConfig, params: dict) -> list[float]:
    """Construct the y0 vector from params and config."""
    if config.n_substrates == 1:
        return [params["X0"], params["S_in"], params.get("y0", 0.0)]

    state = [params["X0"], params["S1"], params["S2"]]
    if config.enzyme_mode == "enzyme":
        state.append(params.get("Z0", 0.0))
    elif config.enzyme_mode == "kompala":
        state.append(params.get("Z1", 0.0))
        state.append(params.get("Z2", 0.0))
    state.append(params.get("y0", 0.0))
    return state


def _unpack_result(
    yobs: np.ndarray,
    times: np.ndarray,
    config: ModelConfig,
    params: dict,
) -> SimulationResult:
    """Convert raw ODE output matrix into :class:`SimulationResult`."""
    X = yobs[:, 0]

    if config.n_substrates == 1:
        substrates = {"S": yobs[:, 1]}
        enzymes: dict[str, np.ndarray] = {}
        total_output = yobs[:, 2]
    else:
        substrates = {"S1": yobs[:, 1], "S2": yobs[:, 2]}
        col = 3
        enzymes = {}
        if config.enzyme_mode == "enzyme":
            enzymes["Z"] = yobs[:, col]
            col += 1
        elif config.enzyme_mode == "kompala":
            enzymes["Z1"] = yobs[:, col]
            enzymes["Z2"] = yobs[:, col + 1]
            col += 2
        total_output = yobs[:, col]

    return SimulationResult(
        times=times,
        X=X,
        substrates=substrates,
        enzymes=enzymes,
        total_output=total_output,
        config=config,
        params=params,
    )


def simulate(
    config: ModelConfig,
    params: dict,
    times: np.ndarray,
    *,
    method: str = "odeint",
) -> SimulationResult:
    """Run a fermentation simulation.

    Parameters
    ----------
    config : ModelConfig
        Model configuration.
    params : dict
        Kinetic parameters (keys match the original notebook conventions).
    times : array_like
        Time evaluation points (h).
    method : str
        ``"odeint"`` (default, LSODA) or any valid ``solve_ivp`` method
        string such as ``"RK45"``, ``"BDF"``, etc.

    Returns
    -------
    SimulationResult
    """
    times = np.asarray(times, dtype=float)
    ode = FermentationODE(config, params)
    y0 = _build_initial_state(config, params)

    if method == "odeint":
        yobs = odeint(ode, y0, times, tfirst=False)
    else:
        sol = solve_ivp(
            ode.solve_ivp_rhs,
            t_span=(times[0], times[-1]),
            y0=y0,
            t_eval=times,
            method=method,
        )
        yobs = sol.y.T

    # Clamp negative concentrations to zero (solver artefact)
    yobs = np.clip(yobs, 0.0, None)

    return _unpack_result(yobs, times, config, params)

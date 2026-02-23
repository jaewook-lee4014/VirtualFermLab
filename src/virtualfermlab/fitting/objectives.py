"""Objective functions for parameter estimation.

Re-implements the **missing** ``calc_all_errors``, ``calc_total_error`` and
``calc_BIC`` functions that are called but never defined in the ESCAPE25
notebook.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import odeint

from virtualfermlab.models.ode_systems import FermentationODE, ModelConfig


def _build_config_from_params(params: dict) -> ModelConfig:
    """Infer a ModelConfig from the flat params dict used in the notebook."""
    ei = params.get("enzyme_induction", False)
    if ei is True:
        enzyme_mode = "enzyme"
    elif ei == "Kompala":
        enzyme_mode = "kompala"
    else:
        enzyme_mode = "direct"
    return ModelConfig(
        n_substrates=params.get("n_substrates", 2),
        n_feeds=params.get("n_feeds", 1),
        growth_model=params.get("growth_model", "Monod"),
        enzyme_mode=enzyme_mode,
    )


def _simulate_for_params(params: dict, times: np.ndarray) -> np.ndarray:
    """Run the ODE with notebook-style params and return the state matrix."""
    config = _build_config_from_params(params)
    ode = FermentationODE(config, params)

    # Build initial state
    ei = params.get("enzyme_induction", False)
    if config.n_substrates == 1:
        y0 = [params["X0"], params["S_in"], params.get("y0", 0.0)]
    else:
        y0 = [params["X0"], params["S1"], params["S2"]]
        if ei is True:
            y0.append(params.get("Z0", 0.0))
        elif ei == "Kompala":
            y0.append(params.get("Z1", 0.0))
            y0.append(params.get("Z2", 0.0))
        y0.append(params.get("y0", 0.0))

    return odeint(ode, y0, times, tfirst=False)


def calc_all_errors(
    params: dict,
    times: np.ndarray,
    exp_data: "pd.DataFrame",
) -> dict[str, np.ndarray]:
    """Compute per-variable residuals between simulation and experiment.

    The experimental data is grouped by time and the median at each unique
    time point is compared against the simulation prediction at that time.

    Parameters
    ----------
    params : dict
        Model parameters (notebook convention).
    times : array_like
        Time points for simulation (typically ``exp_data["Time (h)"]``).
    exp_data : DataFrame
        Experimental data with columns ``"Time (h)"``, ``"[Biomass] (g/L)"``,
        ``"[Glucose] (g/L)"``, ``"[Xylose] (g/L)"``.

    Returns
    -------
    dict
        ``{"biomass_error": ..., "S1_error": ..., "S2_error": ...}``
    """
    import pandas as pd

    times = np.asarray(times, dtype=float)

    # Simulate at all unique time points
    unique_times = np.sort(np.unique(times))
    yobs = _simulate_for_params(params, unique_times)

    X_sim = yobs[:, 0]
    S1_sim = yobs[:, 1]  # Glucose
    S2_sim = yobs[:, 2]  # Xylose

    # Build lookup: time -> index
    time_to_idx = {t: i for i, t in enumerate(unique_times)}

    # Group experimental data by time (median)
    grouped = exp_data.groupby("Time (h)").median(numeric_only=True).reset_index()

    biomass_errors = []
    s1_errors = []
    s2_errors = []

    for _, row in grouped.iterrows():
        t = row["Time (h)"]
        idx = time_to_idx.get(t)
        if idx is None:
            continue
        if "[Biomass] (g/L)" in grouped.columns:
            biomass_errors.append(X_sim[idx] - row["[Biomass] (g/L)"])
        if "[Glucose] (g/L)" in grouped.columns:
            s1_errors.append(S1_sim[idx] - row["[Glucose] (g/L)"])
        if "[Xylose] (g/L)" in grouped.columns:
            s2_errors.append(S2_sim[idx] - row["[Xylose] (g/L)"])

    return {
        "biomass_error": np.array(biomass_errors),
        "S1_error": np.array(s1_errors),
        "S2_error": np.array(s2_errors),
    }


def calc_total_error(
    params: dict,
    times: np.ndarray,
    exp_data: "pd.DataFrame",
    error_type: str,
) -> float:
    """Scalar objective for parameter estimation.

    Parameters
    ----------
    params : dict
        Model parameters.
    times : array_like
        Time points.
    exp_data : DataFrame
        Experimental data.
    error_type : str
        ``"MAE"`` → mean absolute error of all residuals.
        ``"likelihood"`` → Gaussian negative log-likelihood.

    Returns
    -------
    float
        Scalar error value (lower is better).
    """
    errors = calc_all_errors(params, times, exp_data)
    all_errors = np.concatenate([errors["biomass_error"], errors["S1_error"], errors["S2_error"]])

    if error_type == "MAE":
        return float(np.mean(np.abs(all_errors)))
    elif error_type == "likelihood":
        # Gaussian NLL with unit variance
        return float(0.5 * np.sum(all_errors**2))
    else:
        raise ValueError(f"Unknown error_type: {error_type!r}")


def calc_BIC(
    params: dict,
    times: np.ndarray,
    exp_data: "pd.DataFrame",
    N_params: int,
    std_S: float = 1.0,
    std_X: float = 1.0,
) -> float:
    """Bayesian Information Criterion.

    BIC = -2 * ln(L) + k * ln(N)

    Parameters
    ----------
    params : dict
        Model parameters.
    times : array_like
        Time points.
    exp_data : DataFrame
        Experimental data.
    N_params : int
        Number of fitted parameters *k*.
    std_S, std_X : float
        Assumed standard deviation for substrates and biomass.

    Returns
    -------
    float
        BIC value (lower is better).
    """
    errors = calc_all_errors(params, times, exp_data)

    # Gaussian log-likelihood
    biomass_ll = -0.5 * np.sum((errors["biomass_error"] / std_X) ** 2)
    s1_ll = -0.5 * np.sum((errors["S1_error"] / std_S) ** 2)
    s2_ll = -0.5 * np.sum((errors["S2_error"] / std_S) ** 2)
    total_ll = biomass_ll + s1_ll + s2_ll

    N = len(errors["biomass_error"]) + len(errors["S1_error"]) + len(errors["S2_error"])
    return float(-2.0 * total_ll + N_params * np.log(max(N, 1)))

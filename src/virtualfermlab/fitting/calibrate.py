"""Calibration (full parameter estimation) via differential evolution.

Ported from ESCAPE25 ``calibration_function()``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from virtualfermlab.fitting.bounds import get_bounds, get_param_name_order
from virtualfermlab.fitting.objectives import calc_total_error


def calibration_objective(x: np.ndarray, *args: Any) -> float:
    """Objective function for calibration.

    Parameters
    ----------
    x : array
        Parameter vector.
    args : tuple
        ``(params_template, exp_data, error_type)``

    Returns
    -------
    float
        Scalar error.
    """
    params = args[0].copy()
    exp_data = args[1]
    error_type = args[2]

    param_names = get_param_name_order(params, "calibration")
    for name, val in zip(param_names, x):
        params[name] = val

    times = exp_data["Time (h)"]
    return calc_total_error(params, times, exp_data, error_type)


def run_calibration(
    params: dict,
    exp_data: pd.DataFrame,
    error_type: str = "MAE",
    maxiter: int = 100,
    init: str = "latinhypercube",
    seed: int | None = None,
    **de_kwargs: Any,
) -> dict:
    """Run differential-evolution calibration.

    Parameters
    ----------
    params : dict
        Base parameters (``enzyme_induction``, ``n_feeds``, etc.).
    exp_data : DataFrame
        Experimental data.
    error_type : str
        ``"MAE"`` or ``"likelihood"``.
    maxiter : int
        Maximum DE iterations.
    init : str
        DE initialisation strategy.
    seed : int or None
        Random seed.
    **de_kwargs
        Extra keyword arguments for ``differential_evolution``.

    Returns
    -------
    dict
        ``{"params": fitted_dict, "error": float, "result": OptimizeResult}``
    """
    bounds = get_bounds(params, "calibration")
    result = differential_evolution(
        calibration_objective,
        bounds=bounds,
        args=(params, exp_data, error_type),
        maxiter=maxiter,
        init=init,
        seed=seed,
        **de_kwargs,
    )

    param_names = get_param_name_order(params, "calibration")
    fitted = dict(zip(param_names, result.x))

    return {
        "params": fitted,
        "error": float(result.fun),
        "result": result,
    }

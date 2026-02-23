"""Validation (initial-condition-only re-fit) via differential evolution.

Ported from ESCAPE25 ``validation_function()``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from virtualfermlab.fitting.bounds import get_bounds, get_param_name_order
from virtualfermlab.fitting.objectives import calc_total_error


def validation_objective(x: np.ndarray, *args: Any) -> float:
    """Objective function for validation (only initial conditions re-fit).

    Parameters
    ----------
    x : array
        Parameter vector (only X0, Z0, Z1, Z2 depending on model).
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

    param_names = get_param_name_order(params, "validation")
    for name, val in zip(param_names, x):
        params[name] = val

    times = exp_data["Time (h)"]
    return calc_total_error(params, times, exp_data, error_type)


def run_validation(
    params: dict,
    exp_data: pd.DataFrame,
    error_type: str = "MAE",
    maxiter: int = 100,
    init: str = "latinhypercube",
    seed: int | None = None,
    **de_kwargs: Any,
) -> dict:
    """Run differential-evolution validation.

    Only initial conditions (X0, and optionally Z0 or Z1/Z2) are re-fit.

    Parameters
    ----------
    params : dict
        Parameters from calibration merged with new substrate concentrations.
    exp_data : DataFrame
        Validation experimental data.
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
    bounds = get_bounds(params, "validation")

    # Build x0 from current params
    param_names = get_param_name_order(params, "validation")
    x0 = [params.get(name, 0.1) for name in param_names]

    result = differential_evolution(
        validation_objective,
        bounds=bounds,
        args=(params, exp_data, error_type),
        maxiter=maxiter,
        init=init,
        seed=seed,
        x0=x0,
        **de_kwargs,
    )

    fitted = dict(zip(param_names, result.x))

    return {
        "params": fitted,
        "error": float(result.fun),
        "result": result,
    }

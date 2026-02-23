"""Model-specific fitters for pH growth experiments.

Ported from MPR notebook ``fit_monod_lag_fixedYxs()`` and
``fit_lagged_exponential_cap_fixedYxs()``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from virtualfermlab.data.transforms import prepare_time_series
from virtualfermlab.fitting.metrics import metrics
from virtualfermlab.models.analytical import predict_lagged_exponential_cap
from virtualfermlab.models.ph_model import lag_switch


def _monod_lag_odes(
    t: float,
    y: list[float],
    mu_max: float,
    Ks: float,
    Yxs: float,
    lag: float,
) -> list[float]:
    """Single-substrate Monod ODE with lag phase."""
    X, S = y
    if t < lag:
        mu = 0.0
    else:
        mu = mu_max * (S / (Ks + S)) if S > 0 else 0.0
    dXdt = mu * X
    dSdt = -(1.0 / Yxs) * dXdt
    return [dXdt, dSdt]


def simulate_monod_lag(
    time: np.ndarray,
    X0: float,
    S0: float,
    mu_max: float,
    Ks: float,
    Yxs: float,
    lag: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate single-substrate Monod growth with lag.

    Returns
    -------
    tuple of ndarray
        ``(X, S)`` biomass and substrate arrays.
    """
    t = np.asarray(time)
    sort_idx = np.argsort(t)
    t_sorted = t[sort_idx]

    sol = solve_ivp(
        _monod_lag_odes,
        t_span=(t_sorted.min(), t_sorted.max()),
        y0=[X0, S0],
        t_eval=t_sorted,
        args=(mu_max, Ks, Yxs, lag),
        method="RK45",
    )

    X = np.empty_like(sol.y[0])
    S = np.empty_like(sol.y[1])
    X[sort_idx] = sol.y[0]
    S[sort_idx] = sol.y[1]
    return X, S


def fit_monod_lag(
    time: np.ndarray,
    X_obs: np.ndarray,
    *,
    S0: float = 15.0,
    Yxs_fixed: float = 0.5,
    lag_grid: np.ndarray | None = None,
    loss: str = "soft_l1",
    f_scale: float = 0.1,
) -> dict[str, Any]:
    """Fit Monod+lag model with Yxs fixed.

    Fits ``(mu_max, Ks, lag)`` using multi-start least-squares.

    Parameters
    ----------
    time, X_obs : array_like
        Time and observed biomass.
    S0 : float
        Initial substrate concentration.
    Yxs_fixed : float
        Fixed biomass yield.
    lag_grid : array or None
        Grid of initial lag guesses.
    loss, f_scale : str, float
        Robust loss settings for ``least_squares``.

    Returns
    -------
    dict
        ``{"model", "params", "opt", "fit_metrics", "time", "X_obs", "X_pred"}``
    """
    time, X_obs = prepare_time_series(time, X_obs)
    X0 = float(X_obs[0])

    if lag_grid is None:
        lag_grid = np.linspace(
            time.min(), time.min() + 0.33 * (time.max() - time.min()), 40
        )

    bounds = (
        np.array([1e-4, 1e-6, 0.0]),       # mu_max, Ks, lag
        np.array([2.0, 100.0, time.max()]),
    )

    best = None

    def resid(p):
        mu, Ks, lag = p
        X_pred, _ = simulate_monod_lag(time, X0, S0, mu, Ks, Yxs_fixed, lag)
        return X_pred - X_obs

    for lag0 in lag_grid:
        p0 = np.array([0.2, 1.0, float(lag0)])
        p0 = np.maximum(p0, bounds[0] + 1e-12)
        p0 = np.minimum(p0, bounds[1] - 1e-12)

        try:
            res = least_squares(resid, p0, bounds=bounds, loss=loss, f_scale=f_scale)
        except Exception:
            continue

        if best is None or res.cost < best["opt"].cost:
            X_fit, _ = simulate_monod_lag(
                time, X0, S0, res.x[0], res.x[1], Yxs_fixed, res.x[2]
            )
            best = {
                "model": "monod",
                "params": {
                    "mu_max": float(res.x[0]),
                    "Ks": float(res.x[1]),
                    "lag": float(res.x[2]),
                    "Yxs": float(Yxs_fixed),
                },
                "opt": res,
                "fit_metrics": metrics(X_obs, X_fit),
                "time": time,
                "X_obs": X_obs,
                "X_pred": X_fit,
            }

    return best


def fit_exponential_cap(
    time: np.ndarray,
    X_obs: np.ndarray,
    *,
    S0: float = 15.0,
    Yxs_fixed: float = 0.5,
    lag_grid: np.ndarray | None = None,
    loss: str = "soft_l1",
    f_scale: float = 0.1,
) -> dict[str, Any]:
    """Fit lagged-exponential-cap model with Yxs fixed.

    Fits ``(mu_max, lag)``.

    Parameters
    ----------
    time, X_obs : array_like
        Time and observed biomass.
    S0 : float
        Initial substrate concentration.
    Yxs_fixed : float
        Fixed biomass yield.
    lag_grid : array or None
        Grid of initial lag guesses.
    loss, f_scale : str, float
        Robust loss settings for ``least_squares``.

    Returns
    -------
    dict
        ``{"model", "params", "opt", "fit_metrics", "time", "X_obs", "X_pred"}``
    """
    time, X_obs = prepare_time_series(time, X_obs)
    X0 = float(X_obs[0])

    if lag_grid is None:
        lag_grid = np.linspace(
            time.min(), time.min() + 0.33 * (time.max() - time.min()), 40
        )

    bounds = (
        np.array([1e-4, 0.0]),       # mu_max, lag
        np.array([2.0, time.max()]),
    )

    best = None

    def resid(p):
        mu, lag = p
        X_pred = predict_lagged_exponential_cap(time, X0, S0, mu, lag, Yxs_fixed)
        return X_pred - X_obs

    for lag0 in lag_grid:
        p0 = np.array([0.2, float(lag0)])
        p0 = np.maximum(p0, bounds[0] + 1e-12)
        p0 = np.minimum(p0, bounds[1] - 1e-12)

        res = least_squares(resid, p0, bounds=bounds, loss=loss, f_scale=f_scale)

        if best is None or res.cost < best["opt"].cost:
            X_fit = predict_lagged_exponential_cap(
                time, X0, S0, res.x[0], res.x[1], Yxs_fixed
            )
            best = {
                "model": "exp_cap",
                "params": {
                    "mu_max": float(res.x[0]),
                    "lag": float(res.x[1]),
                    "Yxs": float(Yxs_fixed),
                },
                "opt": res,
                "fit_metrics": metrics(X_obs, X_fit),
                "time": time,
                "X_obs": X_obs,
                "X_pred": X_fit,
            }

    return best


def fit_compare_models(
    time: np.ndarray,
    X_obs: np.ndarray,
    *,
    S0: float = 15.0,
    Yxs_fixed: float = 0.5,
) -> tuple[dict, dict[str, dict]]:
    """Fit both Monod+lag and exponential-cap models, return the best.

    Returns
    -------
    tuple
        ``(chosen, {"monod": fit_monod, "exp_cap": fit_exp})``
    """
    fit_m = fit_monod_lag(time, X_obs, S0=S0, Yxs_fixed=Yxs_fixed)
    fit_e = fit_exponential_cap(time, X_obs, S0=S0, Yxs_fixed=Yxs_fixed)

    all_fits = {"monod": fit_m, "exp_cap": fit_e}

    if fit_m is None and fit_e is None:
        return None, all_fits
    if fit_m is None:
        return fit_e, all_fits
    if fit_e is None:
        return fit_m, all_fits

    chosen = fit_m if fit_m["fit_metrics"]["MAE"] <= fit_e["fit_metrics"]["MAE"] else fit_e
    return chosen, all_fits

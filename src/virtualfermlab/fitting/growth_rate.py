"""Model-free exponential growth rate estimation.

Ported from MPR notebook ``fit_exponential_growth()`` and
``sweep_growth_rates()``.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import linregress


def fit_exponential_growth(
    time: np.ndarray,
    biomass: np.ndarray,
    min_frac: float,
    max_frac: float,
) -> dict | None:
    """Fit ln(X) = mu*t + b over a fractional biomass window.

    Parameters
    ----------
    time, biomass : array_like
        Time and biomass arrays.
    min_frac, max_frac : float
        Lower and upper fractions of max biomass defining the fitting window.

    Returns
    -------
    dict or None
        ``{"mu_pct", "r2", "duration", "stderr", "n"}`` or None if too few points.
    """
    time = np.asarray(time)
    biomass = np.asarray(biomass)
    Xmax = biomass.max()
    mask = (biomass > min_frac * Xmax) & (biomass < max_frac * Xmax)

    t_sel = time[mask]
    X_sel = biomass[mask]

    if len(t_sel) < 2:
        return None

    lnX = np.log(X_sel)
    res = linregress(t_sel, lnX)

    return {
        "mu_pct": res.slope * 100.0,
        "r2": res.rvalue**2,
        "duration": float(t_sel.max() - t_sel.min()),
        "stderr": res.stderr,
        "n": len(t_sel),
    }


def sweep_growth_rates(
    time: np.ndarray,
    biomass_dict: dict[str, np.ndarray],
    beta_grid: np.ndarray,
    alpha_grid: np.ndarray,
    min_points: int = 8,
    min_duration: float = 8.0,
    r2_threshold: float = 0.8,
) -> tuple[dict[str, list], dict[str, list]]:
    """Sweep (beta, alpha) windows and collect growth rates.

    Parameters
    ----------
    time : array_like
        Common time axis.
    biomass_dict : dict
        ``{condition_name: biomass_array}``.
    beta_grid, alpha_grid : array_like
        Lower / upper biomass fraction grids.
    min_points : int
        Minimum data points in window.
    min_duration : float
        Minimum time span (h).
    r2_threshold : float
        Minimum R^2 for acceptance.

    Returns
    -------
    tuple of dicts
        ``(mu_abs, mu_norm)`` where each maps condition name to list of values.
    """
    time = np.asarray(time)
    conditions = list(biomass_dict.keys())

    mu_abs = {c: [] for c in conditions}
    mu_norm = {c: [] for c in conditions}

    for beta in beta_grid:
        for alpha in alpha_grid:
            mu_current = {}
            valid = True

            for c in conditions:
                fit = fit_exponential_growth(time, biomass_dict[c], beta, alpha)
                if (
                    fit is None
                    or fit["n"] < min_points
                    or fit["duration"] < min_duration
                    or fit["r2"] < r2_threshold
                ):
                    valid = False
                    break
                mu_current[c] = fit["mu_pct"]

            if not valid:
                continue

            mu_max_val = max(mu_current.values())
            for c, mu in mu_current.items():
                mu_abs[c].append(mu)
                mu_norm[c].append(mu / mu_max_val if mu_max_val > 0 else 0.0)

    return mu_abs, mu_norm

"""Analytical (non-ODE) growth models.

Extracted from the MPR pH experiment notebook.
"""

from __future__ import annotations

import numpy as np


def predict_lagged_exponential_cap(
    time: np.ndarray,
    X0: float,
    S0: float,
    mu_max: float,
    lag: float,
    Yxs: float,
) -> np.ndarray:
    """Lagged exponential growth with a yield-based biomass cap.

    During the lag phase (``t <= lag``) biomass stays at *X0*.  After the lag
    phase biomass grows exponentially at rate *mu_max*, capped at
    ``X0 + Yxs * S0``.

    Parameters
    ----------
    time : array_like
        Time points (h).
    X0 : float
        Initial biomass concentration (g/L).
    S0 : float
        Initial substrate concentration (g/L).
    mu_max : float
        Maximum specific growth rate (1/h).
    lag : float
        Lag phase duration (h).
    Yxs : float
        Biomass yield on substrate (g/g).

    Returns
    -------
    np.ndarray
        Predicted biomass at each time point.
    """
    t = np.asarray(time, dtype=float)
    X_cap = X0 + Yxs * S0
    X_pred = np.full_like(t, fill_value=X0)
    grow = t > lag
    X_pred[grow] = X0 * np.exp(mu_max * (t[grow] - lag))
    return np.minimum(X_pred, X_cap)

"""Data transformation utilities.

Ported from ESCAPE25 (``tidy_data``, ``od600_to_biomass``) and MPR
(``prepare_time_series``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Default OD600 → biomass polynomial coefficients from ESCAPE25
_DEFAULT_OD_COEFFICIENTS = (0.2927, 0.2631, 1.2808, -0.1596)


def od600_to_biomass(
    od600: np.ndarray | float,
    coeff: tuple[float, ...] = _DEFAULT_OD_COEFFICIENTS,
) -> np.ndarray | float:
    """Convert OD600 readings to biomass concentration.

    Uses a cubic polynomial:
    ``X = coeff[0]*OD^3 + coeff[1]*OD^2 + coeff[2]*OD + coeff[3]``

    Parameters
    ----------
    od600 : array_like
        Optical density at 600 nm.
    coeff : tuple of float
        Polynomial coefficients ``(a3, a2, a1, a0)``.

    Returns
    -------
    array_like
        Biomass concentration (g/L).
    """
    od = np.asarray(od600, dtype=float)
    a3, a2, a1, a0 = coeff
    return a3 * od**3 + a2 * od**2 + a1 * od + a0


def tidy_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean experimental data: add biomass column, normalise column names.

    Parameters
    ----------
    df : DataFrame
        Raw experimental data with ``"OD600 (-)"`` column.

    Returns
    -------
    DataFrame
        Copy with ``"[Biomass] (g/L)"`` column added.
    """
    df = df.copy()
    df["[Biomass] (g/L)"] = od600_to_biomass(df["OD600 (-)"])
    # Normalise whitespace in xylose column name
    df.rename(columns={"[Xylose] (g/L) ": "[Xylose] (g/L)"}, inplace=True)
    return df


def prepare_time_series(
    time: np.ndarray,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort, deduplicate and filter a time/biomass series.

    Parameters
    ----------
    time, X : array_like
        Time and biomass arrays.

    Returns
    -------
    tuple of ndarray
        Cleaned ``(time, X)`` arrays.
    """
    tmp = pd.DataFrame({"time": np.asarray(time), "X": np.asarray(X)})
    tmp = tmp.sort_values("time").drop_duplicates(subset="time")
    tmp = tmp[np.isfinite(tmp["X"]) & np.isfinite(tmp["time"])]
    tmp = tmp[tmp["X"] > 0]
    return tmp["time"].values, tmp["X"].values

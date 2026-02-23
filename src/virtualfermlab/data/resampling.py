"""Data resampling utilities.

Ported from ESCAPE25 ``resample_data()`` with additional bootstrap support.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def resample_data(
    exp_data: pd.DataFrame,
    random_state: int = 0,
    N: int = 1,
) -> pd.DataFrame:
    """Remove *N* random rows from experimental data (leave-one-out style).

    Parameters
    ----------
    exp_data : DataFrame
        Experimental data.
    random_state : int
        Seed for reproducibility.
    N : int
        Number of rows to drop.

    Returns
    -------
    DataFrame
        Resampled data with dropped rows removed.
    """
    return (
        exp_data.drop(exp_data.sample(N, random_state=random_state).index)
        .reset_index(drop=True)
    )


def bootstrap(
    exp_data: pd.DataFrame,
    n_bootstrap: int = 100,
    random_state: int = 0,
    frac: float = 1.0,
) -> list[pd.DataFrame]:
    """Generate bootstrap resamples (with replacement).

    Parameters
    ----------
    exp_data : DataFrame
        Source data.
    n_bootstrap : int
        Number of bootstrap samples.
    random_state : int
        Base seed.
    frac : float
        Fraction of original data to draw per sample.

    Returns
    -------
    list of DataFrame
        Bootstrap samples.
    """
    rng = np.random.default_rng(random_state)
    samples = []
    n = len(exp_data)
    size = max(1, int(n * frac))
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=size)
        samples.append(exp_data.iloc[idx].reset_index(drop=True))
    return samples

"""CSV data loaders.

Thin wrappers around :func:`pandas.read_csv` with domain-specific defaults.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_growth_csv(path: str | Path) -> pd.DataFrame:
    """Load a microplate growth CSV file.

    Parameters
    ----------
    path : str or Path
        File path.

    Returns
    -------
    DataFrame
    """
    return pd.read_csv(path)

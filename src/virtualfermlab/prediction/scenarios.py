"""Data loading and scenario-specific train/test splitting."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from virtualfermlab.data.loaders import load_growth_csv
from virtualfermlab.data.transforms import tidy_data
from virtualfermlab.prediction.features import build_features

# Default data directory (relative to this file → project root)
_DATA_DIR = Path(__file__).resolve().parents[3] / "0.TomsCode" / "0.Data"


def load_datasets(
    data_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and tidy both growth-curve CSV datasets.

    Returns
    -------
    df_1to1, df_2to1 : DataFrame
        Tidied DataFrames for the 1:1 and 2:1 glucose:xylose ratios.
    """
    data_dir = Path(data_dir) if data_dir else _DATA_DIR
    df_11 = tidy_data(
        load_growth_csv(
            data_dir / "1-to-1-GlucoseXyloseMicroplateGrowth.csv"
        )
    )
    df_21 = tidy_data(
        load_growth_csv(
            data_dir / "2-to-1-GlucoseXyloseMicroplateGrowth.csv"
        )
    )
    return df_11, df_21


def cross_ratio_split(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    scenario: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build features for one ratio as train, another as test.

    Returns ``(X_train, y_train, X_test, y_test)``.
    """
    X_train, y_train = build_features(df_train, scenario)
    X_test, y_test = build_features(df_test, scenario)
    return X_train, y_train, X_test, y_test


def combined_groupkfold(
    df_11: pd.DataFrame,
    df_21: pd.DataFrame,
    scenario: int,
    n_splits: int = 5,
) -> Iterator[
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
]:
    """GroupKFold CV on both datasets combined, grouped by time point.

    Features are built per-ratio (medians are ratio-specific) then
    concatenated.  Yields ``(X_train, y_train, X_test, y_test)`` tuples.
    """
    X_11, y_11 = build_features(df_11, scenario)
    X_21, y_21 = build_features(df_21, scenario)

    X_all = pd.concat([X_11, X_21], ignore_index=True)
    y_all = pd.concat([y_11, y_21], ignore_index=True)

    groups = X_all["time"].values
    n_groups = len(np.unique(groups))
    gkf = GroupKFold(n_splits=min(n_splits, n_groups))

    for train_idx, test_idx in gkf.split(X_all, y_all, groups):
        yield (
            X_all.iloc[train_idx].reset_index(drop=True),
            y_all.iloc[train_idx].reset_index(drop=True),
            X_all.iloc[test_idx].reset_index(drop=True),
            y_all.iloc[test_idx].reset_index(drop=True),
        )

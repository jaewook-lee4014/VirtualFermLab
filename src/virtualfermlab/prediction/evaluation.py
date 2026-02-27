"""Model training, evaluation, and scenario comparison."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from virtualfermlab.fitting.metrics import metrics as _compute_metrics

# ── Default XGBoost hyper-parameters (tuned for small datasets) ────────

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "max_depth": 3,
    "n_estimators": 100,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbosity": 0,
    "tree_method": "hist",
    "device": "cpu",
    "n_jobs": 1,
}


# ── Model factories ───────────────────────────────────────────────────

def make_xgboost(params: dict | None = None) -> XGBRegressor:
    """Create an :class:`XGBRegressor` with small-dataset defaults."""
    p = {**DEFAULT_XGB_PARAMS, **(params or {})}
    return XGBRegressor(**p)


def make_linear() -> LinearRegression:
    """Create a plain :class:`LinearRegression`."""
    return LinearRegression()


# ── Scenario runner ───────────────────────────────────────────────────

def run_scenario(
    scenario: int,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    xgb_params: dict | None = None,
) -> dict[str, Any]:
    """Train XGBoost, Linear, and Naive models; return metrics + predictions.

    Returns
    -------
    dict
        Keys: ``scenario``, ``models``, ``predictions``, ``metrics``
        (DataFrame with columns *model*, *target*, *MAE*, *RMSE*, *R2*).
    """
    results: dict[str, Any] = {
        "scenario": scenario,
        "models": {},
        "predictions": {},
    }
    metric_rows: list[dict] = []

    for target in ("glucose", "xylose"):
        y_tr = y_train[target].values
        y_te = y_test[target].values

        # -- XGBoost --
        xgb = make_xgboost(xgb_params)
        xgb.fit(X_train, y_tr)
        pred_xgb = xgb.predict(X_test)
        results["models"][f"xgb_{target}"] = xgb
        results["predictions"][f"xgb_{target}"] = pred_xgb
        m = _compute_metrics(y_te, pred_xgb)
        metric_rows.append({"model": "XGBoost", "target": target, **m})

        # -- Linear Regression --
        lr = make_linear()
        lr.fit(X_train, y_tr)
        pred_lr = lr.predict(X_test)
        results["predictions"][f"lr_{target}"] = pred_lr
        metric_rows.append({"model": "Linear", "target": target, **_compute_metrics(y_te, pred_lr)})

        # -- Naive mean baseline --
        pred_naive = np.full(len(y_te), np.mean(y_tr))
        results["predictions"][f"naive_{target}"] = pred_naive
        metric_rows.append({"model": "Naive Mean", "target": target, **_compute_metrics(y_te, pred_naive)})

        # -- Last-HPLC baseline (when available) --
        col = f"last_known_{target}"
        if col in X_test.columns:
            pred_last = X_test[col].values
            results["predictions"][f"last_hplc_{target}"] = pred_last
            metric_rows.append({"model": "Last HPLC", "target": target, **_compute_metrics(y_te, pred_last)})

    results["metrics"] = pd.DataFrame(metric_rows)
    return results


# ── Scenario comparison ───────────────────────────────────────────────

def compare_scenarios(
    all_results: dict[int, dict],
) -> pd.DataFrame:
    """Aggregate metrics across scenarios into one summary table.

    Parameters
    ----------
    all_results : dict
        Mapping ``{scenario_number: run_scenario output, …}``.

    Returns
    -------
    DataFrame
        Columns: *scenario*, *model*, *target*, *MAE*, *RMSE*, *R2*.
    """
    frames = []
    for sc, res in sorted(all_results.items()):
        df = res["metrics"].copy()
        df.insert(0, "scenario", sc)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

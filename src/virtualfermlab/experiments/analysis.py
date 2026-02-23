"""Post-processing analysis for virtual experiment results.

Pareto front extraction, heatmap data, and condition ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from virtualfermlab.experiments.monte_carlo import MonteCarloResult


def pareto_front(
    results: list[MonteCarloResult],
    obj1: str = "mean_yield",
    obj2: str = "mean_mu_max",
) -> pd.DataFrame:
    """Extract non-dominated solutions (maximising both objectives).

    Parameters
    ----------
    results : list of MonteCarloResult
    obj1, obj2 : str
        Attribute names on :class:`MonteCarloResult` to maximise.

    Returns
    -------
    DataFrame
        Rows corresponding to Pareto-optimal conditions.
    """
    rows = []
    for r in results:
        rows.append({
            "strain": r.condition.strain,
            "substrate_A": r.condition.substrate_A,
            "substrate_B": r.condition.substrate_B,
            "ratio": r.condition.ratio,
            "pH": r.condition.pH,
            obj1: getattr(r, obj1),
            obj2: getattr(r, obj2),
            "std_yield": r.std_yield,
        })
    df = pd.DataFrame(rows)

    # Non-dominated sorting (maximise both)
    dominated = set()
    for i in range(len(df)):
        for j in range(len(df)):
            if i == j:
                continue
            if (
                df.iloc[j][obj1] >= df.iloc[i][obj1]
                and df.iloc[j][obj2] >= df.iloc[i][obj2]
                and (df.iloc[j][obj1] > df.iloc[i][obj1] or df.iloc[j][obj2] > df.iloc[i][obj2])
            ):
                dominated.add(i)
                break

    return df.drop(index=list(dominated)).reset_index(drop=True)


def heatmap_data(
    results: list[MonteCarloResult],
    x: str = "pH",
    y: str = "ratio",
    z: str = "mean_yield",
) -> pd.DataFrame:
    """Pivot Monte Carlo results into a heatmap-ready DataFrame.

    Parameters
    ----------
    results : list of MonteCarloResult
    x, y : str
        Condition attributes for axes.
    z : str
        Result attribute for colour.

    Returns
    -------
    DataFrame
        Pivoted with *x* as columns and *y* as index.
    """
    rows = []
    for r in results:
        rows.append({
            x: getattr(r.condition, x),
            y: getattr(r.condition, y),
            z: getattr(r, z),
        })
    df = pd.DataFrame(rows)
    return df.pivot_table(index=y, columns=x, values=z, aggfunc="mean")


def rank_conditions(
    results: list[MonteCarloResult],
    score_fn: callable | None = None,
) -> pd.DataFrame:
    """Rank conditions by a composite score.

    Default score: ``mean_yield - 0.5 * std_yield``.

    Parameters
    ----------
    results : list of MonteCarloResult
    score_fn : callable or None
        ``f(MonteCarloResult) -> float``.

    Returns
    -------
    DataFrame
        Sorted by score descending.
    """
    if score_fn is None:
        def score_fn(r):
            return r.mean_yield - 0.5 * r.std_yield

    rows = []
    for r in results:
        rows.append({
            "strain": r.condition.strain,
            "substrate_A": r.condition.substrate_A,
            "substrate_B": r.condition.substrate_B,
            "ratio": r.condition.ratio,
            "pH": r.condition.pH,
            "mean_yield": r.mean_yield,
            "std_yield": r.std_yield,
            "mean_mu_max": r.mean_mu_max,
            "score": score_fn(r),
        })
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

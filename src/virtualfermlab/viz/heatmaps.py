"""Heatmap and Pareto front visualisations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from virtualfermlab.experiments.monte_carlo import MonteCarloResult


def plot_pH_ratio_heatmap(
    heatmap_df: pd.DataFrame,
    *,
    title: str = "pH x Ratio Heatmap",
    cmap: str = "viridis",
    figsize: tuple[float, float] = (8, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a pH × ratio heatmap.

    Parameters
    ----------
    heatmap_df : DataFrame
        Pivoted DataFrame (index=ratio, columns=pH, values=metric).
    title : str
    cmap : str
    figsize : tuple

    Returns
    -------
    tuple
        ``(fig, ax)``
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        heatmap_df.values,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        extent=[
            heatmap_df.columns.min(),
            heatmap_df.columns.max(),
            heatmap_df.index.min(),
            heatmap_df.index.max(),
        ],
    )
    ax.set_xlabel("pH")
    ax.set_ylabel("Ratio (substrate A fraction)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    return fig, ax


def plot_pareto_front(
    pareto_df: pd.DataFrame,
    obj1: str = "mean_yield",
    obj2: str = "mean_mu_max",
    *,
    figsize: tuple[float, float] = (8, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a 2-objective Pareto front.

    Parameters
    ----------
    pareto_df : DataFrame
        Output of :func:`analysis.pareto_front`.
    obj1, obj2 : str
        Column names for the two objectives.
    figsize : tuple

    Returns
    -------
    tuple
        ``(fig, ax)``
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(pareto_df[obj1], pareto_df[obj2], c="tab:blue", edgecolors="k", s=60)
    ax.set_xlabel(obj1)
    ax.set_ylabel(obj2)
    ax.set_title("Pareto Front")
    ax.grid(True, alpha=0.3)
    return fig, ax

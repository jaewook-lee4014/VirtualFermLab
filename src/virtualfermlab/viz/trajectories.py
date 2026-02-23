"""Simulation trajectory plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from virtualfermlab.simulator.integrator import SimulationResult


def plot_trajectory(
    result: SimulationResult,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot biomass and substrate trajectories.

    Parameters
    ----------
    result : SimulationResult
    title : str or None
    figsize : tuple

    Returns
    -------
    tuple
        ``(fig, ax)``
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(result.times, result.X, label="Biomass (X)", linewidth=2)
    for name, series in result.substrates.items():
        ax.plot(result.times, series, label=name, linestyle="--")

    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Concentration (g/L)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)

    return fig, ax


def plot_results(
    times: np.ndarray,
    X: np.ndarray,
    S: np.ndarray,
    u: np.ndarray,
    *,
    title: str | None = None,
    one_plot: bool = False,
    batchBool: bool = False,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes | list[plt.Axes]]:
    """General-purpose results plotter (compatible with notebook calls).

    Parameters
    ----------
    times : array_like
    X : array_like
        Biomass.
    S : array_like
        Substrates (1D or 2D).
    u : array_like
        Dilution rate(s).
    title : str or None
    one_plot : bool
        If True, combine all on one axis.
    batchBool : bool
        If True, skip dilution rate subplot.
    figsize : tuple or None

    Returns
    -------
    tuple
        ``(fig, axes)``
    """
    S = np.squeeze(S)
    u = np.squeeze(u)

    if one_plot:
        if figsize is None:
            figsize = (8, 5)
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(times, X, label="Predicted X")
        if S.ndim == 1:
            ax.plot(times, S, label="S", linestyle="--")
        else:
            ax.plot(times, S[:, 0], label="S1 (Glucose)", linestyle="--")
            if S.shape[1] > 1:
                ax.plot(times, S[:, 1], label="S2 (Xylose)", linestyle=":")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Concentration (g/L)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if title:
            ax.set_title(title)
        return fig, ax

    n_plots = 2 if batchBool else 3
    if figsize is None:
        figsize = (16, 5)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    axes[0].plot(times, X, label="Predicted X")
    axes[0].set_ylabel("Biomass (g/L)")
    axes[0].set_xlabel("Time (h)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if S.ndim == 1:
        axes[1].plot(times, S, label="S")
    else:
        axes[1].plot(times, S[:, 0], label="S1 (Glucose)")
        if S.shape[1] > 1:
            axes[1].plot(times, S[:, 1], label="S2 (Xylose)")
    axes[1].set_ylabel("Substrate (g/L)")
    axes[1].set_xlabel("Time (h)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if not batchBool:
        axes[2].plot(times, u)
        axes[2].set_ylabel("Dilution Rate (1/h)")
        axes[2].set_xlabel("Time (h)")
        axes[2].grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)

    return fig, axes

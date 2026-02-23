"""pH-dependent parameter trend plots.

Recreates the multi-axis parameter vs pH plots from the MPR notebook.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_pH_parameter_trends(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (7, 4),
) -> tuple[plt.Figure, plt.Axes]:
    """Multi-axis plot of Monod parameters vs pH.

    Parameters
    ----------
    df : DataFrame
        Must have columns ``"pH"``, ``"mu_max"``, ``"Yxs"``, ``"lag"``,
        and optionally ``"Ks"``.
    figsize : tuple

    Returns
    -------
    tuple
        ``(fig, ax_mu)``
    """
    df = df.sort_values("pH").copy()

    fig, ax_mu = plt.subplots(figsize=figsize)

    # mu_max
    ax_mu.scatter(df["pH"], df["mu_max"], color="C0", zorder=3)
    ax_mu.plot(df["pH"], df["mu_max"], color="C0", alpha=0.7)
    ax_mu.set_xlabel("pH")
    ax_mu.set_ylabel("mu_max (1/h)", color="C0")
    ax_mu.tick_params(axis="y", colors="C0")

    # Ks (if available)
    if "Ks" in df.columns:
        ax_Ks = ax_mu.twinx()
        ax_Ks.scatter(df["pH"], df["Ks"], color="C1")
        ax_Ks.plot(df["pH"], df["Ks"], color="C1", alpha=0.7)
        ax_Ks.set_ylabel("Ks (g/L)", color="C1")
        ax_Ks.tick_params(axis="y", colors="C1")

    # Yxs
    ax_Y = ax_mu.twinx()
    ax_Y.spines["right"].set_position(("axes", 1.2))
    ax_Y.scatter(df["pH"], df["Yxs"], color="C2", zorder=3)
    ax_Y.plot(df["pH"], df["Yxs"], color="C2", alpha=0.7)
    ax_Y.set_ylabel("Yxs (g/g)", color="C2")
    ax_Y.tick_params(axis="y", colors="C2")

    # lag
    if "lag" in df.columns:
        ax_lag = ax_mu.twinx()
        ax_lag.spines["right"].set_position(("axes", 1.45))
        ax_lag.scatter(df["pH"], df["lag"], color="C3", zorder=3)
        ax_lag.plot(df["pH"], df["lag"], color="C3", alpha=0.7)
        ax_lag.set_ylabel("Lag (h)", color="C3")
        ax_lag.tick_params(axis="y", colors="C3")

    ax_mu.set_title("Monod parameter estimates vs pH")
    fig.tight_layout()
    return fig, ax_mu

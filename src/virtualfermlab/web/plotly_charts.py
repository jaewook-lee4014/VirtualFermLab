"""Plotly chart functions for the web UI.

Returns JSON-serialisable dicts ``{"data": [...], "layout": {...}}``
that the client renders via ``Plotly.newPlot(div, data, layout)``.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from virtualfermlab.simulator.integrator import SimulationResult
from virtualfermlab.experiments.monte_carlo import MonteCarloResult


def _fig_to_json(fig: go.Figure) -> dict:
    """Convert a Plotly figure to a JSON-safe dict with data + layout."""
    return json.loads(fig.to_json())


def plot_trajectory_plotly(result: SimulationResult) -> dict:
    """Plot biomass and substrate trajectories."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=result.times.tolist(),
        y=result.X.tolist(),
        name="Biomass (X)",
        line=dict(width=2.5),
    ))

    for name, series in result.substrates.items():
        fig.add_trace(go.Scatter(
            x=result.times.tolist(),
            y=series.tolist(),
            name=name,
            line=dict(dash="dash"),
        ))

    for name, series in result.enzymes.items():
        fig.add_trace(go.Scatter(
            x=result.times.tolist(),
            y=series.tolist(),
            name=name,
            line=dict(dash="dot"),
        ))

    fig.update_layout(
        xaxis_title="Time (h)",
        yaxis_title="Concentration (g/L)",
        template="plotly_white",
        height=450,
        margin=dict(t=30, b=50, l=60, r=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return _fig_to_json(fig)


_OBJ_LABELS: dict[str, str] = {
    "mean_yield": "Mean Y_X/S (g/g)",
    "std_yield": "Std Y_X/S",
    "mean_mu_max": "Mean \u03bc_max (1/h)",
    "std_mu_max": "Std \u03bc_max",
}


def plot_heatmap_plotly(
    heatmap_df: pd.DataFrame,
    title: str = "pH x Ratio Heatmap",
) -> dict:
    """Plot a pH x ratio heatmap."""
    # Extract metric name from title if present
    colorbar_title = "Value"
    for key, label in _OBJ_LABELS.items():
        if key in title:
            colorbar_title = label
            break

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_df.values.tolist(),
        x=[str(c) for c in heatmap_df.columns],
        y=[str(r) for r in heatmap_df.index],
        colorscale="Viridis",
        colorbar=dict(title=colorbar_title),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="pH",
        yaxis_title="Ratio",
        template="plotly_white",
        height=450,
        margin=dict(t=50, b=50, l=60, r=30),
    )

    return _fig_to_json(fig)


def plot_pareto_plotly(
    pareto_df: pd.DataFrame,
    obj1: str = "mean_yield",
    obj2: str = "mean_mu_max",
) -> dict:
    """Plot a Pareto front scatter."""
    label1 = _OBJ_LABELS.get(obj1, obj1)
    label2 = _OBJ_LABELS.get(obj2, obj2)

    hover_text = []
    for _, row in pareto_df.iterrows():
        text = (
            f"Strain: {row.get('strain', 'N/A')}<br>"
            f"pH: {row.get('pH', 'N/A'):.2f}<br>"
            f"Ratio: {row.get('ratio', 'N/A'):.2f}<br>"
            f"{label1}: {row[obj1]:.4f}<br>"
            f"{label2}: {row[obj2]:.4f}"
        )
        hover_text.append(text)

    fig = go.Figure(data=go.Scatter(
        x=pareto_df[obj1].tolist(),
        y=pareto_df[obj2].tolist(),
        mode="markers",
        marker=dict(size=10, color="steelblue", line=dict(width=1, color="navy")),
        text=hover_text,
        hoverinfo="text",
    ))

    fig.update_layout(
        xaxis_title=label1,
        yaxis_title=label2,
        title="Pareto Front",
        template="plotly_white",
        height=450,
        margin=dict(t=50, b=50, l=60, r=30),
    )

    return _fig_to_json(fig)


def plot_mc_envelope_plotly(trajectories: list[SimulationResult]) -> dict | None:
    """Plot Monte Carlo trajectory envelope (median + 5th/95th percentile bands)."""
    if not trajectories:
        return None

    times = trajectories[0].times
    all_X = np.array([r.X for r in trajectories])

    sub_keys = list(trajectories[0].substrates.keys())
    all_subs = {}
    for key in sub_keys:
        all_subs[key] = np.array([r.substrates[key] for r in trajectories])

    fig = go.Figure()

    # Biomass envelope
    X_median = np.nanmedian(all_X, axis=0)
    X_p5 = np.nanpercentile(all_X, 5, axis=0)
    X_p95 = np.nanpercentile(all_X, 95, axis=0)

    fig.add_trace(go.Scatter(
        x=times.tolist(), y=X_p95.tolist(),
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=times.tolist(), y=X_p5.tolist(),
        mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(31,119,180,0.2)", showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=times.tolist(), y=X_median.tolist(),
        name="Biomass (median)", line=dict(color="rgb(31,119,180)", width=2.5),
    ))

    # Substrate envelopes
    colors = [
        ("rgb(255,127,14)", "rgba(255,127,14,0.15)"),
        ("rgb(44,160,44)", "rgba(44,160,44,0.15)"),
    ]
    for i, key in enumerate(sub_keys):
        arr = all_subs[key]
        med = np.nanmedian(arr, axis=0)
        p5 = np.nanpercentile(arr, 5, axis=0)
        p95 = np.nanpercentile(arr, 95, axis=0)
        c_line, c_fill = colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=times.tolist(), y=p95.tolist(),
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=times.tolist(), y=p5.tolist(),
            mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor=c_fill, showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=times.tolist(), y=med.tolist(),
            name=f"{key} (median)", line=dict(color=c_line, width=2, dash="dash"),
        ))

    fig.update_layout(
        xaxis_title="Time (h)",
        yaxis_title="Concentration (g/L)",
        title="Monte Carlo Trajectory Envelope (5th-95th percentile)",
        template="plotly_white",
        height=480,
        margin=dict(t=50, b=50, l=60, r=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return _fig_to_json(fig)


def plot_mc_distributions_plotly(mc_result: MonteCarloResult) -> dict:
    """Histogram of Monte Carlo yield and mu_max distributions."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Y_X/S Distribution", "\u03bc_max Distribution"))

    yields = mc_result.yields[np.isfinite(mc_result.yields)]
    mu_maxs = mc_result.mu_max_values[np.isfinite(mc_result.mu_max_values)]

    fig.add_trace(
        go.Histogram(x=yields.tolist(), nbinsx=30, marker_color="steelblue", name="Y_X/S"),
        row=1, col=1,
    )

    fig.add_trace(
        go.Histogram(x=mu_maxs.tolist(), nbinsx=30, marker_color="coral", name="\u03bc_max"),
        row=1, col=2,
    )

    ci_lo, ci_hi = mc_result.ci95_yield

    fig.add_annotation(
        text=(
            f"Mean: {mc_result.mean_yield:.4f} g/g<br>"
            f"Std: {mc_result.std_yield:.4f}<br>"
            f"95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]"
        ),
        xref="x1", yref="y1",
        x=float(np.mean(yields)) if len(yields) > 0 else 0,
        y=0, showarrow=False,
        bgcolor="white", bordercolor="steelblue",
        xanchor="center", yanchor="bottom",
    )

    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(t=50, b=50, l=50, r=30),
        showlegend=False,
    )
    fig.update_xaxes(title_text="Y_X/S (g/g)", row=1, col=1)
    fig.update_xaxes(title_text="\u03bc_max (1/h)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)

    return _fig_to_json(fig)

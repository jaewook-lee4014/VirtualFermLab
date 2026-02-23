"""Result export to CSV and JSON."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from virtualfermlab.simulator.integrator import SimulationResult


def export_results_csv(
    result: SimulationResult,
    path: str | Path,
) -> None:
    """Export a simulation result to CSV.

    Parameters
    ----------
    result : SimulationResult
    path : str or Path
    """
    data = {"time": result.times, "X": result.X}
    for name, series in result.substrates.items():
        data[name] = series
    for name, series in result.enzymes.items():
        data[name] = series
    data["totalOutput"] = result.total_output

    pd.DataFrame(data).to_csv(path, index=False)


def export_results_json(
    result: SimulationResult,
    path: str | Path,
) -> None:
    """Export a simulation result to JSON.

    Parameters
    ----------
    result : SimulationResult
    path : str or Path
    """
    data = {
        "config": {
            "n_substrates": result.config.n_substrates,
            "n_feeds": result.config.n_feeds,
            "growth_model": result.config.growth_model,
            "enzyme_mode": result.config.enzyme_mode,
            "use_cardinal_pH": result.config.use_cardinal_pH,
            "pH": result.config.pH,
        },
        "params": {
            k: float(v) if isinstance(v, (np.floating, float, int)) else v
            for k, v in result.params.items()
        },
        "summary": {
            "yield_biomass": result.yield_biomass,
            "mu_max_effective": result.mu_max_effective,
        },
        "n_timepoints": len(result.times),
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

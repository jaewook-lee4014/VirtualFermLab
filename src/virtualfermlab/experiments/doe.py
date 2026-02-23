"""Design of Experiments: condition generation.

Latin Hypercube Sampling for continuous factors, full factorial for discrete
factors, and a combined generator for virtual experiment designs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np


@dataclass
class ExperimentCondition:
    """A single experimental condition.

    Attributes
    ----------
    strain : str
        Strain identifier.
    substrate_A, substrate_B : str
        Substrate names (B can be ``None`` for single-substrate).
    ratio : float
        Fraction of substrate A (0–1).
    pH : float
        Culture pH.
    total_concentration : float
        Total substrate concentration (g/L).
    dilution_rate : float
        Dilution rate (1/h), 0 for batch.
    """

    strain: str
    substrate_A: str
    substrate_B: str | None = None
    ratio: float = 1.0
    pH: float = 6.0
    total_concentration: float = 30.0
    dilution_rate: float = 0.0


def latin_hypercube_design(
    continuous_ranges: dict[str, tuple[float, float]],
    n_samples: int,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Generate LHS samples.

    Parameters
    ----------
    continuous_ranges : dict
        ``{factor_name: (low, high)}``.
    n_samples : int
        Number of samples.
    seed : int
        Random seed.

    Returns
    -------
    list of dict
        Each dict maps factor names to sampled values.
    """
    rng = np.random.default_rng(seed)
    factors = list(continuous_ranges.keys())
    n_factors = len(factors)

    samples = []
    for j in range(n_factors):
        low, high = continuous_ranges[factors[j]]
        perm = rng.permutation(n_samples)
        col = low + (perm + rng.random(n_samples)) / n_samples * (high - low)
        samples.append(col)

    samples = np.column_stack(samples)
    return [dict(zip(factors, row)) for row in samples]


def full_factorial_design(
    discrete_levels: dict[str, list],
) -> list[dict]:
    """Generate full factorial design over discrete factors.

    Parameters
    ----------
    discrete_levels : dict
        ``{factor_name: [level1, level2, ...]}``.

    Returns
    -------
    list of dict
    """
    factors = list(discrete_levels.keys())
    values = [discrete_levels[f] for f in factors]
    return [dict(zip(factors, combo)) for combo in product(*values)]


def generate_conditions(
    strains: list[str],
    substrates: list[tuple[str, str | None]],
    pH_range: tuple[float, float] = (4.0, 7.0),
    ratio_range: tuple[float, float] = (0.0, 1.0),
    total_concentration: float = 30.0,
    n_continuous: int = 20,
    seed: int = 42,
) -> list[ExperimentCondition]:
    """Generate combined factorial × LHS condition set.

    Discrete factors (strain, substrate pair) are crossed with continuous
    factors (pH, ratio) via LHS.

    Parameters
    ----------
    strains : list of str
    substrates : list of (str, str | None)
        Substrate pairs.
    pH_range : tuple
        (min_pH, max_pH)
    ratio_range : tuple
        (min_ratio, max_ratio) for substrate A fraction.
    total_concentration : float
        Total substrate (g/L).
    n_continuous : int
        Number of LHS samples for (pH, ratio).
    seed : int
        Random seed.

    Returns
    -------
    list of ExperimentCondition
    """
    discrete = full_factorial_design({
        "strain": strains,
        "substrates": substrates,
    })

    continuous = latin_hypercube_design(
        {"pH": pH_range, "ratio": ratio_range},
        n_samples=n_continuous,
        seed=seed,
    )

    conditions = []
    for d in discrete:
        for c in continuous:
            sub_A, sub_B = d["substrates"]
            conditions.append(ExperimentCondition(
                strain=d["strain"],
                substrate_A=sub_A,
                substrate_B=sub_B,
                ratio=c["ratio"],
                pH=c["pH"],
                total_concentration=total_concentration,
            ))

    return conditions

"""Monte Carlo simulation runner.

Runs replicated simulations with sampled parameters for uncertainty
quantification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from virtualfermlab.experiments.doe import ExperimentCondition
from virtualfermlab.models.ode_systems import ModelConfig
from virtualfermlab.parameters.distributions import sample_params
from virtualfermlab.parameters.schema import StrainProfile
from virtualfermlab.simulator.integrator import simulate


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo runs for a single condition.

    Attributes
    ----------
    condition : ExperimentCondition
    yields : np.ndarray
        Final biomass for each replicate.
    mu_max_values : np.ndarray
        Effective mu_max for each replicate.
    substrate_conversions : np.ndarray
        Substrate conversion fractions for each replicate.
    """

    condition: ExperimentCondition
    yields: np.ndarray
    mu_max_values: np.ndarray
    substrate_conversions: np.ndarray

    @property
    def mean_yield(self) -> float:
        return float(np.nanmean(self.yields))

    @property
    def std_yield(self) -> float:
        return float(np.nanstd(self.yields))

    @property
    def mean_mu_max(self) -> float:
        return float(np.nanmean(self.mu_max_values))

    @property
    def std_mu_max(self) -> float:
        return float(np.nanstd(self.mu_max_values))

    @property
    def ci95_yield(self) -> tuple[float, float]:
        return (
            float(np.nanpercentile(self.yields, 2.5)),
            float(np.nanpercentile(self.yields, 97.5)),
        )


def _run_single(
    condition: ExperimentCondition,
    profile: StrainProfile,
    config: ModelConfig,
    times: np.ndarray,
    seed: int,
) -> dict:
    """Run one simulation replicate."""
    rng = np.random.default_rng(seed)
    sampled = sample_params(profile, rng)

    S_total = condition.total_concentration
    S1 = S_total * condition.ratio
    S2 = S_total * (1.0 - condition.ratio)

    params = {
        **sampled,
        "S1": S1,
        "S2": S2,
        "S_in1": S1,
        "S_in2": S2,
        "X0": 0.03,
        "y0": 0.0,
        "dilutionRate": condition.dilution_rate,
    }

    if config.enzyme_mode == "enzyme":
        params.setdefault("Z0", 0.01)
    elif config.enzyme_mode == "kompala":
        params.setdefault("Z1", 0.01)
        params.setdefault("Z2", 0.01)

    # Apply cardinal pH
    if config.use_cardinal_pH and condition.pH is not None:
        config_copy = ModelConfig(
            n_substrates=config.n_substrates,
            n_feeds=config.n_feeds,
            growth_model=config.growth_model,
            enzyme_mode=config.enzyme_mode,
            use_cardinal_pH=True,
            pH=condition.pH,
            pH_min=sampled.get("pH_min", config.pH_min),
            pH_opt=sampled.get("pH_opt", config.pH_opt),
            pH_max=sampled.get("pH_max", config.pH_max),
            use_lag=config.use_lag,
            lag=config.lag,
        )
    else:
        config_copy = config

    # Physical upper bound: biomass cannot exceed initial biomass + all
    # substrate converted at maximum theoretical yield (~1 g/g).
    max_plausible_biomass = params["X0"] + S_total * 2.0

    try:
        result = simulate(config_copy, params, times)
        yield_val = result.yield_biomass
        mu_val = result.mu_max_effective
        # Substrate conversion
        S1_final = result.substrates.get("S1", result.substrates.get("S", np.array([S1])))[-1]
        conversion = 1.0 - S1_final / S1 if S1 > 0 else 0.0

        # Hard reject: non-finite state or biomass exceeding physical bound
        if (
            np.any(~np.isfinite(result.X))
            or not np.isfinite(yield_val)
            or yield_val > max_plausible_biomass
        ):
            yield_val = np.nan
            mu_val = np.nan
            conversion = np.nan
        else:
            # Soft filter: mu_max_effective can be noisy from stiff
            # solver artifacts — keep yield but cap the mu value.
            if not np.isfinite(mu_val) or mu_val > 2.0:
                mu_val = np.nan
    except Exception:
        yield_val = np.nan
        mu_val = np.nan
        conversion = np.nan

    return {"yield": yield_val, "mu_max": mu_val, "conversion": conversion}


def run_monte_carlo(
    condition: ExperimentCondition,
    profile: StrainProfile,
    config: ModelConfig,
    times: np.ndarray,
    n_samples: int = 200,
    n_jobs: int = -1,
    seed: int = 42,
) -> MonteCarloResult:
    """Run Monte Carlo simulations for one condition.

    Parameters
    ----------
    condition : ExperimentCondition
    profile : StrainProfile
    config : ModelConfig
    times : array_like
    n_samples : int
        Number of Monte Carlo replicates.
    n_jobs : int
        Parallelism (``-1`` = all cores).
    seed : int
        Base random seed.

    Returns
    -------
    MonteCarloResult
    """
    times = np.asarray(times)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_single)(condition, profile, config, times, seed + i)
        for i in range(n_samples)
    )

    return MonteCarloResult(
        condition=condition,
        yields=np.array([r["yield"] for r in results]),
        mu_max_values=np.array([r["mu_max"] for r in results]),
        substrate_conversions=np.array([r["conversion"] for r in results]),
    )


class VirtualExperiment:
    """Orchestrate Monte Carlo runs across multiple conditions.

    Parameters
    ----------
    conditions : list of ExperimentCondition
    profile : StrainProfile
    config : ModelConfig
    times : array_like
    n_samples : int
    n_jobs : int
    seed : int
    """

    def __init__(
        self,
        conditions: list[ExperimentCondition],
        profile: StrainProfile,
        config: ModelConfig,
        times: np.ndarray,
        n_samples: int = 200,
        n_jobs: int = 1,
        seed: int = 42,
    ) -> None:
        self.conditions = conditions
        self.profile = profile
        self.config = config
        self.times = np.asarray(times)
        self.n_samples = n_samples
        self.n_jobs = n_jobs
        self.seed = seed

    def run_all(self) -> list[MonteCarloResult]:
        """Run Monte Carlo for every condition."""
        results = []
        for i, cond in enumerate(self.conditions):
            r = run_monte_carlo(
                cond,
                self.profile,
                self.config,
                self.times,
                n_samples=self.n_samples,
                n_jobs=self.n_jobs,
                seed=self.seed + i * self.n_samples,
            )
            results.append(r)
        return results

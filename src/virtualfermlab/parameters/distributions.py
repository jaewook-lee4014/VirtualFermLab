"""Monte-Carlo sampling from parameter distributions.

Draws random parameter values from the distributions defined in a
:class:`StrainProfile`.
"""

from __future__ import annotations

import numpy as np

from virtualfermlab.parameters.schema import DistributionSpec, StrainProfile


_EPS = 1e-12  # Floor to prevent zero-division (e.g. 1/Yx, S+Ks)


def sample_value(spec: DistributionSpec, rng: np.random.Generator) -> float:
    """Draw one sample from a :class:`DistributionSpec`.

    All kinetic/stoichiometric parameters are physically non-negative.
    Sampled values are clamped to a small positive floor to prevent
    negative draws (possible with normal/triangular tails) and
    zero-division in ODE terms like ``1/Yx`` or ``S/(S+Ks)``.

    Parameters
    ----------
    spec : DistributionSpec
    rng : numpy Generator

    Returns
    -------
    float
        Always >= _EPS.
    """
    if spec.type == "fixed":
        return max(spec.value, _EPS)
    elif spec.type == "normal":
        return max(float(rng.normal(spec.value, spec.std)), _EPS)
    elif spec.type == "uniform":
        return max(float(rng.uniform(spec.low, spec.high)), _EPS)
    elif spec.type == "lognormal":
        # Parameterised so that the *mean* is spec.value
        mu = np.log(spec.value**2 / np.sqrt(spec.value**2 + spec.std**2))
        sigma = np.sqrt(np.log(1 + (spec.std / spec.value) ** 2))
        return float(rng.lognormal(mu, sigma))  # always positive
    elif spec.type == "triangular":
        return max(float(rng.triangular(spec.low, spec.value, spec.high)), _EPS)
    else:
        raise ValueError(f"Unknown distribution type: {spec.type!r}")


def sample_params(
    profile: StrainProfile,
    rng: np.random.Generator,
) -> dict:
    """Sample a complete parameter set from a strain profile.

    Parameters
    ----------
    profile : StrainProfile
    rng : numpy Generator

    Returns
    -------
    dict
        Flat dictionary of sampled parameter values, keyed to match the
        notebook convention (e.g. ``mu_max1``, ``K_s1``, ``Yx1``).
    """
    params: dict = {}

    substrates = list(profile.substrates.values())
    for i, sub in enumerate(substrates, start=1):
        suffix = str(i) if len(substrates) > 1 else ""
        params[f"mu_max{suffix}"] = sample_value(sub.mu_max, rng)
        params[f"K_s{suffix}"] = sample_value(sub.Ks, rng)
        params[f"Yx{suffix}"] = sample_value(sub.Yxs, rng)

    for inh in profile.inhibitions:
        params["K_I"] = sample_value(inh.K_I, rng)

    if profile.enzyme_params is not None:
        ep = profile.enzyme_params
        params["K_Z_c"] = sample_value(ep.K_Z_c, rng)
        params["K_Z_S"] = sample_value(ep.K_Z_S, rng)
        params["K_Z_d"] = sample_value(ep.K_Z_d, rng)

    if profile.cardinal_pH is not None:
        cp = profile.cardinal_pH
        params["pH_min"] = sample_value(cp.pH_min, rng)
        params["pH_opt"] = sample_value(cp.pH_opt, rng)
        params["pH_max"] = sample_value(cp.pH_max, rng)

    return params

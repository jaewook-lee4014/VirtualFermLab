"""Tests for parameters/distributions.py."""

import numpy as np
import pytest

from virtualfermlab.parameters.distributions import sample_value, sample_params
from virtualfermlab.parameters.library import load_strain_profile
from virtualfermlab.parameters.schema import DistributionSpec


class TestSampleValue:
    def test_fixed(self, rng):
        spec = DistributionSpec(type="fixed", value=0.5)
        assert sample_value(spec, rng) == 0.5

    def test_normal(self, rng):
        spec = DistributionSpec(type="normal", value=10.0, std=1.0)
        vals = [sample_value(spec, rng) for _ in range(1000)]
        assert abs(np.mean(vals) - 10.0) < 0.2

    def test_uniform(self, rng):
        spec = DistributionSpec(type="uniform", value=0.5, low=0.0, high=1.0)
        vals = [sample_value(spec, rng) for _ in range(1000)]
        assert all(0.0 <= v <= 1.0 for v in vals)

    def test_lognormal(self, rng):
        spec = DistributionSpec(type="lognormal", value=1.0, std=0.2)
        vals = [sample_value(spec, rng) for _ in range(1000)]
        assert all(v > 0 for v in vals)

    def test_triangular(self, rng):
        spec = DistributionSpec(type="triangular", value=0.5, low=0.0, high=1.0)
        vals = [sample_value(spec, rng) for _ in range(1000)]
        assert all(0.0 <= v <= 1.0 for v in vals)


class TestSampleParams:
    def test_from_default_profile(self, rng):
        profile = load_strain_profile("F_venenatum_A35")
        params = sample_params(profile, rng)
        assert "mu_max1" in params
        assert "mu_max2" in params
        assert "K_s1" in params
        assert "Yx1" in params
        assert "K_I" in params
        assert "pH_min" in params
        assert "pH_opt" in params
        # Values should be positive (or at least finite)
        for k, v in params.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"

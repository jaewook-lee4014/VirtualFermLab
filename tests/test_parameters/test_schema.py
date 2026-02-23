"""Tests for parameters/schema.py."""

import pytest
from virtualfermlab.parameters.schema import (
    DistributionSpec,
    SubstrateParams,
    StrainProfile,
    CardinalPHParams,
)


class TestDistributionSpec:
    def test_fixed(self):
        d = DistributionSpec(type="fixed", value=0.1)
        assert d.value == 0.1
        assert d.type == "fixed"

    def test_normal(self):
        d = DistributionSpec(type="normal", value=0.1, std=0.02, confidence="B")
        assert d.std == 0.02
        assert d.confidence == "B"

    def test_uniform(self):
        d = DistributionSpec(type="uniform", value=0.5, low=0.1, high=0.9)
        assert d.low == 0.1
        assert d.high == 0.9


class TestSubstrateParams:
    def test_creation(self):
        sp = SubstrateParams(
            name="glucose",
            mu_max=DistributionSpec(type="fixed", value=0.1),
            Ks=DistributionSpec(type="fixed", value=0.2),
            Yxs=DistributionSpec(type="fixed", value=0.29),
        )
        assert sp.name == "glucose"


class TestStrainProfile:
    def test_minimal(self):
        sp = StrainProfile(name="test_strain")
        assert sp.name == "test_strain"
        assert sp.cardinal_pH is None
        assert len(sp.substrates) == 0

    def test_with_pH(self):
        sp = StrainProfile(
            name="test",
            cardinal_pH=CardinalPHParams(
                pH_min=DistributionSpec(type="fixed", value=3.5),
                pH_opt=DistributionSpec(type="fixed", value=6.0),
                pH_max=DistributionSpec(type="fixed", value=7.5),
            ),
        )
        assert sp.cardinal_pH.pH_opt.value == 6.0

"""Tests for models/enzyme_regulation.py."""

import pytest
from virtualfermlab.models.enzyme_regulation import (
    direct_inhibition,
    enzyme_induction_factor,
    kompala_matching_law,
    kompala_proportional_law,
)


class TestDirectInhibition:
    def test_no_inhibitor(self):
        assert direct_inhibition(0, 0.5) == pytest.approx(1.0)

    def test_equal_to_KI(self):
        assert direct_inhibition(0.5, 0.5) == pytest.approx(0.5)

    def test_high_inhibitor(self):
        assert direct_inhibition(100, 0.5) < 0.01


class TestEnzymeInductionFactor:
    def test_zero_enzyme(self):
        assert enzyme_induction_factor(0, 0.2) == pytest.approx(0.0)

    def test_high_enzyme(self):
        assert enzyme_induction_factor(100, 0.2) == pytest.approx(100 / 100.2, rel=1e-6)


class TestKompala:
    def test_matching_equal_rates(self):
        v1, v2 = kompala_matching_law(0.1, 0.1)
        assert v1 == pytest.approx(1.0)
        assert v2 == pytest.approx(1.0)

    def test_matching_unequal(self):
        v1, v2 = kompala_matching_law(0.2, 0.1)
        assert v1 == pytest.approx(1.0)
        assert v2 == pytest.approx(0.5)

    def test_matching_zero(self):
        v1, v2 = kompala_matching_law(0, 0)
        assert v1 == 0.0 and v2 == 0.0

    def test_proportional_equal(self):
        u1, u2 = kompala_proportional_law(0.1, 0.1)
        assert u1 == pytest.approx(0.5)
        assert u2 == pytest.approx(0.5)

    def test_proportional_sum_to_one(self):
        u1, u2 = kompala_proportional_law(0.3, 0.7)
        assert u1 + u2 == pytest.approx(1.0)

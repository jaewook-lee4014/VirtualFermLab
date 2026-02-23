"""Tests for models/kinetics.py."""

import pytest
from virtualfermlab.models.kinetics import monod_factor, contois_factor, substrate_factor


class TestMonodFactor:
    def test_basic(self):
        assert monod_factor(10, 5) == pytest.approx(10 / 15, rel=1e-10)

    def test_zero_substrate(self):
        assert monod_factor(0, 5) == 0.0

    def test_high_substrate(self):
        assert monod_factor(1000, 1) == pytest.approx(1000 / 1001, rel=1e-6)

    def test_equal_S_Ks(self):
        assert monod_factor(5, 5) == pytest.approx(0.5, rel=1e-10)


class TestContoisFactor:
    def test_basic(self):
        assert contois_factor(10, 5, 2) == pytest.approx(10 / (10 + 10), rel=1e-10)

    def test_reduces_to_monod_at_unit_X(self):
        assert contois_factor(10, 5, 1) == pytest.approx(monod_factor(10, 5))


class TestSubstrateFactor:
    def test_monod_dispatch(self):
        assert substrate_factor(10, 5, "Monod") == pytest.approx(monod_factor(10, 5))

    def test_contois_dispatch(self):
        assert substrate_factor(10, 5, "Contois", X=2) == pytest.approx(contois_factor(10, 5, 2))

    def test_invalid_model(self):
        with pytest.raises(ValueError, match="Unknown growth model"):
            substrate_factor(10, 5, "Invalid")

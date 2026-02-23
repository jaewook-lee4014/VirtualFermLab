"""Tests for models/ph_model.py."""

import pytest
from virtualfermlab.models.ph_model import cardinal_pH_factor, lag_switch


class TestCardinalPH:
    def test_at_optimum(self):
        assert cardinal_pH_factor(6.0, 3.5, 6.0, 7.5) == pytest.approx(1.0, abs=1e-10)

    def test_at_minimum(self):
        assert cardinal_pH_factor(3.5, 3.5, 6.0, 7.5) == 0.0

    def test_at_maximum(self):
        assert cardinal_pH_factor(7.5, 3.5, 6.0, 7.5) == 0.0

    def test_below_minimum(self):
        assert cardinal_pH_factor(2.0, 3.5, 6.0, 7.5) == 0.0

    def test_above_maximum(self):
        assert cardinal_pH_factor(9.0, 3.5, 6.0, 7.5) == 0.0

    def test_between_min_and_opt(self):
        val = cardinal_pH_factor(5.0, 3.5, 6.0, 7.5)
        assert 0.0 < val < 1.0

    def test_symmetric_around_opt(self):
        # The Cardinal pH model is NOT symmetric, but factor should be
        # between 0 and 1 on both sides
        val_low = cardinal_pH_factor(4.5, 3.5, 6.0, 7.5)
        val_high = cardinal_pH_factor(7.0, 3.5, 6.0, 7.5)
        assert 0.0 < val_low < 1.0
        assert 0.0 < val_high < 1.0


class TestLagSwitch:
    def test_during_lag(self):
        assert lag_switch(5.0, 10.0) == 0.0

    def test_after_lag(self):
        assert lag_switch(15.0, 10.0) == 1.0

    def test_at_lag(self):
        assert lag_switch(10.0, 10.0) == 1.0

    def test_zero_lag(self):
        assert lag_switch(0.0, 0.0) == 1.0

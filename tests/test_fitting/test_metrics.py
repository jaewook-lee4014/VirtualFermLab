"""Tests for fitting/metrics.py."""

import numpy as np
import pytest

from virtualfermlab.fitting.metrics import mae, rmse, r_squared, metrics


class TestMAE:
    def test_perfect(self):
        assert mae([1, 2, 3], [1, 2, 3]) == 0.0

    def test_known(self):
        assert mae([1, 2, 3], [2, 3, 4]) == pytest.approx(1.0)


class TestRMSE:
    def test_perfect(self):
        assert rmse([1, 2, 3], [1, 2, 3]) == 0.0

    def test_known(self):
        assert rmse([0, 0], [1, 1]) == pytest.approx(1.0)


class TestR2:
    def test_perfect(self):
        assert r_squared([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_mean_prediction(self):
        # Predicting mean gives R2 = 0
        y = [1, 2, 3]
        y_mean = [2, 2, 2]
        assert r_squared(y, y_mean) == pytest.approx(0.0)


class TestMetrics:
    def test_returns_all(self):
        m = metrics([1, 2, 3], [1.1, 2.1, 3.1])
        assert "MAE" in m
        assert "RMSE" in m
        assert "R2" in m

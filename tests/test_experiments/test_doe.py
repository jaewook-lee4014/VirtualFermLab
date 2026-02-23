"""Tests for experiments/doe.py."""

import pytest
from virtualfermlab.experiments.doe import (
    ExperimentCondition,
    latin_hypercube_design,
    full_factorial_design,
    generate_conditions,
)


class TestLatinHypercube:
    def test_n_samples(self):
        samples = latin_hypercube_design({"pH": (4.0, 7.0), "ratio": (0.0, 1.0)}, 10)
        assert len(samples) == 10

    def test_within_bounds(self):
        samples = latin_hypercube_design({"pH": (4.0, 7.0)}, 50)
        for s in samples:
            assert 4.0 <= s["pH"] <= 7.0


class TestFullFactorial:
    def test_count(self):
        designs = full_factorial_design({"A": [1, 2], "B": ["x", "y", "z"]})
        assert len(designs) == 6

    def test_all_combos(self):
        designs = full_factorial_design({"A": [1, 2], "B": [3, 4]})
        combos = [(d["A"], d["B"]) for d in designs]
        assert (1, 3) in combos
        assert (2, 4) in combos


class TestGenerateConditions:
    def test_generates_conditions(self):
        conditions = generate_conditions(
            strains=["A35"],
            substrates=[("glucose", "xylose")],
            n_continuous=5,
        )
        assert len(conditions) == 5
        assert all(isinstance(c, ExperimentCondition) for c in conditions)
        assert all(c.strain == "A35" for c in conditions)

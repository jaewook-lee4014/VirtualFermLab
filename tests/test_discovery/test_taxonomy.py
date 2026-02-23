"""Tests for the taxonomy similarity module."""

from __future__ import annotations

from virtualfermlab.discovery.taxonomy import lineage_distance


class TestLineageDistance:
    def test_identical(self):
        lin = ["Eukaryota", "Fungi", "Ascomycota", "Saccharomyces"]
        assert lineage_distance(lin, lin) == 0.0

    def test_completely_different(self):
        lin_a = ["Bacteria", "Firmicutes"]
        lin_b = ["Eukaryota", "Fungi"]
        assert lineage_distance(lin_a, lin_b) == 1.0

    def test_partial_overlap(self):
        lin_a = ["Eukaryota", "Fungi", "Ascomycota", "Saccharomyces"]
        lin_b = ["Eukaryota", "Fungi", "Ascomycota", "Pichia"]
        dist = lineage_distance(lin_a, lin_b)
        assert 0.0 < dist < 1.0
        # 3 shared out of max 4
        assert dist == pytest.approx(1.0 - 3.0 / 4.0)

    def test_empty_lineage(self):
        assert lineage_distance([], ["Eukaryota"]) == 1.0
        assert lineage_distance(["Eukaryota"], []) == 1.0

    def test_different_lengths(self):
        lin_a = ["Eukaryota", "Fungi"]
        lin_b = ["Eukaryota", "Fungi", "Ascomycota", "Saccharomyces"]
        dist = lineage_distance(lin_a, lin_b)
        # 2 shared, max len = 4
        assert dist == pytest.approx(1.0 - 2.0 / 4.0)


import pytest

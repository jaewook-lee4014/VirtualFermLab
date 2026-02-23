"""Tests for the organism name resolver."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from virtualfermlab.discovery import db
from virtualfermlab.discovery import name_resolver
from virtualfermlab.discovery.name_resolver import (
    _parse_abbreviated,
    resolve_name,
)


@pytest.fixture(autouse=True)
def _clean_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Use a temp DB and clear the in-memory cache for every test."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    name_resolver._resolve_cache.clear()


class TestParseAbbreviated:
    def test_standard(self):
        result = _parse_abbreviated("F_venenatum_A35")
        assert result is not None
        species_query, strain = result
        assert species_query == "F venenatum"
        assert strain == "A35"

    def test_with_dots(self):
        result = _parse_abbreviated("S.cerevisiae.S288C")
        assert result is not None
        species_query, strain = result
        assert species_query == "S cerevisiae"
        assert strain == "S288C"

    def test_with_hyphens(self):
        result = _parse_abbreviated("P-pastoris-GS115")
        assert result is not None
        species_query, strain = result
        assert species_query == "P pastoris"
        assert strain == "GS115"

    def test_no_strain(self):
        result = _parse_abbreviated("F_venenatum")
        assert result is not None
        species_query, strain = result
        assert species_query == "F venenatum"
        assert strain == ""

    def test_full_name_not_abbreviated(self):
        """Full binomial names should NOT match the abbreviated pattern."""
        result = _parse_abbreviated("Fusarium venenatum")
        assert result is None

    def test_random_string(self):
        assert _parse_abbreviated("hello world") is None
        assert _parse_abbreviated("123") is None


class TestResolveName:
    @mock.patch("virtualfermlab.discovery.name_resolver._resolve_abbreviated")
    def test_abbreviated_resolved(self, mock_resolve_abbrev):
        """Abbreviated names go through _resolve_abbreviated."""
        mock_resolve_abbrev.return_value = "Fusarium venenatum"
        result = resolve_name("F_venenatum_A35")
        assert result == "Fusarium venenatum"
        mock_resolve_abbrev.assert_called_once_with("F", "venenatum")

    @mock.patch("virtualfermlab.discovery.name_resolver._ena_suggest")
    @mock.patch("virtualfermlab.discovery.name_resolver._ena_scientific_name")
    def test_full_binomial_via_scientific_name(self, mock_sci, mock_suggest):
        mock_sci.return_value = "Saccharomyces cerevisiae"
        result = resolve_name("Saccharomyces cerevisiae")
        assert result == "Saccharomyces cerevisiae"
        mock_sci.assert_called_once()

    @mock.patch("virtualfermlab.discovery.name_resolver._ncbi_spell")
    @mock.patch("virtualfermlab.discovery.name_resolver._ncbi_taxonomy_search")
    @mock.patch("virtualfermlab.discovery.name_resolver._ena_suggest")
    @mock.patch("virtualfermlab.discovery.name_resolver._ena_scientific_name")
    def test_falls_through_to_ncbi_spell(self, mock_sci, mock_suggest, mock_ncbi, mock_spell):
        mock_sci.return_value = None
        mock_suggest.return_value = None
        mock_ncbi.return_value = None
        mock_spell.return_value = "Pichia pastoris"
        result = resolve_name("Picha pastris")
        assert result == "Pichia pastoris"

    @mock.patch("virtualfermlab.discovery.name_resolver._ncbi_spell")
    @mock.patch("virtualfermlab.discovery.name_resolver._ncbi_taxonomy_search")
    @mock.patch("virtualfermlab.discovery.name_resolver._ena_suggest")
    @mock.patch("virtualfermlab.discovery.name_resolver._ena_scientific_name")
    def test_returns_original_when_all_fail(self, mock_sci, mock_suggest, mock_ncbi, mock_spell):
        mock_sci.return_value = None
        mock_suggest.return_value = None
        mock_ncbi.return_value = None
        mock_spell.return_value = None
        result = resolve_name("Completely Unknown Organism XYZ")
        assert result == "Completely Unknown Organism XYZ"

    @mock.patch("virtualfermlab.discovery.name_resolver._resolve_abbreviated")
    def test_caches_result(self, mock_resolve_abbrev):
        mock_resolve_abbrev.return_value = "Fusarium venenatum"
        resolve_name("F_venenatum_A35")
        resolve_name("F_venenatum_A35")  # second call
        # Should only call resolve_abbreviated once — second call hits cache
        mock_resolve_abbrev.assert_called_once()

    def test_empty_string(self):
        assert resolve_name("") == ""
        assert resolve_name("  ") == ""

    @mock.patch("virtualfermlab.discovery.name_resolver._ncbi_taxonomy_search")
    @mock.patch("virtualfermlab.discovery.name_resolver._ena_suggest")
    @mock.patch("virtualfermlab.discovery.name_resolver._ena_scientific_name")
    def test_tier4_ncbi_taxonomy_search(self, mock_sci, mock_suggest, mock_ncbi):
        """NCBI taxonomy search (Tier 4) is used when ENA fails."""
        mock_sci.return_value = None
        mock_suggest.return_value = None
        mock_ncbi.return_value = "Komagataella pastoris"
        result = resolve_name("Pichia pastoris")
        assert result == "Komagataella pastoris"

    @mock.patch("virtualfermlab.discovery.name_resolver._resolve_abbreviated")
    def test_abbreviated_falls_through_when_resolve_fails(self, mock_resolve_abbrev):
        """When _resolve_abbreviated returns None, fall through to later tiers."""
        mock_resolve_abbrev.return_value = None
        with mock.patch("virtualfermlab.discovery.name_resolver._ena_scientific_name") as mock_sci:
            mock_sci.return_value = "Fusarium venenatum"
            result = resolve_name("F_venenatum_A35")
            assert result == "Fusarium venenatum"


class TestResolveAbbreviated:
    """Test _resolve_abbreviated directly with mocked NCBI/ENA calls."""

    @mock.patch("virtualfermlab.discovery.name_resolver._ncbi_taxonomy_search")
    def test_common_genera_hit(self, mock_ncbi):
        """Strategy 0: common genera table resolves known abbreviations."""
        mock_ncbi.return_value = "Fusarium venenatum"
        from virtualfermlab.discovery.name_resolver import _resolve_abbreviated
        result = _resolve_abbreviated("F", "venenatum")
        assert result == "Fusarium venenatum"
        # Should have been called with "Fusarium venenatum" (first common genus for F)
        mock_ncbi.assert_called_once_with("Fusarium venenatum")

    @mock.patch("virtualfermlab.discovery.name_resolver.requests")
    @mock.patch("virtualfermlab.discovery.name_resolver._ncbi_taxonomy_search")
    def test_falls_through_to_ena_strategies(self, mock_ncbi, mock_requests):
        """When common genera fail, ENA strategies are tried."""
        # NCBI returns None for all common genera, then returns a hit for
        # an ENA-discovered candidate
        from virtualfermlab.discovery.name_resolver import _resolve_abbreviated

        call_count = [0]

        def ncbi_side_effect(query):
            call_count[0] += 1
            # Return None for common genera attempts, then succeed for validation
            if query == "Xylaria venenatum":
                return "Xylaria venenatum"
            return None

        mock_ncbi.side_effect = ncbi_side_effect

        # Mock ENA responses
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"scientificName": "Xylaria venenatum"}
        ]
        mock_requests.get.return_value = mock_resp
        mock_requests.utils.quote = lambda s: s

        result = _resolve_abbreviated("X", "venenatum")
        # Should resolve via ENA strategy since no common genera for X
        assert result is not None

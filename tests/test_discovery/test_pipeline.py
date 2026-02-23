"""Tests for the discovery pipeline orchestrator."""

from __future__ import annotations

import queue
from pathlib import Path
from unittest import mock

import pytest

from virtualfermlab.discovery import db
from virtualfermlab.discovery.paper_search import _SENTINEL
from virtualfermlab.discovery.pipeline import DiscoveryResult, _build_profile, run_discovery
from virtualfermlab.parameters.schema import (
    DistributionSpec,
    StrainProfile,
    SubstrateParams,
)


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")


def _dummy_profile(name: str = "DummyStrain") -> StrainProfile:
    return StrainProfile(
        name=name,
        substrates={
            "glucose": SubstrateParams(
                name="glucose",
                mu_max=DistributionSpec(type="fixed", value=0.12),
                Ks=DistributionSpec(type="fixed", value=0.18),
                Yxs=DistributionSpec(type="fixed", value=0.32),
            ),
            "xylose": SubstrateParams(
                name="xylose",
                mu_max=DistributionSpec(type="fixed", value=0.06),
                Ks=DistributionSpec(type="fixed", value=0.15),
                Yxs=DistributionSpec(type="fixed", value=0.35),
            ),
        },
    )


class TestBuildProfile:
    def test_uses_extracted_params(self):
        extracted = [
            {"name": "mu_max", "value": 0.3, "substrate": "glucose"},
            {"name": "Ks", "value": 0.05, "substrate": "glucose"},
        ]
        profile = _build_profile("NewStrain", extracted, None, None)
        assert profile.name == "NewStrain"
        assert profile.substrates["glucose"].mu_max.value == 0.3
        assert profile.substrates["glucose"].Ks.value == 0.05
        # Yxs not extracted → biological default
        assert profile.substrates["glucose"].Yxs.value == 0.3

    def test_falls_back_to_similar_strain(self):
        similar = _dummy_profile()
        profile = _build_profile("Unknown", [], "DummyStrain", similar)
        # Should pick up values from similar strain
        assert profile.substrates["glucose"].mu_max.value == 0.12
        assert profile.substrates["glucose"].mu_max.confidence == "C"

    def test_biological_defaults_when_nothing(self):
        profile = _build_profile("Mystery", [], None, None)
        assert profile.substrates["glucose"].mu_max.value == 0.1
        assert profile.substrates["glucose"].Ks.value == 0.2
        assert profile.substrates["glucose"].Yxs.value == 0.3


def _mock_search_into_queue(return_count=0, papers=None):
    """Create a side_effect for search_papers_into_queue that pushes papers + sentinel."""
    if papers is None:
        papers = []

    def _side_effect(strain_name, paper_queue, **kwargs):
        for p in papers:
            paper_queue.put(p)
        paper_queue.put(_SENTINEL)
        return len(papers)

    return _side_effect


def _mock_extract_from_queue(return_params=None):
    """Create a side_effect for extract_from_queue that drains the queue."""
    if return_params is None:
        return_params = []

    def _side_effect(paper_queue, sentinel, client=None):
        while True:
            item = paper_queue.get()
            if item is sentinel:
                break
        return return_params

    return _side_effect


class TestRunDiscovery:
    @mock.patch("virtualfermlab.discovery.llm_extraction.fetch_full_text", return_value=None)
    @mock.patch("virtualfermlab.discovery.pipeline.search_papers_into_queue")
    @mock.patch("virtualfermlab.discovery.pipeline.extract_from_queue")
    @mock.patch("virtualfermlab.discovery.pipeline.find_most_similar", return_value=(None, 0.0))
    @mock.patch("virtualfermlab.discovery.pipeline.list_available_strains", return_value=[])
    def test_full_pipeline_with_no_results(self, mock_list, mock_sim, mock_extract, mock_search, mock_ft):
        """Pipeline should still produce a usable profile even when everything returns empty."""
        mock_search.side_effect = _mock_search_into_queue()
        mock_extract.side_effect = _mock_extract_from_queue()

        result = run_discovery("Unknown Microbe")
        assert isinstance(result, DiscoveryResult)
        assert result.profile is not None
        assert result.profile.name == "Unknown Microbe"
        assert result.source == "default_fallback"
        assert len(result.stages) == 5

    @mock.patch("virtualfermlab.discovery.llm_extraction.fetch_full_text", return_value=None)
    @mock.patch("virtualfermlab.discovery.pipeline.search_papers_into_queue")
    @mock.patch("virtualfermlab.discovery.pipeline.extract_from_queue")
    @mock.patch("virtualfermlab.discovery.pipeline.find_most_similar")
    @mock.patch("virtualfermlab.discovery.pipeline.list_available_strains")
    @mock.patch("virtualfermlab.discovery.pipeline.load_strain_profile")
    def test_pipeline_with_taxonomy_match(
        self, mock_load, mock_list, mock_sim, mock_extract, mock_search, mock_ft
    ):
        mock_search.side_effect = _mock_search_into_queue(
            papers=[{"title": "Paper", "abstract": "text", "doi": "10/a"}],
        )
        mock_extract.side_effect = _mock_extract_from_queue()
        mock_list.return_value = ["DummyStrain"]
        mock_sim.return_value = ("DummyStrain", 0.75)
        mock_load.return_value = _dummy_profile()

        result = run_discovery("Pichia pastoris")
        assert result.source == "taxonomy_match"
        assert result.similar_strain == "DummyStrain"
        assert result.profile.substrates["glucose"].mu_max.value == 0.12

    @mock.patch("virtualfermlab.discovery.llm_extraction.fetch_full_text", return_value=None)
    @mock.patch("virtualfermlab.discovery.pipeline.search_papers_into_queue")
    @mock.patch("virtualfermlab.discovery.pipeline.extract_from_queue")
    @mock.patch("virtualfermlab.discovery.pipeline.find_most_similar", return_value=(None, 0.0))
    @mock.patch("virtualfermlab.discovery.pipeline.list_available_strains", return_value=[])
    def test_pipeline_with_literature(self, mock_list, mock_sim, mock_extract, mock_search, mock_ft):
        mock_search.side_effect = _mock_search_into_queue(
            papers=[{"title": "P", "abstract": "t"}],
        )
        mock_extract.side_effect = _mock_extract_from_queue(
            return_params=[{"name": "mu_max", "value": 0.45, "substrate": "methanol"}],
        )

        result = run_discovery("Pichia pastoris")
        assert result.source == "literature"
        assert result.params_extracted == 1
        assert "methanol" in result.profile.substrates

    def test_progress_callback_called(self):
        stages_seen = []

        def cb(info):
            stages_seen.append(info["stage"])

        with mock.patch("virtualfermlab.discovery.llm_extraction.fetch_full_text", return_value=None), \
             mock.patch("virtualfermlab.discovery.pipeline.search_papers_into_queue",
                        side_effect=_mock_search_into_queue()), \
             mock.patch("virtualfermlab.discovery.pipeline.extract_from_queue",
                        side_effect=_mock_extract_from_queue()), \
             mock.patch("virtualfermlab.discovery.pipeline.find_most_similar", return_value=(None, 0.0)), \
             mock.patch("virtualfermlab.discovery.pipeline.list_available_strains", return_value=[]):
            run_discovery("Test", progress_cb=cb)

        assert "search" in stages_seen
        assert "build_profile" in stages_seen

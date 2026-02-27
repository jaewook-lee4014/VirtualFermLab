"""Tests for the discovery SQLite database layer."""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from unittest import mock

import pytest

from virtualfermlab.discovery import db
from virtualfermlab.parameters.schema import (
    DistributionSpec,
    StrainProfile,
    SubstrateParams,
)


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect DB_PATH to a temp file for every test."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test_strains.db")


class TestPapers:
    def test_save_and_deduplicate(self):
        paper = {
            "doi": "10.1234/test",
            "pmid": "12345",
            "title": "Test Paper",
            "authors": "Smith J",
            "journal": "J Test",
            "year": 2024,
            "abstract": "Abstract text.",
            "source": "pubmed",
        }
        pid1 = db.save_paper(paper)
        pid2 = db.save_paper(paper)  # same DOI → same id
        assert pid1 == pid2

    def test_save_paper_no_doi(self):
        paper = {"title": "No DOI Paper", "source": "semantic_scholar"}
        pid = db.save_paper(paper)
        assert isinstance(pid, int)


class TestExtractedParams:
    def test_save_and_query(self):
        pid = db.save_paper({"doi": "10.9999/x", "title": "T", "source": "pubmed"})
        params = [
            {
                "strain_name": "Pichia pastoris",
                "substrate": "glucose",
                "name": "mu_max",
                "value": 0.25,
                "unit": "1/h",
                "conditions": {"pH": 5.5, "temperature": 30},
                "confidence": "B",
                "kinetic_model": "Monod",
            },
            {
                "strain_name": "Pichia pastoris",
                "substrate": "glucose",
                "name": "Ks",
                "value": 0.1,
                "unit": "g/L",
                "kinetic_model": "Monod",
            },
        ]
        db.save_extracted_params(pid, params)
        rows = db.get_params_for_strain("pichia")
        assert len(rows) == 2
        names = {r["parameter_name"] for r in rows}
        assert names == {"mu_max", "Ks"}

    def test_save_and_query_with_kinetic_model(self):
        pid = db.save_paper({"doi": "10.9999/km", "title": "T", "source": "pubmed"})
        params = [
            {
                "strain_name": "F. venenatum",
                "substrate": "glucose",
                "name": "mu_max",
                "value": 0.18,
                "unit": "1/h",
                "conditions": {
                    "pH": 6.0, "temperature": 28,
                    "fermentation_mode": "batch",
                    "substrate_type": "glucose",
                    "initial_substrate_conc_g_L": 20,
                    "reactor_type": "bioreactor",
                    "fermentation_duration_h": 120,
                    "final_biomass_g_L": 6.8,
                },
                "kinetic_model": "Monod",
                "confidence": "A",
                "evidence": "mu_max = 0.18 h-1",
            },
        ]
        db.save_extracted_params(pid, params)
        rows = db.get_params_for_strain("F. venenatum")
        assert len(rows) == 1
        assert rows[0]["kinetic_model"] == "Monod"

    def test_save_and_query_with_evidence(self):
        pid = db.save_paper({"doi": "10.9999/ev", "title": "T", "source": "pubmed"})
        params = [
            {
                "strain_name": "E. coli",
                "substrate": "glucose",
                "name": "mu_max",
                "value": 0.6,
                "unit": "1/h",
                "conditions": {"pH": 7.0, "temperature": 37},
                "evidence": "The mu_max was 0.6 h-1 on glucose",
                "confidence": "A",
            },
        ]
        db.save_extracted_params(pid, params)
        rows = db.get_params_for_strain("E. coli")
        assert len(rows) == 1
        assert rows[0]["evidence"] == "The mu_max was 0.6 h-1 on glucose"
        assert rows[0]["confidence"] == "A"

    def test_save_without_evidence_defaults(self):
        pid = db.save_paper({"doi": "10.9999/noev", "title": "T2", "source": "pubmed"})
        params = [
            {
                "strain_name": "Yeast",
                "name": "Ks",
                "value": 0.2,
                "unit": "g/L",
            },
        ]
        db.save_extracted_params(pid, params)
        rows = db.get_params_for_strain("yeast")
        assert len(rows) == 1
        assert rows[0]["evidence"] is None
        assert rows[0]["confidence"] == "B"
        assert rows[0]["kinetic_model"] is None


class TestTaxonomy:
    def test_save_and_retrieve(self):
        lineage = ["Eukaryota", "Fungi", "Ascomycota", "Pichia pastoris"]
        db.save_taxonomy(4922, "Pichia pastoris", lineage, "species")
        cached = db.get_taxonomy("Pichia pastoris")
        assert cached is not None
        assert cached["lineage"] == lineage

    def test_cache_miss(self):
        assert db.get_taxonomy("Nonexistent organism") is None


class TestStrainProfileCache:
    def _make_profile(self, name: str = "test_strain") -> StrainProfile:
        return StrainProfile(
            name=name,
            substrates={
                "glucose": SubstrateParams(
                    name="glucose",
                    mu_max=DistributionSpec(type="fixed", value=0.1),
                    Ks=DistributionSpec(type="fixed", value=0.2),
                    Yxs=DistributionSpec(type="fixed", value=0.3),
                )
            },
        )

    def test_round_trip(self):
        profile = self._make_profile()
        db.save_strain_profile_cache("test_strain", profile, "literature", "F_venenatum_A35", 0.75)
        loaded = db.load_strain_profile_cache("test_strain")
        assert loaded is not None
        assert loaded.name == "test_strain"
        assert loaded.substrates["glucose"].mu_max.value == 0.1

    def test_list_cached(self):
        profile = self._make_profile("strain_a")
        db.save_strain_profile_cache("strain_a", profile, "taxonomy_match", None, None)
        names = db.list_cached_strains()
        assert "strain_a" in names

    def test_cache_miss(self):
        assert db.load_strain_profile_cache("nonexistent") is None


class TestSaveExtractedParamsDedup:
    def test_duplicate_params_not_inserted_twice(self):
        pid = db.save_paper({"doi": "10.1234/dup1", "title": "T", "source": "pubmed"})
        params = [
            {"strain_name": "Org", "name": "mu_max", "value": 0.23, "unit": "1/h", "substrate": "glycerol"},
        ]
        db.save_extracted_params(pid, params)
        db.save_extracted_params(pid, params)  # same params again
        rows = db.get_params_for_strain("Org")
        assert len(rows) == 1

    def test_duplicate_updates_confidence_when_better(self):
        pid = db.save_paper({"doi": "10.1234/dup2", "title": "T", "source": "pubmed"})
        db.save_extracted_params(pid, [
            {"strain_name": "Org", "name": "mu_max", "value": 0.3, "unit": "1/h", "confidence": "B"},
        ])
        db.save_extracted_params(pid, [
            {"strain_name": "Org", "name": "mu_max", "value": 0.3, "unit": "1/h",
             "evidence": "mu_max = 0.3", "confidence": "A"},
        ])
        rows = db.get_params_for_strain("Org")
        assert len(rows) == 1
        assert rows[0]["confidence"] == "A"
        assert rows[0]["evidence"] == "mu_max = 0.3"

    def test_different_values_are_not_deduped(self):
        pid = db.save_paper({"doi": "10.1234/dup3", "title": "T", "source": "pubmed"})
        db.save_extracted_params(pid, [
            {"strain_name": "Org", "name": "mu_max", "value": 0.23, "unit": "1/h", "substrate": "glycerol"},
            {"strain_name": "Org", "name": "mu_max", "value": 0.45, "unit": "1/h", "substrate": "glucose"},
        ])
        rows = db.get_params_for_strain("Org")
        assert len(rows) == 2

    def test_paper_has_params(self):
        pid = db.save_paper({"doi": "10.1234/hp", "title": "T", "source": "pubmed"})
        assert db.paper_has_params(pid) is False
        db.save_extracted_params(pid, [
            {"strain_name": "Org", "name": "Ks", "value": 0.1, "unit": "g/L"},
        ])
        assert db.paper_has_params(pid) is True


class TestGetPapersWithParams:
    def test_returns_papers_with_params(self):
        pid = db.save_paper({
            "doi": "10.1234/pwp1",
            "title": "Growth of E. coli on glucose",
            "authors": "Smith J, Doe A",
            "journal": "J Microbiol",
            "year": 2023,
            "source": "pubmed",
        })
        db.save_extracted_params(pid, [
            {
                "strain_name": "E. coli",
                "substrate": "glucose",
                "name": "mu_max",
                "value": 0.6,
                "unit": "1/h",
                "evidence": "mu_max was 0.6 h-1",
                "confidence": "A",
                "kinetic_model": "Monod",
                "conditions": {"temperature": 37, "pH": 7.0, "fermentation_mode": "batch"},
            },
            {
                "strain_name": "E. coli",
                "substrate": "glucose",
                "name": "Ks",
                "value": 0.1,
                "unit": "g/L",
                "confidence": "B",
                "kinetic_model": "Monod",
            },
        ])
        result = db.get_papers_with_params("E. coli")
        assert len(result) == 1
        paper = result[0]
        assert paper["title"] == "Growth of E. coli on glucose"
        assert paper["doi"] == "10.1234/pwp1"
        assert paper["journal"] == "J Microbiol"
        assert paper["year"] == 2023
        assert len(paper["params"]) == 2
        names = {p["name"] for p in paper["params"]}
        assert names == {"mu_max", "Ks"}
        # Check new fields
        mu_param = [p for p in paper["params"] if p["name"] == "mu_max"][0]
        assert mu_param["kinetic_model"] == "Monod"
        assert mu_param["conditions"]["temperature"] == 37
        assert mu_param["conditions"]["pH"] == 7.0

    def test_case_insensitive_match(self):
        pid = db.save_paper({"doi": "10.1234/pwp2", "title": "T", "source": "pubmed"})
        db.save_extracted_params(pid, [
            {"strain_name": "Pichia Pastoris", "name": "mu_max", "value": 0.25, "unit": "1/h"},
        ])
        result = db.get_papers_with_params("pichia pastoris")
        assert len(result) == 1
        assert len(result[0]["params"]) == 1

    def test_multiple_papers(self):
        pid1 = db.save_paper({"doi": "10.1234/mp1", "title": "Paper 1", "year": 2022, "source": "pubmed"})
        pid2 = db.save_paper({"doi": "10.1234/mp2", "title": "Paper 2", "year": 2024, "source": "pubmed"})
        db.save_extracted_params(pid1, [
            {"strain_name": "Yeast X", "name": "mu_max", "value": 0.3, "unit": "1/h"},
        ])
        db.save_extracted_params(pid2, [
            {"strain_name": "Yeast X", "name": "Ks", "value": 0.05, "unit": "g/L"},
            {"strain_name": "Yeast X", "name": "Yxs", "value": 0.4, "unit": "g/g"},
        ])
        result = db.get_papers_with_params("Yeast X")
        assert len(result) == 2
        # Ordered by year DESC
        assert result[0]["year"] == 2024
        assert result[1]["year"] == 2022

    def test_no_matching_strain_returns_empty(self):
        pid = db.save_paper({"doi": "10.1234/nope", "title": "T", "source": "pubmed"})
        db.save_extracted_params(pid, [
            {"strain_name": "Other Organism", "name": "mu_max", "value": 0.1, "unit": "1/h"},
        ])
        result = db.get_papers_with_params("Nonexistent Strain")
        assert result == []

    def test_evidence_and_confidence_included(self):
        pid = db.save_paper({"doi": "10.1234/ec", "title": "T", "source": "pubmed"})
        db.save_extracted_params(pid, [
            {
                "strain_name": "TestOrg",
                "name": "mu_max",
                "value": 0.5,
                "unit": "1/h",
                "evidence": "Table 2 reports mu_max = 0.5",
                "confidence": "A",
            },
        ])
        result = db.get_papers_with_params("TestOrg")
        param = result[0]["params"][0]
        assert param["evidence"] == "Table 2 reports mu_max = 0.5"
        assert param["confidence"] == "A"


class TestThreadSafety:
    def test_concurrent_save_paper(self):
        """Multiple threads saving papers concurrently should not raise."""
        errors: list[Exception] = []

        def _save(i: int) -> None:
            try:
                db.save_paper({
                    "doi": f"10.9999/thread-{i}",
                    "title": f"Thread Paper {i}",
                    "source": "test",
                })
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_save, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors during concurrent writes: {errors}"
        # Verify all papers were saved
        db.init_db()
        conn = db._connect()
        try:
            count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            assert count == 20
        finally:
            conn.close()

    def test_concurrent_save_extracted_params(self):
        """Concurrent extracted_params writes should not corrupt the DB."""
        pid = db.save_paper({"doi": "10.9999/base", "title": "Base", "source": "test"})
        errors: list[Exception] = []

        def _save_params(i: int) -> None:
            try:
                db.save_extracted_params(pid, [{
                    "strain_name": f"Strain_{i}",
                    "name": "mu_max",
                    "value": 0.1 + i * 0.01,
                    "unit": "1/h",
                }])
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_save_params, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors during concurrent writes: {errors}"

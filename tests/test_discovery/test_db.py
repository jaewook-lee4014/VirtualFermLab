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
                "conditions": {"pH": 5.5},
                "confidence": "B",
            },
            {
                "strain_name": "Pichia pastoris",
                "substrate": "glucose",
                "name": "Ks",
                "value": 0.1,
                "unit": "g/L",
            },
        ]
        db.save_extracted_params(pid, params)
        rows = db.get_params_for_strain("pichia")
        assert len(rows) == 2
        names = {r["parameter_name"] for r in rows}
        assert names == {"mu_max", "Ks"}


    def test_save_and_query_with_evidence(self):
        pid = db.save_paper({"doi": "10.9999/ev", "title": "T", "source": "pubmed"})
        params = [
            {
                "strain_name": "E. coli",
                "substrate": "glucose",
                "name": "mu_max",
                "value": 0.6,
                "unit": "1/h",
                "conditions": {"pH": 7.0},
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

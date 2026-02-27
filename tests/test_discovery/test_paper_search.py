"""Tests for paper_search with mocked HTTP."""

from __future__ import annotations

import json
import queue
from pathlib import Path
from unittest import mock

import pytest

from virtualfermlab.discovery import db
from virtualfermlab.discovery.paper_search import (
    _SENTINEL,
    _parse_jats_sections,
    fetch_full_text,
    search_papers,
    search_papers_into_queue,
    search_pubmed,
    search_semantic_scholar,
)


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")


_PUBMED_ESEARCH_RESPONSE = json.dumps({
    "esearchresult": {"idlist": ["12345"]}
})

_PUBMED_EFETCH_XML = """\
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345</PMID>
      <Article>
        <ArticleTitle>Fermentation kinetics of Pichia</ArticleTitle>
        <Journal>
          <Title>J Biotech</Title>
          <JournalIssue><PubDate><Year>2023</Year></PubDate></JournalIssue>
        </Journal>
        <Abstract><AbstractText>We studied Pichia pastoris growth.</AbstractText></Abstract>
        <AuthorList>
          <Author><LastName>Smith</LastName><ForeName>John</ForeName></Author>
        </AuthorList>
        <ELocationID EIdType="doi">10.1234/pichia</ELocationID>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">12345</ArticleId>
        <ArticleId IdType="pmc">PMC9999999</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""


class TestSearchPubmed:
    @mock.patch("virtualfermlab.discovery.paper_search.requests.get")
    @mock.patch("virtualfermlab.discovery.paper_search.time.sleep")
    def test_returns_papers(self, mock_sleep, mock_get):
        resp_search = mock.Mock()
        resp_search.status_code = 200
        resp_search.raise_for_status = mock.Mock()
        resp_search.json.return_value = {"esearchresult": {"idlist": ["12345"]}}

        resp_fetch = mock.Mock()
        resp_fetch.status_code = 200
        resp_fetch.raise_for_status = mock.Mock()
        resp_fetch.text = _PUBMED_EFETCH_XML

        mock_get.side_effect = [resp_search, resp_fetch]

        papers = search_pubmed("Pichia pastoris fermentation", max_results=5)
        assert len(papers) == 1
        assert papers[0]["pmid"] == "12345"
        assert papers[0]["doi"] == "10.1234/pichia"

    @mock.patch("virtualfermlab.discovery.paper_search.requests.get")
    @mock.patch("virtualfermlab.discovery.paper_search.time.sleep")
    def test_extracts_pmcid(self, mock_sleep, mock_get):
        resp_search = mock.Mock()
        resp_search.status_code = 200
        resp_search.raise_for_status = mock.Mock()
        resp_search.json.return_value = {"esearchresult": {"idlist": ["12345"]}}

        resp_fetch = mock.Mock()
        resp_fetch.status_code = 200
        resp_fetch.raise_for_status = mock.Mock()
        resp_fetch.text = _PUBMED_EFETCH_XML

        mock_get.side_effect = [resp_search, resp_fetch]

        papers = search_pubmed("Pichia", max_results=5)
        assert papers[0]["pmcid"] == "PMC9999999"

    @mock.patch("virtualfermlab.discovery.paper_search.requests.get")
    def test_handles_network_error(self, mock_get):
        mock_get.side_effect = ConnectionError("no network")
        papers = search_pubmed("anything")
        assert papers == []


class TestSearchSemanticScholar:
    @mock.patch("virtualfermlab.discovery.paper_search.requests.get")
    def test_returns_papers(self, mock_get):
        resp = mock.Mock()
        resp.status_code = 200
        resp.raise_for_status = mock.Mock()
        resp.json.return_value = {
            "data": [
                {
                    "title": "Growth of Pichia",
                    "authors": [{"name": "J Smith"}],
                    "year": 2023,
                    "abstract": "We studied growth.",
                    "externalIds": {"DOI": "10.5678/pichia", "PubMed": "67890"},
                    "journal": {"name": "Biotech J"},
                }
            ]
        }
        mock_get.return_value = resp

        papers = search_semantic_scholar("Pichia", max_results=5)
        assert len(papers) == 1
        assert papers[0]["doi"] == "10.5678/pichia"


class TestSearchPapers:
    @mock.patch("virtualfermlab.discovery.paper_search.search_openalex", return_value=[])
    @mock.patch("virtualfermlab.discovery.paper_search.search_europe_pmc", return_value=[])
    @mock.patch("virtualfermlab.discovery.paper_search.search_semantic_scholar")
    @mock.patch("virtualfermlab.discovery.paper_search.search_pubmed")
    def test_deduplicates_by_doi(self, mock_pm, mock_s2, mock_epmc, mock_oa):
        paper_common = {
            "pmid": "111",
            "title": "Paper A",
            "authors": "",
            "journal": "",
            "year": 2023,
            "abstract": "text",
            "doi": "10.1/shared",
            "source": "pubmed",
        }
        mock_pm.return_value = [paper_common]
        mock_s2.return_value = [{**paper_common, "source": "semantic_scholar"}]

        papers = search_papers("Pichia pastoris")
        assert len(papers) == 1  # deduplicated


class TestSearchPapersIntoQueue:
    @mock.patch("virtualfermlab.discovery.paper_search.search_openalex", return_value=[])
    @mock.patch("virtualfermlab.discovery.paper_search.search_europe_pmc", return_value=[])
    @mock.patch("virtualfermlab.discovery.paper_search.search_semantic_scholar")
    @mock.patch("virtualfermlab.discovery.paper_search.search_pubmed")
    def test_pushes_papers_and_sentinel(self, mock_pm, mock_s2, mock_epmc, mock_oa):
        paper_a = {
            "pmid": "1", "title": "Paper A", "authors": "", "journal": "",
            "year": 2023, "abstract": "text", "doi": "10.1/a", "source": "pubmed",
        }
        paper_b = {
            "pmid": "2", "title": "Paper B", "authors": "", "journal": "",
            "year": 2023, "abstract": "text", "doi": "10.2/b", "source": "semantic_scholar",
        }
        mock_pm.return_value = [paper_a]
        mock_s2.return_value = [paper_b]

        q: queue.Queue = queue.Queue()
        count = search_papers_into_queue("Pichia pastoris", q)

        assert count == 2
        items = []
        while True:
            item = q.get_nowait()
            if item is _SENTINEL:
                break
            items.append(item)
        assert len(items) == 2

    @mock.patch("virtualfermlab.discovery.paper_search.search_openalex", return_value=[])
    @mock.patch("virtualfermlab.discovery.paper_search.search_europe_pmc", return_value=[])
    @mock.patch("virtualfermlab.discovery.paper_search.search_semantic_scholar")
    @mock.patch("virtualfermlab.discovery.paper_search.search_pubmed")
    def test_deduplicates_by_doi(self, mock_pm, mock_s2, mock_epmc, mock_oa):
        paper = {
            "pmid": "1", "title": "Same", "authors": "", "journal": "",
            "year": 2023, "abstract": "text", "doi": "10.1/same", "source": "pubmed",
        }
        mock_pm.return_value = [paper]
        mock_s2.return_value = [{**paper, "source": "semantic_scholar"}]

        q: queue.Queue = queue.Queue()
        count = search_papers_into_queue("Test", q)

        assert count == 1

    @mock.patch("virtualfermlab.discovery.paper_search.search_openalex", return_value=[])
    @mock.patch("virtualfermlab.discovery.paper_search.search_europe_pmc", return_value=[])
    @mock.patch("virtualfermlab.discovery.paper_search.search_semantic_scholar")
    @mock.patch("virtualfermlab.discovery.paper_search.search_pubmed")
    def test_sentinel_sent_even_when_no_papers(self, mock_pm, mock_s2, mock_epmc, mock_oa):
        mock_pm.return_value = []
        mock_s2.return_value = []

        q: queue.Queue = queue.Queue()
        count = search_papers_into_queue("Empty", q)

        assert count == 0
        assert q.get_nowait() is _SENTINEL


# --------------------------------------------------------------------------
# Full-text fetch tests
# --------------------------------------------------------------------------

_JATS_XML = """\
<article>
  <body>
    <sec sec-type="results">
      <title>Results</title>
      <p>The maximum specific growth rate was 0.25 h-1.</p>
      <p>Biomass yield was 0.45 g/g on glucose.</p>
    </sec>
    <sec sec-type="discussion">
      <title>Discussion</title>
      <p>These values are consistent with literature.</p>
    </sec>
    <sec sec-type="methods">
      <title>Methods</title>
      <p>This section should not appear in the output.</p>
    </sec>
  </body>
  <table-wrap>
    <label>Table 1</label>
    <caption><p>Kinetic parameters</p></caption>
    <table>
      <tr><th>Parameter</th><th>Value</th></tr>
      <tr><td>mu_max</td><td>0.25</td></tr>
      <tr><td>Ks</td><td>0.15</td></tr>
    </table>
  </table-wrap>
</article>
"""


class TestParseJatsSections:
    def test_extracts_results_and_discussion(self):
        text = _parse_jats_sections(_JATS_XML)
        assert "Results" in text
        assert "0.25 h-1" in text
        assert "Discussion" in text
        assert "consistent with literature" in text

    def test_excludes_methods(self):
        text = _parse_jats_sections(_JATS_XML)
        assert "Methods" not in text
        assert "should not appear" not in text

    def test_extracts_tables(self):
        text = _parse_jats_sections(_JATS_XML)
        assert "Table 1" in text
        assert "mu_max" in text
        assert "0.15" in text

    def test_returns_empty_for_no_relevant_sections(self):
        xml = "<article><body><sec sec-type='methods'><title>Methods</title><p>foo</p></sec></body></article>"
        assert _parse_jats_sections(xml) == ""

    def test_truncation(self):
        from virtualfermlab.discovery.paper_search import _FULLTEXT_MAX_CHARS
        # Build XML with a very long results section
        long_text = "A" * (_FULLTEXT_MAX_CHARS + 1000)
        xml = f"<article><body><sec sec-type='results'><title>Results</title><p>{long_text}</p></sec></body></article>"
        text = _parse_jats_sections(xml)
        assert text.endswith("[truncated]")
        assert len(text) <= _FULLTEXT_MAX_CHARS + 50  # allow for the truncation marker

    def test_results_by_title_keyword(self):
        """Sections identified by title text (not sec-type attribute)."""
        xml = '<article><body><sec><title>3. Results and Discussion</title><p>mu_max was 0.3</p></sec></body></article>'
        text = _parse_jats_sections(xml)
        assert "mu_max was 0.3" in text


class TestFetchFullText:
    @mock.patch("virtualfermlab.discovery.paper_search.requests.get")
    @mock.patch("virtualfermlab.discovery.paper_search.time.sleep")
    def test_fetches_with_existing_pmcid(self, mock_sleep, mock_get):
        """When paper already has pmcid, skip ELink and go straight to EFetch."""
        resp_efetch = mock.Mock()
        resp_efetch.status_code = 200
        resp_efetch.raise_for_status = mock.Mock()
        resp_efetch.text = _JATS_XML
        mock_get.return_value = resp_efetch

        paper = {"pmid": "12345", "pmcid": "PMC9999999", "title": "Test"}
        result = fetch_full_text(paper)

        assert result is not None
        assert "0.25 h-1" in result
        # Should only call EFetch, not ELink
        assert mock_get.call_count == 1

    @mock.patch("virtualfermlab.discovery.paper_search.requests.get")
    @mock.patch("virtualfermlab.discovery.paper_search.time.sleep")
    def test_converts_pmid_to_pmcid(self, mock_sleep, mock_get):
        """When no pmcid, use ELink to convert PMID → PMCID, then EFetch."""
        resp_elink = mock.Mock()
        resp_elink.status_code = 200
        resp_elink.raise_for_status = mock.Mock()
        resp_elink.json.return_value = {
            "linksets": [{
                "linksetdbs": [{
                    "dbto": "pmc",
                    "links": ["9999999"],
                }]
            }]
        }

        resp_efetch = mock.Mock()
        resp_efetch.status_code = 200
        resp_efetch.raise_for_status = mock.Mock()
        resp_efetch.text = _JATS_XML

        mock_get.side_effect = [resp_elink, resp_efetch]

        paper = {"pmid": "12345", "title": "Test"}
        result = fetch_full_text(paper)

        assert result is not None
        assert "0.25 h-1" in result
        assert mock_get.call_count == 2

    def test_returns_none_without_pmid_or_pmcid(self):
        paper = {"title": "No IDs", "doi": "10.1/x"}
        assert fetch_full_text(paper) is None

    @mock.patch("virtualfermlab.discovery.paper_search.requests.get")
    @mock.patch("virtualfermlab.discovery.paper_search.time.sleep")
    def test_returns_none_when_no_pmc_link(self, mock_sleep, mock_get):
        """Paper exists in PubMed but is not in PMC."""
        resp_elink = mock.Mock()
        resp_elink.status_code = 200
        resp_elink.raise_for_status = mock.Mock()
        resp_elink.json.return_value = {"linksets": [{"linksetdbs": []}]}
        mock_get.return_value = resp_elink

        paper = {"pmid": "12345", "title": "No PMC"}
        assert fetch_full_text(paper) is None

    @mock.patch("virtualfermlab.discovery.paper_search.requests.get")
    @mock.patch("virtualfermlab.discovery.paper_search.time.sleep")
    def test_returns_none_on_network_error(self, mock_sleep, mock_get):
        mock_get.side_effect = ConnectionError("no network")
        paper = {"pmid": "12345", "pmcid": "PMC123", "title": "Test"}
        assert fetch_full_text(paper) is None

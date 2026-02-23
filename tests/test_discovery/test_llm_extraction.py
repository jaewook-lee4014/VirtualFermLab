"""Tests for LLM extraction (with mocked HTTP)."""

from __future__ import annotations

import json
import queue
from pathlib import Path
from unittest import mock

import pytest

from virtualfermlab.discovery import db
from virtualfermlab.discovery.llm_extraction import (
    LLMClient,
    _chunk_text,
    _dedup_params,
    _evidence_contains_value,
    _extract_json,
    _validate_param,
    extract_from_papers,
    extract_from_queue,
)


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")


class TestExtractJson:
    def test_clean_json(self):
        raw = '{"strain_name": "E. coli", "parameters": []}'
        assert _extract_json(raw) == {"strain_name": "E. coli", "parameters": []}

    def test_markdown_fences(self):
        raw = '```json\n{"strain_name": "E. coli", "parameters": []}\n```'
        assert _extract_json(raw) == {"strain_name": "E. coli", "parameters": []}

    def test_surrounding_text(self):
        raw = 'Here is the result:\n{"strain_name": "E. coli", "parameters": []}\nDone.'
        assert _extract_json(raw) == {"strain_name": "E. coli", "parameters": []}

    def test_trailing_comma(self):
        raw = '{"strain_name": "E. coli", "parameters": [{"name": "mu_max", "value": 0.5,},],}'
        result = _extract_json(raw)
        assert result is not None
        assert result["strain_name"] == "E. coli"

    def test_invalid_json(self):
        assert _extract_json("not json at all") is None

    def test_empty_string(self):
        assert _extract_json("") is None


class TestValidateParam:
    def test_valid_mu_max(self):
        assert _validate_param({"name": "mu_max", "value": 0.25}) is True

    def test_mu_max_too_high(self):
        assert _validate_param({"name": "mu_max", "value": 100.0}) is False

    def test_mu_max_negative(self):
        assert _validate_param({"name": "mu_max", "value": -0.1}) is False

    def test_unknown_param_name(self):
        assert _validate_param({"name": "unknown_thing", "value": 1.0}) is False

    def test_missing_value(self):
        assert _validate_param({"name": "mu_max"}) is False

    def test_string_value(self):
        assert _validate_param({"name": "mu_max", "value": "fast"}) is False

    def test_valid_pH(self):
        assert _validate_param({"name": "pH_opt", "value": 6.5}) is True

    def test_pH_out_of_range(self):
        assert _validate_param({"name": "pH_opt", "value": 15.0}) is False

    def test_valid_Yxs(self):
        assert _validate_param({"name": "Yxs", "value": 0.45}) is True

    def test_Yxs_too_high(self):
        assert _validate_param({"name": "Yxs", "value": 3.0}) is False


class TestLLMClient:
    def test_is_available_false_when_no_server(self):
        client = LLMClient(base_url="http://localhost:99999/v1")
        assert client.is_available() is False

    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.post")
    def test_extract_params_parses_json(self, mock_post):
        llm_response = {
            "strain_name": "Pichia pastoris",
            "substrates": ["glucose"],
            "parameters": [
                {"name": "mu_max", "value": 0.25, "unit": "1/h", "substrate": "glucose",
                 "evidence": "growth rate was 0.25 h-1"},
                {"name": "Ks", "value": 0.15, "unit": "g/L", "substrate": "glucose",
                 "evidence": "Ks = 0.15 g/L"},
            ],
            "conditions": {"pH": 5.5, "temperature": 30},
        }
        mock_resp = mock.Mock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = mock.Mock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps(llm_response)}}]
        }
        mock_post.return_value = mock_resp

        client = LLMClient()
        paper = {"title": "Test", "abstract": "Some text about Pichia."}
        params = client.extract_params(paper)

        assert len(params) == 2
        assert params[0]["name"] == "mu_max"
        assert params[0]["value"] == 0.25
        assert params[0]["strain_name"] == "Pichia pastoris"
        assert params[0]["evidence"] == "growth rate was 0.25 h-1"
        assert params[0]["confidence"] == "A"  # value 0.25 found in evidence

    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.post")
    def test_extract_params_handles_bad_json(self, mock_post):
        mock_resp = mock.Mock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = mock.Mock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "not valid json"}}]
        }
        mock_post.return_value = mock_resp

        client = LLMClient()
        params = client.extract_params({"title": "Test", "abstract": "abc"})
        assert params == []

    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.post")
    def test_extract_params_handles_markdown_fenced(self, mock_post):
        """LLM wrapping JSON in markdown code fences should still parse."""
        llm_response = {
            "strain_name": "S. cerevisiae",
            "substrates": ["glucose"],
            "parameters": [
                {"name": "mu_max", "value": 0.4, "unit": "1/h", "substrate": "glucose"},
            ],
            "conditions": {"pH": 5.0, "temperature": 30},
        }
        mock_resp = mock.Mock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = mock.Mock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "```json\n" + json.dumps(llm_response) + "\n```"}}]
        }
        mock_post.return_value = mock_resp

        client = LLMClient()
        params = client.extract_params({"title": "Test", "abstract": "yeast"})
        assert len(params) == 1
        assert params[0]["value"] == 0.4

    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.post")
    def test_extract_params_filters_invalid(self, mock_post):
        """Invalid parameter values should be silently discarded."""
        llm_response = {
            "strain_name": "E. coli",
            "substrates": ["glucose"],
            "parameters": [
                {"name": "mu_max", "value": 0.6, "unit": "1/h", "substrate": "glucose"},
                {"name": "mu_max", "value": 999.0, "unit": "1/h", "substrate": "glucose"},
                {"name": "made_up", "value": 1.0, "unit": "", "substrate": "glucose"},
            ],
            "conditions": {},
        }
        mock_resp = mock.Mock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = mock.Mock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps(llm_response)}}]
        }
        mock_post.return_value = mock_resp

        client = LLMClient()
        params = client.extract_params({"title": "Test", "abstract": "text"})
        assert len(params) == 1
        assert params[0]["value"] == 0.6

    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.post")
    def test_uses_fulltext_prompt_when_full_text_present(self, mock_post):
        """When paper has full_text, the full-text prompt template should be used."""
        llm_response = {
            "strain_name": "E. coli",
            "substrates": ["glucose"],
            "parameters": [
                {"name": "mu_max", "value": 0.5, "unit": "1/h", "substrate": "glucose"},
            ],
            "conditions": {},
        }
        mock_resp = mock.Mock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = mock.Mock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps(llm_response)}}]
        }
        mock_post.return_value = mock_resp

        client = LLMClient()
        paper = {
            "title": "E. coli growth",
            "abstract": "abstract text",
            "full_text": "## Results\nmu_max was 0.5 h-1 on glucose.",
        }
        params = client.extract_params(paper)

        # Verify the prompt sent to LLM contains the full text, not abstract
        call_args = mock_post.call_args
        messages = call_args.kwargs.get("json", call_args[1].get("json", {}))["messages"]
        user_msg = messages[1]["content"]
        assert "Full text (Results, Discussion, Tables):" in user_msg
        assert "mu_max was 0.5 h-1 on glucose" in user_msg
        assert len(params) == 1

    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.post")
    def test_uses_abstract_prompt_when_no_full_text(self, mock_post):
        """When paper lacks full_text, the abstract prompt template should be used."""
        llm_response = {
            "strain_name": "E. coli",
            "substrates": [],
            "parameters": [],
            "conditions": {},
        }
        mock_resp = mock.Mock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = mock.Mock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps(llm_response)}}]
        }
        mock_post.return_value = mock_resp

        client = LLMClient()
        paper = {"title": "Test", "abstract": "Some abstract."}
        client.extract_params(paper)

        call_args = mock_post.call_args
        messages = call_args.kwargs.get("json", call_args[1].get("json", {}))["messages"]
        user_msg = messages[1]["content"]
        assert "Abstract:" in user_msg
        assert "Full text" not in user_msg


class TestExtractFromPapers:
    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.post")
    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.get")
    @mock.patch("virtualfermlab.discovery.llm_extraction.fetch_full_text", return_value=None)
    def test_skips_when_unavailable(self, mock_ft, mock_get, mock_post):
        mock_get.side_effect = ConnectionError("no server")
        papers = [{"title": "Test", "abstract": "text", "doi": "10.1/a", "source": "pubmed"}]
        result = extract_from_papers(papers)
        assert result == []
        mock_post.assert_not_called()


_SENTINEL = object()


class TestExtractFromQueue:
    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.post")
    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.get")
    @mock.patch("virtualfermlab.discovery.llm_extraction.fetch_full_text", return_value=None)
    def test_processes_papers_from_queue(self, mock_ft, mock_get, mock_post):
        # Mark LLM as available
        resp_models = mock.Mock()
        resp_models.status_code = 200
        mock_get.return_value = resp_models

        llm_response = {
            "strain_name": "Pichia pastoris",
            "parameters": [
                {"name": "mu_max", "value": 0.25, "unit": "1/h", "substrate": "glucose"},
            ],
            "conditions": {"pH": 5.5},
        }
        mock_resp = mock.Mock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = mock.Mock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps(llm_response)}}]
        }
        mock_post.return_value = mock_resp

        q: queue.Queue = queue.Queue()
        q.put({"title": "Paper A", "abstract": "Pichia study", "doi": "10.1/a", "source": "pubmed"})
        q.put(_SENTINEL)

        params = extract_from_queue(q, _SENTINEL)
        assert len(params) == 1
        assert params[0]["name"] == "mu_max"

    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.get")
    def test_drains_queue_when_unavailable(self, mock_get):
        mock_get.side_effect = ConnectionError("no server")

        q: queue.Queue = queue.Queue()
        q.put({"title": "Paper", "abstract": "text", "doi": "10.1/x"})
        q.put({"title": "Paper2", "abstract": "text2", "doi": "10.1/y"})
        q.put(_SENTINEL)

        result = extract_from_queue(q, _SENTINEL)
        assert result == []
        assert q.empty()

    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.get")
    def test_handles_empty_queue_with_sentinel(self, mock_get):
        resp_models = mock.Mock()
        resp_models.status_code = 200
        mock_get.return_value = resp_models

        q: queue.Queue = queue.Queue()
        q.put(_SENTINEL)

        result = extract_from_queue(q, _SENTINEL)
        assert result == []

    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.post")
    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.get")
    @mock.patch("virtualfermlab.discovery.llm_extraction.fetch_full_text")
    def test_fetches_full_text_before_extraction(self, mock_ft, mock_get, mock_post):
        """Consumer should attempt full-text fetch for each paper."""
        mock_ft.return_value = "## Results\nmu_max = 0.3 h-1"

        resp_models = mock.Mock()
        resp_models.status_code = 200
        mock_get.return_value = resp_models

        llm_response = {
            "strain_name": "Yeast",
            "parameters": [
                {"name": "mu_max", "value": 0.3, "unit": "1/h", "substrate": None},
            ],
            "conditions": {},
        }
        mock_resp = mock.Mock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = mock.Mock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps(llm_response)}}]
        }
        mock_post.return_value = mock_resp

        q: queue.Queue = queue.Queue()
        paper = {"title": "Test", "abstract": "text", "doi": "10.1/z", "pmid": "999"}
        q.put(paper)
        q.put(_SENTINEL)

        params = extract_from_queue(q, _SENTINEL)
        assert len(params) == 1

        # Verify fetch_full_text was called
        mock_ft.assert_called_once_with(paper)

        # Verify full-text prompt was used (the paper dict gets mutated)
        call_args = mock_post.call_args
        messages = call_args.kwargs.get("json", call_args[1].get("json", {}))["messages"]
        user_msg = messages[1]["content"]
        assert "Full text (Results, Discussion, Tables):" in user_msg


class TestChunkText:
    def test_short_text_single_chunk(self):
        chunks = _chunk_text("short text", max_model_tokens=4096)
        assert len(chunks) == 1
        assert chunks[0] == "short text"

    def test_long_text_splits(self):
        # With default overhead (~1200 tok) and 4096 limit, available ~2896 tok
        # At 3.5 chars/tok = ~10136 chars per chunk
        text = "A" * 25000
        chunks = _chunk_text(text, max_model_tokens=4096)
        assert len(chunks) >= 2
        # All text should be covered
        combined_len = sum(len(c) for c in chunks)
        assert combined_len >= len(text)

    def test_prefers_paragraph_boundary(self):
        block_a = "First paragraph content.\n\nSecond paragraph content."
        # Make it just long enough to need splitting, with a clear \n\n boundary
        padding = "X" * 15000
        text = block_a + "\n\n" + padding
        chunks = _chunk_text(text, max_model_tokens=4096)
        assert len(chunks) >= 2
        # First chunk should end at a natural boundary (not mid-word)
        assert chunks[0].endswith(".") or chunks[0].endswith("\n") or chunks[0][-1].isalpha()

    def test_overlap_between_chunks(self):
        text = "A" * 25000
        chunks = _chunk_text(text, max_model_tokens=4096, overlap_chars=300)
        # With overlap, total chars > original text
        total_chars = sum(len(c) for c in chunks)
        assert total_chars > len(text)


class TestDedupParams:
    def test_removes_exact_duplicates(self):
        params = [
            {"name": "mu_max", "value": 0.25, "unit": "1/h", "substrate": "glucose"},
            {"name": "mu_max", "value": 0.25, "unit": "1/h", "substrate": "glucose"},
        ]
        assert len(_dedup_params(params)) == 1

    def test_keeps_different_values(self):
        params = [
            {"name": "mu_max", "value": 0.25, "unit": "1/h", "substrate": "glucose"},
            {"name": "mu_max", "value": 0.30, "unit": "1/h", "substrate": "glucose"},
        ]
        assert len(_dedup_params(params)) == 2

    def test_keeps_different_substrates(self):
        params = [
            {"name": "mu_max", "value": 0.25, "unit": "1/h", "substrate": "glucose"},
            {"name": "mu_max", "value": 0.25, "unit": "1/h", "substrate": "xylose"},
        ]
        assert len(_dedup_params(params)) == 2

    def test_empty_list(self):
        assert _dedup_params([]) == []


class TestChunkedExtraction:
    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.post")
    def test_multi_chunk_merges_results(self, mock_post):
        """Long full-text should be chunked, each chunk extracted, results merged."""
        # Chunk 1 response: mu_max on glucose
        resp_chunk1 = {
            "strain_name": "E. coli",
            "parameters": [
                {"name": "mu_max", "value": 0.6, "unit": "1/h", "substrate": "glucose"},
            ],
            "conditions": {},
        }
        # Chunk 2 response: Ks on glucose + duplicate mu_max
        resp_chunk2 = {
            "strain_name": "E. coli",
            "parameters": [
                {"name": "mu_max", "value": 0.6, "unit": "1/h", "substrate": "glucose"},
                {"name": "Ks", "value": 0.12, "unit": "g/L", "substrate": "glucose"},
            ],
            "conditions": {},
        }

        def _make_mock_resp(data):
            r = mock.Mock()
            r.status_code = 200
            r.raise_for_status = mock.Mock()
            r.json.return_value = {
                "choices": [{"message": {"content": json.dumps(data)}}]
            }
            return r

        # Figure out how many chunks will be produced and provide enough responses
        client = LLMClient(max_model_tokens=2000)
        long_text = "Results section.\n\n" + ("Data point. " * 500) + "\n\nTable data here."

        from virtualfermlab.discovery.llm_extraction import _chunk_text
        n_chunks = len(_chunk_text(long_text, max_model_tokens=2000))

        # First chunk returns resp_chunk1, last returns resp_chunk2,
        # middle chunks (if any) return empty params
        empty_resp = {"strain_name": "E. coli", "parameters": [], "conditions": {}}
        responses = [_make_mock_resp(resp_chunk1)]
        for _ in range(n_chunks - 2):
            responses.append(_make_mock_resp(empty_resp))
        responses.append(_make_mock_resp(resp_chunk2))
        mock_post.side_effect = responses

        paper = {"title": "E. coli study", "full_text": long_text}
        params = client.extract_params(paper)

        assert mock_post.call_count == n_chunks
        # mu_max duplicate should be deduped, so we get mu_max + Ks = 2
        assert len(params) == 2
        names = {p["name"] for p in params}
        assert names == {"mu_max", "Ks"}


class TestEvidenceContainsValue:
    def test_exact_match(self):
        assert _evidence_contains_value("mu_max was 0.25 h-1", 0.25) is True

    def test_integer_value(self):
        assert _evidence_contains_value("K_I = 5 g/L", 5.0) is True

    def test_middle_dot_separator(self):
        assert _evidence_contains_value("growth rate of 0\u00b725 h-1", 0.25) is True

    def test_comma_separator(self):
        assert _evidence_contains_value("mu_max = 0,25 h-1", 0.25) is True

    def test_value_not_present(self):
        assert _evidence_contains_value("growth rate was high", 0.25) is False

    def test_none_evidence(self):
        assert _evidence_contains_value(None, 0.25) is False

    def test_empty_evidence(self):
        assert _evidence_contains_value("", 0.25) is False


class TestEvidenceInExtraction:
    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.post")
    def test_confidence_A_when_value_in_evidence(self, mock_post):
        """Confidence should be 'A' when value appears in evidence text."""
        llm_response = {
            "strain_name": "E. coli",
            "parameters": [
                {"name": "mu_max", "value": 0.5, "unit": "1/h", "substrate": "glucose",
                 "evidence": "The mu_max was 0.5 h-1 on glucose"},
            ],
            "conditions": {},
        }
        mock_resp = mock.Mock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = mock.Mock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps(llm_response)}}]
        }
        mock_post.return_value = mock_resp

        client = LLMClient()
        params = client.extract_params({"title": "Test", "abstract": "text"})
        assert len(params) == 1
        assert params[0]["confidence"] == "A"
        assert params[0]["evidence"] == "The mu_max was 0.5 h-1 on glucose"

    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.post")
    def test_confidence_B_when_value_not_in_evidence(self, mock_post):
        """Confidence should be 'B' when evidence doesn't contain the value."""
        llm_response = {
            "strain_name": "E. coli",
            "parameters": [
                {"name": "mu_max", "value": 0.5, "unit": "1/h", "substrate": "glucose",
                 "evidence": "Growth was observed on glucose"},
            ],
            "conditions": {},
        }
        mock_resp = mock.Mock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = mock.Mock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps(llm_response)}}]
        }
        mock_post.return_value = mock_resp

        client = LLMClient()
        params = client.extract_params({"title": "Test", "abstract": "text"})
        assert len(params) == 1
        assert params[0]["confidence"] == "B"

    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.post")
    def test_confidence_B_when_no_evidence(self, mock_post):
        """Confidence should be 'B' when evidence is missing entirely."""
        llm_response = {
            "strain_name": "E. coli",
            "parameters": [
                {"name": "mu_max", "value": 0.5, "unit": "1/h", "substrate": "glucose"},
            ],
            "conditions": {},
        }
        mock_resp = mock.Mock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = mock.Mock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps(llm_response)}}]
        }
        mock_post.return_value = mock_resp

        client = LLMClient()
        params = client.extract_params({"title": "Test", "abstract": "text"})
        assert len(params) == 1
        assert params[0]["confidence"] == "B"
        assert params[0]["evidence"] is None


class TestDedupWithEvidence:
    def test_prefers_entry_with_evidence(self):
        params = [
            {"name": "mu_max", "value": 0.25, "substrate": "glucose",
             "confidence": "B", "evidence": None},
            {"name": "mu_max", "value": 0.25, "substrate": "glucose",
             "confidence": "A", "evidence": "mu_max was 0.25"},
        ]
        result = _dedup_params(params)
        assert len(result) == 1
        assert result[0]["confidence"] == "A"
        assert result[0]["evidence"] == "mu_max was 0.25"

    def test_prefers_higher_confidence(self):
        params = [
            {"name": "Ks", "value": 0.15, "substrate": "glucose",
             "confidence": "B", "evidence": "some text"},
            {"name": "Ks", "value": 0.15, "substrate": "glucose",
             "confidence": "A", "evidence": "Ks = 0.15 g/L"},
        ]
        result = _dedup_params(params)
        assert len(result) == 1
        assert result[0]["confidence"] == "A"

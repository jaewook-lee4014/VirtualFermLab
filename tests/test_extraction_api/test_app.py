"""Tests for the extraction API endpoints (mocked vLLM)."""

from __future__ import annotations

import json
from unittest import mock

import pytest

from virtualfermlab.extraction_api.app import create_app
from virtualfermlab.discovery.llm_extraction import LLMClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_response(strain="E. coli", params=None, conditions=None):
    """Build a fake vLLM chat/completions response body."""
    llm_json = {
        "strain_name": strain,
        "substrates": [],
        "parameters": params or [],
        "conditions": conditions or {},
    }
    return {
        "choices": [{"message": {"content": json.dumps(llm_json)}}]
    }


def _mock_post_response(llm_body):
    """Return a mock requests.Response for a successful LLM call."""
    resp = mock.Mock()
    resp.status_code = 200
    resp.raise_for_status = mock.Mock()
    resp.json.return_value = llm_body
    return resp


def _mock_models_ok():
    """Return a mock for GET /v1/models (available)."""
    resp = mock.Mock()
    resp.status_code = 200
    return resp


def _mock_models_fail():
    """Simulate unreachable vLLM server."""
    raise ConnectionError("no server")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client_app():
    """Yield a Flask test client with a real (unmocked) LLMClient."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as tc:
        yield tc


# ---------------------------------------------------------------------------
# TestHealth
# ---------------------------------------------------------------------------

class TestHealth:
    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.get")
    def test_healthy(self, mock_get, client_app):
        mock_get.return_value = _mock_models_ok()
        resp = client_app.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"

    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.get")
    def test_unhealthy(self, mock_get, client_app):
        mock_get.side_effect = ConnectionError("no server")
        resp = client_app.get("/health")
        assert resp.status_code == 503
        data = resp.get_json()
        assert data["status"] == "unavailable"


# ---------------------------------------------------------------------------
# TestModelInfo
# ---------------------------------------------------------------------------

class TestModelInfo:
    def test_returns_model_info(self, client_app):
        resp = client_app.get("/model-info")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "model" in data
        assert "max_model_tokens" in data
        assert "valid_parameters" in data
        assert "mu_max" in data["valid_parameters"]


# ---------------------------------------------------------------------------
# TestExtractAbstract
# ---------------------------------------------------------------------------

class TestExtractAbstract:
    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.post")
    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.get")
    def test_success(self, mock_get, mock_post, client_app):
        mock_get.return_value = _mock_models_ok()
        llm_body = _make_llm_response(
            strain="E. coli",
            params=[
                {"name": "mu_max", "value": 0.6, "unit": "1/h",
                 "substrate": "glucose", "evidence": "mu_max was 0.6 h-1"},
            ],
        )
        mock_post.return_value = _mock_post_response(llm_body)

        resp = client_app.post(
            "/extract/abstract",
            json={"title": "Growth of E. coli", "abstract": "mu_max was 0.6 h-1 on glucose"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["n_parameters"] == 1
        assert data["source"] == "abstract"
        assert data["parameters"][0]["name"] == "mu_max"
        assert data["parameters"][0]["value"] == 0.6
        assert data["parameters"][0]["confidence"] == "A"

    def test_missing_title(self, client_app):
        resp = client_app.post(
            "/extract/abstract",
            json={"abstract": "some text"},
        )
        assert resp.status_code == 400
        assert "title" in resp.get_json()["error"]

    def test_missing_abstract(self, client_app):
        resp = client_app.post(
            "/extract/abstract",
            json={"title": "Test"},
        )
        assert resp.status_code == 400
        assert "abstract" in resp.get_json()["error"]

    def test_not_json(self, client_app):
        resp = client_app.post(
            "/extract/abstract",
            data="plain text",
            content_type="text/plain",
        )
        assert resp.status_code == 400

    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.get")
    def test_vllm_unavailable(self, mock_get, client_app):
        mock_get.side_effect = ConnectionError("no server")
        resp = client_app.post(
            "/extract/abstract",
            json={"title": "Test", "abstract": "text"},
        )
        assert resp.status_code == 503

    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.post")
    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.get")
    def test_empty_result(self, mock_get, mock_post, client_app):
        mock_get.return_value = _mock_models_ok()
        llm_body = _make_llm_response(params=[])
        mock_post.return_value = _mock_post_response(llm_body)

        resp = client_app.post(
            "/extract/abstract",
            json={"title": "Test", "abstract": "No kinetic data here."},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["n_parameters"] == 0
        assert data["parameters"] == []

    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.post")
    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.get")
    def test_validation_filters_bad_params(self, mock_get, mock_post, client_app):
        mock_get.return_value = _mock_models_ok()
        llm_body = _make_llm_response(
            params=[
                {"name": "mu_max", "value": 0.3, "unit": "1/h",
                 "substrate": "glucose", "evidence": "mu_max 0.3"},
                {"name": "mu_max", "value": 999.0, "unit": "1/h",
                 "substrate": "glucose"},
                {"name": "made_up", "value": 1.0, "unit": ""},
            ],
        )
        mock_post.return_value = _mock_post_response(llm_body)

        resp = client_app.post(
            "/extract/abstract",
            json={"title": "Test", "abstract": "mu_max 0.3"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["n_parameters"] == 1
        assert data["parameters"][0]["value"] == 0.3

    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.post")
    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.get")
    def test_confidence_B_without_evidence(self, mock_get, mock_post, client_app):
        mock_get.return_value = _mock_models_ok()
        llm_body = _make_llm_response(
            params=[
                {"name": "mu_max", "value": 0.5, "unit": "1/h", "substrate": "glucose"},
            ],
        )
        mock_post.return_value = _mock_post_response(llm_body)

        resp = client_app.post(
            "/extract/abstract",
            json={"title": "Test", "abstract": "growth observed"},
        )
        data = resp.get_json()
        assert data["parameters"][0]["confidence"] == "B"


# ---------------------------------------------------------------------------
# TestExtractFulltext
# ---------------------------------------------------------------------------

class TestExtractFulltext:
    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.post")
    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.get")
    def test_success(self, mock_get, mock_post, client_app):
        mock_get.return_value = _mock_models_ok()
        llm_body = _make_llm_response(
            params=[
                {"name": "mu_max", "value": 0.4, "unit": "1/h",
                 "substrate": "glucose", "evidence": "mu_max = 0.4 h-1"},
            ],
        )
        mock_post.return_value = _mock_post_response(llm_body)

        resp = client_app.post(
            "/extract/fulltext",
            json={"title": "Test", "full_text": "Results: mu_max = 0.4 h-1 on glucose."},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["n_parameters"] == 1
        assert data["source"] == "fulltext"
        assert data["n_chunks"] == 1

    def test_missing_full_text(self, client_app):
        resp = client_app.post(
            "/extract/fulltext",
            json={"title": "Test"},
        )
        assert resp.status_code == 400
        assert "full_text" in resp.get_json()["error"]

    def test_missing_title(self, client_app):
        resp = client_app.post(
            "/extract/fulltext",
            json={"full_text": "some text"},
        )
        assert resp.status_code == 400
        assert "title" in resp.get_json()["error"]

    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.get")
    def test_vllm_unavailable(self, mock_get, client_app):
        mock_get.side_effect = ConnectionError("no server")
        resp = client_app.post(
            "/extract/fulltext",
            json={"title": "Test", "full_text": "some content"},
        )
        assert resp.status_code == 503

    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.post")
    @mock.patch("virtualfermlab.discovery.llm_extraction.requests.get")
    def test_chunking_with_long_text(self, mock_get, mock_post, client_app):
        """Long full_text should be chunked; n_chunks reflects this."""
        mock_get.return_value = _mock_models_ok()

        # Chunk 1: mu_max
        resp_chunk1 = _make_llm_response(
            params=[
                {"name": "mu_max", "value": 0.6, "unit": "1/h", "substrate": "glucose"},
            ],
        )
        # Chunk 2: Ks + duplicate mu_max
        resp_chunk2 = _make_llm_response(
            params=[
                {"name": "mu_max", "value": 0.6, "unit": "1/h", "substrate": "glucose"},
                {"name": "Ks", "value": 0.12, "unit": "g/L", "substrate": "glucose"},
            ],
        )
        empty_resp = _make_llm_response(params=[])

        # Build a text long enough to need multiple chunks at default 4096 tokens
        long_text = "Results section.\n\n" + ("Data point. " * 3000) + "\n\nTable data here."

        from virtualfermlab.discovery.llm_extraction import _chunk_text
        n_chunks = len(_chunk_text(long_text, max_model_tokens=4096))
        assert n_chunks >= 2

        responses = [_mock_post_response(resp_chunk1)]
        for _ in range(n_chunks - 2):
            responses.append(_mock_post_response(empty_resp))
        responses.append(_mock_post_response(resp_chunk2))
        mock_post.side_effect = responses

        resp = client_app.post(
            "/extract/fulltext",
            json={"title": "Long paper", "full_text": long_text},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["n_chunks"] == n_chunks
        # mu_max duplicate deduped → mu_max + Ks = 2
        assert data["n_parameters"] == 2
        names = {p["name"] for p in data["parameters"]}
        assert names == {"mu_max", "Ks"}

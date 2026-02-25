"""Standalone REST API for LLM-based kinetic parameter extraction.

Reuses the LLMClient and prompt templates from the discovery pipeline
but exposes them as a stateless HTTP service — no database, no paper
search, no numpy/plotly dependency.

Usage:
    python -m virtualfermlab.extraction_api.app [--host HOST] [--port PORT]

Environment variables:
    VLLM_BASE_URL   (default: http://localhost:8000/v1)
    VLLM_MODEL      (default: Qwen/Qwen2.5-32B-Instruct)
    VLLM_MAX_TOKENS (default: 4096)
"""

from __future__ import annotations

import argparse
import logging
import os

from flask import Flask, jsonify, request

from virtualfermlab.discovery.llm_extraction import (
    LLMClient,
    _chunk_text,
    _dedup_params,
    _VALID_PARAM_NAMES,
)

logger = logging.getLogger(__name__)


def _get_client() -> LLMClient:
    """Build an LLMClient from environment variables."""
    return LLMClient(
        base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        model=os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-32B-Instruct"),
        max_model_tokens=int(os.environ.get("VLLM_MAX_TOKENS", "4096")),
    )


def create_app(client: LLMClient | None = None) -> Flask:
    """Application factory.

    Parameters
    ----------
    client : LLMClient, optional
        Injected for testing. When *None*, a client is built from env vars
        on every request that needs one.
    """
    app = Flask(__name__)

    def _client() -> LLMClient:
        return client if client is not None else _get_client()

    # ----- GET /health ------------------------------------------------

    @app.route("/health", methods=["GET"])
    def health():
        c = _client()
        if c.is_available():
            return jsonify({"status": "ok", "model": c.model}), 200
        return jsonify({"status": "unavailable", "error": "vLLM endpoint not reachable"}), 503

    # ----- GET /model-info --------------------------------------------

    @app.route("/model-info", methods=["GET"])
    def model_info():
        c = _client()
        return jsonify({
            "model": c.model,
            "max_model_tokens": c.max_model_tokens,
            "vllm_base_url": c.base_url,
            "valid_parameters": sorted(_VALID_PARAM_NAMES),
        }), 200

    # ----- POST /extract/abstract -------------------------------------

    @app.route("/extract/abstract", methods=["POST"])
    def extract_abstract():
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Request body must be JSON"}), 400

        title = data.get("title")
        abstract = data.get("abstract")
        if not title or not abstract:
            return jsonify({"error": "Both 'title' and 'abstract' are required"}), 400

        c = _client()
        if not c.is_available():
            return jsonify({"error": "vLLM endpoint not reachable"}), 503

        paper = {"title": title, "abstract": abstract}
        params = c.extract_params(paper)

        return jsonify({
            "parameters": params,
            "n_parameters": len(params),
            "source": "abstract",
        }), 200

    # ----- POST /extract/fulltext -------------------------------------

    @app.route("/extract/fulltext", methods=["POST"])
    def extract_fulltext():
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Request body must be JSON"}), 400

        title = data.get("title")
        full_text = data.get("full_text")
        if not title or not full_text:
            return jsonify({"error": "Both 'title' and 'full_text' are required"}), 400

        c = _client()
        if not c.is_available():
            return jsonify({"error": "vLLM endpoint not reachable"}), 503

        paper = {"title": title, "full_text": full_text}
        n_chunks = len(_chunk_text(full_text, c.max_model_tokens))
        params = c.extract_params(paper)

        return jsonify({
            "parameters": params,
            "n_parameters": len(params),
            "n_chunks": n_chunks,
            "source": "fulltext",
        }), 200

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Parameter Extraction API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("API_PORT", "5001")))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

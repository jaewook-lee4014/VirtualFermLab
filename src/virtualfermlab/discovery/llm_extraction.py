"""LLM-based kinetic parameter extraction via vLLM (OpenAI-compatible API)."""

from __future__ import annotations

import json
import logging
import os
import queue
import re

import requests

from virtualfermlab.discovery import db
from virtualfermlab.discovery.paper_search import fetch_full_text
from virtualfermlab.discovery.prompts import (
    EXTRACTION_PROMPT_ABSTRACT,
    EXTRACTION_PROMPT_FULLTEXT,
    EXTRACTION_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# Allowed parameter names — anything outside this set is discarded.
_VALID_PARAM_NAMES = frozenset({
    "mu_max", "Ks", "Yxs", "K_I",
    "pH_opt", "pH_min", "pH_max", "lag_time",
})


def _extract_json(text: str) -> dict | None:
    """Best-effort extraction of a JSON object from LLM output.

    Handles common issues:
    - Markdown code fences (```json ... ```)
    - Leading/trailing text around the JSON
    - Trailing commas (common LLM mistake)
    """
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find the outermost { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        # Remove trailing commas before } or ]
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


def _validate_param(p: dict) -> bool:
    """Return True if a parameter dict looks reasonable."""
    name = p.get("name", "")
    if name not in _VALID_PARAM_NAMES:
        return False
    try:
        val = float(p["value"])
    except (KeyError, TypeError, ValueError):
        return False
    # Sanity bounds
    if name == "mu_max" and not (0.001 <= val <= 5.0):
        return False
    if name == "Ks" and not (0.0001 <= val <= 500.0):
        return False
    if name == "Yxs" and not (0.001 <= val <= 2.0):
        return False
    if name == "K_I" and not (0.001 <= val <= 500.0):
        return False
    if name in ("pH_opt", "pH_min", "pH_max") and not (0.0 <= val <= 14.0):
        return False
    if name == "lag_time" and not (0.0 <= val <= 200.0):
        return False
    return True


def _evidence_contains_value(evidence: str | None, value: float) -> bool:
    """Check whether *evidence* text contains the numeric *value*.

    Handles common formatting variants: ``0.25``, ``0·25``, ``0,25``.
    """
    if not evidence:
        return False
    # Build candidate string representations of the value
    val_str = f"{value:g}"  # e.g. "0.25"
    if val_str in evidence:
        return True
    # Try with middle-dot or comma as decimal separator
    if "." in val_str:
        if val_str.replace(".", "\u00b7") in evidence:
            return True
        if val_str.replace(".", ",") in evidence:
            return True
    return False


# Characters per token estimate — conservative for scientific text with
# tables, units and special characters (measured range: 2.4–4.2 chars/tok).
_CHARS_PER_TOKEN = 2.4

# Overhead tokens: prompt template (with few-shot examples) + system prompt
# + output reserve.  Measured on Qwen2.5-32B tokeniser:
#   system prompt  ~  60 tok
#   template chrome~ 900 tok  (rules + schema + few-shot, without {full_text})
#   output reserve ~ 512 tok
_PROMPT_OVERHEAD_TOKENS = 1500


def _chunk_text(text: str, max_model_tokens: int, overlap_chars: int = 200) -> list[str]:
    """Split *text* into chunks that fit within the LLM token budget.

    Each chunk is sized so that ``overhead + chunk`` stays within
    *max_model_tokens*.  Adjacent chunks overlap by *overlap_chars* to
    avoid splitting a sentence or table row at a boundary.
    """
    available_tokens = max_model_tokens - _PROMPT_OVERHEAD_TOKENS
    if available_tokens <= 0:
        available_tokens = 512  # absolute minimum

    chunk_chars = int(available_tokens * _CHARS_PER_TOKEN)

    if len(text) <= chunk_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_chars
        if end >= len(text):
            chunks.append(text[start:])
            break

        # Try to break at a paragraph boundary (\n\n) near the end
        split_at = text.rfind("\n\n", start + chunk_chars // 2, end)
        if split_at == -1:
            # Fall back to line break
            split_at = text.rfind("\n", start + chunk_chars // 2, end)
        if split_at == -1:
            split_at = end

        chunks.append(text[start:split_at])
        start = max(start + 1, split_at - overlap_chars)

    return chunks


def _dedup_params(params: list[dict]) -> list[dict]:
    """Remove duplicate parameters (same name + value + substrate).

    When duplicates exist, the entry with higher confidence (A > B) or
    with evidence text is preferred.
    """
    best: dict[tuple, dict] = {}
    for p in params:
        key = (p["name"], round(p["value"], 6), p.get("substrate"))
        existing = best.get(key)
        if existing is None:
            best[key] = p
        elif p.get("confidence", "B") < existing.get("confidence", "B"):
            # "A" < "B" lexicographically, so lower = better
            best[key] = p
        elif p.get("evidence") and not existing.get("evidence"):
            best[key] = p
    return list(best.values())


class LLMClient:
    """Thin client for an OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str = "/dev/shm/models/Qwen2.5-32B-Instruct",
        max_model_tokens: int = 4096,
    ) -> None:
        self.base_url = base_url or os.environ.get(
            "VLLM_BASE_URL", "http://erc-hpc-comp247:8000/v1"
        )
        self.model = model
        self.max_model_tokens = max_model_tokens

    def is_available(self) -> bool:
        """Return ``True`` if the vLLM endpoint is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/models", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def _call_llm(self, prompt: str) -> str | None:
        """Send a single prompt to the LLM and return the raw response text."""
        output_budget = min(
            1024,
            self.max_model_tokens - _PROMPT_OVERHEAD_TOKENS,
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": max(output_budget, 256),
        }

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except (requests.RequestException, KeyError) as exc:
            logger.warning("LLM request failed: %s", exc)
            return None

    def _parse_response(self, content: str, title: str) -> list[dict]:
        """Parse LLM response text into validated parameter dicts.

        Each returned dict includes ``evidence`` (verbatim quote from the
        paper, or ``None``) and ``confidence`` (``"A"`` when the numeric
        value appears inside the evidence text, ``"B"`` otherwise).
        """
        data = _extract_json(content)
        if data is None:
            logger.warning(
                "Could not parse JSON from LLM for '%s'. Raw: %.200s",
                title, content,
            )
            return []

        strain_name = data.get("strain_name", "")
        conditions = data.get("conditions")
        params: list[dict] = []
        for p in data.get("parameters", []):
            if not _validate_param(p):
                logger.debug("Discarding invalid param: %s", p)
                continue
            value = float(p["value"])
            evidence = p.get("evidence") or None
            confidence = (
                "A" if _evidence_contains_value(evidence, value) else "B"
            )
            params.append({
                "name": p.get("name", ""),
                "value": value,
                "unit": p.get("unit", ""),
                "substrate": p.get("substrate"),
                "strain_name": strain_name,
                "conditions": conditions,
                "evidence": evidence,
                "confidence": confidence,
            })
        return params

    def extract_params(self, paper: dict) -> list[dict]:
        """Extract kinetic parameters from a paper.

        If *paper* contains a ``full_text`` key, the text is split into
        chunks that fit within the model's context window.  Each chunk
        is sent independently and results are merged with deduplication.
        Falls back to the abstract-only prompt when no full-text is
        available.

        Returns a list of parameter dicts, each with keys:
        ``name``, ``value``, ``unit``, ``substrate``, ``strain_name``,
        ``conditions``, ``evidence``, ``confidence``.
        """
        title = paper.get("title", "")
        full_text = paper.get("full_text")

        if full_text:
            chunks = _chunk_text(full_text, self.max_model_tokens)
            logger.info(
                "Processing '%s' in %d chunk(s)",
                title[:60], len(chunks),
            )

            all_params: list[dict] = []
            for i, chunk in enumerate(chunks):
                prompt = EXTRACTION_PROMPT_FULLTEXT.format(
                    title=title,
                    full_text=chunk,
                )
                content = self._call_llm(prompt)
                if content is None:
                    continue
                chunk_params = self._parse_response(content, title)
                logger.debug(
                    "Chunk %d/%d: %d params", i + 1, len(chunks), len(chunk_params),
                )
                all_params.extend(chunk_params)

            return _dedup_params(all_params)

        # Abstract-only fallback
        prompt = EXTRACTION_PROMPT_ABSTRACT.format(
            title=title,
            abstract=paper.get("abstract", ""),
        )
        content = self._call_llm(prompt)
        if content is None:
            return []
        return self._parse_response(content, title)


def extract_from_papers(papers: list[dict], client: LLMClient | None = None) -> list[dict]:
    """Extract kinetic parameters from a list of papers.

    Saves results to the SQLite database as a side effect.
    """
    if client is None:
        client = LLMClient()

    if not client.is_available():
        logger.warning("LLM endpoint not available — skipping extraction")
        return []

    all_params: list[dict] = []

    for paper in papers:
        if not paper.get("abstract"):
            continue

        # Attempt full-text retrieval from PMC before LLM call
        if not paper.get("full_text"):
            ft = fetch_full_text(paper)
            if ft:
                paper["full_text"] = ft
                logger.info("Fetched full-text for '%s'", paper.get("title", "?")[:60])

        paper_id = db.save_paper(paper)
        extracted = client.extract_params(paper)

        if extracted:
            db.save_extracted_params(paper_id, extracted)
            all_params.extend(extracted)
            logger.info(
                "Extracted %d params from '%s'",
                len(extracted),
                paper.get("title", "?")[:60],
            )

    return all_params


def extract_from_queue(
    paper_queue: queue.Queue,
    sentinel: object,
    client: LLMClient | None = None,
) -> list[dict]:
    """Consume papers from *paper_queue* and extract parameters sequentially.

    Blocks on ``queue.get()`` until the *sentinel* object is received,
    indicating that the producer has finished.

    If the LLM endpoint is unreachable, drains the queue and returns an
    empty list so the pipeline can degrade gracefully.
    """
    if client is None:
        client = LLMClient()

    if not client.is_available():
        logger.warning("LLM endpoint not available — draining queue and skipping extraction")
        while True:
            item = paper_queue.get()
            if item is sentinel:
                break
        return []

    all_params: list[dict] = []

    while True:
        item = paper_queue.get()
        if item is sentinel:
            break

        paper: dict = item
        if not paper.get("abstract"):
            continue

        # Attempt full-text retrieval from PMC before LLM call
        if not paper.get("full_text"):
            ft = fetch_full_text(paper)
            if ft:
                paper["full_text"] = ft
                logger.info("Fetched full-text for '%s'", paper.get("title", "?")[:60])

        paper_id = db.save_paper(paper)
        extracted = client.extract_params(paper)

        if extracted:
            db.save_extracted_params(paper_id, extracted)
            all_params.extend(extracted)
            logger.info(
                "Extracted %d params from '%s'",
                len(extracted),
                paper.get("title", "?")[:60],
            )

    return all_params

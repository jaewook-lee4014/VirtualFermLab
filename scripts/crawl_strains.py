#!/usr/bin/env python3
"""Batch literature crawling for kinetic parameters of filamentous fungi.

Targets ~10 strains taxonomically similar to Fusarium venenatum.
For each strain, searches PubMed + Semantic Scholar with multiple queries,
fetches full-text from PMC where available, and extracts kinetic parameters
(mu_max, Ks, Yxs) via vLLM (Qwen2.5-32B).

Usage (run on the server where vLLM is running):
    export VLLM_BASE_URL=http://localhost:8000/v1
    python scripts/crawl_strains.py [--strains "Fusarium oxysporum,Trichoderma reesei"]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from virtualfermlab.discovery import db
from virtualfermlab.discovery.paper_search import (
    search_pubmed,
    search_semantic_scholar,
    fetch_full_text,
)
from virtualfermlab.discovery.llm_extraction import LLMClient
from virtualfermlab.discovery.pipeline import run_discovery

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("crawl_strains.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ── Target strains ──────────────────────────────────────────────────
# Filamentous fungi taxonomically close to F. venenatum, with published
# kinetic data on glucose and/or xylose fermentation.

DEFAULT_STRAINS = [
    "Fusarium oxysporum",          # Same genus, known xylose fermenter
    "Fusarium graminearum",        # Same genus, plant pathogen with growth data
    "Trichoderma reesei",          # Industrial cellulase producer, uses both sugars
    "Aspergillus niger",           # Industrial workhorse, well-characterised
    "Aspergillus oryzae",          # Food-grade, koji fermentation
    "Neurospora crassa",           # Model filamentous fungus, xylose metaboliser
    "Rhizopus oryzae",             # Filamentous, lactic acid / ethanol producer
    "Mucor indicus",               # Filamentous, ethanol from xylose
    "Paecilomyces variotii",       # SCP producer (competitor to Quorn)
    "Myceliophthora thermophila",  # Thermophilic filamentous fungus
]

# Multiple query templates to maximise coverage
QUERY_TEMPLATES = [
    '"{strain}" fermentation kinetics Monod glucose',
    '"{strain}" growth rate kinetics xylose fermentation',
    '"{strain}" biomass yield glucose xylose',
    '"{strain}" specific growth rate substrate consumption batch',
    '"{strain}" mu_max Ks Yxs kinetic parameter',
]

MAX_RESULTS_PER_QUERY = 15


def _search_all_queries(
    strain: str, max_per_query: int = MAX_RESULTS_PER_QUERY
) -> list[dict]:
    """Run multiple search queries for a single strain, deduplicate by DOI."""
    seen_dois: set[str] = set()
    all_papers: list[dict] = []

    for template in QUERY_TEMPLATES:
        query = template.format(strain=strain)
        logger.info("  Query: %s", query)

        # PubMed
        try:
            pm = search_pubmed(query, max_per_query)
            logger.info("    PubMed: %d results", len(pm))
        except Exception:
            logger.exception("    PubMed failed")
            pm = []

        time.sleep(0.5)

        # Semantic Scholar
        try:
            s2 = search_semantic_scholar(query, max_per_query)
            logger.info("    S2: %d results", len(s2))
        except Exception:
            logger.exception("    S2 failed")
            s2 = []

        time.sleep(0.5)

        for paper in pm + s2:
            doi = paper.get("doi")
            if doi and doi in seen_dois:
                continue
            if doi:
                seen_dois.add(doi)
            all_papers.append(paper)

    logger.info("  Total unique papers for %s: %d", strain, len(all_papers))
    return all_papers


def _extract_params_batch(
    papers: list[dict], client: LLMClient, strain: str
) -> list[dict]:
    """Extract parameters from a batch of papers, saving to DB."""
    all_params: list[dict] = []
    total = len(papers)

    for i, paper in enumerate(papers, 1):
        title = paper.get("title", "?")[:70]
        abstract = paper.get("abstract", "")

        if not abstract or len(abstract) < 50:
            logger.info("  [%d/%d] Skipping (no abstract): %s", i, total, title)
            continue

        # Fetch full-text from PMC if available
        if not paper.get("full_text"):
            try:
                ft = fetch_full_text(paper)
                if ft:
                    paper["full_text"] = ft
                    logger.info("  [%d/%d] Got full-text for: %s", i, total, title)
            except Exception:
                pass

        # Save paper to DB
        try:
            paper_id = db.save_paper(paper)
        except Exception:
            logger.warning("  [%d/%d] Failed to save paper: %s", i, total, title)
            continue

        # Skip if already extracted
        if db.paper_has_params(paper_id):
            logger.info("  [%d/%d] Already extracted, skipping: %s", i, total, title)
            continue

        # Extract parameters via LLM
        logger.info("  [%d/%d] Extracting from: %s", i, total, title)
        try:
            extracted = client.extract_params(paper)
        except Exception:
            logger.exception("  [%d/%d] Extraction failed: %s", i, total, title)
            continue

        if extracted:
            db.save_extracted_params(paper_id, extracted)
            all_params.extend(extracted)
            for p in extracted:
                logger.info(
                    "    -> %s: %s = %.4f %s (substrate=%s, conf=%s)",
                    p.get("strain_name", strain),
                    p["name"],
                    p["value"],
                    p.get("unit", ""),
                    p.get("substrate", "N/A"),
                    p.get("confidence", "?"),
                )
        else:
            logger.info("  [%d/%d] No params extracted", i, total)

        # Rate limiting
        time.sleep(0.3)

    return all_params


def crawl_strain(strain: str, client: LLMClient) -> dict:
    """Full crawling pipeline for a single strain.

    Returns summary dict with papers_found, params_extracted, param_details.
    """
    logger.info("=" * 70)
    logger.info("CRAWLING: %s", strain)
    logger.info("=" * 70)
    t0 = time.monotonic()

    # 1. Search papers
    papers = _search_all_queries(strain)

    # 2. Extract parameters
    params = _extract_params_batch(papers, client, strain)

    elapsed = time.monotonic() - t0
    logger.info(
        "DONE: %s — %d papers, %d params extracted in %.1f seconds",
        strain, len(papers), len(params), elapsed,
    )

    return {
        "strain": strain,
        "papers_found": len(papers),
        "params_extracted": len(params),
        "params": params,
        "elapsed_sec": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Crawl literature for kinetic parameters")
    parser.add_argument(
        "--strains",
        type=str,
        default=None,
        help="Comma-separated list of strain names (default: all 10 target strains)",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM API base URL",
    )
    args = parser.parse_args()

    if args.strains:
        strains = [s.strip() for s in args.strains.split(",")]
    else:
        strains = DEFAULT_STRAINS

    # Initialise DB
    db.init_db()

    # Create LLM client
    client = LLMClient(base_url=args.vllm_url)
    if not client.is_available():
        logger.error("vLLM endpoint not available at %s", args.vllm_url)
        logger.error("Set VLLM_BASE_URL or pass --vllm-url")
        sys.exit(1)
    logger.info("vLLM endpoint OK: %s", args.vllm_url)

    # Run crawling for each strain
    results = []
    total_t0 = time.monotonic()

    for i, strain in enumerate(strains, 1):
        logger.info("\n>>> Strain %d/%d: %s <<<", i, len(strains), strain)
        result = crawl_strain(strain, client)
        results.append(result)

    total_elapsed = time.monotonic() - total_t0

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("CRAWLING COMPLETE — Total time: %.1f minutes", total_elapsed / 60)
    logger.info("=" * 70)
    total_papers = sum(r["papers_found"] for r in results)
    total_params = sum(r["params_extracted"] for r in results)
    logger.info("Total papers searched: %d", total_papers)
    logger.info("Total params extracted: %d", total_params)

    for r in results:
        logger.info(
            "  %-35s: %3d papers, %3d params (%.0fs)",
            r["strain"], r["papers_found"], r["params_extracted"], r["elapsed_sec"],
        )

    # Also run discovery pipeline (taxonomy match + profile building) for each strain
    logger.info("\n" + "=" * 70)
    logger.info("BUILDING STRAIN PROFILES")
    logger.info("=" * 70)
    for strain in strains:
        try:
            disc = run_discovery(strain)
            logger.info(
                "  %s: source=%s, similar=%s (sim=%.2f), params=%d",
                strain,
                disc.source,
                disc.similar_strain or "none",
                disc.similarity_score or 0,
                disc.params_extracted,
            )
        except Exception:
            logger.exception("  Failed to build profile for %s", strain)

    logger.info("\nAll done.")


if __name__ == "__main__":
    main()

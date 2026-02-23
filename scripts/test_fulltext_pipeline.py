"""Manual integration test: full-text fetch + LLM extraction on two real papers.

Paper 1: ESCAPE conference proceedings (no PMID/PMCID — tests abstract fallback)
Paper 2: PMC Open Access (PMID 39555020, PMC11565039 — tests full-text path)

Usage:
    python scripts/test_fulltext_pipeline.py
"""

from __future__ import annotations

import json
import sys
import textwrap

# Ensure the project is importable
sys.path.insert(0, "src")

from virtualfermlab.discovery.paper_search import fetch_full_text, search_pubmed
from virtualfermlab.discovery.llm_extraction import LLMClient


def _sep(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def main() -> None:
    # ------------------------------------------------------------------
    # Paper 1: ESCAPE proceedings (no PMC full-text expected)
    # ------------------------------------------------------------------
    paper1 = {
        "pmid": None,
        "pmcid": None,
        "title": "Parameter estimation of multi-substrate biokinetic models "
                 "of lignocellulosic microbial protein systems",
        "abstract": (
            "This work presents a parameter estimation framework for "
            "multi-substrate biokinetic models of lignocellulosic microbial "
            "protein systems, specifically Fusarium venenatum A3/5 grown on "
            "glucose and xylose. Three enzyme induction models (Direct "
            "Inhibition, Enzyme Inhibition, Optimal Enzyme Production) based "
            "on Monod and Contois kinetics were calibrated against microplate "
            "growth data. Parameter estimation was performed using nonlinear "
            "least squares with random subsampling for uncertainty "
            "quantification. Model comparison via BIC, MAE, and log-likelihood "
            "identified the Optimal Enzyme Production model with Contois "
            "kinetics as providing the best fit."
        ),
        "doi": "10.1016/B978-0-443-28824-1.50427-0",
        "source": "manual",
    }

    _sep("Paper 1: ESCAPE proceedings (abstract-only path)")
    print(f"Title : {paper1['title']}")
    print(f"PMID  : {paper1['pmid']}")
    print(f"PMCID : {paper1['pmcid']}")
    print()

    ft1 = fetch_full_text(paper1)
    print(f"Full-text fetch result: {'None (expected — no PMCID)' if ft1 is None else f'{len(ft1)} chars'}")

    # ------------------------------------------------------------------
    # Paper 2: PMC Open Access
    # ------------------------------------------------------------------
    _sep("Paper 2: PMC Open Access (full-text path)")

    # First try searching PubMed to get metadata
    print("Searching PubMed for paper 2...")
    pm_results = search_pubmed(
        "High throughput parameter estimation mycoprotein lignocellulosic",
        max_results=5,
    )

    paper2 = None
    for p in pm_results:
        if p.get("pmid") == "39555020":
            paper2 = p
            break

    if paper2 is None:
        # Fallback: construct manually
        print("PubMed search didn't return exact hit. Using manual metadata.")
        paper2 = {
            "pmid": "39555020",
            "pmcid": "PMC11565039",
            "title": "High throughput parameter estimation and uncertainty analysis "
                     "applied to the production of mycoprotein from synthetic "
                     "lignocellulosic hydrolysates",
            "abstract": "",
            "doi": "10.1016/j.crfs.2024.100908",
            "source": "manual",
        }
    else:
        print(f"Found via PubMed. PMCID from search: {paper2.get('pmcid')}")

    print(f"Title : {paper2['title']}")
    print(f"PMID  : {paper2['pmid']}")
    print(f"PMCID : {paper2.get('pmcid')}")
    print()

    ft2 = fetch_full_text(paper2)
    if ft2 is None:
        print("Full-text fetch: None (unexpected!)")
    else:
        print(f"Full-text fetch: {len(ft2)} chars")
        print("\n--- Preview (first 1500 chars) ---")
        print(ft2[:1500])
        print("--- End preview ---\n")
        paper2["full_text"] = ft2

    # ------------------------------------------------------------------
    # LLM extraction (only if vLLM is available)
    # ------------------------------------------------------------------
    _sep("LLM Extraction")

    client = LLMClient()
    if not client.is_available():
        print(f"vLLM endpoint not available at {client.base_url}")
        print("Skipping LLM extraction. Full-text fetch results are above.")
        return

    print(f"vLLM endpoint available at {client.base_url}")
    print()

    for label, paper in [("Paper 1 (abstract-only)", paper1), ("Paper 2 (full-text)", paper2)]:
        print(f"--- {label} ---")
        params = client.extract_params(paper)
        if params:
            print(f"Extracted {len(params)} parameters:")
            print(json.dumps(params, indent=2))
        else:
            print("No parameters extracted.")
        print()


if __name__ == "__main__":
    main()

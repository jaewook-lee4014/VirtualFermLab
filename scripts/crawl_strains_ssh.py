#!/usr/bin/env python3
"""Batch literature crawling via the remote VirtualFermLab discovery API.

Calls the /api/discovery/start endpoint on the remote server for each
target strain. The server handles paper search, full-text retrieval,
LLM-based parameter extraction, taxonomy matching, and profile building.

Usage:
    python scripts/crawl_strains.py
    python scripts/crawl_strains.py --strains "Fusarium oxysporum,Trichoderma reesei"
    python scripts/crawl_strains.py --api-host localhost --api-port 8080  # if SSH-tunnelled
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("crawl_strains.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ── Target strains (30 total: 15 mandatory + 15 new candidates) ────
DEFAULT_STRAINS = [
    # ── Mandatory 15 (user-specified) ──
    "Fusarium oxysporum",
    "Fusarium graminearum",
    "Trichoderma reesei",
    "Aspergillus niger",
    "Aspergillus oryzae",
    "Neurospora crassa",
    "Rhizopus oryzae",
    "Mucor indicus",
    "Paecilomyces variotii",
    "Myceliophthora thermophila",
    "Penicillium cyclopium",
    "Rhizopus chinensis",
    "Candida intermedia",
    "Aspergillus fumigatus",
    "Torula sp.",
    # ── Tier 1: published growth kinetics ──
    "Fusarium venenatum",
    "Mucor circinelloides",
    "Neurospora intermedia",
    "Aspergillus terreus",
    "Penicillium brevicompactum",
    "Umbelopsis isabellina",
    "Rhizopus oligosporus",
    # ── Tier 2: SCP demonstrated, some kinetics ──
    "Trichoderma harzianum",
    "Aspergillus awamori",
    "Pleurotus ostreatus",
    "Monascus purpureus",
    "Aspergillus nidulans",
    "Fusarium solani",
    # ── Tier 3: supplementary ──
    "Lichtheimia corymbifera",
    "Rhizomucor pusillus",
]

SSH_KEY = "~/.ssh/LLM4param-home.pem"
SSH_HOST = "ubuntu@10.211.117.207"


def _ssh_curl(url: str, method: str = "GET", data: dict | None = None) -> dict | None:
    """Execute curl on the remote server via SSH and return parsed JSON."""
    if method == "POST" and data:
        curl_cmd = (
            f"curl -s -X POST {url} "
            f"-H 'Content-Type: application/json' "
            f"-d '{json.dumps(data)}'"
        )
    else:
        curl_cmd = f"curl -s {url}"

    ssh_cmd = [
        "ssh", "-i", SSH_KEY,
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=no",
        SSH_HOST,
        curl_cmd,
    ]
    try:
        result = subprocess.run(
            ssh_cmd, capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            logger.error("SSH failed: %s", result.stderr)
            return None
        if not result.stdout.strip():
            return None
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        logger.error("SSH+curl timed out for %s", url)
        return None
    except json.JSONDecodeError:
        logger.error("Invalid JSON response: %s", result.stdout[:200])
        return None


def discover_strain(strain: str, api_base: str) -> dict | None:
    """Start discovery for a strain, poll until complete, return result."""
    logger.info("Starting discovery for: %s", strain)

    # 1. Start
    resp = _ssh_curl(
        f"{api_base}/api/discovery/start",
        method="POST",
        data={"strain_name": strain},
    )
    if not resp or "task_id" not in resp:
        logger.error("  Failed to start discovery: %s", resp)
        return None

    task_id = resp["task_id"]
    logger.info("  Task started: %s", task_id)

    # 2. Poll status
    while True:
        time.sleep(30)
        status = _ssh_curl(f"{api_base}/api/discovery/status/{task_id}")
        if not status:
            logger.warning("  Status check failed, retrying...")
            continue

        st = status.get("status", "unknown")
        progress = status.get("progress", 0)
        total = status.get("total", 5)
        stage = status.get("current_stage", {})
        stage_name = stage.get("stage", "") if isinstance(stage, dict) else ""

        logger.info(
            "  [%s] %d/%d — stage: %s",
            st, progress, total, stage_name,
        )

        if st == "completed":
            break
        elif st == "failed":
            logger.error("  Discovery failed: %s", status.get("error", ""))
            return None

    # 3. Fetch result
    result = _ssh_curl(f"{api_base}/api/discovery/result/{task_id}")
    if not result:
        logger.error("  Failed to fetch result")
        return None

    # Log summary
    logger.info(
        "  Result: source=%s, papers=%d, params=%d, similar=%s (%.2f)",
        result.get("source", "?"),
        result.get("papers_found", 0),
        result.get("params_extracted", 0),
        result.get("similar_strain", "none"),
        result.get("similarity_score", 0) or 0,
    )

    # Log extracted parameters
    papers = result.get("papers", [])
    for paper in papers:
        params = paper.get("params", [])
        if params:
            logger.info("    Paper: %s", paper.get("title", "?")[:70])
            for p in params:
                logger.info(
                    "      %s: %s = %s %s (substrate=%s)",
                    p.get("strain_name", ""),
                    p.get("parameter_name", ""),
                    p.get("value", ""),
                    p.get("unit", ""),
                    p.get("substrate", "N/A"),
                )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Crawl literature for kinetic parameters via remote discovery API"
    )
    parser.add_argument(
        "--strains", type=str, default=None,
        help="Comma-separated strain names (default: 10 target strains)",
    )
    parser.add_argument(
        "--api-base", type=str, default="http://localhost:8080",
        help="API base URL on the remote server (default: http://localhost:8080)",
    )
    args = parser.parse_args()

    strains = (
        [s.strip() for s in args.strains.split(",")]
        if args.strains
        else DEFAULT_STRAINS
    )

    api_base = args.api_base
    logger.info("API base: %s (via SSH to %s)", api_base, SSH_HOST)
    logger.info("Target strains (%d): %s", len(strains), strains)

    # Verify connectivity
    test = _ssh_curl(f"{api_base}/api/strains")
    if not test:
        logger.error("Cannot reach API at %s via SSH", api_base)
        sys.exit(1)
    logger.info("API OK. Available strains: %s", test.get("strains", []))

    # Run discovery for each strain
    results = []
    t0 = time.monotonic()

    for i, strain in enumerate(strains, 1):
        logger.info("\n" + "=" * 60)
        logger.info(">>> [%d/%d] %s <<<", i, len(strains), strain)
        logger.info("=" * 60)

        result = discover_strain(strain, api_base)
        results.append({"strain": strain, "result": result})

        # Brief pause between strains for rate limiting
        if i < len(strains):
            time.sleep(2)

    elapsed = time.monotonic() - t0

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("CRAWLING COMPLETE — %.1f minutes", elapsed / 60)
    logger.info("=" * 60)

    total_papers = 0
    total_params = 0
    for r in results:
        res = r["result"]
        if res:
            papers = res.get("papers_found", 0)
            params = res.get("params_extracted", 0)
            total_papers += papers
            total_params += params
            logger.info(
                "  %-35s: %3d papers, %3d params, source=%s",
                r["strain"], papers, params, res.get("source", "?"),
            )
        else:
            logger.info("  %-35s: FAILED", r["strain"])

    logger.info("Total: %d papers, %d params", total_papers, total_params)

    # Save results to JSON
    output_path = "crawl_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()

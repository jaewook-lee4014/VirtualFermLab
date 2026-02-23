"""NCBI Taxonomy lineage lookup and similarity scoring."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

import requests

from virtualfermlab.discovery import db
from virtualfermlab.discovery.name_resolver import _ncbi_get, resolve_name

logger = logging.getLogger(__name__)

_NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def get_lineage(organism_name: str) -> list[str]:
    """Return the taxonomic lineage for *organism_name*.

    First resolves the name to a proper scientific name (handles
    abbreviations like ``F_venenatum_A35`` → ``Fusarium venenatum``).
    Then checks the local SQLite cache, and on a miss queries NCBI
    Taxonomy E-utilities.

    Returns an empty list on failure.
    """
    # Check cache under original name first
    cached = db.get_taxonomy(organism_name)
    if cached is not None:
        return cached["lineage"]

    # Resolve to proper scientific name
    resolved = resolve_name(organism_name)

    # Check cache under resolved name too
    if resolved != organism_name:
        cached = db.get_taxonomy(resolved)
        if cached is not None:
            # Also cache under the original name for next time
            db.save_taxonomy(cached["taxid"], organism_name, cached["lineage"], cached["rank"])
            return cached["lineage"]

    try:
        # ESearch – find taxid (use resolved name)
        resp = _ncbi_get(
            f"{_NCBI_BASE}/esearch.fcgi",
            params={"db": "taxonomy", "term": resolved, "retmode": "json"},
            timeout=15,
        )
        resp.raise_for_status()
        id_list = resp.json().get("esearchresult", {}).get("idlist", [])
        if not id_list:
            logger.warning("No NCBI Taxonomy entry for '%s' (resolved from '%s')", resolved, organism_name)
            return []

        taxid = int(id_list[0])

        # EFetch – get lineage XML
        resp = _ncbi_get(
            f"{_NCBI_BASE}/efetch.fcgi",
            params={"db": "taxonomy", "id": taxid, "retmode": "xml"},
            timeout=15,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.text)

        taxon = root.find(".//Taxon")
        if taxon is None:
            return []

        # Parse lineage from <LineageEx>
        lineage: list[str] = []
        for lin_taxon in taxon.findall(".//LineageEx/Taxon"):
            sci_name = lin_taxon.findtext("ScientificName", "")
            if sci_name:
                lineage.append(sci_name)

        # Append the organism itself
        sci = taxon.findtext("ScientificName", organism_name)
        lineage.append(sci)

        rank = taxon.findtext("Rank", "")

        # Cache under both the resolved and original names
        db.save_taxonomy(taxid, resolved, lineage, rank)
        if organism_name != resolved:
            db.save_taxonomy(taxid, organism_name, lineage, rank)
        return lineage

    except Exception:
        logger.exception("Taxonomy lookup failed for '%s' (resolved: '%s')", organism_name, resolved)
        return []


def lineage_distance(lineage_a: list[str], lineage_b: list[str]) -> float:
    """Compute a normalised distance between two lineages.

    Returns a float in ``[0, 1]`` where 0 means identical and 1 means
    completely unrelated.
    """
    if not lineage_a or not lineage_b:
        return 1.0

    shared = 0
    for a, b in zip(lineage_a, lineage_b):
        if a == b:
            shared += 1
        else:
            break

    max_len = max(len(lineage_a), len(lineage_b))
    return 1.0 - shared / max_len


def find_most_similar(
    target_name: str,
    known_strains: list[str],
) -> tuple[str | None, float]:
    """Find the known strain most taxonomically similar to *target_name*.

    Parameters
    ----------
    target_name:
        The unknown organism to match.
    known_strains:
        List of organism names that have known profiles.

    Returns
    -------
    (best_name, similarity_score) where similarity = 1 - distance.
    ``(None, 0.0)`` if no match could be established.
    """
    target_lineage = get_lineage(target_name)
    if not target_lineage:
        return None, 0.0

    best_name: str | None = None
    best_similarity = 0.0

    for name in known_strains:
        lin = get_lineage(name)
        if not lin:
            continue
        dist = lineage_distance(target_lineage, lin)
        sim = 1.0 - dist
        if sim > best_similarity:
            best_similarity = sim
            best_name = name

    return best_name, best_similarity

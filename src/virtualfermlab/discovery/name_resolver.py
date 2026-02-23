"""Resolve informal/abbreviated strain names to proper scientific names.

Uses a three-tier strategy:

1. **Local heuristic** — parse abbreviated forms like ``F_venenatum_A35``
   and expand the genus initial via EBI ENA suggest.
2. **EBI ENA Taxonomy suggest-for-submission** — robust fuzzy matching
   that handles abbreviations, partial names, and common names.
3. **NCBI ESearch spelling suggestion** — fallback for typos.

All tiers use only ``requests``; no extra packages required.
"""

from __future__ import annotations

import logging
import re
import time

import requests

from virtualfermlab.discovery import db

logger = logging.getLogger(__name__)

_ENA_BASE = "https://www.ebi.ac.uk/ena/taxonomy/rest"
_NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# NCBI rate limit: 3 requests/sec without API key.
# Track the last request time so we can throttle.
_last_ncbi_request: float = 0.0
_NCBI_MIN_INTERVAL = 0.35  # seconds between requests (~3/sec)


def _ncbi_get(url: str, **kwargs) -> requests.Response:
    """Rate-limited GET to NCBI with automatic retry on 429."""
    global _last_ncbi_request
    elapsed = time.monotonic() - _last_ncbi_request
    if elapsed < _NCBI_MIN_INTERVAL:
        time.sleep(_NCBI_MIN_INTERVAL - elapsed)
    _last_ncbi_request = time.monotonic()
    resp = requests.get(url, **kwargs)
    if resp.status_code == 429:
        # Retry once after a longer pause
        time.sleep(1.0)
        _last_ncbi_request = time.monotonic()
        resp = requests.get(url, **kwargs)
    return resp

# Cache resolved names in-memory for the lifetime of the process so we
# don't re-query for the same input within one pipeline run.
_resolve_cache: dict[str, str | None] = {}

# Common microbial genus abbreviations used in fermentation / bioprocess
# literature.  When an abbreviated name starts with one of these initials
# and the species epithet matches, the full genus is used directly.
_COMMON_GENERA: dict[str, list[str]] = {
    "A": ["Aspergillus", "Acetobacter", "Aureobasidium"],
    "B": ["Bacillus", "Brettanomyces", "Bifidobacterium"],
    "C": ["Candida", "Clostridium", "Corynebacterium", "Cryptococcus"],
    "D": ["Debaryomyces", "Deinococcus"],
    "E": ["Escherichia", "Enterococcus", "Enterobacter"],
    "F": ["Fusarium"],
    "G": ["Geobacillus", "Gluconobacter"],
    "H": ["Hansenula", "Halobacterium"],
    "K": ["Kluyveromyces", "Komagataella", "Klebsiella"],
    "L": ["Lactobacillus", "Lactococcus", "Leuconostoc"],
    "M": ["Methanobacterium", "Mortierella", "Mucor", "Myceliophthora"],
    "N": ["Neurospora"],
    "O": ["Ogataea"],
    "P": ["Pichia", "Pseudomonas", "Penicillium", "Propionibacterium"],
    "R": ["Rhizopus", "Rhodotorula", "Rhodococcus"],
    "S": ["Saccharomyces", "Streptomyces", "Schizosaccharomyces", "Streptococcus"],
    "T": ["Trichoderma", "Thermus", "Torulaspora"],
    "Y": ["Yarrowia"],
    "Z": ["Zymomonas", "Zygosaccharomyces"],
}


# ------------------------------------------------------------------
# Heuristic parser for abbreviated strain identifiers
# ------------------------------------------------------------------

# Matches patterns like:
#   F_venenatum_A35  →  genus_initial="F", species="venenatum", strain="A35"
#   S_cerevisiae_S288C
#   P_pastoris_GS115
_ABBREV_RE = re.compile(
    r"^([A-Z])[\s_.-]+"           # single capital letter = genus initial
    r"([a-z][a-z_]+?)"            # species epithet (lowercase)
    r"(?:[\s_.-]+(.+))?$"         # optional strain designator
)


def _parse_abbreviated(name: str) -> tuple[str, str] | None:
    """Try to parse an abbreviated strain name.

    Returns ``(species_query, strain_part)`` or ``None`` if the format
    doesn't match.  ``species_query`` will be something like
    ``"F venenatum"`` — still needs genus expansion.
    """
    m = _ABBREV_RE.match(name.strip())
    if m is None:
        return None
    genus_initial = m.group(1)
    species = m.group(2).replace("_", " ")
    strain = (m.group(3) or "").replace("_", " ").strip()
    return f"{genus_initial} {species}", strain


# ------------------------------------------------------------------
# EBI ENA taxonomy suggestion
# ------------------------------------------------------------------


def _ena_suggest(query: str) -> str | None:
    """Query EBI ENA ``suggest-for-submission`` and return the best scientific name."""
    try:
        resp = requests.get(
            f"{_ENA_BASE}/suggest-for-submission/{requests.utils.quote(query)}",
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not data:
            return None
        # The endpoint returns a list; first hit is best match.
        return data[0].get("scientificName")
    except Exception:
        logger.debug("ENA suggest failed for '%s'", query)
        return None


def _ena_scientific_name(query: str) -> str | None:
    """Try an exact scientific-name lookup on ENA."""
    try:
        resp = requests.get(
            f"{_ENA_BASE}/scientific-name/{requests.utils.quote(query)}",
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        if isinstance(data, list) and data:
            return data[0].get("scientificName")
        if isinstance(data, dict):
            return data.get("scientificName")
        return None
    except Exception:
        logger.debug("ENA scientific-name lookup failed for '%s'", query)
        return None


# ------------------------------------------------------------------
# NCBI spelling suggestion
# ------------------------------------------------------------------


def _ncbi_spell(query: str) -> str | None:
    """Use NCBI ESpell to get a corrected spelling."""
    try:
        resp = _ncbi_get(
            f"{_NCBI_BASE}/espell.fcgi",
            params={"db": "taxonomy", "term": query},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        # Response is XML — extract <CorrectedQuery>
        import xml.etree.ElementTree as ET

        root = ET.fromstring(resp.text)
        corrected = root.findtext(".//CorrectedQuery")
        if corrected and corrected.lower() != query.lower():
            return corrected
        return None
    except Exception:
        logger.debug("NCBI ESpell failed for '%s'", query)
        return None


def _ncbi_taxonomy_search(query: str) -> str | None:
    """Search NCBI Taxonomy and return the scientific name of the first hit.

    Useful for wildcard queries like ``"Fusarium venenatum"`` or even
    ``"F* venenatum"`` (genus-initial wildcard).
    """
    import xml.etree.ElementTree as ET

    try:
        # Step 1: ESearch to get taxid
        resp = _ncbi_get(
            f"{_NCBI_BASE}/esearch.fcgi",
            params={"db": "taxonomy", "term": query, "retmode": "json"},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        id_list = resp.json().get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return None

        # Step 2: EFetch to get scientific name
        resp = _ncbi_get(
            f"{_NCBI_BASE}/efetch.fcgi",
            params={"db": "taxonomy", "id": id_list[0], "retmode": "xml"},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        root = ET.fromstring(resp.text)
        sci_name = root.findtext(".//Taxon/ScientificName")
        return sci_name
    except Exception:
        logger.debug("NCBI taxonomy search failed for '%s'", query)
        return None


def _resolve_abbreviated(genus_initial: str, species: str) -> str | None:
    """Expand an abbreviated name like ``("F", "venenatum")`` to a full binomial.

    Uses a multi-strategy approach combining ENA and NCBI to handle
    abbreviated genus initials robustly.
    """
    genus_initial_upper = genus_initial[0].upper()
    # Two candidate lists: high-confidence (from "initial species" query)
    # and lower-confidence (from species-only query).
    hi_candidates: list[str] = []
    lo_candidates: list[str] = []
    genera_seen: list[str] = []

    # ---- Strategy 0: common genera lookup table ----
    # Try well-known microbial genera matching the initial.
    for genus in _COMMON_GENERA.get(genus_initial_upper, []):
        constructed = f"{genus} {species}"
        ncbi_name = _ncbi_taxonomy_search(constructed)
        if ncbi_name:
            return ncbi_name

    def _collect_hits(resp_json: list, target: list[str]) -> None:
        for hit in resp_json:
            sci_name = hit.get("scientificName", "")
            if sci_name and sci_name[0].upper() == genus_initial_upper:
                parts = sci_name.split()
                if len(parts) >= 2:
                    binomial = f"{parts[0]} {parts[1]}"
                    if binomial not in target and binomial not in hi_candidates and binomial not in lo_candidates:
                        target.append(binomial)
                    if parts[0] not in genera_seen:
                        genera_seen.append(parts[0])

    # ---- Strategy A (high confidence): ENA suggest "initial species" ----
    # Most specific query — "E coli" → "Escherichia coli".
    try:
        query_a = f"{genus_initial} {species}"
        resp = requests.get(
            f"{_ENA_BASE}/suggest-for-submission/{requests.utils.quote(query_a)}",
            timeout=10,
        )
        if resp.status_code == 200:
            _collect_hits(resp.json(), hi_candidates)
    except Exception:
        logger.debug("ENA suggest for '%s %s' failed", genus_initial, species)

    # ---- Strategy B (lower confidence): ENA suggest species epithet only ----
    # Less specific — "venenatum" → filter by initial.
    try:
        resp = requests.get(
            f"{_ENA_BASE}/suggest-for-submission/{requests.utils.quote(species)}",
            timeout=10,
        )
        if resp.status_code == 200:
            _collect_hits(resp.json(), lo_candidates)
    except Exception:
        logger.debug("ENA suggest for species '%s' failed", species)

    # ---- Strategy C: ENA genus lookup ----
    # Discover genus names starting with the initial, then construct
    # "<Genus> <species>" for NCBI validation later.
    try:
        resp = requests.get(
            f"{_ENA_BASE}/suggest-for-submission/{requests.utils.quote(genus_initial)}",
            timeout=10,
        )
        if resp.status_code == 200:
            for hit in resp.json():
                sci_name = hit.get("scientificName", "")
                if sci_name:
                    genus = sci_name.split()[0]
                    if genus[0].upper() == genus_initial_upper and genus not in genera_seen:
                        genera_seen.append(genus)
    except Exception:
        pass

    # ---- Validate high-confidence candidates (from "initial species" query) ----
    for candidate in hi_candidates:
        ncbi_name = _ncbi_taxonomy_search(candidate)
        if ncbi_name:
            return ncbi_name

    # ---- Try "<Genus> <species>" for each discovered genus via NCBI ----
    # Do this BEFORE validating low-confidence candidates, because
    # genus-constructed names using the original species epithet are
    # more likely correct than ENA hits from a species-only search.
    already_tried = set(hi_candidates + lo_candidates)
    for genus in genera_seen:
        constructed = f"{genus} {species}"
        if constructed not in already_tried:
            ncbi_name = _ncbi_taxonomy_search(constructed)
            if ncbi_name:
                return ncbi_name

    # ---- Validate low-confidence candidates ----
    for candidate in lo_candidates:
        ncbi_name = _ncbi_taxonomy_search(candidate)
        if ncbi_name:
            return ncbi_name

    # Return first candidate even without NCBI validation
    all_candidates = hi_candidates + lo_candidates
    if all_candidates:
        return all_candidates[0]

    return None


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def resolve_name(raw_name: str) -> str:
    """Resolve *raw_name* to a proper NCBI-compatible scientific name.

    Returns the best scientific name found, or the original *raw_name*
    unchanged if resolution fails at every tier.

    Resolution tiers (tried in order):

    1. Check in-memory and SQLite caches.
    2. If the name looks abbreviated (e.g. ``F_venenatum_A35``), parse it
       and expand the genus via EBI ENA suggest.
    3. Direct EBI ENA scientific-name lookup (handles clean binomials).
    4. EBI ENA suggest-for-submission (fuzzy, handles common names).
    5. NCBI ESpell (spelling correction fallback).
    """
    raw_name = raw_name.strip()
    if not raw_name:
        return raw_name

    # In-memory cache
    if raw_name in _resolve_cache:
        cached = _resolve_cache[raw_name]
        return cached if cached is not None else raw_name

    # SQLite cache — if we already have taxonomy for this exact name
    cached_tax = db.get_taxonomy(raw_name)
    if cached_tax is not None:
        _resolve_cache[raw_name] = raw_name
        return raw_name

    resolved: str | None = None

    # Tier 1: abbreviated format heuristic (e.g. "F_venenatum_A35")
    parsed = _parse_abbreviated(raw_name)
    if parsed is not None:
        partial_query, _strain = parsed  # e.g. "F venenatum"
        genus_initial = partial_query.split()[0]
        species = " ".join(partial_query.split()[1:])
        resolved = _resolve_abbreviated(genus_initial, species)
        if resolved:
            logger.info("Resolved '%s' -> '%s' (abbreviated expansion)", raw_name, resolved)

    # Tier 2: direct ENA scientific-name lookup (clean binomials)
    if resolved is None:
        clean = raw_name.replace("_", " ")
        resolved = _ena_scientific_name(clean)
        if resolved:
            logger.info("Resolved '%s' -> '%s' (ENA scientific-name)", raw_name, resolved)

    # Tier 3: ENA suggest-for-submission (fuzzy / common names)
    if resolved is None:
        clean = raw_name.replace("_", " ")
        resolved = _ena_suggest(clean)
        if resolved:
            logger.info("Resolved '%s' -> '%s' (ENA suggest)", raw_name, resolved)

    # Tier 4: NCBI taxonomy direct search
    if resolved is None:
        clean = raw_name.replace("_", " ")
        resolved = _ncbi_taxonomy_search(clean)
        if resolved:
            logger.info("Resolved '%s' -> '%s' (NCBI taxonomy search)", raw_name, resolved)

    # Tier 5: NCBI spelling correction
    if resolved is None:
        clean = raw_name.replace("_", " ")
        corrected = _ncbi_spell(clean)
        if corrected:
            resolved = corrected
            logger.info("Resolved '%s' -> '%s' (NCBI ESpell)", raw_name, resolved)

    _resolve_cache[raw_name] = resolved
    return resolved if resolved is not None else raw_name

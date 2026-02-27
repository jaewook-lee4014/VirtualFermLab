"""Literature search via PubMed, Semantic Scholar, Europe PMC, and OpenAlex.

Includes PubTator 3.0 pre-screening, Unpaywall OA lookup, and BioC full-text.
"""

from __future__ import annotations

import json as _json
import logging
import os
import queue
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import requests

from virtualfermlab.discovery import db

logger = logging.getLogger(__name__)

_NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_S2_BASE = "https://api.semanticscholar.org/graph/v1"
_EPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
_OPENALEX_BASE = "https://api.openalex.org"
_PUBTATOR_BASE = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"
_UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
_BIOC_BASE = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi"

_UNPAYWALL_EMAIL = os.environ.get("UNPAYWALL_EMAIL", "virtualfermlab@example.com")

_SENTINEL = object()

_FULLTEXT_MAX_CHARS = 30000  # generous limit; chunking happens in LLM layer

# --------------------------------------------------------------------------
# PubMed
# --------------------------------------------------------------------------


def search_pubmed(query: str, max_results: int = 20) -> list[dict]:
    """Search PubMed via NCBI E-utilities and return paper metadata."""
    results: list[dict] = []
    try:
        # ESearch
        resp = requests.get(
            f"{_NCBI_BASE}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
            },
            timeout=30,
        )
        resp.raise_for_status()
        id_list = resp.json().get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return results

        # Rate-limit: 3 req/sec without API key
        time.sleep(0.35)

        # EFetch – get summaries
        resp = requests.get(
            f"{_NCBI_BASE}/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(id_list),
                "rettype": "abstract",
                "retmode": "xml",
            },
            timeout=30,
        )
        resp.raise_for_status()

        # Parse XML response
        root = ET.fromstring(resp.text)
        for article in root.findall(".//PubmedArticle"):
            medline = article.find("MedlineCitation")
            if medline is None:
                continue
            pmid_el = medline.find("PMID")
            art = medline.find("Article")
            if art is None:
                continue

            title_el = art.find("ArticleTitle")
            journal_el = art.find("Journal/Title")
            year_el = art.find("Journal/JournalIssue/PubDate/Year")
            abstract_el = art.find("Abstract/AbstractText")

            # Authors
            authors_list: List[str] = []
            author_list_el = art.find("AuthorList")
            if author_list_el is not None:
                for auth in author_list_el.findall("Author"):
                    last = auth.findtext("LastName", "")
                    fore = auth.findtext("ForeName", "")
                    if last:
                        authors_list.append(f"{last} {fore}".strip())

            # DOI
            doi = None
            for eid in art.findall("ELocationID"):
                if eid.get("EIdType") == "doi":
                    doi = eid.text

            # PMCID (from PubmedData/ArticleIdList)
            pmcid = None
            pubmed_data = article.find("PubmedData")
            if pubmed_data is not None:
                for aid in pubmed_data.findall("ArticleIdList/ArticleId"):
                    if aid.get("IdType") == "pmc":
                        pmcid = aid.text
                        break

            results.append({
                "pmid": pmid_el.text if pmid_el is not None else None,
                "pmcid": pmcid,
                "title": title_el.text if title_el is not None else "",
                "authors": "; ".join(authors_list),
                "journal": journal_el.text if journal_el is not None else "",
                "year": int(year_el.text) if year_el is not None and year_el.text else None,
                "abstract": abstract_el.text if abstract_el is not None else "",
                "doi": doi,
                "source": "pubmed",
            })
    except Exception:
        logger.exception("PubMed search failed for query: %s", query)

    return results


# --------------------------------------------------------------------------
# Semantic Scholar
# --------------------------------------------------------------------------


def search_semantic_scholar(query: str, max_results: int = 20) -> list[dict]:
    """Search Semantic Scholar Academic Graph API."""
    results: list[dict] = []
    try:
        resp = requests.get(
            f"{_S2_BASE}/paper/search",
            params={
                "query": query,
                "limit": min(max_results, 100),
                "fields": "title,authors,year,abstract,externalIds,journal",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])

        for paper in data:
            ext = paper.get("externalIds") or {}
            authors_raw = paper.get("authors") or []
            authors_str = "; ".join(a.get("name", "") for a in authors_raw)
            journal_info = paper.get("journal") or {}

            results.append({
                "pmid": ext.get("PubMed"),
                "title": paper.get("title", ""),
                "authors": authors_str,
                "journal": journal_info.get("name", ""),
                "year": paper.get("year"),
                "abstract": paper.get("abstract") or "",
                "doi": ext.get("DOI"),
                "source": "semantic_scholar",
            })
    except Exception:
        logger.exception("Semantic Scholar search failed for query: %s", query)

    return results


# --------------------------------------------------------------------------
# Europe PMC
# --------------------------------------------------------------------------


def search_europe_pmc(query: str, max_results: int = 20) -> list[dict]:
    """Search Europe PMC REST API.

    Europe PMC covers PubMed, PMC, and additional European sources
    (e.g. EBI, Agricola) that may not appear in NCBI searches.
    """
    results: list[dict] = []
    try:
        resp = requests.get(
            f"{_EPMC_BASE}/search",
            params={
                "query": query,
                "resultType": "core",
                "format": "json",
                "pageSize": min(max_results, 100),
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json().get("resultList", {}).get("result", [])

        for paper in data:
            # Authors
            authors_raw = paper.get("authorString", "")
            results.append({
                "pmid": paper.get("pmid"),
                "pmcid": paper.get("pmcid"),
                "title": paper.get("title", ""),
                "authors": authors_raw,
                "journal": paper.get("journalTitle", ""),
                "year": int(paper["pubYear"]) if paper.get("pubYear") else None,
                "abstract": paper.get("abstractText") or "",
                "doi": paper.get("doi"),
                "source": "europe_pmc",
            })
    except Exception:
        logger.exception("Europe PMC search failed for query: %s", query)

    return results


# --------------------------------------------------------------------------
# OpenAlex
# --------------------------------------------------------------------------


def search_openalex(query: str, max_results: int = 20) -> list[dict]:
    """Search OpenAlex API (250M+ works, successor to Microsoft Academic).

    Uses polite pool (mailto header) for better rate limits.
    """
    results: list[dict] = []
    try:
        resp = requests.get(
            f"{_OPENALEX_BASE}/works",
            params={
                "search": query,
                "per_page": min(max_results, 50),
                "select": "id,doi,title,authorships,publication_year,"
                          "primary_location,abstract_inverted_index",
            },
            headers={"User-Agent": f"VirtualFermLab/1.0 (mailto:{_UNPAYWALL_EMAIL})"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json().get("results", [])

        for work in data:
            # Reconstruct abstract from inverted index
            abstract = _openalex_invert_abstract(
                work.get("abstract_inverted_index")
            )
            # Authors
            authors_parts = []
            for authorship in (work.get("authorships") or []):
                author = authorship.get("author", {})
                name = author.get("display_name", "")
                if name:
                    authors_parts.append(name)
            # Journal
            loc = work.get("primary_location") or {}
            source = loc.get("source") or {}
            journal = source.get("display_name", "")
            # DOI (strip https://doi.org/ prefix)
            doi_raw = work.get("doi") or ""
            doi = doi_raw.replace("https://doi.org/", "") if doi_raw else None

            results.append({
                "pmid": None,
                "title": work.get("title", ""),
                "authors": "; ".join(authors_parts),
                "journal": journal,
                "year": work.get("publication_year"),
                "abstract": abstract,
                "doi": doi,
                "source": "openalex",
            })
    except Exception:
        logger.exception("OpenAlex search failed for query: %s", query)

    return results


def _openalex_invert_abstract(inverted_index: dict | None) -> str:
    """Convert OpenAlex inverted abstract index back to plain text."""
    if not inverted_index:
        return ""
    # {word: [pos1, pos2, ...], ...} → reconstruct by position
    word_positions: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(w for _, w in word_positions)


# --------------------------------------------------------------------------
# PubTator 3.0 — entity-based pre-screening
# --------------------------------------------------------------------------


def pubtator_check_relevance(
    pmids: list[str],
    target_species: list[str],
    target_chemicals: list[str] | None = None,
) -> dict[str, dict]:
    """Check papers for species/chemical entity mentions via PubTator 3.0.

    Parameters
    ----------
    pmids : list[str]
        PubMed IDs to check.
    target_species : list[str]
        Species names to look for (case-insensitive substring match).
    target_chemicals : list[str] or None
        Chemical/substrate names to look for. If None, defaults to common
        fermentation substrates.

    Returns
    -------
    dict
        ``{pmid: {"species": [...], "chemicals": [...], "relevant": bool}}``
    """
    if target_chemicals is None:
        target_chemicals = [
            "glucose", "xylose", "fructose", "sucrose", "cellulose",
            "starch", "lactose", "arabinose", "maltose",
        ]

    target_sp_lower = [s.lower() for s in target_species]
    target_ch_lower = [c.lower() for c in target_chemicals]
    result: dict[str, dict] = {}

    # PubTator accepts up to 100 PMIDs per request
    batch_size = 100
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        pmid_str = ",".join(batch)
        try:
            time.sleep(0.5)
            resp = requests.get(
                f"{_PUBTATOR_BASE}/publications/export/biocjson",
                params={"pmids": pmid_str},
                timeout=60,
            )
            if resp.status_code == 404:
                logger.debug("PubTator: no annotations for PMIDs %s", pmid_str[:50])
                continue
            resp.raise_for_status()
            # Response may be multiple JSON objects (one per PMID), newline-delimited
            for line in resp.text.strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    doc = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                pmid = doc.get("pmid") or doc.get("_id", "")
                species_found: list[str] = []
                chemicals_found: list[str] = []
                for passage in doc.get("passages", []):
                    for ann in passage.get("annotations", []):
                        infons = ann.get("infons", {})
                        ann_type = infons.get("type", "")
                        ann_text = ann.get("text", "").lower()
                        if ann_type == "Species":
                            if any(sp in ann_text for sp in target_sp_lower):
                                species_found.append(ann.get("text", ""))
                        elif ann_type == "Chemical":
                            if any(ch in ann_text for ch in target_ch_lower):
                                chemicals_found.append(ann.get("text", ""))
                result[str(pmid)] = {
                    "species": list(set(species_found)),
                    "chemicals": list(set(chemicals_found)),
                    "relevant": bool(species_found),  # at minimum the species must match
                }
        except Exception:
            logger.exception("PubTator request failed for batch starting at %d", i)

    return result


# --------------------------------------------------------------------------
# Unpaywall — OA full-text discovery
# --------------------------------------------------------------------------


def unpaywall_find_oa(doi: str) -> str | None:
    """Look up an OA PDF/HTML URL for a DOI via Unpaywall.

    Returns the best OA URL, or ``None`` if the paper is not available OA.
    """
    if not doi:
        return None
    try:
        time.sleep(0.2)
        resp = requests.get(
            f"{_UNPAYWALL_BASE}/{doi}",
            params={"email": _UNPAYWALL_EMAIL},
            timeout=15,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        if not data.get("is_oa"):
            return None
        best = data.get("best_oa_location") or {}
        return best.get("url_for_pdf") or best.get("url") or None
    except Exception:
        logger.debug("Unpaywall lookup failed for DOI %s", doi)
        return None


# --------------------------------------------------------------------------
# BioC API — structured full-text from PMC OA
# --------------------------------------------------------------------------


def fetch_full_text_bioc(pmcid: str) -> str | None:
    """Fetch structured full-text from PMC via BioC API.

    Returns Results/Discussion sections and tables as plain text,
    or ``None`` if the paper is not available.
    """
    if not pmcid:
        return None
    # Normalise: ensure "PMC" prefix is absent for BioC API
    pmc_num = pmcid.replace("PMC", "")
    try:
        time.sleep(0.35)
        resp = requests.get(
            f"{_BIOC_BASE}/BioC_json/PMC{pmc_num}/unicode",
            timeout=60,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.debug("BioC fetch failed for %s", pmcid)
        return None

    # Extract relevant passages
    parts: list[str] = []
    for doc in data.get("documents", []):
        for passage in doc.get("passages", []):
            infons = passage.get("infons", {})
            section = infons.get("section_type", "").lower()
            sec_name = infons.get("section", "").lower()
            ptype = infons.get("type", "").lower()

            is_relevant = (
                section in ("results", "discussion", "results and discussion")
                or any(kw in sec_name for kw in ("result", "discussion", "kinetic"))
                or ptype == "table"
            )
            if is_relevant:
                text = passage.get("text", "").strip()
                if text:
                    parts.append(text)

    if not parts:
        return None
    full = "\n\n".join(parts)
    if len(full) > _FULLTEXT_MAX_CHARS:
        full = full[:_FULLTEXT_MAX_CHARS] + "\n[truncated]"
    return full


# --------------------------------------------------------------------------
# Full-text retrieval via PMC
# --------------------------------------------------------------------------


def _pmid_to_pmcid(pmid: str) -> str | None:
    """Convert a PubMed ID to a PMC ID using the NCBI ELink API."""
    try:
        time.sleep(0.35)
        resp = requests.get(
            f"{_NCBI_BASE}/elink.fcgi",
            params={
                "dbfrom": "pubmed",
                "db": "pmc",
                "id": pmid,
                "retmode": "json",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        linksets = data.get("linksets", [])
        if not linksets:
            return None
        linksetdbs = linksets[0].get("linksetdbs", [])
        for lsdb in linksetdbs:
            if lsdb.get("dbto") == "pmc":
                links = lsdb.get("links", [])
                if links:
                    return f"PMC{links[0]}"
        return None
    except Exception:
        logger.debug("ELink PMID→PMCID failed for %s", pmid)
        return None


def _parse_jats_sections(xml_text: str) -> str:
    """Extract Results, Discussion text and Tables from JATS XML.

    Returns a plain-text / markdown representation of the relevant sections.
    """
    root = ET.fromstring(xml_text)
    parts: list[str] = []

    # Helper to recursively get all text from an element
    def _itertext(el: ET.Element) -> str:
        return "".join(el.itertext()).strip()

    # --- Sections: Results and Discussion ---
    body = root.find(".//body")
    if body is not None:
        for sec in body.iter("sec"):
            sec_type = (sec.get("sec-type") or "").lower()
            title_el = sec.find("title")
            title_text = (_itertext(title_el) if title_el is not None else "").lower()

            is_relevant = sec_type in ("results", "discussion", "results and discussion") or any(
                kw in title_text for kw in ("result", "discussion")
            )
            if is_relevant:
                heading = _itertext(title_el) if title_el is not None else sec_type.title()
                paragraphs = []
                for p in sec.findall("p"):
                    paragraphs.append(_itertext(p))
                if paragraphs:
                    parts.append(f"## {heading}\n" + "\n".join(paragraphs))

    # --- Tables ---
    for tw in root.iter("table-wrap"):
        caption_el = tw.find("caption")
        label_el = tw.find("label")
        caption_parts = []
        if label_el is not None:
            caption_parts.append(_itertext(label_el))
        if caption_el is not None:
            caption_parts.append(_itertext(caption_el))
        caption = " ".join(caption_parts).strip()

        table = tw.find(".//table")
        if table is None:
            continue

        rows: list[list[str]] = []
        for tr in table.iter("tr"):
            cells = []
            for cell in tr:
                if cell.tag in ("th", "td"):
                    cells.append(_itertext(cell))
            if cells:
                rows.append(cells)

        if rows:
            table_lines = []
            if caption:
                table_lines.append(caption)
            # Header row
            table_lines.append("| " + " | ".join(rows[0]) + " |")
            table_lines.append("| " + " | ".join("---" for _ in rows[0]) + " |")
            for row in rows[1:]:
                # Pad or trim to match header width
                padded = row + [""] * (len(rows[0]) - len(row))
                table_lines.append("| " + " | ".join(padded[: len(rows[0])]) + " |")
            parts.append("\n".join(table_lines))

    text = "\n\n".join(parts)
    if not text:
        return ""
    if len(text) > _FULLTEXT_MAX_CHARS:
        text = text[:_FULLTEXT_MAX_CHARS] + "\n[truncated]"
    return text


def fetch_full_text(paper: dict) -> str | None:
    """Retrieve full-text with multi-source fallback.

    Tries sources in order:
    1. PMC JATS XML (NCBI E-Utilities) — most structured
    2. BioC API (PMC OA) — paragraph-level with annotations
    3. Unpaywall — OA PDF URL (returns URL string, not text)

    Returns the extracted text, or ``None`` if no source succeeded.
    """
    pmcid = paper.get("pmcid")

    # If no PMCID stored, try converting from PMID
    if not pmcid and paper.get("pmid"):
        pmcid = _pmid_to_pmcid(paper["pmid"])

    # Strategy 1: PMC JATS XML (best structure for tables)
    if pmcid:
        try:
            time.sleep(0.35)
            resp = requests.get(
                f"{_NCBI_BASE}/efetch.fcgi",
                params={"db": "pmc", "id": pmcid, "retmode": "xml"},
                timeout=60,
            )
            resp.raise_for_status()
            text = _parse_jats_sections(resp.text)
            if text:
                logger.debug("Full-text via PMC JATS for %s", pmcid)
                return text
        except Exception:
            logger.debug("PMC EFetch failed for %s", pmcid)

    # Strategy 2: BioC API (structured paragraphs)
    if pmcid:
        text = fetch_full_text_bioc(pmcid)
        if text:
            logger.debug("Full-text via BioC for %s", pmcid)
            return text

    # Strategy 3: Unpaywall OA URL
    doi = paper.get("doi")
    if doi:
        oa_url = unpaywall_find_oa(doi)
        if oa_url:
            logger.info("OA URL found via Unpaywall for DOI %s: %s", doi, oa_url)
            # Store the URL so the caller can fetch the PDF if needed
            paper["oa_url"] = oa_url

    return None


# --------------------------------------------------------------------------
# Combined search
# --------------------------------------------------------------------------


def search_papers(
    strain_name: str,
    substrates: list[str] | None = None,
    max_results: int = 20,
) -> list[dict]:
    """Search PubMed, Semantic Scholar, Europe PMC, and OpenAlex.

    Results are deduplicated by DOI and persisted to the database.

    Parameters
    ----------
    strain_name:
        Organism or strain name to search for.
    substrates:
        Optional substrates to refine the query.
    max_results:
        Max results per source.

    Returns
    -------
    list of paper dicts.
    """
    query_parts = [f'"{strain_name}"', "fermentation", "growth kinetics"]
    if substrates:
        query_parts.extend(substrates)
    query = " ".join(query_parts)

    logger.info("Searching papers: %s", query)

    pubmed_papers = search_pubmed(query, max_results)
    s2_papers = search_semantic_scholar(query, max_results)
    epmc_papers = search_europe_pmc(query, max_results)
    oa_papers = search_openalex(query, max_results)

    # Deduplicate by DOI
    seen_dois: set[str] = set()
    combined: list[dict] = []

    for paper in pubmed_papers + s2_papers + epmc_papers + oa_papers:
        doi = paper.get("doi")
        if doi and doi in seen_dois:
            continue
        if doi:
            seen_dois.add(doi)
        combined.append(paper)

    # Persist to SQLite
    for paper in combined:
        try:
            db.save_paper(paper)
        except Exception:
            logger.warning("Failed to save paper: %s", paper.get("title", "?"))

    logger.info(
        "Found %d papers (PubMed=%d, S2=%d, EPMC=%d, OpenAlex=%d)",
        len(combined), len(pubmed_papers), len(s2_papers),
        len(epmc_papers), len(oa_papers),
    )
    return combined


# --------------------------------------------------------------------------
# Parallel search with queue producer
# --------------------------------------------------------------------------


def search_papers_into_queue(
    strain_name: str,
    paper_queue: queue.Queue,
    substrates: list[str] | None = None,
    max_results: int = 20,
) -> int:
    """Search all API sources in parallel and push papers into *paper_queue*.

    Each unique paper (by DOI) is pushed as it becomes available.  When all
    sources have completed, a :data:`_SENTINEL` is pushed so the consumer
    knows to stop.

    Returns the total number of papers pushed (excluding sentinel).
    """
    query_parts = [f'"{strain_name}"', "fermentation", "growth kinetics"]
    if substrates:
        query_parts.extend(substrates)
    query = " ".join(query_parts)

    logger.info("Searching papers (parallel): %s", query)

    seen_dois: set[str] = set()
    seen_lock = threading.Lock()
    pushed = 0

    search_fns = {
        "pubmed": lambda: search_pubmed(query, max_results),
        "semantic_scholar": lambda: search_semantic_scholar(query, max_results),
        "europe_pmc": lambda: search_europe_pmc(query, max_results),
        "openalex": lambda: search_openalex(query, max_results),
    }

    with ThreadPoolExecutor(max_workers=len(search_fns)) as executor:
        futures = {
            executor.submit(fn): name for name, fn in search_fns.items()
        }

        for future in as_completed(futures):
            source_name = futures[future]
            try:
                papers = future.result()
            except Exception:
                logger.exception("Search source %s failed", source_name)
                continue

            for paper in papers:
                doi = paper.get("doi")
                with seen_lock:
                    if doi and doi in seen_dois:
                        continue
                    if doi:
                        seen_dois.add(doi)

                # Persist to DB
                try:
                    db.save_paper(paper)
                except Exception:
                    logger.warning("Failed to save paper: %s", paper.get("title", "?"))

                paper_queue.put(paper)
                pushed += 1

    paper_queue.put(_SENTINEL)
    logger.info("Parallel search complete: pushed %d papers", pushed)
    return pushed

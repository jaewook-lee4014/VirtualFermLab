"""Literature search via PubMed and Semantic Scholar."""

from __future__ import annotations

import logging
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
    """Attempt to retrieve full-text (Results/Discussion + Tables) from PMC.

    Returns the extracted text, or ``None`` if the paper is not available
    in PMC Open Access.
    """
    pmcid = paper.get("pmcid")

    # If no PMCID stored, try converting from PMID
    if not pmcid and paper.get("pmid"):
        pmcid = _pmid_to_pmcid(paper["pmid"])

    if not pmcid:
        return None

    try:
        time.sleep(0.35)
        resp = requests.get(
            f"{_NCBI_BASE}/efetch.fcgi",
            params={
                "db": "pmc",
                "id": pmcid,
                "retmode": "xml",
            },
            timeout=60,
        )
        resp.raise_for_status()
    except Exception:
        logger.debug("PMC EFetch failed for %s", pmcid)
        return None

    text = _parse_jats_sections(resp.text)
    return text if text else None


# --------------------------------------------------------------------------
# Combined search
# --------------------------------------------------------------------------


def search_papers(
    strain_name: str,
    substrates: list[str] | None = None,
    max_results: int = 20,
) -> list[dict]:
    """Search both PubMed and Semantic Scholar, deduplicate, and persist.

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

    # Deduplicate by DOI
    seen_dois: set[str] = set()
    combined: list[dict] = []

    for paper in pubmed_papers + s2_papers:
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

    logger.info("Found %d papers (%d PubMed, %d S2)", len(combined), len(pubmed_papers), len(s2_papers))
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

"""SQLite database layer for discovered strain data."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from virtualfermlab.parameters.schema import StrainProfile

DB_PATH = Path(__file__).parent / "strains.db"

_db_lock = threading.Lock()

_CREATE_TABLES_SQL = """\
CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doi TEXT UNIQUE,
    pmid TEXT,
    pmcid TEXT,
    title TEXT NOT NULL,
    authors TEXT,
    journal TEXT,
    year INTEGER,
    abstract TEXT,
    full_text TEXT,
    source TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS extracted_params (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER REFERENCES papers(id),
    strain_name TEXT NOT NULL,
    substrate TEXT,
    parameter_name TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT,
    conditions TEXT,
    evidence TEXT,
    confidence TEXT DEFAULT 'B',
    kinetic_model TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS taxonomy_cache (
    taxid INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    lineage TEXT NOT NULL,
    rank TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS strain_profiles_cache (
    name TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    source TEXT,
    similar_to TEXT,
    similarity_score REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they do not exist.

    Also migrates existing databases by adding new columns when needed.
    """
    conn = _connect()
    try:
        conn.executescript(_CREATE_TABLES_SQL)
        conn.commit()
        # Migrate existing databases: add columns that may not exist yet
        for col, col_type in [("pmcid", "TEXT"), ("full_text", "TEXT")]:
            try:
                conn.execute(f"ALTER TABLE papers ADD COLUMN {col} {col_type}")
                conn.commit()
            except sqlite3.OperationalError:
                # Column already exists
                pass
        # Migrate extracted_params: add columns that may not exist yet
        for col, col_type in [("evidence", "TEXT"), ("kinetic_model", "TEXT")]:
            try:
                conn.execute(f"ALTER TABLE extracted_params ADD COLUMN {col} {col_type}")
                conn.commit()
            except sqlite3.OperationalError:
                pass
    finally:
        conn.close()


def save_paper(paper: dict) -> int:
    """Insert or update a paper record, return its ``id``."""
    with _db_lock:
        init_db()
        conn = _connect()
        try:
            doi = paper.get("doi")
            if doi:
                row = conn.execute("SELECT id FROM papers WHERE doi = ?", (doi,)).fetchone()
                if row:
                    return row["id"]
            cur = conn.execute(
                "INSERT INTO papers (doi, pmid, pmcid, title, authors, journal, year, abstract, full_text, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doi,
                    paper.get("pmid"),
                    paper.get("pmcid"),
                    paper.get("title", ""),
                    paper.get("authors", ""),
                    paper.get("journal", ""),
                    paper.get("year"),
                    paper.get("abstract", ""),
                    paper.get("full_text"),
                    paper.get("source", ""),
                ),
            )
            conn.commit()
            return cur.lastrowid  # type: ignore[return-value]
        finally:
            conn.close()


def save_extracted_params(paper_id: int, params: list[dict]) -> None:
    """Bulk-insert extracted parameter rows, skipping duplicates.

    A row is considered a duplicate when ``(paper_id, parameter_name,
    value, substrate)`` already exists.  When a duplicate is found, the
    existing row is updated only if the new entry has higher confidence
    (``"A"`` beats ``"B"``) or provides evidence text that was missing.
    """
    with _db_lock:
        init_db()
        conn = _connect()
        try:
            for p in params:
                conditions_json = json.dumps(p["conditions"]) if p.get("conditions") else None
                name = p["name"]
                value = p["value"]
                substrate = p.get("substrate")
                evidence = p.get("evidence")
                confidence = p.get("confidence", "B")
                kinetic_model = p.get("kinetic_model")

                # Check for existing duplicate
                existing = conn.execute(
                    "SELECT id, evidence, confidence FROM extracted_params "
                    "WHERE paper_id = ? AND parameter_name = ? "
                    "AND value = ? AND COALESCE(substrate, '') = ?",
                    (paper_id, name, value, substrate or ""),
                ).fetchone()

                if existing is not None:
                    # Update only if new entry is strictly better
                    old_conf = existing["confidence"] or "B"
                    new_is_better = (
                        (confidence < old_conf)  # "A" < "B" lexicographically
                        or (evidence and not existing["evidence"])
                    )
                    if new_is_better:
                        conn.execute(
                            "UPDATE extracted_params SET evidence = ?, confidence = ?, kinetic_model = ? WHERE id = ?",
                            (evidence, confidence, kinetic_model, existing["id"]),
                        )
                    continue

                conn.execute(
                    "INSERT INTO extracted_params "
                    "(paper_id, strain_name, substrate, parameter_name, value, unit, "
                    "conditions, evidence, confidence, kinetic_model) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        paper_id,
                        p.get("strain_name", ""),
                        substrate,
                        name,
                        value,
                        p.get("unit"),
                        conditions_json,
                        evidence,
                        confidence,
                        kinetic_model,
                    ),
                )
            conn.commit()
        finally:
            conn.close()


def paper_has_params(paper_id: int) -> bool:
    """Return ``True`` if *paper_id* already has extracted parameters."""
    init_db()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT 1 FROM extracted_params WHERE paper_id = ? LIMIT 1",
            (paper_id,),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def get_params_for_strain(strain_name: str) -> list[dict]:
    """Return extracted parameter rows matching *strain_name* (case-insensitive)."""
    init_db()
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM extracted_params WHERE LOWER(strain_name) LIKE ?",
            (f"%{strain_name.lower()}%",),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def save_taxonomy(taxid: int, name: str, lineage: list[str], rank: str) -> None:
    """Insert or replace a taxonomy cache entry."""
    with _db_lock:
        init_db()
        conn = _connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO taxonomy_cache (taxid, name, lineage, rank) "
                "VALUES (?, ?, ?, ?)",
                (taxid, name, json.dumps(lineage), rank),
            )
            conn.commit()
        finally:
            conn.close()


def get_taxonomy(name: str) -> dict | None:
    """Retrieve cached taxonomy by organism *name*."""
    init_db()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM taxonomy_cache WHERE LOWER(name) = ?",
            (name.lower(),),
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["lineage"] = json.loads(d["lineage"])
        return d
    finally:
        conn.close()


def save_strain_profile_cache(
    name: str,
    profile: StrainProfile,
    source: str,
    similar_to: str | None,
    score: float | None,
) -> None:
    """Cache a built :class:`StrainProfile`."""
    with _db_lock:
        init_db()
        conn = _connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO strain_profiles_cache "
                "(name, profile_json, source, similar_to, similarity_score) "
                "VALUES (?, ?, ?, ?, ?)",
                (name, profile.model_dump_json(), source, similar_to, score),
            )
            conn.commit()
        finally:
            conn.close()


def load_strain_profile_cache(name: str) -> StrainProfile | None:
    """Load a cached :class:`StrainProfile` or return ``None``."""
    from virtualfermlab.parameters.schema import StrainProfile

    init_db()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT profile_json FROM strain_profiles_cache WHERE name = ?",
            (name,),
        ).fetchone()
        if row is None:
            return None
        return StrainProfile.model_validate_json(row["profile_json"])
    finally:
        conn.close()


def get_papers_with_params(strain_name: str) -> list[dict]:
    """Return papers and their extracted parameters for *strain_name*.

    Results are grouped by paper.  Papers without extracted parameters
    are included with an empty ``params`` list.

    Returns a list of dicts, each with keys:
    ``title``, ``doi``, ``journal``, ``year``, ``authors``, ``params``.
    Each entry in ``params`` has keys:
    ``name``, ``value``, ``unit``, ``substrate``, ``evidence``, ``confidence``.
    """
    init_db()
    conn = _connect()
    try:
        like_pat = f"%{strain_name.lower()}%"
        rows = conn.execute(
            """
            SELECT p.id   AS paper_id,
                   p.title,
                   p.doi,
                   p.journal,
                   p.year,
                   p.authors,
                   ep.parameter_name,
                   ep.value,
                   ep.unit,
                   ep.substrate,
                   ep.evidence,
                   ep.confidence,
                   ep.kinetic_model,
                   ep.conditions
              FROM papers p
              LEFT JOIN extracted_params ep
                ON ep.paper_id = p.id
               AND LOWER(ep.strain_name) LIKE ?
             WHERE p.id IN (
                       SELECT DISTINCT paper_id FROM extracted_params
                        WHERE LOWER(strain_name) LIKE ?
                   )
             ORDER BY p.year DESC, p.id, ep.id
            """,
            (like_pat, like_pat),
        ).fetchall()

        papers: dict[int, dict] = {}
        for r in rows:
            pid = r["paper_id"]
            if pid not in papers:
                papers[pid] = {
                    "title": r["title"],
                    "doi": r["doi"],
                    "journal": r["journal"],
                    "year": r["year"],
                    "authors": r["authors"],
                    "params": [],
                }
            if r["parameter_name"] is not None:
                papers[pid]["params"].append({
                    "name": r["parameter_name"],
                    "value": r["value"],
                    "unit": r["unit"],
                    "substrate": r["substrate"],
                    "evidence": r["evidence"],
                    "confidence": r["confidence"],
                    "kinetic_model": r["kinetic_model"],
                    "conditions": json.loads(r["conditions"]) if r["conditions"] else None,
                })
        return list(papers.values())
    finally:
        conn.close()


def list_cached_strains() -> list[str]:
    """Return names of all cached strain profiles."""
    init_db()
    conn = _connect()
    try:
        rows = conn.execute("SELECT name FROM strain_profiles_cache").fetchall()
        return [r["name"] for r in rows]
    finally:
        conn.close()

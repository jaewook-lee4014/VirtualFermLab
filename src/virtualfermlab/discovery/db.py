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
        # Migrate extracted_params: add evidence column
        try:
            conn.execute("ALTER TABLE extracted_params ADD COLUMN evidence TEXT")
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
    """Bulk-insert extracted parameter rows."""
    with _db_lock:
        init_db()
        conn = _connect()
        try:
            for p in params:
                conditions_json = json.dumps(p["conditions"]) if p.get("conditions") else None
                conn.execute(
                    "INSERT INTO extracted_params "
                    "(paper_id, strain_name, substrate, parameter_name, value, unit, conditions, evidence, confidence) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        paper_id,
                        p.get("strain_name", ""),
                        p.get("substrate"),
                        p["name"],
                        p["value"],
                        p.get("unit"),
                        conditions_json,
                        p.get("evidence"),
                        p.get("confidence", "B"),
                    ),
                )
            conn.commit()
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


def list_cached_strains() -> list[str]:
    """Return names of all cached strain profiles."""
    init_db()
    conn = _connect()
    try:
        rows = conn.execute("SELECT name FROM strain_profiles_cache").fetchall()
        return [r["name"] for r in rows]
    finally:
        conn.close()

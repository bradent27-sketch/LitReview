"""SQLite schema and connection helper. Schema follows spec section 4, with a
few additions the spec calls a "starting point" for: a `rankings` table (the
spec's required explainability feature needs somewhere to record nearest-seed
per run) and a `notifications` table (Phase 5 preprint->published alerts)."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY,
    doi TEXT UNIQUE,                  -- normalized: lowercase, no https://doi.org/ prefix
    pmid TEXT,
    pmcid TEXT,
    source TEXT NOT NULL,             -- 'europepmc' | 'biorxiv' | 'medrxiv'
    is_preprint INTEGER NOT NULL DEFAULT 0,
    published_doi TEXT,               -- set when a preprint later appears in a journal
    title TEXT NOT NULL,
    abstract TEXT,
    authors TEXT,                     -- JSON array
    journal TEXT,
    date_published TEXT,              -- ISO date, normalized
    date_published_kind TEXT,         -- 'posted' | 'epub' | 'print' -- which semantic it was
    date_first_seen TEXT,             -- earliest date seen across preprint+journal versions
    date_ingested TEXT NOT NULL,
    url TEXT,
    pdf_url TEXT,
    mesh_terms TEXT,                  -- JSON array
    dedup_title_key TEXT,             -- normalized_title|first_author_surname|year
    retracted INTEGER NOT NULL DEFAULT 0,
    retraction_note TEXT,
    raw JSON
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi) WHERE doi IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_papers_pmid ON papers(pmid) WHERE pmid IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_papers_dedup_title_key ON papers(dedup_title_key) WHERE dedup_title_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_papers_published_doi ON papers(published_doi) WHERE published_doi IS NOT NULL;

CREATE TABLE IF NOT EXISTS embeddings (
    paper_id INTEGER PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    vec BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS ratings (
    id INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    label INTEGER NOT NULL,           -- +1 / -1
    rated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ratings_paper_id ON ratings(paper_id);

CREATE TABLE IF NOT EXISTS seeds (
    id INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL UNIQUE REFERENCES papers(id) ON DELETE CASCADE,
    note TEXT,
    added_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS digests (
    id INTEGER PRIMARY KEY,
    run_date TEXT NOT NULL,
    paper_ids JSON NOT NULL,
    model_version TEXT,
    is_catchup INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rankings (
    id INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    run_date TEXT NOT NULL,
    score REAL NOT NULL,
    method TEXT NOT NULL,             -- 'centroid' | 'classifier'
    nearest_paper_id INTEGER REFERENCES papers(id),
    nearest_similarity REAL,
    nearest_label TEXT,               -- 'seed' | 'rated_up' | 'rated_down'
    model_version TEXT
);
CREATE INDEX IF NOT EXISTS idx_rankings_run_date ON rankings(run_date);
CREATE INDEX IF NOT EXISTS idx_rankings_paper_id ON rankings(paper_id);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    source TEXT NOT NULL,             -- 'europepmc' | 'biorxiv' | 'medrxiv' | 'embed' | 'rank' | 'digest' | 'retrain'
    n_fetched INTEGER NOT NULL DEFAULT 0,
    n_new INTEGER NOT NULL DEFAULT 0,
    n_deduped INTEGER NOT NULL DEFAULT 0,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at);

CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY,
    paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    type TEXT NOT NULL,               -- 'preprint_published' | 'scoop_alarm' | 'watchlist' | 'retraction'
    message TEXT NOT NULL,
    created_at TEXT NOT NULL,
    seen INTEGER NOT NULL DEFAULT 0
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


@contextmanager
def open_db(db_path: str | Path):
    conn = connect(db_path)
    try:
        init_db(conn)
        yield conn
    finally:
        conn.close()

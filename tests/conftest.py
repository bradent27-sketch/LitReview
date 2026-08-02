from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from litdesk.db import init_db

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def db_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    init_db(conn)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def load_fixture():
    def _load(name: str) -> dict:
        with open(FIXTURES_DIR / name) as fh:
            return json.load(fh)
    return _load


class FakeSession:
    """Stands in for CachedSession in source-client tests: returns queued
    fixture bodies in order instead of making real HTTP calls."""

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.calls: list[tuple[str, dict | None]] = []

    def get_json(self, url: str, params: dict | None = None, use_cache: bool = True) -> dict:
        self.calls.append((url, params))
        if not self._responses:
            raise AssertionError(f"FakeSession ran out of queued responses (call #{len(self.calls)} to {url})")
        return self._responses.pop(0)


@pytest.fixture
def make_fake_session():
    return FakeSession


@pytest.fixture
def insert_paper():
    def _insert(conn, **overrides):
        fields = dict(
            doi=None, pmid=None, pmcid=None, source="europepmc", is_preprint=0,
            published_doi=None, title="Untitled", abstract=None, authors="[]",
            journal=None, date_published="2026-01-01", date_published_kind="epub",
            date_first_seen="2026-01-01", date_ingested="2026-01-01T00:00:00", url=None,
            pdf_url=None, mesh_terms="[]", dedup_title_key=None, raw=None,
        )
        fields.update(overrides)
        cols = ", ".join(fields)
        placeholders = ", ".join("?" for _ in fields)
        cur = conn.execute(f"INSERT INTO papers ({cols}) VALUES ({placeholders})", list(fields.values()))  # noqa: S608
        conn.commit()
        return cur.lastrowid
    return _insert


@pytest.fixture
def insert_embedding():
    def _insert(conn, paper_id, vec, model="test-model"):
        import numpy as np

        from litdesk.embeddings import vec_to_blob

        conn.execute(
            "INSERT OR REPLACE INTO embeddings (paper_id, model, vec) VALUES (?, ?, ?)",
            (paper_id, model, vec_to_blob(np.array(vec, dtype="float32"))),
        )
        conn.commit()
    return _insert


@pytest.fixture
def insert_seed():
    def _insert(conn, paper_id, note=None):
        import datetime as dt

        conn.execute(
            "INSERT INTO seeds (paper_id, note, added_at) VALUES (?, ?, ?)",
            (paper_id, note, dt.datetime.utcnow().isoformat()),
        )
        conn.commit()
    return _insert

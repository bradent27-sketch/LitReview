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

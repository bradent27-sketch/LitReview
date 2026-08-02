import pytest
from fastapi.testclient import TestClient

from litdesk.config import Config
from litdesk.db import open_db
from litdesk.server import create_app


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.db_path = str(tmp_path / "test.db")
    c.digest.output_dir = str(tmp_path / "digests")
    with open_db(c.db_path):
        pass  # just ensure schema exists
    return c


@pytest.fixture
def client(cfg):
    return TestClient(create_app(cfg))


def _insert_paper(cfg, **overrides):
    fields = dict(
        doi=None, pmid=None, pmcid=None, source="europepmc", is_preprint=0,
        published_doi=None, title="Untitled", abstract=None, authors="[]",
        journal=None, date_published="2026-01-01", date_published_kind="epub",
        date_first_seen="2026-01-01", date_ingested="2026-01-01T00:00:00", url=None,
        pdf_url=None, mesh_terms="[]", dedup_title_key=None, raw=None,
    )
    fields.update(overrides)
    with open_db(cfg.db_path) as conn:
        cols = ", ".join(fields)
        placeholders = ", ".join("?" for _ in fields)
        cur = conn.execute(f"INSERT INTO papers ({cols}) VALUES ({placeholders})", list(fields.values()))
        conn.commit()
        return cur.lastrowid


def test_index_with_no_digest_yet(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "No digest yet" in resp.text


def test_index_serves_latest_digest(cfg, client):
    from pathlib import Path
    Path(cfg.digest.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.digest.output_dir) / "latest.html").write_text("<html><body>Hello Digest</body></html>")

    resp = client.get("/")
    assert resp.status_code == 200
    assert "Hello Digest" in resp.text


def test_rate_valid_paper_persists(cfg, client):
    paper_id = _insert_paper(cfg, title="Rate me")

    resp = client.post("/rate", json={"paper_id": paper_id, "label": 1})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["paper_id"] == paper_id
    assert body["total_rated_papers"] == 1

    with open_db(cfg.db_path) as conn:
        row = conn.execute("SELECT label FROM ratings WHERE paper_id = ?", (paper_id,)).fetchone()
    assert row["label"] == 1


def test_rate_rejects_invalid_label(cfg, client):
    paper_id = _insert_paper(cfg, title="Rate me")
    resp = client.post("/rate", json={"paper_id": paper_id, "label": 5})
    assert resp.status_code == 400


def test_rate_rejects_unknown_paper(client):
    resp = client.post("/rate", json={"paper_id": 999999, "label": 1})
    assert resp.status_code == 404


def test_rate_twice_allows_changing_your_mind(cfg, client):
    paper_id = _insert_paper(cfg, title="Rate me")
    client.post("/rate", json={"paper_id": paper_id, "label": -1})
    client.post("/rate", json={"paper_id": paper_id, "label": 1})

    with open_db(cfg.db_path) as conn:
        rows = conn.execute("SELECT label FROM ratings WHERE paper_id = ? ORDER BY id", (paper_id,)).fetchall()
    assert [r["label"] for r in rows] == [-1, 1]  # both kept as history; classifier.py takes the latest

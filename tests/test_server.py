import os
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from litdesk.config import Config, load_config
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
def config_path(tmp_path):
    return tmp_path / "config.yaml"


@pytest.fixture
def client(cfg, config_path):
    return TestClient(create_app(cfg, config_path=config_path))


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


# --- Settings page ---

def test_settings_page_renders_current_values(client):
    resp = client.get("/settings")
    assert resp.status_code == 200
    assert 'name="queries"' in resp.text
    assert 'name="email_smtp_password_session_only"' in resp.text


def test_settings_page_shows_saved_banner_only_with_query_param(client):
    assert "Settings saved" not in client.get("/settings").text
    assert "Settings saved" in client.get("/settings?saved=1").text


def test_save_settings_writes_yaml_and_redirects(client, config_path):
    resp = client.post(
        "/settings",
        data={
            "queries": "query one\nquery two",
            "project_description": "hexokinase work",
            "lookback_days": "10",
            "sources_europepmc": "on",
            "llm_enabled": "on",
            "llm_provider": "claude_code",
            "llm_tldr_top_n": "10",
            "email_from_addr": "me@gmail.com",
            "email_to_addr": "me@gmail.com",
        },
        follow_redirects=False,
    )

    assert resp.status_code == 303
    assert resp.headers["location"] == "/settings?saved=1"
    assert config_path.exists()

    reloaded = load_config(config_path)
    assert reloaded.queries == ["query one", "query two"]
    assert reloaded.project_description == "hexokinase work"
    assert reloaded.lookback_days == 10
    assert reloaded.llm.enabled is True
    assert reloaded.sources.europepmc is True
    assert reloaded.sources.biorxiv is False  # omitted checkbox -> unchecked


def test_save_settings_uploads_bibtex_file(client, config_path):
    bib_content = b"@article{x, doi = {10.1/from-upload}}"
    resp = client.post(
        "/settings",
        data={"queries": "q"},
        files={"seeds_bibtex_file": ("library.bib", bib_content, "text/plain")},
        follow_redirects=False,
    )
    assert resp.status_code == 303

    reloaded = load_config(config_path)
    assert reloaded.seeds.bibtex_path is not None
    assert Path(reloaded.seeds.bibtex_path).read_bytes() == bib_content


def test_save_settings_sets_session_only_smtp_password(client, monkeypatch):
    monkeypatch.delenv("LITDESK_SMTP_PASSWORD", raising=False)
    client.post("/settings", data={"queries": "q", "email_smtp_password_session_only": "app-password-123"})
    assert os.environ.get("LITDESK_SMTP_PASSWORD") == "app-password-123"


def test_save_settings_blank_password_does_not_clear_existing_env_var(client, monkeypatch):
    monkeypatch.setenv("LITDESK_SMTP_PASSWORD", "already-set")
    client.post("/settings", data={"queries": "q", "email_smtp_password_session_only": ""})
    assert os.environ.get("LITDESK_SMTP_PASSWORD") == "already-set"


# --- Control panel: page + notifications ---

def test_control_page_shows_recent_runs_and_notifications(cfg, client):
    with open_db(cfg.db_path) as conn:
        conn.execute(
            "INSERT INTO runs (started_at, finished_at, source, n_fetched, n_new, n_deduped, error) "
            "VALUES (?,?,?,?,?,?,?)",
            ("2026-08-01T00:00:00", "2026-08-01T00:01:00", "europepmc", 5, 2, 3, None),
        )
        conn.execute(
            "INSERT INTO notifications (paper_id, type, message, created_at) "
            "VALUES (NULL, 'scoop_alarm', 'test note', '2026-08-01T00:00:00')"
        )
        conn.commit()

    resp = client.get("/control")
    assert resp.status_code == 200
    assert "europepmc" in resp.text
    assert "test note" in resp.text
    assert "1 new" in resp.text


def test_control_mark_notifications_seen(cfg, client):
    with open_db(cfg.db_path) as conn:
        conn.execute(
            "INSERT INTO notifications (paper_id, type, message, created_at) "
            "VALUES (NULL, 'scoop_alarm', 'note', '2026-08-01T00:00:00')"
        )
        conn.commit()

    resp = client.post("/control/notifications/mark-seen", follow_redirects=False)
    assert resp.status_code == 303

    with open_db(cfg.db_path) as conn:
        unseen = conn.execute("SELECT COUNT(*) AS n FROM notifications WHERE seen = 0").fetchone()["n"]
    assert unseen == 0


# --- Control panel: tool buttons ---

def test_control_retrain_not_enough_ratings(client):
    client.post("/control/retrain", follow_redirects=False)
    assert "Not enough ratings" in client.get("/control").text


def test_control_export_writes_file_and_shows_result(cfg, client):
    import json as _json

    paper_id = _insert_paper(cfg, title="Liked Paper", authors=_json.dumps(["Smith JA"]))
    with open_db(cfg.db_path) as conn:
        conn.execute("INSERT INTO ratings (paper_id, label, rated_at) VALUES (?, 1, '2026-01-01')", (paper_id,))
        conn.commit()

    resp = client.post("/control/export", follow_redirects=False)
    assert resp.status_code == 303

    page = client.get("/control")
    assert "Liked Paper" in page.text
    assert "/files/export-" in page.text
    assert len(list(Path(cfg.digest.output_dir).glob("export-*.bib"))) == 1


def test_control_export_nothing_rated(client):
    client.post("/control/export", follow_redirects=False)
    assert "Nothing thumbed up yet" in client.get("/control").text


def test_control_check_retractions_reports_none(client):
    client.post("/control/check-retractions", follow_redirects=False)
    assert "No new retractions" in client.get("/control").text


def test_control_synthesis_disabled_shows_message_without_calling_llm(client):
    client.post("/control/synthesis", follow_redirects=False)
    assert "turn them on in Settings" in client.get("/control").text


# --- File downloads ---

def test_get_file_serves_existing_file(cfg, client):
    Path(cfg.digest.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.digest.output_dir) / "export-2026-08-03.bib").write_text("@article{x,}")

    resp = client.get("/files/export-2026-08-03.bib")
    assert resp.status_code == 200
    assert "@article" in resp.text


def test_get_file_rejects_unknown_file(client):
    assert client.get("/files/does-not-exist.bib").status_code == 404


def test_get_file_path_traversal_cannot_escape_digest_dir(cfg, client, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("hunter2")

    resp = client.get("/files/..%2Fsecret.txt")
    assert resp.status_code == 404


# --- Control panel: background "Run now" job ---

def test_control_run_completes_and_status_reflects_success(cfg, client, monkeypatch):
    import time

    import numpy as np

    from litdesk.embeddings import vec_to_blob

    seed_id = _insert_paper(cfg, title="Seed Paper")
    candidate_id = _insert_paper(cfg, title="Candidate Paper", doi="10.1/candidate")
    with open_db(cfg.db_path) as conn:
        for pid, vec in ((seed_id, [1.0, 0.0]), (candidate_id, [0.9, 0.1])):
            conn.execute(
                "INSERT INTO embeddings (paper_id, model, vec) VALUES (?, ?, ?)",
                (pid, "sentence-transformers/all-MiniLM-L6-v2", vec_to_blob(np.array(vec, dtype="float32"))),
            )
        conn.execute("INSERT INTO seeds (paper_id, note, added_at) VALUES (?, NULL, '2026-01-01T00:00:00')", (seed_id,))
        conn.commit()

    import litdesk.pipeline as pipeline_mod

    monkeypatch.setattr(
        pipeline_mod, "run_ingest",
        lambda *a, **k: [{"source": "europepmc", "fetched": 0, "new": 0, "deduped": 0, "error": None}],
    )
    monkeypatch.setattr(pipeline_mod, "embed_missing", lambda *a, **k: 0)

    resp = client.post("/control/run")
    assert resp.json()["started"] is True

    status = None
    for _ in range(100):
        status = client.get("/control/run-status").json()
        if not status["running"]:
            break
        time.sleep(0.02)

    assert status["running"] is False
    assert status["had_error"] is False
    assert any("paper(s) ->" in line for line in status["log"])


def test_control_run_second_call_while_running_does_not_start_twice(client, monkeypatch):
    import time

    import litdesk.pipeline as pipeline_mod

    release = threading.Event()

    def blocking_ingest(*a, **k):
        release.wait(timeout=5)
        return [{"source": "europepmc", "fetched": 0, "new": 0, "deduped": 0, "error": None}]

    monkeypatch.setattr(pipeline_mod, "run_ingest", blocking_ingest)
    monkeypatch.setattr(pipeline_mod, "embed_missing", lambda *a, **k: 0)

    try:
        resp1 = client.post("/control/run")
        assert resp1.json()["started"] is True

        for _ in range(50):
            if client.get("/control/run-status").json()["running"]:
                break
            time.sleep(0.02)

        resp2 = client.post("/control/run")
        assert resp2.json()["started"] is False
    finally:
        release.set()
        for _ in range(50):
            if not client.get("/control/run-status").json()["running"]:
                break
            time.sleep(0.02)

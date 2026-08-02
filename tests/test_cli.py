import argparse
import sqlite3
import textwrap

import pytest

from litdesk import cli
from litdesk import config as config_mod
from litdesk.models import RawPaper
from litdesk.sources import biorxiv, europepmc


@pytest.fixture
def temp_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(textwrap.dedent(f"""\
        db_path: {tmp_path / "test.db"}
        cache_dir: {tmp_path / "cache"}
        log_dir: {tmp_path / "logs"}
        queries: ["q1"]
        sources:
          europepmc: true
          biorxiv: true
          medrxiv: false
        digest:
          output_dir: {tmp_path / "digests"}
    """))
    return config_path


def test_cli_ingest_prints_summary(temp_config, monkeypatch, capsys):
    monkeypatch.setattr(europepmc, "search", lambda session, query, lookback_days=7:
                         [RawPaper(source="europepmc", title="P1", doi="10.1/p1")])
    monkeypatch.setattr(biorxiv, "fetch", lambda session, server="biorxiv", lookback_days=7, category=None:
                         [RawPaper(source=server, title="P2", doi="10.1101/p2", is_preprint=True)])

    exit_code = cli.main(["--config", str(temp_config), "ingest"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "europepmc" in out
    assert "biorxiv" in out
    assert "TOTAL" in out


def test_cli_ingest_is_idempotent_across_reruns(temp_config, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(europepmc, "search", lambda session, query, lookback_days=7:
                         [RawPaper(source="europepmc", title="P1", doi="10.1/p1")])
    monkeypatch.setattr(biorxiv, "fetch", lambda session, server="biorxiv", lookback_days=7, category=None:
                         [RawPaper(source=server, title="P2", doi="10.1101/p2", is_preprint=True)])

    cli.main(["--config", str(temp_config), "ingest"])
    capsys.readouterr()
    cli.main(["--config", str(temp_config), "ingest"])
    capsys.readouterr()

    conn = sqlite3.connect(tmp_path / "test.db")
    n = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    conn.close()
    assert n == 2  # not 4 — second run deduped against the first


def test_cli_ingest_returns_nonzero_on_source_error(temp_config, monkeypatch, capsys):
    def broken_search(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(europepmc, "search", broken_search)
    monkeypatch.setattr(biorxiv, "fetch", lambda *a, **k: [])

    exit_code = cli.main(["--config", str(temp_config), "ingest"])
    assert exit_code == 1


def test_cli_runs_shows_history(temp_config, monkeypatch, capsys):
    monkeypatch.setattr(europepmc, "search", lambda *a, **k: [])
    monkeypatch.setattr(biorxiv, "fetch", lambda *a, **k: [])

    cli.main(["--config", str(temp_config), "ingest"])
    capsys.readouterr()
    cli.main(["--config", str(temp_config), "runs"])
    out = capsys.readouterr().out
    assert "europepmc" in out
    assert "biorxiv" in out


def _seed_db(db_path, **overrides):
    """Minimal direct-SQL paper insert for CLI tests that don't need the
    full ingest pipeline — mirrors conftest.py's insert_paper fixture.
    Initializes the schema itself (via open_db) so tests never have to
    route through the network-hitting `ingest` command just to get a
    usable database."""
    from litdesk.db import open_db

    fields = dict(
        doi=None, pmid=None, pmcid=None, source="europepmc", is_preprint=0,
        published_doi=None, title="Untitled", abstract=None, authors="[]",
        journal=None, date_published="2026-01-01", date_published_kind="epub",
        date_first_seen="2026-01-01", date_ingested="2026-01-01T00:00:00", url=None,
        pdf_url=None, mesh_terms="[]", dedup_title_key=None, raw=None,
    )
    fields.update(overrides)
    with open_db(db_path) as conn:
        cols = ", ".join(fields)
        placeholders = ", ".join("?" for _ in fields)
        cur = conn.execute(f"INSERT INTO papers ({cols}) VALUES ({placeholders})", list(fields.values()))
        conn.commit()
        return cur.lastrowid


def test_cli_export_writes_bibtex_for_rated_up_papers(temp_config, tmp_path, capsys):
    import json

    from litdesk.db import open_db

    pid = _seed_db(tmp_path / "test.db", title="Liked Paper", authors=json.dumps(["Smith JA"]))
    with open_db(tmp_path / "test.db") as conn:
        conn.execute("INSERT INTO ratings (paper_id, label, rated_at) VALUES (?, 1, '2026-01-01')", (pid,))
        conn.commit()

    exit_code = cli.main(["--config", str(temp_config), "export"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Liked Paper" in out
    bib_files = list((tmp_path / "digests").glob("export-*.bib"))
    assert len(bib_files) == 1
    assert "Liked Paper" in bib_files[0].read_text()


def test_cli_export_handles_nothing_rated(temp_config, tmp_path, capsys):
    from litdesk.db import open_db

    with open_db(tmp_path / "test.db"):
        pass  # just initialize the schema, nothing rated

    exit_code = cli.main(["--config", str(temp_config), "export"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Nothing thumbed up" in out


def test_cli_check_retractions_flags_and_reports(temp_config, tmp_path, capsys):
    import json

    _seed_db(
        tmp_path / "test.db", title="Retracted Paper",
        raw=json.dumps({"pubTypeList": {"pubType": ["Retracted Publication"]}}),
    )

    exit_code = cli.main(["--config", str(temp_config), "check-retractions"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Retracted Paper" in out

    conn = sqlite3.connect(tmp_path / "test.db")
    retracted = conn.execute("SELECT retracted FROM papers WHERE title = 'Retracted Paper'").fetchone()[0]
    conn.close()
    assert retracted == 1


def test_cli_notifications_shows_and_marks_seen(temp_config, tmp_path, capsys):
    from litdesk.db import open_db

    with open_db(tmp_path / "test.db") as conn:
        conn.execute(
            "INSERT INTO notifications (paper_id, type, message, created_at) VALUES (NULL, 'scoop_alarm', 'test message', '2026-01-01T00:00:00')"
        )
        conn.commit()

    exit_code = cli.main(["--config", str(temp_config), "notifications"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "test message" in out

    # Second call: already marked seen, shouldn't show again without --all
    cli.main(["--config", str(temp_config), "notifications"])
    out2 = capsys.readouterr().out
    assert "test message" not in out2

    cli.main(["--config", str(temp_config), "notifications", "--all"])
    out3 = capsys.readouterr().out
    assert "test message" in out3


def test_cli_init_writes_config_and_creates_dirs(tmp_path, monkeypatch):
    # init resolves everything relative to REPO_ROOT, so give it an isolated
    # fake repo root (with a copy of the real example config) instead of
    # letting it touch this actual checkout's config/config.yaml.
    real_example = (config_mod.REPO_ROOT / "config" / "config.example.yaml").read_text()
    fake_config_dir = tmp_path / "config"
    fake_config_dir.mkdir()
    (fake_config_dir / "config.example.yaml").write_text(real_example)
    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)

    exit_code = cli.cmd_init(argparse.Namespace())
    assert exit_code == 0
    assert (tmp_path / "config" / "config.yaml").exists()
    assert (tmp_path / "data").is_dir()
    assert (tmp_path / "digests").is_dir()

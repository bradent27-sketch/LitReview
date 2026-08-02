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

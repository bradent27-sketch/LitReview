import litdesk.ingest as ingest_mod
from litdesk.config import Config
from litdesk.models import RawPaper
from litdesk.sources import biorxiv, europepmc


def _base_config(tmp_path):
    cfg = Config()
    cfg.queries = ["q1", "q2"]
    cfg.sources.medrxiv = False
    cfg.cache_dir = str(tmp_path / "cache")
    return cfg


def test_run_ingest_happy_path(db_conn, monkeypatch, tmp_path):
    cfg = _base_config(tmp_path)
    call_count = {"europepmc": 0}

    def fake_search(session, query, lookback_days=7):
        call_count["europepmc"] += 1
        return [RawPaper(source="europepmc", title=f"{query} paper", doi=f"10.1/{query}")]

    def fake_fetch(session, server="biorxiv", lookback_days=7, category=None):
        return [RawPaper(source=server, title="preprint x", doi="10.1101/x", is_preprint=True)]

    monkeypatch.setattr(europepmc, "search", fake_search)
    monkeypatch.setattr(biorxiv, "fetch", fake_fetch)

    results = ingest_mod.run_ingest(db_conn, cfg)

    assert call_count["europepmc"] == 2  # once per configured query
    by_source = {r["source"]: r for r in results}
    assert by_source["europepmc"]["fetched"] == 2
    assert by_source["europepmc"]["new"] == 2
    assert by_source["biorxiv"]["fetched"] == 1
    assert by_source["biorxiv"]["new"] == 1

    runs = db_conn.execute("SELECT source, n_fetched, n_new, error FROM runs ORDER BY id").fetchall()
    assert [r["source"] for r in runs] == ["europepmc", "biorxiv"]
    assert all(r["error"] is None for r in runs)


def test_run_ingest_one_source_failing_does_not_block_others(db_conn, monkeypatch, tmp_path):
    cfg = _base_config(tmp_path)
    cfg.queries = ["q1"]

    def broken_search(session, query, lookback_days=7):
        raise RuntimeError("europepmc changed their schema again")

    def fake_fetch(session, server="biorxiv", lookback_days=7, category=None):
        return [RawPaper(source=server, title="preprint x", doi="10.1101/x", is_preprint=True)]

    monkeypatch.setattr(europepmc, "search", broken_search)
    monkeypatch.setattr(biorxiv, "fetch", fake_fetch)

    results = ingest_mod.run_ingest(db_conn, cfg)
    by_source = {r["source"]: r for r in results}
    assert by_source["europepmc"]["error"] is not None
    assert by_source["biorxiv"]["error"] is None
    assert by_source["biorxiv"]["new"] == 1

    runs = db_conn.execute("SELECT source, error FROM runs ORDER BY id").fetchall()
    errored = [r for r in runs if r["error"]]
    assert len(errored) == 1
    assert errored[0]["source"] == "europepmc"


def test_run_ingest_dedups_across_sources_via_doi(db_conn, monkeypatch, tmp_path):
    cfg = _base_config(tmp_path)
    cfg.queries = ["q1"]
    shared_doi = "10.1101/shared"

    def fake_search(session, query, lookback_days=7):
        return [RawPaper(source="europepmc", title="Shared paper", doi=shared_doi, is_preprint=True)]

    def fake_fetch(session, server="biorxiv", lookback_days=7, category=None):
        return [RawPaper(source=server, title="Shared paper", doi=shared_doi, is_preprint=True)]

    monkeypatch.setattr(europepmc, "search", fake_search)
    monkeypatch.setattr(biorxiv, "fetch", fake_fetch)

    ingest_mod.run_ingest(db_conn, cfg)
    count = db_conn.execute("SELECT COUNT(*) AS n FROM papers WHERE doi = ?", (shared_doi,)).fetchone()["n"]
    assert count == 1

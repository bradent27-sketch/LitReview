from litdesk import seeds
from litdesk.config import Config
from litdesk.models import RawPaper
from litdesk.sources import europepmc


def test_extract_dois_from_bibtex_handles_braces_and_quotes():
    text = """
    @article{doe2026,
      title = {Some Paper},
      doi = {10.1016/j.jbc.2026.100001},
    }
    @article{roe2026,
      title = {Another},
      doi = "10.1101/2026.01.01.500000",
    }
    """
    assert seeds.extract_dois_from_bibtex(text) == [
        "10.1016/j.jbc.2026.100001",
        "10.1101/2026.01.01.500000",
    ]


def test_add_seed_by_doi_links_existing_paper(db_conn, insert_paper):
    pid = insert_paper(db_conn, doi="10.1/existing", title="Existing paper")
    cfg = Config()

    paper_id, status = seeds.add_seed_by_doi(db_conn, cfg, "10.1/existing")

    assert paper_id == pid
    assert status == "linked_existing"
    row = db_conn.execute("SELECT * FROM seeds WHERE paper_id = ?", (pid,)).fetchone()
    assert row is not None


def test_add_seed_by_doi_fetches_new_paper(db_conn, monkeypatch, make_fake_session):
    cfg = Config()

    def fake_fetch_by_doi(session, doi):
        return RawPaper(source="europepmc", title="Fetched Seed", doi=doi)

    monkeypatch.setattr(europepmc, "fetch_by_doi", fake_fetch_by_doi)

    paper_id, status = seeds.add_seed_by_doi(db_conn, cfg, "10.1/new-seed", session=make_fake_session([]))

    assert status == "fetched"
    row = db_conn.execute("SELECT title FROM papers WHERE id = ?", (paper_id,)).fetchone()
    assert row["title"] == "Fetched Seed"
    seed_row = db_conn.execute("SELECT * FROM seeds WHERE paper_id = ?", (paper_id,)).fetchone()
    assert seed_row is not None


def test_add_seed_by_doi_not_found(db_conn, monkeypatch, make_fake_session):
    cfg = Config()
    monkeypatch.setattr(europepmc, "fetch_by_doi", lambda session, doi: None)

    paper_id, status = seeds.add_seed_by_doi(db_conn, cfg, "10.1/missing", session=make_fake_session([]))

    assert paper_id is None
    assert status == "not_found"
    assert db_conn.execute("SELECT COUNT(*) AS n FROM seeds").fetchone()["n"] == 0


def test_add_seed_by_doi_is_idempotent(db_conn, insert_paper):
    pid = insert_paper(db_conn, doi="10.1/existing", title="Existing paper")
    cfg = Config()

    seeds.add_seed_by_doi(db_conn, cfg, "10.1/existing")
    paper_id, status = seeds.add_seed_by_doi(db_conn, cfg, "10.1/existing")

    assert paper_id == pid
    assert status == "already_seed"
    assert db_conn.execute("SELECT COUNT(*) AS n FROM seeds WHERE paper_id = ?", (pid,)).fetchone()["n"] == 1


def test_load_seeds_from_config_combines_dois_and_bibtex(db_conn, insert_paper, monkeypatch, tmp_path):
    pid = insert_paper(db_conn, doi="10.1/from-config", title="From config list")
    bibtex_path = tmp_path / "library.bib"
    bibtex_path.write_text('@article{x, doi = {10.1/from-bibtex}}')

    cfg = Config()
    cfg.seeds.dois = ["10.1/from-config"]
    cfg.seeds.bibtex_path = str(bibtex_path)

    monkeypatch.setattr(europepmc, "fetch_by_doi", lambda session, doi: RawPaper(source="europepmc", title="From bibtex", doi=doi))

    results = seeds.load_seeds_from_config(db_conn, cfg)

    statuses = {r["doi"]: r["status"] for r in results}
    assert statuses["10.1/from-config"] == "linked_existing"
    assert statuses["10.1/from-bibtex"] == "fetched"
    assert db_conn.execute("SELECT COUNT(*) AS n FROM seeds").fetchone()["n"] == 2

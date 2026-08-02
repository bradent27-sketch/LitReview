import json

from litdesk import export


def test_bibtex_escape_handles_braces_and_backslash():
    assert export._bibtex_escape("A {special} title \\ here") == "A \\{special\\} title \\\\ here"


def test_cite_key_uses_surname_year_titleword():
    key = export._cite_key("The Kinase Mechanism", ["Smith JA"], "2026", set())
    assert key == "smith2026kinase"


def test_cite_key_dedupes_collisions():
    used = set()
    key1 = export._cite_key("Kinase Studies", ["Smith JA"], "2026", used)
    key2 = export._cite_key("Kinase Structures", ["Smith AB"], "2026", used)
    assert key1 != key2
    assert key1 == "smith2026kinase"
    assert key2 == "smith2026kinasea"


def test_paper_to_bibtex_journal_article(db_conn, insert_paper):
    pid = insert_paper(
        db_conn, title="A Great Paper", authors=json.dumps(["Smith JA", "Doe R"]),
        journal="JBC", date_published="2026-03-01", doi="10.1/x", url="https://doi.org/10.1/x",
        is_preprint=0,
    )
    row = db_conn.execute("SELECT * FROM papers WHERE id = ?", (pid,)).fetchone()
    bib = export.paper_to_bibtex(row, set())
    assert bib.startswith("@article{smith2026great,")
    assert "title = {A Great Paper}" in bib
    assert "author = {Smith JA and Doe R}" in bib
    assert "journal = {JBC}" in bib
    assert "doi = {10.1/x}" in bib
    assert "note" not in bib


def test_paper_to_bibtex_preprint_uses_unpublished_and_note(db_conn, insert_paper):
    pid = insert_paper(
        db_conn, title="A Preprint", authors=json.dumps(["Lin C"]),
        date_published="2026-01-01", doi="10.1101/x", is_preprint=1,
    )
    row = db_conn.execute("SELECT * FROM papers WHERE id = ?", (pid,)).fetchone()
    bib = export.paper_to_bibtex(row, set())
    assert bib.startswith("@unpublished{")
    assert "note = {Preprint}" in bib
    assert "journal" not in bib


def test_export_rated_up_bibtex_only_includes_thumbs_up(db_conn, insert_paper, insert_rating):
    up_id = insert_paper(db_conn, title="Liked Paper", authors=json.dumps(["Smith JA"]), date_published="2026-01-01")
    down_id = insert_paper(db_conn, title="Disliked Paper", authors=json.dumps(["Doe R"]), date_published="2026-01-01")
    unrated_id = insert_paper(db_conn, title="Unrated Paper")
    insert_rating(db_conn, up_id, 1)
    insert_rating(db_conn, down_id, -1)

    bibtex = export.export_rated_up_bibtex(db_conn)

    assert "Liked Paper" in bibtex
    assert "Disliked Paper" not in bibtex
    assert "Unrated Paper" not in bibtex


def test_export_rated_up_bibtex_empty_when_nothing_rated(db_conn):
    assert export.export_rated_up_bibtex(db_conn) == ""


def test_export_rated_up_bibtex_uses_latest_rating(db_conn, insert_paper, insert_rating):
    pid = insert_paper(db_conn, title="Changed Mind", date_published="2026-01-01")
    insert_rating(db_conn, pid, 1, rated_at="2026-01-01T00:00:00")
    insert_rating(db_conn, pid, -1, rated_at="2026-01-02T00:00:00")
    assert export.export_rated_up_bibtex(db_conn) == ""

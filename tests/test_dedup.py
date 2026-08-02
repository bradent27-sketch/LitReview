from litdesk import dedup
from litdesk.models import RawPaper


def test_normalize_doi_strips_prefixes_and_lowercases():
    assert dedup.normalize_doi("https://doi.org/10.1016/J.JBC.2026.100001") == "10.1016/j.jbc.2026.100001"
    assert dedup.normalize_doi("  DOI:10.1101/2026.01.01.500000/  ") == "10.1101/2026.01.01.500000"
    assert dedup.normalize_doi(None) is None
    assert dedup.normalize_doi("") is None


def test_normalize_title_strips_punctuation_and_collapses_whitespace():
    assert dedup.normalize_title("The Kinase's Role:  Structure & Function!") == "the kinases role structure function"
    assert dedup.normalize_title(None) == ""


def test_first_author_surname_handles_both_conventions():
    assert dedup.first_author_surname(["Doe, J."]) == "doe"          # bioRxiv "Last, First"
    assert dedup.first_author_surname(["Smith JA"]) == "smith"        # Europe PMC "Last Initials"
    assert dedup.first_author_surname(["van der Berg JA"]) == "van der berg"
    assert dedup.first_author_surname([]) is None
    assert dedup.first_author_surname([""]) is None


def test_extract_year():
    assert dedup.extract_year("2026-07-28") == "2026"
    assert dedup.extract_year(None) is None
    assert dedup.extract_year("not-a-date") is None


def test_compute_dedup_title_key_is_stable_across_equivalent_inputs():
    key1 = dedup.compute_dedup_title_key("The Kinase's Role!", ["Smith JA"], "2026-07-28")
    key2 = dedup.compute_dedup_title_key("the kinases role", ["Smith AB"], "2026-01-01")
    assert key1 == key2 == "the kinases role|smith|2026"


def _insert_paper(conn, **overrides):
    fields = dict(
        doi="10.1000/example", pmid=None, pmcid=None, source="europepmc", is_preprint=0,
        published_doi=None, title="Example Paper", abstract="abstract", authors="[]",
        journal="J", date_published="2026-01-01", date_published_kind="epub",
        date_first_seen="2026-01-01", date_ingested="2026-01-01T00:00:00", url=None,
        pdf_url=None, mesh_terms="[]",
        dedup_title_key=dedup.compute_dedup_title_key("Example Paper", ["Smith JA"], "2026-01-01"),
        raw=None,
    )
    fields.update(overrides)
    cols = ", ".join(fields)
    placeholders = ", ".join("?" for _ in fields)
    cur = conn.execute(f"INSERT INTO papers ({cols}) VALUES ({placeholders})", list(fields.values()))
    conn.commit()
    return cur.lastrowid


def test_find_match_by_doi(db_conn):
    _insert_paper(db_conn, doi="10.1000/match-me")
    hit = dedup.find_match(db_conn, RawPaper(source="europepmc", title="Different title", doi="10.1000/match-me"))
    assert hit is not None


def test_find_match_by_pmid(db_conn):
    _insert_paper(db_conn, doi="10.1000/x", pmid="999")
    hit = dedup.find_match(db_conn, RawPaper(source="europepmc", title="Different title", doi=None, pmid="999"))
    assert hit is not None


def test_find_match_by_title_author_year(db_conn):
    _insert_paper(db_conn, doi=None, title="Example Paper")
    hit = dedup.find_match(
        db_conn,
        RawPaper(source="europepmc", title="Example, Paper!!", doi=None, authors=["Smith JA"], date_published="2026-01-15"),
    )
    assert hit is not None


def test_find_match_via_published_doi_link(db_conn):
    # Row is a preprint whose journal DOI we already know.
    _insert_paper(db_conn, doi="10.1101/preprint-doi", published_doi="10.1016/journal-doi", is_preprint=1)
    hit = dedup.find_match(db_conn, RawPaper(source="europepmc", title="x", doi="10.1016/journal-doi"))
    assert hit is not None


def test_no_match_for_unrelated_paper(db_conn):
    _insert_paper(db_conn, doi="10.1000/x", title="Totally unrelated")
    hit = dedup.find_match(
        db_conn,
        RawPaper(source="europepmc", title="Something else entirely", doi="10.9999/nope", authors=["Nobody N"], date_published="2020-01-01"),
    )
    assert hit is None

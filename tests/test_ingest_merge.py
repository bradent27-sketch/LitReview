import json

from litdesk.ingest import upsert_paper
from litdesk.models import RawPaper


def _get(conn, paper_id):
    return conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()


def test_new_paper_is_inserted(db_conn):
    paper_id, is_new = upsert_paper(db_conn, RawPaper(
        source="europepmc", title="Brand New Paper", doi="10.1000/new-1",
        authors=["Doe J"], date_published="2026-07-01",
    ))
    assert is_new is True
    row = _get(db_conn, paper_id)
    assert row["title"] == "Brand New Paper"
    assert row["date_first_seen"] == "2026-07-01"


def test_exact_doi_reingestion_is_deduped_not_new(db_conn):
    id1, is_new1 = upsert_paper(db_conn, RawPaper(source="europepmc", title="P", doi="10.1000/dup", date_published="2026-01-01"))
    id2, is_new2 = upsert_paper(db_conn, RawPaper(source="europepmc", title="P", doi="10.1000/dup", date_published="2026-01-01"))
    assert is_new1 is True
    assert is_new2 is False
    assert id1 == id2


def test_preprint_then_published_link_then_journal_metadata_merge(db_conn):
    # 1. Preprint first seen — no journal DOI known yet.
    pid, is_new = upsert_paper(db_conn, RawPaper(
        source="biorxiv", title="Chaperone folding kinetics (preprint)", doi="10.1101/preprint-1",
        is_preprint=True, published_doi=None, abstract="preprint abstract",
        authors=["Lin, C."], date_published="2026-05-01",
    ))
    assert is_new is True
    row = _get(db_conn, pid)
    assert row["is_preprint"] == 1
    assert row["published_doi"] is None

    # 2. A later bioRxiv poll reveals the journal DOI (still the preprint record).
    pid2, is_new2 = upsert_paper(db_conn, RawPaper(
        source="biorxiv", title="Chaperone folding kinetics (preprint)", doi="10.1101/preprint-1",
        is_preprint=True, published_doi="10.1016/journal-1", abstract="preprint abstract",
        authors=["Lin, C."], date_published="2026-05-01",
    ))
    assert is_new2 is False
    assert pid2 == pid
    row = _get(db_conn, pid)
    assert row["published_doi"] == "10.1016/journal-1"
    assert row["is_preprint"] == 1  # still a preprint until the journal metadata itself arrives

    # 3. Europe PMC returns the actual journal (MED) record for that DOI.
    pid3, is_new3 = upsert_paper(db_conn, RawPaper(
        source="europepmc", title="Chaperone folding kinetics: full study", doi="10.1016/journal-1",
        is_preprint=False, abstract="full peer-reviewed abstract", journal="JBC",
        authors=["Lin C", "Ahmed F"], date_published="2026-07-15",
    ))
    assert is_new3 is False
    assert pid3 == pid  # merged into the same row, not a new one

    row = _get(db_conn, pid)
    assert row["is_preprint"] == 0
    assert row["title"] == "Chaperone folding kinetics: full study"       # journal metadata wins
    assert row["abstract"] == "full peer-reviewed abstract"
    assert row["journal"] == "JBC"
    assert row["date_published"] == "2026-07-15"                          # shown date is now the journal date
    assert row["date_first_seen"] == "2026-05-01"                         # but first-seen stays the preprint date
    assert row["doi"] == "10.1101/preprint-1"                             # stable identity: never changes
    assert row["published_doi"] == "10.1016/journal-1"
    assert json.loads(row["authors"]) == ["Lin C", "Ahmed F"]


def test_journal_first_then_preprint_arrives_does_not_clobber_journal_metadata(db_conn):
    # Journal version ingested first, with no known preprint.
    jid, is_new = upsert_paper(db_conn, RawPaper(
        source="europepmc", title="Journal-first Paper", doi="10.1016/journal-2",
        is_preprint=False, abstract="journal abstract", journal="Nature",
        authors=["Roe R"], date_published="2025-12-01",
    ))
    assert is_new is True

    # A preprint shows up whose bioRxiv `published` field points at that journal DOI.
    pid, is_new2 = upsert_paper(db_conn, RawPaper(
        source="biorxiv", title="Journal-first Paper (preprint)", doi="10.1101/preprint-2",
        is_preprint=True, published_doi="10.1016/journal-2", abstract="preprint abstract",
        authors=["Roe, R."], date_published="2025-11-01",
    ))
    assert is_new2 is False
    assert pid == jid

    row = _get(db_conn, jid)
    # Journal metadata is untouched by the later-arriving preprint record.
    assert row["title"] == "Journal-first Paper"
    assert row["abstract"] == "journal abstract"
    assert row["is_preprint"] == 0
    assert row["doi"] == "10.1016/journal-2"
    # But we now know it once had a preprint stage, and first-seen moves earlier.
    assert row["published_doi"] == "10.1016/journal-2"
    assert row["date_first_seen"] == "2025-11-01"


def test_notifies_when_a_rated_up_preprint_gets_published_doi(db_conn):
    pid, _ = upsert_paper(db_conn, RawPaper(
        source="biorxiv", title="Preprint I like", doi="10.1101/liked-1",
        is_preprint=True, date_published="2026-05-01",
    ))
    db_conn.execute("INSERT INTO ratings (paper_id, label, rated_at) VALUES (?, 1, '2026-05-02')", (pid,))
    db_conn.commit()

    upsert_paper(db_conn, RawPaper(
        source="biorxiv", title="Preprint I like", doi="10.1101/liked-1",
        is_preprint=True, published_doi="10.1016/journal-liked-1", date_published="2026-05-01",
    ))

    notif = db_conn.execute(
        "SELECT * FROM notifications WHERE paper_id = ? AND type = 'preprint_published'", (pid,)
    ).fetchone()
    assert notif is not None
    assert "10.1016/journal-liked-1" in notif["message"]


def test_does_not_notify_for_unrated_or_downvoted_preprints(db_conn):
    unrated_id, _ = upsert_paper(db_conn, RawPaper(
        source="biorxiv", title="Unrated preprint", doi="10.1101/unrated-1",
        is_preprint=True, date_published="2026-05-01",
    ))
    downvoted_id, _ = upsert_paper(db_conn, RawPaper(
        source="biorxiv", title="Downvoted preprint", doi="10.1101/down-1",
        is_preprint=True, date_published="2026-05-01",
    ))
    db_conn.execute("INSERT INTO ratings (paper_id, label, rated_at) VALUES (?, -1, '2026-05-02')", (downvoted_id,))
    db_conn.commit()

    upsert_paper(db_conn, RawPaper(
        source="biorxiv", title="Unrated preprint", doi="10.1101/unrated-1",
        is_preprint=True, published_doi="10.1016/journal-unrated", date_published="2026-05-01",
    ))
    upsert_paper(db_conn, RawPaper(
        source="biorxiv", title="Downvoted preprint", doi="10.1101/down-1",
        is_preprint=True, published_doi="10.1016/journal-down", date_published="2026-05-01",
    ))

    assert db_conn.execute("SELECT COUNT(*) AS n FROM notifications").fetchone()["n"] == 0


def test_does_not_renotify_once_published_doi_already_known(db_conn):
    pid, _ = upsert_paper(db_conn, RawPaper(
        source="biorxiv", title="Preprint I like", doi="10.1101/liked-2",
        is_preprint=True, date_published="2026-05-01",
    ))
    db_conn.execute("INSERT INTO ratings (paper_id, label, rated_at) VALUES (?, 1, '2026-05-02')", (pid,))
    db_conn.commit()

    for _ in range(2):  # simulate the same info arriving on two separate ingest runs
        upsert_paper(db_conn, RawPaper(
            source="biorxiv", title="Preprint I like", doi="10.1101/liked-2",
            is_preprint=True, published_doi="10.1016/journal-liked-2", date_published="2026-05-01",
        ))

    count = db_conn.execute("SELECT COUNT(*) AS n FROM notifications WHERE paper_id = ?", (pid,)).fetchone()["n"]
    assert count == 1


def test_title_author_year_fallback_when_no_doi(db_conn):
    id1, is_new1 = upsert_paper(db_conn, RawPaper(
        source="europepmc", title="No DOI Here", doi=None, authors=["Smith JA"], date_published="2026-02-01",
    ))
    id2, is_new2 = upsert_paper(db_conn, RawPaper(
        source="europepmc", title="No, DOI Here!!", doi=None, authors=["Smith AB"], date_published="2026-02-20",
    ))
    assert is_new1 is True
    assert is_new2 is False
    assert id1 == id2

from litdesk.sources import biorxiv


def test_parse_entry_extracts_expected_fields(load_fixture):
    body = load_fixture("biorxiv_page1.json")
    entry = body["collection"][0]
    paper = biorxiv.parse_entry(entry, "biorxiv")

    assert paper.doi == "10.1101/2026.05.01.500123"
    assert paper.is_preprint is True
    assert paper.authors == ["Lin, C.", "Ahmed, F."]
    assert paper.date_published == "2026-05-01"
    assert paper.date_published_kind == "posted"
    assert paper.published_doi is None  # "NA" in the fixture
    assert paper.url == "https://doi.org/10.1101/2026.05.01.500123"
    assert paper.pdf_url == "https://www.biorxiv.org/content/10.1101/2026.05.01.500123v1.full.pdf"


def test_parse_entry_normalizes_published_doi():
    entry = {
        "doi": "10.1101/x", "title": "T", "authors": "Doe, J.", "date": "2026-01-01",
        "abstract": "a", "published": "10.1016/j.jbc.2026.100777", "version": "2",
    }
    paper = biorxiv.parse_entry(entry, "biorxiv")
    assert paper.published_doi == "10.1016/j.jbc.2026.100777"


def test_fetch_single_page(load_fixture, make_fake_session):
    body = load_fixture("biorxiv_page1.json")
    session = make_fake_session([body])
    papers = biorxiv.fetch(session, server="biorxiv", lookback_days=7)
    assert len(papers) == 2
    assert len(session.calls) == 1
    url, _ = session.calls[0]
    assert url == "https://api.biorxiv.org/details/biorxiv/7d/0"


def test_fetch_stops_on_short_page(make_fake_session):
    body = {"messages": [{"total": "1"}], "collection": [{"doi": "10.1/a", "title": "T", "authors": "Doe, J.", "date": "2026-01-01"}]}
    session = make_fake_session([body])
    papers = biorxiv.fetch(session, server="medrxiv", lookback_days=5)
    assert len(papers) == 1
    assert len(session.calls) == 1  # total reached, no second page fetched


def test_fetch_paginates_when_full_page_returned(make_fake_session):
    def make_page(n):
        return {
            "messages": [{"total": "150"}],
            "collection": [{"doi": f"10.1/{i}", "title": "T", "authors": "Doe, J.", "date": "2026-01-01"} for i in range(n)],
        }
    session = make_fake_session([make_page(100), make_page(50)])
    papers = biorxiv.fetch(session, server="biorxiv", lookback_days=7)
    assert len(papers) == 150
    assert len(session.calls) == 2
    assert session.calls[1][0].endswith("/100")

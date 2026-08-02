from litdesk.sources import europepmc


def test_parse_result_extracts_expected_fields(load_fixture):
    body = load_fixture("europepmc_page1.json")
    results = body["resultList"]["result"]

    journal_paper = europepmc.parse_result(results[0])
    assert journal_paper.doi == "10.1016/j.jbc.2026.100001"
    assert journal_paper.pmid == "38111111"
    assert journal_paper.is_preprint is False
    assert journal_paper.authors == ["Doe J", "Roe R", "Smith AB"]
    assert journal_paper.journal == "Journal of Biological Chemistry"
    assert journal_paper.mesh_terms == ["Enzymes", "Kinetics"]
    assert journal_paper.pdf_url == "https://europepmc.org/articles/PMC10111111/pdf"
    assert journal_paper.date_published == "2026-07-28"
    assert journal_paper.date_published_kind == "epub"

    preprint = europepmc.parse_result(results[1])
    assert preprint.is_preprint is True
    assert preprint.date_published_kind == "posted"
    assert preprint.published_doi is None  # Europe PMC doesn't expose this link


def test_search_single_page_stops_when_hitcount_reached(load_fixture, make_fake_session):
    body = load_fixture("europepmc_page1.json")
    session = make_fake_session([body])
    papers = europepmc.search(session, "some query", lookback_days=7)
    assert len(papers) == 3
    assert len(session.calls) == 1
    # query gets wrapped with a date filter and sort directive
    _, params = session.calls[0]
    assert "FIRST_PDATE:" in params["query"]
    assert params["cursorMark"] == "*"


def test_search_paginates_via_cursor_mark(load_fixture, make_fake_session):
    p1 = load_fixture("europepmc_multipage_p1.json")
    p2 = load_fixture("europepmc_multipage_p2.json")
    session = make_fake_session([p1, p2])
    papers = europepmc.search(session, "some query", lookback_days=7)
    assert [p.doi for p in papers] == ["10.1000/aaa", "10.1000/bbb"]
    assert len(session.calls) == 2
    assert session.calls[0][1]["cursorMark"] == "*"
    assert session.calls[1][1]["cursorMark"] == "PAGE2"


def test_search_stops_on_empty_result_page(make_fake_session):
    empty = {"hitCount": 0, "nextCursorMark": "*", "resultList": {"result": []}}
    session = make_fake_session([empty])
    papers = europepmc.search(session, "some query", lookback_days=7)
    assert papers == []
    assert len(session.calls) == 1

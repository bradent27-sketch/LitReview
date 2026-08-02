import sys

from litdesk.config import Config
from litdesk.llm import client as llm_client
from litdesk.llm import summarize


class FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeResponse:
    def __init__(self, text):
        self.content = [FakeTextBlock(text)]


class FakeClient:
    """Stands in for anthropic.Anthropic() — .messages.create() either
    returns a canned response or raises a canned exception."""

    def __init__(self, response_or_exc):
        self._response_or_exc = response_or_exc

        class _Messages:
            def create(_self, **kwargs):
                if isinstance(self._response_or_exc, Exception):
                    raise self._response_or_exc
                return self._response_or_exc

        self.messages = _Messages()


def _cfg(**overrides):
    cfg = Config()
    cfg.llm.enabled = True
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# --- get_client ---

def test_get_client_returns_none_for_unimplemented_provider():
    cfg = _cfg()
    cfg.llm.provider = "claude_code"
    assert llm_client.get_client(cfg) is None


def test_get_client_returns_none_when_sdk_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    assert llm_client.get_client(_cfg()) is None


def test_get_client_returns_none_when_construction_fails(monkeypatch):
    import anthropic

    def boom(*a, **k):
        raise anthropic.AnthropicError("no credentials found")

    monkeypatch.setattr(anthropic, "Anthropic", boom)
    assert llm_client.get_client(_cfg()) is None


# --- summarize_paper ---

def test_summarize_paper_parses_tldr_and_relevance():
    client = FakeClient(FakeResponse(
        "This paper does X and finds Y. It also shows Z.\n"
        "Relevance: Directly related to enzyme kinetics work."
    ))
    result = summarize.summarize_paper(client, _cfg(), "Title", "Abstract text", "JBC", "enzyme kinetics")
    assert result["tldr"] == "This paper does X and finds Y. It also shows Z."
    assert result["relevance"] == "Directly related to enzyme kinetics work."


def test_summarize_paper_handles_missing_relevance_line():
    client = FakeClient(FakeResponse("Just a plain summary with no relevance line."))
    result = summarize.summarize_paper(client, _cfg(), "Title", None, None, "biochemistry")
    assert result["tldr"] == "Just a plain summary with no relevance line."
    assert result["relevance"] is None


def test_summarize_paper_returns_none_on_request_failure():
    client = FakeClient(RuntimeError("rate limited"))
    result = summarize.summarize_paper(client, _cfg(), "Title", "Abstract", "Journal", "biochemistry")
    assert result is None


def test_summarize_paper_returns_none_on_empty_response():
    client = FakeClient(FakeResponse(""))
    result = summarize.summarize_paper(client, _cfg(), "Title", "Abstract", "Journal", "biochemistry")
    assert result is None


# --- enrich_digest_entries ---

def test_enrich_digest_entries_noop_when_disabled(db_conn, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("get_client should not be called when llm.enabled is false")
    monkeypatch.setattr(summarize, "get_client", boom)

    cfg = Config()
    cfg.llm.enabled = False
    data = {"run_date": "2026-08-02", "entries": [{"id": 1, "title": "T", "abstract": "A", "journal": "J"}]}
    result = summarize.enrich_digest_entries(db_conn, cfg, data)
    assert "tldr" not in result["entries"][0]


def test_enrich_digest_entries_noop_when_no_client(db_conn, monkeypatch):
    monkeypatch.setattr(summarize, "get_client", lambda cfg: None)
    cfg = _cfg()
    data = {"run_date": "2026-08-02", "entries": [{"id": 1, "title": "T", "abstract": "A", "journal": "J"}]}
    result = summarize.enrich_digest_entries(db_conn, cfg, data)
    assert "tldr" not in result["entries"][0]


def test_enrich_digest_entries_updates_entries_and_rankings(db_conn, insert_paper, monkeypatch):
    pid = insert_paper(db_conn, title="Real Paper")
    db_conn.execute(
        "INSERT INTO rankings (paper_id, run_date, score, method, model_version) VALUES (?,?,?,?,?)",
        (pid, "2026-08-02", 0.9, "centroid", "test-model"),
    )
    db_conn.commit()

    fake = FakeClient(FakeResponse("TLDR sentence one. TLDR sentence two.\nRelevance: Very relevant."))
    monkeypatch.setattr(summarize, "get_client", lambda cfg: fake)

    cfg = _cfg()
    cfg.llm.tldr_top_n = 10
    data = {
        "run_date": "2026-08-02",
        "entries": [{"id": pid, "title": "Real Paper", "abstract": "abs", "journal": "J"}],
    }
    result = summarize.enrich_digest_entries(db_conn, cfg, data)

    assert result["entries"][0]["tldr"] == "TLDR sentence one. TLDR sentence two."
    assert result["entries"][0]["relevance"] == "Very relevant."

    row = db_conn.execute("SELECT tldr, tldr_relevance FROM rankings WHERE paper_id = ?", (pid,)).fetchone()
    assert row["tldr"] == "TLDR sentence one. TLDR sentence two."
    assert row["tldr_relevance"] == "Very relevant."


def test_enrich_digest_entries_only_covers_top_n(db_conn, insert_paper, monkeypatch):
    ids = []
    for i in range(3):
        pid = insert_paper(db_conn, title=f"Paper {i}")
        db_conn.execute(
            "INSERT INTO rankings (paper_id, run_date, score, method, model_version) VALUES (?,?,?,?,?)",
            (pid, "2026-08-02", 1.0 - i * 0.1, "centroid", "test-model"),
        )
        ids.append(pid)
    db_conn.commit()

    fake = FakeClient(FakeResponse("A summary.\nRelevance: yes."))
    monkeypatch.setattr(summarize, "get_client", lambda cfg: fake)

    cfg = _cfg()
    cfg.llm.tldr_top_n = 2
    data = {
        "run_date": "2026-08-02",
        "entries": [{"id": pid, "title": f"Paper {i}", "abstract": "a", "journal": "J"} for i, pid in enumerate(ids)],
    }
    result = summarize.enrich_digest_entries(db_conn, cfg, data)

    assert "tldr" in result["entries"][0]
    assert "tldr" in result["entries"][1]
    assert "tldr" not in result["entries"][2]


# --- weekly_synthesis ---

def test_weekly_synthesis_returns_none_for_empty_list():
    assert summarize.weekly_synthesis(FakeClient(FakeResponse("x")), _cfg(), []) is None


def test_weekly_synthesis_formats_paper_listing_and_returns_text():
    fake = FakeClient(FakeResponse("Theme: everyone is studying kinases this week."))
    papers = [{"title": "Paper A", "journal": "JBC", "tldr": "About kinases."}]
    result = summarize.weekly_synthesis(fake, _cfg(), papers)
    assert result == "Theme: everyone is studying kinases this week."


def test_weekly_synthesis_returns_none_on_failure():
    fake = FakeClient(RuntimeError("boom"))
    result = summarize.weekly_synthesis(fake, _cfg(), [{"title": "T", "journal": "J", "tldr": "x"}])
    assert result is None


# --- weekly_synthesis_for_recent_digests ---

def test_weekly_synthesis_for_recent_digests_none_when_disabled(db_conn):
    cfg = Config()
    cfg.llm.enabled = False
    assert summarize.weekly_synthesis_for_recent_digests(db_conn, cfg) is None


def test_weekly_synthesis_for_recent_digests_none_when_no_rows(db_conn, monkeypatch):
    monkeypatch.setattr(summarize, "get_client", lambda cfg: FakeClient(FakeResponse("x")))
    assert summarize.weekly_synthesis_for_recent_digests(db_conn, _cfg()) is None


def test_weekly_synthesis_for_recent_digests_reads_ranked_papers(db_conn, insert_paper, monkeypatch):
    pid = insert_paper(db_conn, title="Ranked Paper", journal="Cell")
    db_conn.execute(
        "INSERT INTO rankings (paper_id, run_date, score, method, model_version, tldr) VALUES (?,?,?,?,?,?)",
        (pid, "2026-08-01", 0.8, "centroid", "test-model", "A tldr."),
    )
    db_conn.commit()

    captured = {}

    def fake_get_client(cfg):
        return FakeClient(FakeResponse("Synthesis text"))

    monkeypatch.setattr(summarize, "get_client", fake_get_client)

    real_weekly_synthesis = summarize.weekly_synthesis

    def spy(client, cfg, papers):
        captured["papers"] = papers
        return real_weekly_synthesis(client, cfg, papers)

    monkeypatch.setattr(summarize, "weekly_synthesis", spy)

    result = summarize.weekly_synthesis_for_recent_digests(db_conn, _cfg(), days=7)
    assert result == "Synthesis text"
    assert captured["papers"] == [{"title": "Ranked Paper", "journal": "Cell", "tldr": "A tldr."}]

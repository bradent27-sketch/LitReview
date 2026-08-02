import datetime as dt
import json

import numpy as np
import pytest

import litdesk.embeddings as litdesk_embeddings
from litdesk import digest
from litdesk.config import Config

MODEL = "test-model"


def _cfg(**overrides):
    cfg = Config()
    cfg.embeddings.model = MODEL
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_candidate_paper_ids_excludes_rated_seeded_and_already_digested(db_conn, insert_paper, insert_seed):
    plain = insert_paper(db_conn, title="Plain candidate")
    rated = insert_paper(db_conn, title="Already rated")
    seeded = insert_paper(db_conn, title="A seed itself")
    shown = insert_paper(db_conn, title="Already in a past digest")

    db_conn.execute("INSERT INTO ratings (paper_id, label, rated_at) VALUES (?, 1, '2026-01-01')", (rated,))
    insert_seed(db_conn, seeded)
    db_conn.execute(
        "INSERT INTO digests (run_date, paper_ids, model_version, is_catchup, created_at) VALUES (?,?,?,0,?)",
        ("2026-07-01", json.dumps([shown]), MODEL, "2026-07-01T00:00:00"),
    )
    db_conn.commit()

    ids = digest.candidate_paper_ids(db_conn)
    assert ids == [plain]


def test_candidate_paper_ids_excludes_retracted_papers(db_conn, insert_paper):
    plain = insert_paper(db_conn, title="Plain candidate")
    insert_paper(db_conn, title="Retracted candidate", retracted=1, retraction_note="Retracted Publication")

    ids = digest.candidate_paper_ids(db_conn)
    assert ids == [plain]


def test_build_digest_raises_without_seeds(db_conn, insert_paper):
    insert_paper(db_conn)
    with pytest.raises(ValueError, match="seed"):
        digest.build_digest(db_conn, _cfg())


def test_build_digest_raises_when_candidates_unembedded(db_conn, insert_paper, insert_embedding, insert_seed):
    seed = insert_paper(db_conn)
    insert_seed(db_conn, seed)
    insert_embedding(db_conn, seed, [1.0, 0.0], model=MODEL)
    insert_paper(db_conn, title="Never embedded")  # candidate with no vector at all

    with pytest.raises(ValueError, match="embed"):
        digest.build_digest(db_conn, _cfg())


def test_build_digest_empty_when_no_candidates(db_conn, insert_paper, insert_embedding, insert_seed):
    seed = insert_paper(db_conn)
    insert_seed(db_conn, seed)
    insert_embedding(db_conn, seed, [1.0, 0.0], model=MODEL)

    data = digest.build_digest(db_conn, _cfg())
    assert data["entries"] == []


def _make_ranked_candidates(db_conn, insert_paper, insert_embedding):
    """6 candidates with strictly decreasing similarity to the seed [1,0],
    plus a 7th far-away one in a journal configured for always-include."""
    vecs = [
        (1.0, 0.0), (0.9, 0.1), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.1, 0.9),
    ]
    ids = []
    for i, v in enumerate(vecs):
        pid = insert_paper(db_conn, title=f"Candidate {i}", doi=f"10.1/c{i}")
        insert_embedding(db_conn, pid, list(v), model=MODEL)
        ids.append(pid)
    forced = insert_paper(db_conn, title="Forced via journal", journal="Nature", doi="10.1/forced")
    insert_embedding(db_conn, forced, [-1.0, -1.0], model=MODEL)  # would rank last on score alone
    return ids, forced


def test_build_digest_bands_and_forced_include(db_conn, insert_paper, insert_embedding, insert_seed):
    seed = insert_paper(db_conn, title="Seed")
    insert_seed(db_conn, seed)
    insert_embedding(db_conn, seed, [1.0, 0.0], model=MODEL)
    ranked_ids, forced_id = _make_ranked_candidates(db_conn, insert_paper, insert_embedding)

    cfg = _cfg()
    cfg.digest.top_n = 6
    cfg.journals_always_include = ["Nature"]

    data = digest.build_digest(db_conn, cfg)
    entries_by_id = {e["id"]: e for e in data["entries"]}

    assert len(data["entries"]) == 7  # top 6 + 1 forced
    assert [entries_by_id[pid]["band"] for pid in ranked_ids] == \
        ["high", "high", "medium", "medium", "low", "low"]

    forced_entry = entries_by_id[forced_id]
    assert forced_entry["forced_include"] is True
    assert forced_entry["band"] is None
    assert forced_entry["journal"] == "Nature"

    # entries are sorted by score descending regardless of forced/ranked status
    scores = [e["score"] for e in data["entries"]]
    assert scores == sorted(scores, reverse=True)

    # explainability: every entry points at the (only) seed
    assert all(e["nearest_title"] == "Seed" for e in entries_by_id.values() if not e["forced_include"])


def test_build_digest_persists_rankings_and_digests_rows(db_conn, insert_paper, insert_embedding, insert_seed):
    seed = insert_paper(db_conn, title="Seed")
    insert_seed(db_conn, seed)
    insert_embedding(db_conn, seed, [1.0, 0.0], model=MODEL)
    candidate = insert_paper(db_conn, title="Candidate")
    insert_embedding(db_conn, candidate, [0.9, 0.1], model=MODEL)

    data = digest.build_digest(db_conn, _cfg())

    digest_rows = db_conn.execute("SELECT * FROM digests").fetchall()
    assert len(digest_rows) == 1
    assert json.loads(digest_rows[0]["paper_ids"]) == [candidate]
    assert digest_rows[0]["run_date"] == data["run_date"]

    ranking_rows = db_conn.execute("SELECT * FROM rankings WHERE paper_id = ?", (candidate,)).fetchall()
    assert len(ranking_rows) == 1
    assert ranking_rows[0]["method"] == "centroid"
    assert ranking_rows[0]["nearest_paper_id"] == seed


def test_build_digest_does_not_repeat_papers_across_runs(db_conn, insert_paper, insert_embedding, insert_seed):
    seed = insert_paper(db_conn, title="Seed")
    insert_seed(db_conn, seed)
    insert_embedding(db_conn, seed, [1.0, 0.0], model=MODEL)
    candidate = insert_paper(db_conn, title="Candidate")
    insert_embedding(db_conn, candidate, [0.9, 0.1], model=MODEL)

    first = digest.build_digest(db_conn, _cfg())
    assert len(first["entries"]) == 1

    second = digest.build_digest(db_conn, _cfg())
    assert second["entries"] == []


def test_build_digest_uses_centroid_before_enough_ratings(db_conn, insert_paper, insert_embedding, insert_seed, insert_rating):
    seed = insert_paper(db_conn, title="Seed")
    insert_seed(db_conn, seed)
    insert_embedding(db_conn, seed, [1.0, 0.0], model=MODEL)
    # A couple of ratings, but nowhere near min_ratings_for_classifier.
    for i in range(2):
        pid = insert_paper(db_conn, title=f"Rated {i}")
        insert_embedding(db_conn, pid, [1.0, 0.0], model=MODEL)
        insert_rating(db_conn, pid, 1)
    candidate = insert_paper(db_conn, title="Candidate")
    insert_embedding(db_conn, candidate, [0.9, 0.1], model=MODEL)

    data = digest.build_digest(db_conn, _cfg())
    ranking_row = db_conn.execute("SELECT method FROM rankings WHERE paper_id = ?", (candidate,)).fetchone()
    assert ranking_row["method"] == "centroid"
    assert data["entries"][0]["nearest_label"] == "seed"


def test_build_digest_switches_to_classifier_once_enough_ratings(
    db_conn, insert_paper, insert_embedding, insert_seed, insert_rating,
):
    seed = insert_paper(db_conn, title="Seed")
    insert_seed(db_conn, seed)
    insert_embedding(db_conn, seed, [1.0, 0.0], model=MODEL)

    cfg = _cfg()
    cfg.ranker.min_ratings_for_classifier = 8
    cfg.ranker.n_random_negatives = 10

    for i in range(4):
        pid = insert_paper(db_conn, title=f"Up {i}", doi=f"10.1/up{i}")
        insert_embedding(db_conn, pid, [1.0 - 0.02 * i, 0.02 * i], model=MODEL)
        insert_rating(db_conn, pid, 1)
    for i in range(4):
        pid = insert_paper(db_conn, title=f"Down {i}", doi=f"10.1/down{i}")
        insert_embedding(db_conn, pid, [0.02 * i, 1.0 - 0.02 * i], model=MODEL)
        insert_rating(db_conn, pid, -1)
    for i in range(10):  # unrated pool for random-negative sampling
        pid = insert_paper(db_conn, title=f"Filler {i}", doi=f"10.1/filler{i}")
        insert_embedding(db_conn, pid, [0.5, 0.5], model=MODEL)

    candidate = insert_paper(db_conn, title="Looks like the ups", doi="10.1/candidate")
    insert_embedding(db_conn, candidate, [0.9, 0.1], model=MODEL)

    data = digest.build_digest(db_conn, cfg)
    ranking_row = db_conn.execute("SELECT method FROM rankings WHERE paper_id = ?", (candidate,)).fetchone()
    assert ranking_row["method"] == "classifier"
    entry = next(e for e in data["entries"] if e["id"] == candidate)
    assert entry["nearest_label"] == "rated_up"


def test_render_html_writes_dated_and_latest_files(tmp_path):
    data = {
        "run_date": "2026-08-02",
        "is_catchup": False,
        "gap_days": 0,
        "n_candidates_total": 1,
        "entries": [{
            "id": 1, "title": "A Great Paper", "journal": "JBC", "date_published": "2026-08-01",
            "abstract": "An interesting abstract.", "url": "https://doi.org/10.1/x",
            "authors": ["Doe J"], "is_preprint": False, "score": 0.87, "band": "high",
            "forced_include": False, "nearest_title": "Seed Paper", "nearest_similarity": 0.91,
        }],
    }
    out_path = digest.render_html(data, tmp_path)
    assert out_path == tmp_path / "2026-08-02.html"
    assert out_path.exists()
    assert (tmp_path / "latest.html").exists()

    html = out_path.read_text()
    assert "A Great Paper" in html
    assert "band-high" in html
    assert "closest to" in html
    assert "Seed Paper" in html


def test_render_html_omits_link_when_url_missing(tmp_path):
    data = {
        "run_date": "2026-08-02", "is_catchup": False, "gap_days": 0, "n_candidates_total": 1,
        "entries": [{
            "id": 1, "title": "No URL Paper", "journal": None, "date_published": None,
            "abstract": None, "url": None, "authors": [], "is_preprint": False, "score": 0.5,
            "band": "medium", "forced_include": False, "nearest_title": None, "nearest_similarity": None,
        }],
    }
    html = digest.render_html(data, tmp_path).read_text()
    assert 'href="None"' not in html


def test_build_digest_forces_watchlist_author(db_conn, insert_paper, insert_embedding, insert_seed):
    seed = insert_paper(db_conn, title="Seed")
    insert_seed(db_conn, seed)
    insert_embedding(db_conn, seed, [1.0, 0.0], model=MODEL)

    watchlisted = insert_paper(db_conn, title="Paper by a watchlisted author", authors=json.dumps(["Famous PI"]))
    insert_embedding(db_conn, watchlisted, [-1.0, -1.0], model=MODEL)  # would never rank in on score alone
    other = insert_paper(db_conn, title="Other candidate")
    insert_embedding(db_conn, other, [0.95, 0.05], model=MODEL)

    cfg = _cfg()
    cfg.watchlists.authors = ["Famous PI"]
    cfg.digest.top_n = 1  # watchlisted paper would not otherwise make the cut

    data = digest.build_digest(db_conn, cfg)
    entries_by_id = {e["id"]: e for e in data["entries"]}

    assert watchlisted in entries_by_id
    assert entries_by_id[watchlisted]["forced_include"] is True
    assert "watchlist" in entries_by_id[watchlisted]["forced_reason"].lower()
    assert "famous pi" in entries_by_id[watchlisted]["forced_reason"].lower()


def test_build_digest_scoop_alert_forces_include_and_notifies(db_conn, insert_paper, insert_embedding, insert_seed, monkeypatch):
    seed = insert_paper(db_conn, title="Seed")
    insert_seed(db_conn, seed)
    insert_embedding(db_conn, seed, [1.0, 0.0], model=MODEL)

    scoopy = insert_paper(db_conn, title="Scoop paper")
    insert_embedding(db_conn, scoopy, [0.0, 1.0], model=MODEL)
    other = insert_paper(db_conn, title="Other")
    insert_embedding(db_conn, other, [0.95, 0.05], model=MODEL)

    monkeypatch.setattr(
        litdesk_embeddings, "embed_texts",
        lambda model_name, texts, batch_size=32: np.array([[0.0, 1.0]]),
    )

    cfg = _cfg()
    cfg.digest.top_n = 1  # scoopy would not otherwise make the cut
    cfg.scoop_alarm.enabled = True
    cfg.scoop_alarm.threshold = 0.9
    cfg.project_description = "my active project"

    data = digest.build_digest(db_conn, cfg)
    entries_by_id = {e["id"]: e for e in data["entries"]}

    assert entries_by_id[scoopy]["scoop_alert"] is True
    assert entries_by_id.get(other, {}).get("scoop_alert", False) is False

    notif = db_conn.execute(
        "SELECT * FROM notifications WHERE paper_id = ? AND type = 'scoop_alarm'", (scoopy,)
    ).fetchone()
    assert notif is not None


def test_scoop_alarm_disabled_by_default_even_with_project_description(db_conn, insert_paper, insert_embedding, insert_seed, monkeypatch):
    seed = insert_paper(db_conn, title="Seed")
    insert_seed(db_conn, seed)
    insert_embedding(db_conn, seed, [1.0, 0.0], model=MODEL)
    candidate = insert_paper(db_conn, title="Candidate")
    insert_embedding(db_conn, candidate, [0.0, 1.0], model=MODEL)

    def boom(*a, **k):
        raise AssertionError("embed_texts should not be called when scoop_alarm.enabled is false")
    monkeypatch.setattr(litdesk_embeddings, "embed_texts", boom)

    cfg = _cfg()
    cfg.project_description = "my active project"  # set, but scoop_alarm.enabled defaults to False

    data = digest.build_digest(db_conn, cfg)
    assert data["entries"][0]["scoop_alert"] is False


def test_build_digest_catchup_mode_after_gap(db_conn, insert_paper, insert_embedding, insert_seed):
    seed = insert_paper(db_conn, title="Seed")
    insert_seed(db_conn, seed)
    insert_embedding(db_conn, seed, [1.0, 0.0], model=MODEL)

    last_run_date = (dt.date.today() - dt.timedelta(days=10)).isoformat()
    db_conn.execute(
        "INSERT INTO digests (run_date, paper_ids, model_version, is_catchup, created_at) VALUES (?,?,?,0,?)",
        (last_run_date, json.dumps([]), MODEL, last_run_date + "T00:00:00"),
    )
    db_conn.commit()

    for i in range(5):
        pid = insert_paper(db_conn, title=f"Candidate {i}", doi=f"10.1/catchup{i}")
        insert_embedding(db_conn, pid, [0.9 - i * 0.05, 0.1], model=MODEL)

    cfg = _cfg()
    cfg.digest.catchup_gap_days = 3
    cfg.digest.catchup_top_n = 2

    data = digest.build_digest(db_conn, cfg)

    assert data["is_catchup"] is True
    assert data["gap_days"] == 10
    assert data["n_candidates_total"] == 5
    assert len(data["entries"]) == 2  # catchup_top_n, not the default top_n


def test_build_digest_not_catchup_within_gap_threshold(db_conn, insert_paper, insert_embedding, insert_seed):
    seed = insert_paper(db_conn, title="Seed")
    insert_seed(db_conn, seed)
    insert_embedding(db_conn, seed, [1.0, 0.0], model=MODEL)

    last_run_date = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    db_conn.execute(
        "INSERT INTO digests (run_date, paper_ids, model_version, is_catchup, created_at) VALUES (?,?,?,0,?)",
        (last_run_date, json.dumps([]), MODEL, last_run_date + "T00:00:00"),
    )
    db_conn.commit()
    candidate = insert_paper(db_conn, title="Candidate")
    insert_embedding(db_conn, candidate, [0.9, 0.1], model=MODEL)

    data = digest.build_digest(db_conn, _cfg())
    assert data["is_catchup"] is False
    assert data["gap_days"] == 1


def test_render_html_shows_scoop_alert_badge(tmp_path):
    data = {
        "run_date": "2026-08-02", "is_catchup": False, "gap_days": 0, "n_candidates_total": 1,
        "entries": [{
            "id": 1, "title": "Scoop Paper", "journal": "JBC", "date_published": "2026-08-01",
            "abstract": "abs", "url": "https://doi.org/10.1/x", "authors": [], "is_preprint": False,
            "score": 0.1, "band": "low", "forced_include": False, "forced_reason": None,
            "scoop_alert": True, "nearest_title": None, "nearest_similarity": None,
        }],
    }
    html = digest.render_html(data, tmp_path).read_text()
    assert "band-scoop" in html
    assert "scoop alert" in html.lower()


def test_render_html_shows_catchup_banner(tmp_path):
    data = {
        "run_date": "2026-08-02", "is_catchup": True, "gap_days": 12, "n_candidates_total": 340,
        "entries": [],
    }
    html = digest.render_html(data, tmp_path).read_text()
    assert "away 12 day" in html
    assert "340" in html

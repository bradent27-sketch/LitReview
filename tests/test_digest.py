import json

import pytest

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
    assert "No URL Paper" in html

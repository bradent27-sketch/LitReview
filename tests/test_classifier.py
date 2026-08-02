from litdesk import classifier
from litdesk.config import Config

MODEL = "test-model"


def _cfg(**overrides):
    cfg = Config()
    cfg.embeddings.model = MODEL
    cfg.ranker.min_ratings_for_classifier = 8
    cfg.ranker.n_random_negatives = 10
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _build_separable_dataset(db_conn, insert_paper, insert_embedding, insert_rating, n_per_class=4):
    positives, negatives = [], []
    for i in range(n_per_class):
        pid = insert_paper(db_conn, title=f"Positive {i}", doi=f"10.1/pos{i}")
        insert_embedding(db_conn, pid, [1.0 - 0.02 * i, 0.0 + 0.02 * i], model=MODEL)
        insert_rating(db_conn, pid, 1)
        positives.append(pid)
    for i in range(n_per_class):
        pid = insert_paper(db_conn, title=f"Negative {i}", doi=f"10.1/neg{i}")
        insert_embedding(db_conn, pid, [0.0 + 0.02 * i, 1.0 - 0.02 * i], model=MODEL)
        insert_rating(db_conn, pid, -1)
        negatives.append(pid)
    # A handful of unrated papers so random-negative sampling has something to draw from.
    for i in range(10):
        pid = insert_paper(db_conn, title=f"Unrated {i}", doi=f"10.1/unrated{i}")
        insert_embedding(db_conn, pid, [0.5, 0.5], model=MODEL)
    return positives, negatives


def test_get_current_ratings_uses_latest_rating_only(db_conn, insert_paper, insert_rating):
    pid = insert_paper(db_conn)
    insert_rating(db_conn, pid, -1, rated_at="2026-01-01T00:00:00")
    insert_rating(db_conn, pid, 1, rated_at="2026-01-02T00:00:00")  # changed their mind

    ratings = classifier.get_current_ratings(db_conn)
    assert ratings[pid] == 1


def test_sample_random_unrated_excludes_given_ids(db_conn, insert_paper):
    excluded = insert_paper(db_conn, title="Excluded")
    included = [insert_paper(db_conn, title=f"Other {i}") for i in range(5)]

    sampled = classifier.sample_random_unrated(db_conn, n=100, exclude_ids={excluded})
    assert excluded not in sampled
    assert set(sampled) == set(included)


def test_sample_random_unrated_respects_n(db_conn, insert_paper):
    for i in range(20):
        insert_paper(db_conn, title=f"Paper {i}")
    sampled = classifier.sample_random_unrated(db_conn, n=5, exclude_ids=set())
    assert len(sampled) == 5


def test_train_returns_none_below_min_ratings(db_conn, insert_paper, insert_embedding, insert_rating):
    pid = insert_paper(db_conn)
    insert_embedding(db_conn, pid, [1.0, 0.0], model=MODEL)
    insert_rating(db_conn, pid, 1)

    assert classifier.train(db_conn, _cfg()) is None


def test_train_returns_none_when_only_one_class_present(db_conn, insert_paper, insert_embedding, insert_rating):
    for i in range(8):
        pid = insert_paper(db_conn, title=f"P{i}")
        insert_embedding(db_conn, pid, [1.0, 0.0], model=MODEL)
        insert_rating(db_conn, pid, 1)  # all thumbs-up, no thumbs-down at all

    assert classifier.train(db_conn, _cfg()) is None


def test_train_and_rank_orders_candidates_sensibly(db_conn, insert_paper, insert_embedding, insert_rating):
    _build_separable_dataset(db_conn, insert_paper, insert_embedding, insert_rating)
    cfg = _cfg()

    trained = classifier.train(db_conn, cfg)
    assert trained is not None

    like_positive = insert_paper(db_conn, title="Looks like the positives")
    insert_embedding(db_conn, like_positive, [0.92, 0.08], model=MODEL)
    like_negative = insert_paper(db_conn, title="Looks like the negatives")
    insert_embedding(db_conn, like_negative, [0.08, 0.92], model=MODEL)

    ranked = classifier.rank_by_classifier(db_conn, cfg, [like_positive, like_negative], trained)
    ranked_ids = [r.paper_id for r in ranked]

    assert ranked_ids[0] == like_positive
    assert ranked_ids[1] == like_negative

    by_id = {r.paper_id: r for r in ranked}
    assert by_id[like_positive].nearest_label == "rated_up"
    assert by_id[like_negative].nearest_label == "rated_down"


def test_evaluate_holdout_returns_none_below_threshold(db_conn, insert_paper, insert_embedding, insert_rating):
    assert classifier.evaluate_holdout(db_conn, _cfg()) is None


def test_evaluate_holdout_reports_metrics_on_separable_data(db_conn, insert_paper, insert_embedding, insert_rating):
    _build_separable_dataset(db_conn, insert_paper, insert_embedding, insert_rating, n_per_class=6)
    cfg = _cfg()
    cfg.ranker.min_ratings_for_classifier = 12

    result = classifier.evaluate_holdout(db_conn, cfg)

    assert result is not None
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["accuracy"] >= 0.5  # trivially separable clusters
    assert result["n_train"] > 0
    assert result["n_test"] > 0

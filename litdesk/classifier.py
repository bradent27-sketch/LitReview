"""Trained ranker (spec section 5, steps 2-4): logistic regression on rated
papers' embeddings, once there are enough ratings to bother. Thumbs-up are
positives, thumbs-down are explicit negatives, plus a large sample of random
unrated papers as implicit negatives to regularize the boundary. Retrained on
every run — a logistic regression on a few thousand vectors takes seconds.

Explicit and random negatives get different sample weights (a thumbs-down is
a much stronger signal than "probably not interesting"); sklearn's
`class_weight` only distinguishes positive/negative as a whole, so the
explicit-vs-random distinction is done via per-sample weights instead.
"""

from __future__ import annotations

import random

import numpy as np
from sklearn.linear_model import LogisticRegression

from litdesk.config import Config
from litdesk.ranking import RankedPaper, load_embeddings, nearest_reference


def get_current_ratings(conn) -> dict[int, int]:
    """Most recent rating per paper — re-rating overwrites, not appends."""
    rows = conn.execute(
        """
        SELECT r.paper_id, r.label FROM ratings r
        JOIN (SELECT paper_id, MAX(id) AS max_id FROM ratings GROUP BY paper_id) latest
          ON latest.paper_id = r.paper_id AND latest.max_id = r.id
        """
    ).fetchall()
    return {row["paper_id"]: row["label"] for row in rows}


def sample_random_unrated(conn, n: int, exclude_ids: set[int]) -> list[int]:
    exclude_list = list(exclude_ids)
    if exclude_list:
        placeholders = ",".join("?" for _ in exclude_list)
        where = f"WHERE id NOT IN ({placeholders})"
        params = (*exclude_list, n)
    else:
        where = ""
        params = (n,)
    rows = conn.execute(f"SELECT id FROM papers {where} ORDER BY RANDOM() LIMIT ?", params).fetchall()  # noqa: S608
    return [r["id"] for r in rows]


def _training_arrays(
    conn, cfg: Config, positive_ids: list[int], explicit_negative_ids: list[int], exclude_from_random: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    model_name = cfg.embeddings.model
    pos_neg_embeddings = load_embeddings(conn, model_name, positive_ids + explicit_negative_ids)

    random_neg_ids = sample_random_unrated(conn, cfg.ranker.n_random_negatives, exclude_from_random)
    random_neg_embeddings = load_embeddings(conn, model_name, random_neg_ids)

    X: list[np.ndarray] = []
    y: list[int] = []
    w: list[float] = []
    for pid in positive_ids:
        if pid in pos_neg_embeddings:
            X.append(pos_neg_embeddings[pid]); y.append(1); w.append(cfg.ranker.weight_positive)
    for pid in explicit_negative_ids:
        if pid in pos_neg_embeddings:
            X.append(pos_neg_embeddings[pid]); y.append(0); w.append(cfg.ranker.weight_explicit_negative)
    for vec in random_neg_embeddings.values():
        X.append(vec); y.append(0); w.append(cfg.ranker.weight_random_negative)

    if len(set(y)) < 2:
        return None
    return np.stack(X), np.array(y), np.array(w)


class TrainedRanker:
    def __init__(self, model: LogisticRegression):
        self.model = model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


def train(conn, cfg: Config) -> TrainedRanker | None:
    """Returns None if there isn't yet at least one thumbs-up and one
    thumbs-down with a stored embedding — sklearn needs both classes."""
    ratings = get_current_ratings(conn)
    if len(ratings) < cfg.ranker.min_ratings_for_classifier:
        return None

    positives = [pid for pid, label in ratings.items() if label == 1]
    explicit_negs = [pid for pid, label in ratings.items() if label == -1]
    arrays = _training_arrays(conn, cfg, positives, explicit_negs, exclude_from_random=set(ratings))
    if arrays is None:
        return None
    X, y, w = arrays

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y, sample_weight=w)
    return TrainedRanker(model)


def rank_by_classifier(
    conn, cfg: Config, candidate_paper_ids: list[int], ranker: TrainedRanker,
) -> list[RankedPaper]:
    """Nearest-neighbor explainability here is against rated papers (up or
    down), not seeds — the classifier has moved past cold start."""
    model_name = cfg.embeddings.model
    candidate_vecs = load_embeddings(conn, model_name, candidate_paper_ids)
    if not candidate_vecs:
        return []

    ratings = get_current_ratings(conn)
    rated_vecs = load_embeddings(conn, model_name, list(ratings))

    ids = list(candidate_vecs)
    X = np.stack([candidate_vecs[pid] for pid in ids])
    scores = ranker.predict_proba(X)

    ranked = []
    for pid, score in zip(ids, scores, strict=True):
        nearest_id, nearest_sim = nearest_reference(candidate_vecs[pid], rated_vecs)
        nearest_label = None
        if nearest_id is not None:
            nearest_label = "rated_up" if ratings[nearest_id] == 1 else "rated_down"
        ranked.append(RankedPaper(pid, float(score), nearest_id, nearest_sim, nearest_label))

    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked


def evaluate_holdout(conn, cfg: Config, seed: int = 0) -> dict | None:
    """Trains on a stratified split of your ratings, reports accuracy /
    precision / recall on the held-out remainder. `litdesk retrain` calls
    this — it never affects the ranker actually used for tomorrow's digest,
    which always trains fresh on the full rating set."""
    ratings = get_current_ratings(conn)
    if len(ratings) < cfg.ranker.min_ratings_for_classifier:
        return None

    positives = [pid for pid, label in ratings.items() if label == 1]
    negatives = [pid for pid, label in ratings.items() if label == -1]
    if len(positives) < 2 or len(negatives) < 2:
        return None  # need at least one of each in both train and test

    rng = random.Random(seed)

    def split(ids: list[int]) -> tuple[list[int], list[int]]:
        shuffled = ids[:]
        rng.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * cfg.ranker.held_out_fraction))
        return shuffled[n_test:], shuffled[:n_test]

    pos_train, pos_test = split(positives)
    neg_train, neg_test = split(negatives)

    arrays = _training_arrays(conn, cfg, pos_train, neg_train, exclude_from_random=set(ratings))
    if arrays is None:
        return None
    X_train, y_train, w_train = arrays

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train, sample_weight=w_train)

    model_name = cfg.embeddings.model
    test_embeddings = load_embeddings(conn, model_name, pos_test + neg_test)
    X_test, y_true = [], []
    for pid in pos_test:
        if pid in test_embeddings:
            X_test.append(test_embeddings[pid]); y_true.append(1)
    for pid in neg_test:
        if pid in test_embeddings:
            X_test.append(test_embeddings[pid]); y_true.append(0)
    if not X_test:
        return None

    y_pred = (model.predict_proba(np.stack(X_test))[:, 1] >= 0.5).astype(int)
    y_true_arr = np.array(y_true)

    tp = int(np.sum((y_pred == 1) & (y_true_arr == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true_arr == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true_arr == 1)))

    return {
        "accuracy": float(np.mean(y_pred == y_true_arr)),
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "n_train": len(y_train),
        "n_test": len(y_true_arr),
    }

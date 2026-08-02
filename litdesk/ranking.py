"""Cold-start ranking (spec section 5): rank candidates by cosine similarity
to the centroid of seed-paper embeddings. Every ranked paper also records its
single nearest seed for explainability — "If I can't tell why something
surfaced, I stop trusting the list and the whole thing dies."

Phase 3 adds a trained-classifier method alongside this once ratings exist;
this module stays the cold-start fallback either way.
"""

from __future__ import annotations

import numpy as np

from litdesk.embeddings import blob_to_vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def load_embeddings(conn, model_name: str, paper_ids: list[int] | None = None) -> dict[int, np.ndarray]:
    if paper_ids is None:
        rows = conn.execute("SELECT paper_id, vec FROM embeddings WHERE model = ?", (model_name,)).fetchall()
    else:
        placeholders = ",".join("?" for _ in paper_ids)
        rows = conn.execute(
            f"SELECT paper_id, vec FROM embeddings WHERE model = ? AND paper_id IN ({placeholders})",  # noqa: S608
            (model_name, *paper_ids),
        ).fetchall()
    return {r["paper_id"]: blob_to_vec(r["vec"]) for r in rows}


def seed_paper_ids(conn) -> list[int]:
    return [r["paper_id"] for r in conn.execute("SELECT paper_id FROM seeds").fetchall()]


def compute_centroid(vectors: list[np.ndarray]) -> np.ndarray:
    centroid = np.stack(vectors).mean(axis=0)
    norm = np.linalg.norm(centroid)
    return centroid / norm if norm > 0 else centroid


class RankedPaper:
    __slots__ = ("paper_id", "score", "nearest_paper_id", "nearest_similarity")

    def __init__(self, paper_id: int, score: float, nearest_paper_id: int | None, nearest_similarity: float | None):
        self.paper_id = paper_id
        self.score = score
        self.nearest_paper_id = nearest_paper_id
        self.nearest_similarity = nearest_similarity


def rank_by_centroid(
    conn,
    model_name: str,
    candidate_paper_ids: list[int],
    reference_paper_ids: list[int],
) -> list[RankedPaper]:
    """Reference papers are typically seeds; Phase 3 may pass seeds + rated-up
    papers instead. Returns candidates sorted by descending score. Candidates
    without a stored embedding (e.g. `litdesk embed` hasn't run yet) are
    silently skipped, not crashed on."""
    candidate_vecs = load_embeddings(conn, model_name, candidate_paper_ids)
    ref_vecs = load_embeddings(conn, model_name, reference_paper_ids)
    if not ref_vecs:
        raise ValueError("No embedded reference papers to rank against — load seeds and run `litdesk embed` first")

    centroid = compute_centroid(list(ref_vecs.values()))

    ranked = []
    for pid in candidate_paper_ids:
        vec = candidate_vecs.get(pid)
        if vec is None:
            continue
        score = cosine_similarity(vec, centroid)
        nearest_id, nearest_sim = None, None
        for ref_id, ref_vec in ref_vecs.items():
            sim = cosine_similarity(vec, ref_vec)
            if nearest_sim is None or sim > nearest_sim:
                nearest_id, nearest_sim = ref_id, sim
        ranked.append(RankedPaper(pid, score, nearest_id, nearest_sim))

    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked

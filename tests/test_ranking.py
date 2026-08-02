import math

import numpy as np
import pytest

from litdesk import ranking

MODEL = "test-model"


def test_cosine_similarity_identical_vectors_is_one():
    a = np.array([1.0, 2.0, 3.0])
    assert ranking.cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert ranking.cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors_is_minus_one():
    a = np.array([1.0, 0.0])
    assert ranking.cosine_similarity(a, -a) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_is_zero_not_nan():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    assert ranking.cosine_similarity(a, b) == 0.0


def test_compute_centroid_normalizes_to_unit_length():
    centroid = ranking.compute_centroid([np.array([2.0, 0.0]), np.array([0.0, 2.0])])
    assert math.isclose(np.linalg.norm(centroid), 1.0, rel_tol=1e-6)


def test_rank_by_centroid_orders_by_similarity_and_finds_nearest_seed(db_conn, insert_paper, insert_embedding, insert_seed):
    seed1 = insert_paper(db_conn, title="Seed near axis A")
    seed2 = insert_paper(db_conn, title="Seed near axis B")
    insert_seed(db_conn, seed1)
    insert_seed(db_conn, seed2)
    insert_embedding(db_conn, seed1, [1.0, 0.0], model=MODEL)
    insert_embedding(db_conn, seed2, [0.0, 1.0], model=MODEL)

    close_to_a = insert_paper(db_conn, title="Close to A")
    close_to_b = insert_paper(db_conn, title="Close to B")
    far_from_both = insert_paper(db_conn, title="Far from both")
    insert_embedding(db_conn, close_to_a, [0.9, 0.1], model=MODEL)
    insert_embedding(db_conn, close_to_b, [0.1, 0.9], model=MODEL)
    insert_embedding(db_conn, far_from_both, [-1.0, -1.0], model=MODEL)

    ranked = ranking.rank_by_centroid(
        db_conn, MODEL,
        candidate_paper_ids=[close_to_a, close_to_b, far_from_both],
        reference_paper_ids=ranking.seed_paper_ids(db_conn),
    )

    assert [r.paper_id for r in ranked][:2] == [close_to_a, close_to_b] or \
           [r.paper_id for r in ranked][:2] == [close_to_b, close_to_a]
    assert ranked[-1].paper_id == far_from_both

    by_id = {r.paper_id: r for r in ranked}
    assert by_id[close_to_a].nearest_paper_id == seed1
    assert by_id[close_to_b].nearest_paper_id == seed2


def test_rank_by_centroid_skips_candidates_without_embeddings(db_conn, insert_paper, insert_embedding, insert_seed):
    seed = insert_paper(db_conn)
    insert_seed(db_conn, seed)
    insert_embedding(db_conn, seed, [1.0, 0.0], model=MODEL)

    embedded = insert_paper(db_conn, title="Has embedding")
    insert_embedding(db_conn, embedded, [1.0, 0.0], model=MODEL)
    not_embedded = insert_paper(db_conn, title="No embedding yet")

    ranked = ranking.rank_by_centroid(db_conn, MODEL, [embedded, not_embedded], [seed])
    assert [r.paper_id for r in ranked] == [embedded]


def test_rank_by_centroid_raises_without_embedded_references(db_conn, insert_paper, insert_seed):
    seed = insert_paper(db_conn)
    insert_seed(db_conn, seed)  # seed exists but was never embedded
    candidate = insert_paper(db_conn)

    with pytest.raises(ValueError, match="reference"):
        ranking.rank_by_centroid(db_conn, MODEL, [candidate], [seed])

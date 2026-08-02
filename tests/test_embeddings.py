import numpy as np

from litdesk import embeddings
from litdesk.config import Config


def test_vec_blob_roundtrip():
    vec = np.array([0.1, -0.2, 3.5], dtype=np.float32)
    blob = embeddings.vec_to_blob(vec)
    assert isinstance(blob, bytes)
    restored = embeddings.blob_to_vec(blob)
    np.testing.assert_allclose(restored, vec, rtol=1e-6)


def test_paper_text_includes_sep_when_abstract_present():
    text = embeddings.paper_text("Title", "Abstract body")
    assert text == "Title [SEP] Abstract body"


def test_paper_text_falls_back_to_title_only_when_abstract_missing():
    assert embeddings.paper_text("Title only", None) == "Title only"
    assert embeddings.paper_text("Title only", "") == "Title only"


def test_embed_missing_only_processes_unembedded_papers(db_conn, insert_paper, insert_embedding, monkeypatch):
    p1 = insert_paper(db_conn, title="Paper One")
    p2 = insert_paper(db_conn, title="Paper Two")
    insert_embedding(db_conn, p1, [1.0, 0.0], model="sentence-transformers/all-MiniLM-L6-v2")

    seen_texts = []

    def fake_embed_texts(model_name, texts, batch_size=32):
        seen_texts.extend(texts)
        return np.array([[0.5, 0.5]] * len(texts), dtype=np.float32)

    monkeypatch.setattr(embeddings, "embed_texts", fake_embed_texts)

    cfg = Config()
    n = embeddings.embed_missing(db_conn, cfg)

    assert n == 1
    assert seen_texts == ["Paper Two"]
    row = db_conn.execute("SELECT model, vec FROM embeddings WHERE paper_id = ?", (p2,)).fetchone()
    assert row["model"] == cfg.embeddings.model
    np.testing.assert_allclose(embeddings.blob_to_vec(row["vec"]), [0.5, 0.5])


def test_embed_missing_reembeds_when_model_changes(db_conn, insert_paper, insert_embedding, monkeypatch):
    p1 = insert_paper(db_conn, title="Paper One")
    insert_embedding(db_conn, p1, [1.0, 0.0], model="old-model")

    monkeypatch.setattr(
        embeddings, "embed_texts",
        lambda model_name, texts, batch_size=32: np.array([[9.0, 9.0]] * len(texts), dtype=np.float32),
    )

    cfg = Config()
    cfg.embeddings.model = "new-model"
    n = embeddings.embed_missing(db_conn, cfg)

    assert n == 1
    row = db_conn.execute("SELECT model FROM embeddings WHERE paper_id = ?", (p1,)).fetchone()
    assert row["model"] == "new-model"


def test_embed_missing_is_noop_when_nothing_to_do(db_conn, insert_paper, insert_embedding, monkeypatch):
    p1 = insert_paper(db_conn)
    insert_embedding(db_conn, p1, [1.0, 0.0], model="sentence-transformers/all-MiniLM-L6-v2")

    def boom(*a, **k):
        raise AssertionError("embed_texts should not be called when nothing is missing")

    monkeypatch.setattr(embeddings, "embed_texts", boom)
    cfg = Config()
    assert embeddings.embed_missing(db_conn, cfg) == 0

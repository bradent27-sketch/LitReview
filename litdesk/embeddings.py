"""Embedding generation (spec section 5, step 1). Default model is MiniLM —
fast, no `adapters` dependency. SPECTER2 is the documented upgrade path once
the pipeline is validated; swap `embeddings.model` in config and rerun
`litdesk embed` — every paper gets re-embedded since the stored `model` name
no longer matches, so mixed embeddings never get compared.
"""

from __future__ import annotations

import logging

import numpy as np

from litdesk.config import Config

logger = logging.getLogger("litdesk.embeddings")

_model_cache: dict[str, object] = {}


def get_model(model_name: str):
    if model_name not in _model_cache:
        # Imported lazily: sentence-transformers (and torch) are the heaviest
        # deps in this project and only Phase 2+ commands need them.
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s (first run downloads model weights)", model_name)
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def paper_text(title: str, abstract: str | None) -> str:
    # spec 5.1: "Embed title + '[SEP]' + abstract". Abstracts are frequently
    # missing (gotcha, section 7) — rank on title alone rather than crashing.
    return f"{title} [SEP] {abstract}" if abstract else title


def embed_texts(model_name: str, texts: list[str], batch_size: int = 32) -> np.ndarray:
    model = get_model(model_name)
    return np.asarray(
        model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True),
        dtype=np.float32,
    )


def vec_to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def embed_missing(conn, cfg: Config) -> int:
    """Embeds every paper that doesn't yet have a vector under the currently
    configured model. Returns the number of papers embedded."""
    model_name = cfg.embeddings.model
    rows = conn.execute(
        """SELECT p.id, p.title, p.abstract FROM papers p
           LEFT JOIN embeddings e ON e.paper_id = p.id AND e.model = ?
           WHERE e.paper_id IS NULL""",
        (model_name,),
    ).fetchall()
    if not rows:
        return 0

    texts = [paper_text(r["title"], r["abstract"]) for r in rows]
    vecs = embed_texts(model_name, texts, batch_size=cfg.embeddings.batch_size)
    for row, vec in zip(rows, vecs, strict=True):
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (paper_id, model, vec) VALUES (?, ?, ?)",
            (row["id"], model_name, vec_to_blob(vec)),
        )
    conn.commit()
    return len(rows)

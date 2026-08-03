from __future__ import annotations

from litdesk import pipeline
from litdesk.config import Config


def _cfg(tmp_path):
    cfg = Config()
    cfg.digest.output_dir = str(tmp_path / "digests")
    cfg.seeds.dois = []
    cfg.seeds.bibtex_path = None
    return cfg


def _fake_ingest_ok(*a, **k):
    return [{"source": "europepmc", "fetched": 1, "new": 1, "deduped": 0, "error": None}]


def _fake_ingest_error(*a, **k):
    return [{"source": "europepmc", "fetched": 0, "new": 0, "deduped": 0, "error": "boom"}]


def _noop_embed(*a, **k):
    return 0


def test_run_full_pipeline_builds_digest_when_seeds_exist(db_conn, insert_paper, insert_embedding, insert_seed, monkeypatch, tmp_path):
    seed_id = insert_paper(db_conn, title="Seed Paper")
    insert_embedding(db_conn, seed_id, [1.0, 0.0], model="sentence-transformers/all-MiniLM-L6-v2")
    insert_seed(db_conn, seed_id)
    candidate_id = insert_paper(db_conn, title="Candidate Paper", doi="10.1/candidate")
    insert_embedding(db_conn, candidate_id, [0.9, 0.1], model="sentence-transformers/all-MiniLM-L6-v2")

    monkeypatch.setattr(pipeline, "run_ingest", _fake_ingest_ok)
    monkeypatch.setattr(pipeline, "embed_missing", _noop_embed)

    logs = []
    cfg = _cfg(tmp_path)
    result = pipeline.run_full_pipeline(db_conn, cfg, log=logs.append)

    assert result["had_error"] is False
    assert result["data"] is not None
    assert result["path"].exists()
    assert any("ingest europepmc" in line for line in logs)
    assert any("paper(s) ->" in line for line in logs)


def test_run_full_pipeline_flags_ingest_errors_but_still_builds_digest(db_conn, insert_paper, insert_embedding, insert_seed, monkeypatch, tmp_path):
    seed_id = insert_paper(db_conn, title="Seed Paper")
    insert_embedding(db_conn, seed_id, [1.0, 0.0], model="sentence-transformers/all-MiniLM-L6-v2")
    insert_seed(db_conn, seed_id)
    candidate_id = insert_paper(db_conn, title="Candidate Paper", doi="10.1/candidate")
    insert_embedding(db_conn, candidate_id, [0.9, 0.1], model="sentence-transformers/all-MiniLM-L6-v2")

    monkeypatch.setattr(pipeline, "run_ingest", _fake_ingest_error)
    monkeypatch.setattr(pipeline, "embed_missing", _noop_embed)

    logs = []
    result = pipeline.run_full_pipeline(db_conn, _cfg(tmp_path), log=logs.append)

    assert result["had_error"] is True
    assert result["data"] is not None  # ingest failing doesn't block a digest from existing candidates
    assert any("ERROR: boom" in line for line in logs)


def test_run_full_pipeline_reports_no_data_when_no_seeds_yet(db_conn, insert_paper, insert_embedding, monkeypatch, tmp_path):
    # A candidate paper exists (so build_digest doesn't just short-circuit on
    # "nothing to rank"), but with zero seeds, ranking has nothing to
    # bootstrap the cold-start centroid from.
    candidate_id = insert_paper(db_conn, title="Candidate Paper", doi="10.1/candidate")
    insert_embedding(db_conn, candidate_id, [0.9, 0.1], model="sentence-transformers/all-MiniLM-L6-v2")

    monkeypatch.setattr(pipeline, "run_ingest", _fake_ingest_ok)
    monkeypatch.setattr(pipeline, "embed_missing", _noop_embed)

    logs = []
    result = pipeline.run_full_pipeline(db_conn, _cfg(tmp_path), log=logs.append)

    assert result["had_error"] is False  # ingest itself succeeded; nothing to build a digest from is separate
    assert result["data"] is None
    assert result["path"] is None
    assert "No seed papers yet" in result["error"]
    assert any("Can't build a digest yet" in line for line in logs)

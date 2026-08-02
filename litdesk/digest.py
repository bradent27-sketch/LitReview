"""Phase 2: candidate selection, ranking persistence, static HTML rendering.

`build_digest` handles ranking + bookkeeping; `render_html` handles output.
`generate_digest` chains both — that's what the CLI calls.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from litdesk import classifier, ranking
from litdesk.config import Config

TEMPLATE_DIR = Path(__file__).parent / "templates"


def candidate_paper_ids(conn) -> list[int]:
    """Papers never rated, never seeded, and never shown in a previous
    digest — decoupled from any date window so a lagged ingest or a gap in
    runs never silently drops or repeats a paper."""
    rows = conn.execute(
        """
        SELECT p.id FROM papers p
        WHERE p.id NOT IN (SELECT paper_id FROM ratings)
          AND p.id NOT IN (SELECT paper_id FROM seeds)
          AND p.id NOT IN (
              SELECT CAST(je.value AS INTEGER)
              FROM digests, json_each(digests.paper_ids) AS je
          )
        ORDER BY p.id
        """
    ).fetchall()
    return [r["id"] for r in rows]


def _band_for_rank(position: int, total: int) -> str:
    if total <= 0:
        return "low"
    fraction = position / total
    if fraction < 1 / 3:
        return "high"
    if fraction < 2 / 3:
        return "medium"
    return "low"


def _forced_include_ids(conn, cfg: Config, candidate_ids: list[int], already_selected: set[int]) -> set[int]:
    if not cfg.journals_always_include or not candidate_ids:
        return set()
    placeholders = ",".join("?" for _ in candidate_ids)
    rows = conn.execute(
        f"SELECT id, journal FROM papers WHERE id IN ({placeholders})",  # noqa: S608
        candidate_ids,
    ).fetchall()
    journal_set = set(cfg.journals_always_include)
    return {r["id"] for r in rows if r["journal"] in journal_set and r["id"] not in already_selected}


def _rank_candidates(conn, cfg: Config, candidate_ids: list[int]) -> tuple[str, list[ranking.RankedPaper]]:
    """Trained classifier once there are enough ratings of both kinds
    (spec section 5, step 2-4); centroid-similarity cold start otherwise.
    Falls back to centroid if training can't proceed yet (e.g. ratings
    exist but are all the same label) rather than erroring."""
    n_rated = len(classifier.get_current_ratings(conn))
    if n_rated >= cfg.ranker.min_ratings_for_classifier:
        trained = classifier.train(conn, cfg)
        if trained is not None:
            return "classifier", classifier.rank_by_classifier(conn, cfg, candidate_ids, trained)

    seeds = ranking.seed_paper_ids(conn)
    if not seeds:
        raise ValueError("No seed papers yet — set seeds.dois in config.yaml and run `litdesk seeds`")
    return "centroid", ranking.rank_by_centroid(conn, cfg.embeddings.model, candidate_ids, seeds)


def build_digest(conn, cfg: Config) -> dict:
    """Ranks today's candidates and persists rankings + a digests row.
    Returns the dict `render_html` expects. Run `litdesk seeds` and
    `litdesk embed` first (or use `litdesk digest`, which chains everything)."""
    run_date = dt.date.today().isoformat()
    candidate_ids = candidate_paper_ids(conn)

    if not candidate_ids:
        return {"entries": [], "run_date": run_date, "is_catchup": False, "gap_days": 0, "n_candidates_total": 0}

    method, ranked = _rank_candidates(conn, cfg, candidate_ids)
    if not ranked:
        raise ValueError(
            "Candidates exist but none have embeddings yet — run `litdesk embed` before `litdesk digest`"
        )

    top = ranked[: cfg.digest.top_n]
    band_by_id = {r.paper_id: _band_for_rank(i, len(top)) for i, r in enumerate(top)}
    selected_ids = set(band_by_id)

    forced_ids = _forced_include_ids(conn, cfg, candidate_ids, selected_ids)
    ranked_by_id = {r.paper_id: r for r in ranked}
    selected = [ranked_by_id[pid] for pid in selected_ids | forced_ids if pid in ranked_by_id]
    selected.sort(key=lambda r: r.score, reverse=True)

    paper_rows = _fetch_papers(conn, [r.paper_id for r in selected])
    nearest_ids = {r.nearest_paper_id for r in selected if r.nearest_paper_id is not None}
    nearest_titles = _fetch_titles(conn, nearest_ids)

    entries = []
    for r in selected:
        p = paper_rows.get(r.paper_id)
        if p is None:
            continue
        entries.append({
            "id": p["id"],
            "title": p["title"],
            "journal": p["journal"] or ("Preprint" if p["is_preprint"] else None),
            "date_published": p["date_published"],
            "abstract": p["abstract"],
            "url": p["url"],
            "authors": json.loads(p["authors"]) if p["authors"] else [],
            "is_preprint": bool(p["is_preprint"]),
            "score": r.score,
            "band": band_by_id.get(r.paper_id),
            "forced_include": r.paper_id in forced_ids and r.paper_id not in selected_ids,
            "nearest_title": nearest_titles.get(r.nearest_paper_id),
            "nearest_similarity": r.nearest_similarity,
            "nearest_label": r.nearest_label,
        })

    model_version = cfg.embeddings.model
    now = dt.datetime.utcnow().isoformat()
    for r in selected:
        conn.execute(
            """INSERT INTO rankings (paper_id, run_date, score, method, nearest_paper_id,
                   nearest_similarity, nearest_label, model_version)
               VALUES (?,?,?,?,?,?,?,?)""",
            (r.paper_id, run_date, r.score, method, r.nearest_paper_id, r.nearest_similarity, r.nearest_label, model_version),
        )
    conn.execute(
        "INSERT INTO digests (run_date, paper_ids, model_version, is_catchup, created_at) VALUES (?,?,?,?,?)",
        (run_date, json.dumps([r.paper_id for r in selected]), model_version, 0, now),
    )
    conn.commit()

    return {
        "entries": entries,
        "run_date": run_date,
        "is_catchup": False,
        "gap_days": 0,
        "n_candidates_total": len(candidate_ids),
    }


def _fetch_papers(conn, paper_ids: list[int]) -> dict:
    if not paper_ids:
        return {}
    placeholders = ",".join("?" for _ in paper_ids)
    rows = conn.execute(f"SELECT * FROM papers WHERE id IN ({placeholders})", paper_ids).fetchall()  # noqa: S608
    return {row["id"]: row for row in rows}


def _fetch_titles(conn, paper_ids: set[int]) -> dict:
    if not paper_ids:
        return {}
    placeholders = ",".join("?" for _ in paper_ids)
    rows = conn.execute(f"SELECT id, title FROM papers WHERE id IN ({placeholders})", list(paper_ids)).fetchall()  # noqa: S608
    return {row["id"]: row["title"] for row in rows}


def render_html(digest_data: dict, output_dir: str | Path) -> Path:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("digest.html.jinja")
    html = template.render(**digest_data)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{digest_data['run_date']}.html"
    out_path.write_text(html)
    (output_dir / "latest.html").write_text(html)
    return out_path

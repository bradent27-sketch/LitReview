"""Candidate selection, ranking persistence, static HTML rendering.

`build_digest` handles ranking + bookkeeping; `render_html` handles output.
cli.py's `_build_and_render_digest` chains both plus the optional Phase 4
LLM enrichment step.
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
    """Papers never rated, never seeded, never flagged retracted, and never
    shown in a previous digest — decoupled from any date window so a lagged
    ingest or a gap in runs never silently drops or repeats a paper."""
    rows = conn.execute(
        """
        SELECT p.id FROM papers p
        WHERE p.retracted = 0
          AND p.id NOT IN (SELECT paper_id FROM ratings)
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


def _gap_days_since_last_digest(conn) -> int | None:
    row = conn.execute("SELECT run_date FROM digests ORDER BY id DESC LIMIT 1").fetchone()
    if row is None:
        return None
    last = dt.date.fromisoformat(row["run_date"])
    return (dt.date.today() - last).days


def _forced_include(
    conn, cfg: Config, candidate_ids: list[int], already_selected: set[int],
) -> dict[int, str]:
    """Returns {paper_id: reason} for candidates that belong in the digest
    regardless of score: a configured always-include journal, or a
    watchlisted author (spec Phase 5). `watchlists.labs` isn't matched here
    yet — affiliation strings aren't normalized enough to match reliably."""
    if (not cfg.journals_always_include and not cfg.watchlists.authors) or not candidate_ids:
        return {}
    placeholders = ",".join("?" for _ in candidate_ids)
    rows = conn.execute(
        f"SELECT id, journal, authors FROM papers WHERE id IN ({placeholders})",  # noqa: S608
        candidate_ids,
    ).fetchall()
    journal_set = set(cfg.journals_always_include)
    watch_authors = {a.lower() for a in cfg.watchlists.authors}

    reasons: dict[int, str] = {}
    for row in rows:
        if row["id"] in already_selected:
            continue
        if row["journal"] in journal_set:
            reasons[row["id"]] = f"always include: {row['journal']}"
            continue
        if not watch_authors:
            continue
        paper_authors = [a.lower() for a in json.loads(row["authors"] or "[]")]
        hit = next((wa for wa in watch_authors if any(wa in pa for pa in paper_authors)), None)
        if hit:
            reasons[row["id"]] = f"watchlist: {hit.title()}"
    return reasons


def _scoop_alerts(conn, cfg: Config, candidate_ids: list[int]) -> set[int]:
    """Candidates unusually similar to `project_description` (spec Phase 5:
    "flag anything above a similarity threshold ... separately and
    loudly"). Embeds the description fresh every run rather than caching it
    — one extra embed call, never a stale comparison after an edit."""
    if not cfg.scoop_alarm.enabled or not cfg.project_description.strip() or not candidate_ids:
        return set()
    from litdesk import embeddings as emb

    model_name = cfg.embeddings.model
    project_vec = emb.embed_texts(model_name, [cfg.project_description])[0]
    candidate_vecs = ranking.load_embeddings(conn, model_name, candidate_ids)
    return {
        pid for pid, vec in candidate_vecs.items()
        if ranking.cosine_similarity(vec, project_vec) >= cfg.scoop_alarm.threshold
    }


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
    gap_days = _gap_days_since_last_digest(conn)
    candidate_ids = candidate_paper_ids(conn)

    if not candidate_ids:
        return {"entries": [], "run_date": run_date, "is_catchup": False, "gap_days": 0, "n_candidates_total": 0}

    method, ranked = _rank_candidates(conn, cfg, candidate_ids)
    if not ranked:
        raise ValueError(
            "Candidates exist but none have embeddings yet — run `litdesk embed` before `litdesk digest`"
        )

    is_catchup = gap_days is not None and gap_days > cfg.digest.catchup_gap_days
    top_n = cfg.digest.catchup_top_n if is_catchup else cfg.digest.top_n

    top = ranked[:top_n]
    band_by_id = {r.paper_id: _band_for_rank(i, len(top)) for i, r in enumerate(top)}
    selected_ids = set(band_by_id)

    forced_reasons = _forced_include(conn, cfg, candidate_ids, selected_ids)
    scoop_ids = _scoop_alerts(conn, cfg, candidate_ids)
    ranked_by_id = {r.paper_id: r for r in ranked}
    selected = [
        ranked_by_id[pid] for pid in selected_ids | set(forced_reasons) | scoop_ids if pid in ranked_by_id
    ]
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
            "forced_include": r.paper_id in forced_reasons and r.paper_id not in selected_ids,
            "forced_reason": forced_reasons.get(r.paper_id),
            "scoop_alert": r.paper_id in scoop_ids,
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
    for pid in scoop_ids:
        conn.execute(
            "INSERT INTO notifications (paper_id, type, message, created_at) VALUES (?, 'scoop_alarm', ?, ?)",
            (pid, f"Similarity to your active project is >= {cfg.scoop_alarm.threshold}", now),
        )
    conn.execute(
        "INSERT INTO digests (run_date, paper_ids, model_version, is_catchup, created_at) VALUES (?,?,?,?,?)",
        (run_date, json.dumps([r.paper_id for r in selected]), model_version, int(is_catchup), now),
    )
    conn.commit()

    return {
        "entries": entries,
        "run_date": run_date,
        "is_catchup": is_catchup,
        "gap_days": gap_days or 0,
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

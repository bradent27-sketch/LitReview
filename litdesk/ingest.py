"""Phase 1: fetch from every configured source, dedup, write to SQLite, log
a `runs` row per source no matter what happens (spec gotcha: a silent cron
failure going unnoticed for weeks is the likeliest way this project dies)."""

from __future__ import annotations

import datetime as dt
import json
import logging

from litdesk import dedup
from litdesk.config import Config, resolve_path
from litdesk.http_client import CachedSession
from litdesk.models import RawPaper
from litdesk.sources import biorxiv, europepmc

logger = logging.getLogger("litdesk.ingest")


def _now_iso() -> str:
    return dt.datetime.utcnow().isoformat()


def upsert_paper(conn, paper: RawPaper) -> tuple[int, bool]:
    """Insert a new paper row, or merge into an existing one. Returns
    (paper_id, is_new). See dedup.py for the merge-direction rules."""
    existing = dedup.find_match(conn, paper)
    doi = dedup.normalize_doi(paper.doi)
    pub_doi = dedup.normalize_doi(paper.published_doi)
    title_key = dedup.compute_dedup_title_key(paper.title, paper.authors, paper.date_published)
    raw_json = json.dumps(paper.raw) if paper.raw is not None else None

    if existing is None:
        cur = conn.execute(
            """INSERT INTO papers (doi, pmid, pmcid, source, is_preprint, published_doi,
                   title, abstract, authors, journal, date_published, date_published_kind,
                   date_first_seen, date_ingested, url, pdf_url, mesh_terms, dedup_title_key, raw)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (doi, paper.pmid, paper.pmcid, paper.source, int(paper.is_preprint), pub_doi,
             paper.title, paper.abstract, json.dumps(paper.authors), paper.journal,
             paper.date_published, paper.date_published_kind, paper.date_published, _now_iso(),
             paper.url, paper.pdf_url, json.dumps(paper.mesh_terms), title_key, raw_json),
        )
        return cur.lastrowid, True

    row_id = existing["id"]
    updates: dict = {}

    if not paper.is_preprint:
        # A journal-quality record always wins the metadata fields, regardless
        # of whether the preprint or the journal version was ingested first.
        updates.update(
            title=paper.title, abstract=paper.abstract, authors=json.dumps(paper.authors),
            journal=paper.journal, date_published=paper.date_published,
            date_published_kind=paper.date_published_kind, source=paper.source,
            is_preprint=0, url=paper.url, pdf_url=paper.pdf_url,
            mesh_terms=json.dumps(paper.mesh_terms),
            dedup_title_key=title_key or existing["dedup_title_key"],
            raw=raw_json if raw_json is not None else existing["raw"],
        )
        if paper.pmid and not existing["pmid"]:
            updates["pmid"] = paper.pmid
        if paper.pmcid and not existing["pmcid"]:
            updates["pmcid"] = paper.pmcid
        if not existing["published_doi"]:
            journal_doi = doi if (doi and doi != existing["doi"]) else pub_doi
            if journal_doi:
                updates["published_doi"] = journal_doi
        if not existing["doi"] and doi:
            updates["doi"] = doi
    else:
        # Incoming record is itself a preprint: never clobber metadata that's
        # already there, just backfill the journal-DOI link if we learned it.
        if not existing["published_doi"] and pub_doi:
            updates["published_doi"] = pub_doi
        if not existing["pmid"] and paper.pmid:
            updates["pmid"] = paper.pmid

    candidate_dates = [d for d in (existing["date_first_seen"], existing["date_published"], paper.date_published) if d]
    if candidate_dates:
        updates["date_first_seen"] = min(candidate_dates)

    if updates:
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        conn.execute(f"UPDATE papers SET {set_clause} WHERE id = ?", (*updates.values(), row_id))  # noqa: S608

        if "published_doi" in updates:
            _notify_if_rated_up(conn, row_id, updates["published_doi"])

    return row_id, False


def _notify_if_rated_up(conn, paper_id: int, published_doi: str) -> None:
    """Preprint->published notifier (spec Phase 5): fires the moment a
    paper's published_doi is set for the first time, if the paper is
    something the user rated up. Silent for everything else — most papers
    are unrated, and this isn't interesting unless you cared enough to
    thumbs-up the preprint."""
    rating = conn.execute(
        "SELECT label FROM ratings WHERE paper_id = ? ORDER BY id DESC LIMIT 1", (paper_id,)
    ).fetchone()
    if rating is None or rating["label"] != 1:
        return
    conn.execute(
        "INSERT INTO notifications (paper_id, type, message, created_at) VALUES (?, 'preprint_published', ?, ?)",
        (paper_id, f"A preprint you rated up now has a journal DOI: {published_doi}", _now_iso()),
    )


def log_run(conn, source: str, started_at: str, n_fetched: int, n_new: int, n_deduped: int, error: str | None = None) -> None:
    conn.execute(
        "INSERT INTO runs (started_at, finished_at, source, n_fetched, n_new, n_deduped, error) VALUES (?,?,?,?,?,?,?)",
        (started_at, _now_iso(), source, n_fetched, n_new, n_deduped, error),
    )
    conn.commit()


def ingest_europepmc(conn, cfg: Config) -> dict:
    started = _now_iso()
    session = CachedSession(
        cache_dir=str(resolve_path(cfg.cache_dir) / "europepmc"),
        requests_per_sec=cfg.rate_limits.europepmc_per_sec,
        ttl_hours=cfg.cache_ttl_hours,
    )
    n_fetched = n_new = n_deduped = 0
    error = None
    try:
        for query in cfg.queries:
            papers = europepmc.search(session, query, lookback_days=cfg.lookback_days)
            n_fetched += len(papers)
            for p in papers:
                _, is_new = upsert_paper(conn, p)
                n_new += is_new
                n_deduped += not is_new
            conn.commit()
    except Exception as exc:  # noqa: BLE001 - one source failing must not kill the run or go unlogged
        error = str(exc)
        logger.exception("europepmc ingest failed")
    log_run(conn, "europepmc", started, n_fetched, n_new, n_deduped, error)
    return {"source": "europepmc", "fetched": n_fetched, "new": n_new, "deduped": n_deduped, "error": error}


def ingest_biorxiv_server(conn, cfg: Config, server: str) -> dict:
    started = _now_iso()
    session = CachedSession(
        cache_dir=str(resolve_path(cfg.cache_dir) / server),
        requests_per_sec=cfg.rate_limits.biorxiv_per_sec,
        ttl_hours=cfg.cache_ttl_hours,
    )
    n_fetched = n_new = n_deduped = 0
    error = None
    try:
        categories = cfg.biorxiv_categories or [None]
        seen: set[str] = set()
        for category in categories:
            papers = biorxiv.fetch(session, server=server, lookback_days=cfg.lookback_days, category=category)
            for p in papers:
                dedup_key = dedup.normalize_doi(p.doi) or f"notitle:{p.title}"
                if dedup_key in seen:
                    continue  # same paper matched by more than one configured category
                seen.add(dedup_key)
                n_fetched += 1
                _, is_new = upsert_paper(conn, p)
                n_new += is_new
                n_deduped += not is_new
            conn.commit()
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        logger.exception("%s ingest failed", server)
    log_run(conn, server, started, n_fetched, n_new, n_deduped, error)
    return {"source": server, "fetched": n_fetched, "new": n_new, "deduped": n_deduped, "error": error}


def run_ingest(conn, cfg: Config) -> list[dict]:
    results = []
    if cfg.sources.europepmc:
        results.append(ingest_europepmc(conn, cfg))
    if cfg.sources.biorxiv:
        results.append(ingest_biorxiv_server(conn, cfg, "biorxiv"))
    if cfg.sources.medrxiv:
        results.append(ingest_biorxiv_server(conn, cfg, "medrxiv"))
    return results

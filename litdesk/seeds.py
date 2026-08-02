"""Seed paper loading for cold-start ranking (spec section 5, "Cold start").

Seeds often predate any lookback window (your own decade-old papers, a
classic from your quals list) and won't match a standing query, so this
fetches them directly by DOI rather than relying on `litdesk ingest` to
have already pulled them in.
"""

from __future__ import annotations

import datetime as dt
import re

from litdesk import dedup, ingest
from litdesk.config import Config, resolve_path
from litdesk.http_client import CachedSession
from litdesk.sources import europepmc

# Matches doi = {10.xxxx/...} or doi = "10.xxxx/..." in a BibTeX entry.
_BIBTEX_DOI_RE = re.compile(r'doi\s*=\s*[{"]([^}",]+)[}"]', re.IGNORECASE)


def extract_dois_from_bibtex(text: str) -> list[str]:
    dois = []
    for match in _BIBTEX_DOI_RE.findall(text):
        normalized = dedup.normalize_doi(match)
        if normalized:
            dois.append(normalized)
    return dois


def make_session(cfg: Config) -> CachedSession:
    return CachedSession(
        cache_dir=str(resolve_path(cfg.cache_dir) / "europepmc"),
        requests_per_sec=cfg.rate_limits.europepmc_per_sec,
        ttl_hours=cfg.cache_ttl_hours,
    )


def add_seed_by_doi(
    conn, cfg: Config, doi: str, note: str | None = None, session: CachedSession | None = None
) -> tuple[int | None, str]:
    """Returns (paper_id, status). status is one of: 'fetched' (new paper
    pulled in just for this), 'linked_existing' (already in `papers`),
    'already_seed' (no-op, safe to rerun), 'not_found'."""
    normalized = dedup.normalize_doi(doi)
    if not normalized:
        return None, "not_found"

    row = conn.execute("SELECT id FROM papers WHERE doi = ?", (normalized,)).fetchone()
    if row is None:
        session = session or make_session(cfg)
        paper = europepmc.fetch_by_doi(session, normalized)
        if paper is None:
            return None, "not_found"
        paper_id, _ = ingest.upsert_paper(conn, paper)
        conn.commit()
        status = "fetched"
    else:
        paper_id = row["id"]
        status = "linked_existing"

    already = conn.execute("SELECT 1 FROM seeds WHERE paper_id = ?", (paper_id,)).fetchone()
    if already:
        return paper_id, "already_seed"

    conn.execute(
        "INSERT INTO seeds (paper_id, note, added_at) VALUES (?, ?, ?)",
        (paper_id, note, dt.datetime.utcnow().isoformat()),
    )
    conn.commit()
    return paper_id, status


def load_seeds_from_config(conn, cfg: Config) -> list[dict]:
    """Adds every DOI in cfg.seeds.dois, plus any extracted from
    cfg.seeds.bibtex_path if set. Idempotent — safe to rerun."""
    dois = list(cfg.seeds.dois)
    if cfg.seeds.bibtex_path:
        bibtex_path = resolve_path(cfg.seeds.bibtex_path)
        if bibtex_path.exists():
            dois.extend(extract_dois_from_bibtex(bibtex_path.read_text()))

    session = make_session(cfg)
    results = []
    for doi in dois:
        paper_id, status = add_seed_by_doi(conn, cfg, doi, session=session)
        results.append({"doi": doi, "paper_id": paper_id, "status": status})
    return results

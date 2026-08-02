"""bioRxiv / medRxiv details API client — see spec section 3.2.

Europe PMC ingests these via Crossref within ~24h, so this client exists
purely for same-day coverage. No API key; throttle to ~1 req/s regardless
(the CachedSession's rate limiter enforces this via config).
"""

from __future__ import annotations

import logging

from litdesk.http_client import CachedSession
from litdesk.models import RawPaper

logger = logging.getLogger("litdesk.sources.biorxiv")

BASE_URL = "https://api.biorxiv.org/details"
PAGE_SIZE = 100
MAX_PAGES_SAFETY = 100

DOMAIN = {"biorxiv": "www.biorxiv.org", "medrxiv": "www.medrxiv.org"}


def _split_authors(authors_field: str) -> list[str]:
    # bioRxiv format: "Doe, J.; Smith, A." -> ["Doe, J.", "Smith, A."]
    return [a.strip() for a in (authors_field or "").split(";") if a.strip()]


def _normalize_published(published: str | None) -> str | None:
    if not published or published.strip().upper() == "NA":
        return None
    return published.strip()


def _pdf_url(server: str, doi: str, version: str | None) -> str | None:
    if not doi:
        return None
    domain = DOMAIN.get(server, DOMAIN["biorxiv"])
    v = version or "1"
    return f"https://{domain}/content/{doi}v{v}.full.pdf"


def parse_entry(entry: dict, server: str) -> RawPaper:
    doi = entry.get("doi")
    published_doi = _normalize_published(entry.get("published"))
    return RawPaper(
        source=server,
        title=entry.get("title") or "(no title)",
        doi=doi,
        pmid=None,
        pmcid=None,
        is_preprint=True,
        published_doi=published_doi,
        abstract=entry.get("abstract"),
        authors=_split_authors(entry.get("authors", "")),
        journal=None,
        date_published=entry.get("date"),
        date_published_kind="posted",
        url=f"https://doi.org/{doi}" if doi else None,
        pdf_url=_pdf_url(server, doi, entry.get("version")),
        mesh_terms=[],
        raw=entry,
    )


def fetch(
    session: CachedSession,
    server: str = "biorxiv",
    lookback_days: int = 7,
    category: str | None = None,
    max_pages: int = MAX_PAGES_SAFETY,
) -> list[RawPaper]:
    papers: list[RawPaper] = []
    cursor = 0
    for page in range(max_pages):
        url = f"{BASE_URL}/{server}/{lookback_days}d/{cursor}"
        params = {"category": category} if category else None
        body = session.get_json(url, params=params)

        collection = body.get("collection", [])
        papers.extend(parse_entry(e, server) for e in collection)

        messages = body.get("messages", [{}])
        total = messages[0].get("total") if messages else None

        if len(collection) == 0:
            break
        cursor += PAGE_SIZE
        if total is not None and cursor >= int(total):
            break
    else:
        logger.warning("biorxiv: hit max_pages=%d safety cap for server=%s", max_pages, server)

    return papers

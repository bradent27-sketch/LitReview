"""Europe PMC REST client. One index over MEDLINE (MED), PMC full text (PMC),
and preprints (PPR, incl. bioRxiv/medRxiv) — see spec section 3.1.

Query syntax is Lucene-ish and case-sensitive on field names (TITLE:,
ABSTRACT:, AUTH:, MESH:, SRC:, ...). We don't force a SRC: filter here —
add one inline in your query string if you want to restrict a particular
standing query to MEDLINE only. Prototype queries at
https://europepmc.org/advancesearch before adding them to config.
"""

from __future__ import annotations

import datetime as dt
import logging

from litdesk.http_client import CachedSession
from litdesk.models import RawPaper

logger = logging.getLogger("litdesk.sources.europepmc")

BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
PAGE_SIZE = 100
MAX_PAGES_SAFETY = 100  # 10k results per query, generous ceiling against runaway loops


def date_range_for_lookback(lookback_days: int, today: dt.date | None = None) -> tuple[str, str]:
    today = today or dt.date.today()
    start = today - dt.timedelta(days=lookback_days)
    return start.isoformat(), today.isoformat()


def build_query(user_query: str, start_date: str, end_date: str) -> str:
    return f'({user_query}) AND FIRST_PDATE:[{start_date} TO {end_date}] sort_date:y'


def _extract_authors(result: dict) -> list[str]:
    author_list = result.get("authorList", {}).get("author", [])
    names = [a["fullName"] for a in author_list if a.get("fullName")]
    if names:
        return names
    author_string = result.get("authorString", "")
    return [a.strip() for a in author_string.split(",") if a.strip()]


def _extract_mesh(result: dict) -> list[str]:
    heading_list = result.get("meshHeadingList", {}).get("meshHeading", [])
    return [h["descriptorName"] for h in heading_list if h.get("descriptorName")]


def _extract_pdf_url(result: dict) -> str | None:
    urls = result.get("fullTextUrlList", {}).get("fullTextUrl", [])
    for u in urls:
        if u.get("documentStyle") == "pdf":
            return u.get("url")
    return None


def _extract_date(result: dict) -> tuple[str | None, str | None]:
    """Returns (date_published, kind). Europe PMC blends epub/print dates
    under firstPublicationDate; we can only reliably distinguish "posted"
    (preprints) from everything else without a second API call."""
    date = result.get("firstPublicationDate")
    if not date and result.get("pubYear"):
        date = f"{result['pubYear']}-01-01"
    kind = "posted" if result.get("source") == "PPR" else "epub"
    return date, kind


def parse_result(result: dict) -> RawPaper:
    date_published, date_kind = _extract_date(result)
    source_code = result.get("source", "")
    ext_id = result.get("id", "")
    return RawPaper(
        source="europepmc",
        title=result.get("title") or "(no title)",
        doi=result.get("doi"),
        pmid=result.get("pmid"),
        pmcid=result.get("pmcid"),
        is_preprint=(source_code == "PPR"),
        published_doi=None,  # Europe PMC doesn't expose this link; bioRxiv client does
        abstract=result.get("abstractText"),
        authors=_extract_authors(result),
        journal=result.get("journalInfo", {}).get("journal", {}).get("title"),
        date_published=date_published,
        date_published_kind=date_kind,
        url=f"https://europepmc.org/article/{source_code}/{ext_id}" if source_code and ext_id else None,
        pdf_url=_extract_pdf_url(result),
        mesh_terms=_extract_mesh(result),
        raw=result,
    )


def search(
    session: CachedSession,
    query: str,
    lookback_days: int = 7,
    max_pages: int = MAX_PAGES_SAFETY,
) -> list[RawPaper]:
    start_date, end_date = date_range_for_lookback(lookback_days)
    full_query = build_query(query, start_date, end_date)

    papers: list[RawPaper] = []
    cursor_mark = "*"
    for page in range(max_pages):
        params = {
            "query": full_query,
            "format": "json",
            "resultType": "core",
            "pageSize": PAGE_SIZE,
            "cursorMark": cursor_mark,
        }
        body = session.get_json(BASE_URL, params=params)
        results = body.get("resultList", {}).get("result", [])
        papers.extend(parse_result(r) for r in results)

        next_cursor = body.get("nextCursorMark")
        hit_count = body.get("hitCount", len(papers))
        if not next_cursor or next_cursor == cursor_mark or len(results) == 0:
            break
        cursor_mark = next_cursor
        if len(papers) >= hit_count:
            break
    else:
        logger.warning("europepmc: hit max_pages=%d safety cap for query %r", max_pages, query)

    return papers

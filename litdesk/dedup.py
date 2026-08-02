"""Identity resolution: normalized DOI -> PMID -> normalized title + first
author surname + year (spec section 4). A preprint and its later journal
version share one row, linked via `published_doi`.

`published_doi` semantics, since a row's `doi` never changes after creation
(stable identity for FK-like references from ratings/seeds/embeddings):
whichever DOI the row was NOT created with, but that we've since learned
belongs to the same work (a preprint's eventual journal DOI, or vice versa),
lives in `published_doi`. Matching therefore checks both of new_paper's DOIs
against both of the existing row's DOIs.
"""

from __future__ import annotations

import re
import sqlite3

from litdesk.models import RawPaper

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")
_YEAR_RE = re.compile(r"(\d{4})")

_DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi:",
)


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    d = doi.strip().lower()
    for prefix in _DOI_PREFIXES:
        if d.startswith(prefix):
            d = d[len(prefix):]
            break
    d = d.strip("/ ")
    return d or None


def normalize_title(title: str | None) -> str:
    if not title:
        return ""
    t = title.lower()
    t = _PUNCT_RE.sub("", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def first_author_surname(authors: list[str]) -> str | None:
    if not authors:
        return None
    first = authors[0].strip()
    if not first:
        return None
    if "," in first:
        # "Doe, J." (bioRxiv style)
        surname = first.split(",")[0].strip()
        return surname.lower() or None
    tokens = first.split()
    if not tokens:
        return None
    if len(tokens) == 1:
        return tokens[0].lower()
    last = tokens[-1]
    if last.isupper() and len(last) <= 4:
        # "Smith JA" (Europe PMC fullName style) -> surname is everything before initials
        return " ".join(tokens[:-1]).lower()
    # "John Smith" (First Last) -> assume last token is the surname
    return last.lower()


def extract_year(date_published: str | None) -> str | None:
    if not date_published:
        return None
    m = _YEAR_RE.match(date_published)
    return m.group(1) if m else None


def compute_dedup_title_key(title: str | None, authors: list[str], date_published: str | None) -> str | None:
    norm_title = normalize_title(title)
    if not norm_title:
        return None
    surname = first_author_surname(authors) or ""
    year = extract_year(date_published) or ""
    return f"{norm_title}|{surname}|{year}"


def find_match(conn: sqlite3.Connection, paper: RawPaper) -> sqlite3.Row | None:
    doi = normalize_doi(paper.doi)
    pub_doi = normalize_doi(paper.published_doi)
    title_key = compute_dedup_title_key(paper.title, paper.authors, paper.date_published)

    clauses: list[str] = []
    params: list[str] = []
    for candidate in (doi, pub_doi):
        if candidate:
            clauses += ["doi = ?", "published_doi = ?"]
            params += [candidate, candidate]
    if paper.pmid:
        clauses.append("pmid = ?")
        params.append(paper.pmid)
    if title_key:
        clauses.append("dedup_title_key = ?")
        params.append(title_key)

    if not clauses:
        return None

    sql = f"SELECT * FROM papers WHERE {' OR '.join(clauses)} LIMIT 1"  # noqa: S608 (params are placeholders, not interpolated)
    return conn.execute(sql, params).fetchone()

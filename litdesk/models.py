"""Normalized intermediate representation that every source client produces,
so ingest/dedup logic never touches source-specific field names."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RawPaper:
    source: str                        # 'europepmc' | 'biorxiv' | 'medrxiv'
    title: str
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    is_preprint: bool = False
    published_doi: str | None = None   # journal DOI, once a preprint is published
    abstract: str | None = None
    authors: list[str] = field(default_factory=list)
    journal: str | None = None
    date_published: str | None = None       # ISO date (YYYY-MM-DD)
    date_published_kind: str | None = None  # 'posted' | 'epub' | 'print'
    url: str | None = None
    pdf_url: str | None = None
    mesh_terms: list[str] = field(default_factory=list)
    raw: dict | None = None

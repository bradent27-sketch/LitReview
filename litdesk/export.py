"""Phase 5: BibTeX export of everything you've thumbed up."""

from __future__ import annotations

import json
import re

from litdesk import classifier, dedup

_STOPWORDS = {"a", "an", "the", "of", "in", "on", "for", "and", "to", "with"}
_NON_ALPHA_RE = re.compile(r"[^a-z]")
_WORD_RE = re.compile(r"[A-Za-z]+")


def _bibtex_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def _cite_key(title: str, authors: list[str], year: str | None, used: set[str]) -> str:
    surname = _NON_ALPHA_RE.sub("", (dedup.first_author_surname(authors) or "anon").lower()) or "anon"
    title_word = next(
        (w.lower() for w in _WORD_RE.findall(title) if w.lower() not in _STOPWORDS),
        "paper",
    )
    base = f"{surname}{year or 'nd'}{title_word}"

    key = base
    suffix = ord("a")
    while key in used:
        key = f"{base}{chr(suffix)}"
        suffix += 1
    used.add(key)
    return key


def paper_to_bibtex(row, used_keys: set[str]) -> str:
    authors = json.loads(row["authors"]) if row["authors"] else []
    year = dedup.extract_year(row["date_published"])
    key = _cite_key(row["title"], authors, year, used_keys)
    entry_type = "unpublished" if row["is_preprint"] else "article"

    fields: dict[str, str | None] = {
        "title": _bibtex_escape(row["title"]),
        "author": " and ".join(authors) if authors else None,
        "year": year,
        "journal": row["journal"] if entry_type == "article" else None,
        "note": "Preprint" if entry_type == "unpublished" else None,
        "doi": row["doi"],
        "url": row["url"],
    }

    lines = [f"@{entry_type}{{{key},"]
    lines += [f"  {k} = {{{v}}}," for k, v in fields.items() if v]
    lines.append("}")
    return "\n".join(lines)


def export_rated_up_bibtex(conn) -> str:
    """BibTeX for every paper whose *current* rating is +1."""
    ratings = classifier.get_current_ratings(conn)
    paper_ids = [pid for pid, label in ratings.items() if label == 1]
    if not paper_ids:
        return ""
    placeholders = ",".join("?" for _ in paper_ids)
    rows = conn.execute(
        f"SELECT * FROM papers WHERE id IN ({placeholders}) ORDER BY date_published DESC",  # noqa: S608
        paper_ids,
    ).fetchall()
    used_keys: set[str] = set()
    return "\n\n".join(paper_to_bibtex(row, used_keys) for row in rows)

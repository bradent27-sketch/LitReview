"""Phase 4: per-paper TLDRs for the top N digest entries, and a weekly
cross-paper synthesis (spec section 6, Phase 4). Both are entirely optional
— gated on `llm.enabled` in config — and every function here degrades to a
no-op rather than raising, so a flaky request never breaks digest
generation. Uses `claude-haiku-4-5-20251001` for the high-volume per-paper
TLDRs and `claude-sonnet-5` for the weekly synthesis by default, per spec.
"""

from __future__ import annotations

import datetime as dt
import logging

from litdesk.config import Config
from litdesk.llm.client import get_client

logger = logging.getLogger("litdesk.llm")


def _research_profile(cfg: Config) -> str:
    return cfg.project_description or "; ".join(cfg.queries) or "general biochemistry"


def summarize_paper(
    client, cfg: Config, title: str, abstract: str | None, journal: str | None, research_profile: str,
) -> dict | None:
    """Returns {"tldr": str, "relevance": str | None}, or None on failure."""
    prompt = (
        f"Title: {title}\n"
        f"Journal: {journal or 'unknown'}\n"
        f"Abstract: {abstract or '(no abstract available)'}\n\n"
        f"My research interests: {research_profile}\n\n"
        "Write a two-sentence, plain-language TLDR of this paper for a biochemistry "
        "PhD student skimming a daily reading digest. Then, on a new line starting "
        "exactly with 'Relevance: ', give one sentence on why it might matter given "
        "my research interests — or say plainly that it's just generally interesting "
        "if you can't find a real connection. Don't invent a connection that isn't there."
    )
    try:
        response = client.messages.create(
            model=cfg.llm.tldr_model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:  # noqa: BLE001 - one paper's TLDR failing must not break the digest
        logger.warning("TLDR request failed for %r: %s", title, exc)
        return None

    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if not text:
        return None
    tldr, _, relevance = text.partition("Relevance:")
    return {"tldr": tldr.strip(), "relevance": relevance.strip() or None}


def enrich_digest_entries(conn, cfg: Config, digest_data: dict) -> dict:
    """Adds 'tldr'/'relevance' to the top llm.tldr_top_n entries of an
    already-built digest, in place, and persists them to `rankings`. A
    complete no-op (no client constructed, no API calls) if llm.enabled is
    false — that's the flag Phase 4 promises to be skippable behind."""
    if not cfg.llm.enabled:
        return digest_data
    client = get_client(cfg)
    if client is None:
        return digest_data

    research_profile = _research_profile(cfg)
    run_date = digest_data["run_date"]
    top_entries = digest_data["entries"][: cfg.llm.tldr_top_n]

    for entry in top_entries:
        result = summarize_paper(client, cfg, entry["title"], entry["abstract"], entry["journal"], research_profile)
        if result is None:
            continue
        entry["tldr"] = result["tldr"]
        entry["relevance"] = result["relevance"]
        conn.execute(
            "UPDATE rankings SET tldr = ?, tldr_relevance = ? WHERE paper_id = ? AND run_date = ?",
            (result["tldr"], result["relevance"], entry["id"], run_date),
        )
    conn.commit()
    return digest_data


def weekly_synthesis(client, cfg: Config, papers: list[dict]) -> str | None:
    """papers: [{"title", "journal", "tldr"}, ...] — typically the week's
    ranked/TLDR'd papers, most relevant first."""
    if not papers:
        return None
    listing = "\n".join(
        f"- {p['title']} ({p.get('journal') or 'unknown journal'}): {p.get('tldr') or '(no summary available)'}"
        for p in papers
    )
    prompt = (
        f"Here are this week's top papers from my literature digest:\n\n{listing}\n\n"
        "Write a short synthesis (a few short paragraphs, or bullet points): what "
        "themes connect these papers, anything that seems to contradict something "
        "else here, and anything you think is worth reading in full. Be specific — "
        "reference papers by title, don't speak in generalities."
    )
    try:
        response = client.messages.create(
            model=cfg.llm.synthesis_model,
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:  # noqa: BLE001 - optional feature, must not raise
        logger.warning("Weekly synthesis request failed: %s", exc)
        return None
    return "".join(block.text for block in response.content if block.type == "text").strip() or None


def weekly_synthesis_for_recent_digests(conn, cfg: Config, days: int = 7) -> str | None:
    """Synthesizes across every ranked paper from the last `days` days of
    digests (highest score first). None if disabled, unavailable, or there's
    nothing to synthesize yet."""
    if not cfg.llm.enabled:
        return None
    client = get_client(cfg)
    if client is None:
        return None

    cutoff = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    rows = conn.execute(
        """SELECT p.title, p.journal, r.tldr FROM rankings r
           JOIN papers p ON p.id = r.paper_id
           WHERE r.run_date >= ?
           ORDER BY r.run_date DESC, r.score DESC""",
        (cutoff,),
    ).fetchall()
    if not rows:
        return None
    papers = [{"title": r["title"], "journal": r["journal"], "tldr": r["tldr"]} for r in rows]
    return weekly_synthesis(client, cfg, papers)

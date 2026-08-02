"""Phase 5: retraction/correction flags, derived from metadata already
stored in `papers.raw` (Europe PMC's `pubTypeList`) — no extra API calls.

This only catches retractions Europe PMC had already recorded as of when
you ingested the paper. It won't retroactively flag something retracted
after you ingested it — rerun `litdesk check-retractions` periodically (a
weekly cron entry alongside `litdesk synthesis` is reasonable) so papers get
re-examined as their raw metadata gets refreshed by later ingest runs.
"""

from __future__ import annotations

import datetime as dt
import json

_RETRACTION_MARKERS = ("retract", "expression of concern", "correction")


def is_retraction_flagged(raw: dict) -> tuple[bool, str | None]:
    if not isinstance(raw, dict):
        return False, None
    pub_types = raw.get("pubTypeList", {}).get("pubType", [])
    for pub_type in pub_types:
        if any(marker in pub_type.lower() for marker in _RETRACTION_MARKERS):
            return True, pub_type
    return False, None


def check_all_papers(conn) -> list[dict]:
    """Re-scans every stored paper's raw metadata; flags any not already
    flagged. Returns the newly-flagged papers as [{"id", "title", "note"}]."""
    rows = conn.execute("SELECT id, title, raw FROM papers WHERE raw IS NOT NULL AND retracted = 0").fetchall()
    now = dt.datetime.utcnow().isoformat()
    newly_flagged = []

    for row in rows:
        try:
            raw = json.loads(row["raw"])
        except (TypeError, ValueError):
            continue
        flagged, note = is_retraction_flagged(raw)
        if not flagged:
            continue
        conn.execute("UPDATE papers SET retracted = 1, retraction_note = ? WHERE id = ?", (note, row["id"]))
        conn.execute(
            "INSERT INTO notifications (paper_id, type, message, created_at) VALUES (?, 'retraction', ?, ?)",
            (row["id"], f"Flagged as: {note}", now),
        )
        newly_flagged.append({"id": row["id"], "title": row["title"], "note": note})

    conn.commit()
    return newly_flagged

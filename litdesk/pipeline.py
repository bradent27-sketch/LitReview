"""The end-to-end ingest -> seeds -> embed -> digest pipeline. Both the
`litdesk run` CLI command and the web control panel's "Run now" button
drive this exact code, so their behavior (and log lines) never drift
apart — the web version just captures `log` into an in-memory buffer
instead of printing to stdout.
"""

from __future__ import annotations

from typing import Callable

from litdesk.config import Config, resolve_path
from litdesk.digest import build_digest, render_html
from litdesk.embeddings import embed_missing
from litdesk.ingest import run_ingest
from litdesk.seeds import load_seeds_from_config


def build_and_render_digest(conn, cfg: Config):
    """build_digest -> Phase 4 TLDR enrichment (no-op if llm.enabled is
    false) -> render_html -> optional email delivery (no-op if
    email.enabled is false)."""
    data = build_digest(conn, cfg)
    if cfg.llm.enabled:
        from litdesk.llm.summarize import enrich_digest_entries
        data = enrich_digest_entries(conn, cfg, data)
    data["api_base"] = f"http://127.0.0.1:{cfg.server.port}"
    path = render_html(data, resolve_path(cfg.digest.output_dir))
    if cfg.email.enabled and data["entries"]:
        from litdesk.email_digest import send_digest_email
        subject = f"LitDesk digest — {data['run_date']} ({len(data['entries'])} papers)"
        send_digest_email(cfg, subject, path.read_text())
    return path, data


def run_full_pipeline(conn, cfg: Config, log: Callable[[str], None] = print) -> dict:
    """Ingest + seeds + embed + digest in one shot — the crontab entry and
    the control panel's "Run now" button both call this. Returns
    {"had_error": bool, "path": Path | None, "data": dict | None,
    "error": str | None} — `path`/`data` are None only when a digest
    couldn't be built at all (e.g. no seeds yet)."""
    had_error = False
    for r in run_ingest(conn, cfg):
        err = f"  ERROR: {r['error']}" if r["error"] else ""
        had_error = had_error or bool(r["error"])
        log(f"ingest {r['source']:<10} fetched={r['fetched']:<5} new={r['new']:<5} deduped={r['deduped']:<5}{err}")

    load_seeds_from_config(conn, cfg)
    embed_missing(conn, cfg)
    try:
        path, data = build_and_render_digest(conn, cfg)
    except ValueError as exc:
        log(f"Can't build a digest yet: {exc}")
        return {"had_error": had_error, "path": None, "data": None, "error": str(exc)}

    log(f"\n{len(data['entries'])} paper(s) -> {path}")
    if had_error:
        log("(one or more sources errored during ingest — see `litdesk runs` / the Control Panel)")
    return {"had_error": had_error, "path": path, "data": data, "error": None}

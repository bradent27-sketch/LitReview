"""Local web control panel: serves the latest digest, accepts thumbs
up/down ratings, and (beyond spec, per user request) a Settings page for
editing config.yaml as a form plus a Control Panel for running the
pipeline and other tools without a terminal. Bind to 127.0.0.1 only (see
cli.cmd_serve): none of this has auth, and spec section 1 is explicit that
this tool has no business being reachable off your machine.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel

from litdesk.config import Config, load_config, resolve_path
from litdesk.config_form import config_to_form_dict, form_to_config_dict, save_config_yaml
from litdesk.db import open_db
from litdesk.digest import TEMPLATE_DIR
from litdesk.email_digest import SMTP_PASSWORD_ENV_VAR
from litdesk.pipeline import run_full_pipeline

logger = logging.getLogger("litdesk.server")


class RatingIn(BaseModel):
    paper_id: int
    label: int  # +1 or -1


class _PipelineRunState:
    """One shared, lock-guarded background-job slot. This app is single-user
    and local, so a full task queue would be solving a problem that doesn't
    exist here — this is just enough to run one pipeline pass at a time and
    let the Control Panel page poll its progress."""

    def __init__(self):
        self._lock = threading.Lock()
        self.running = False
        self.log_lines: list[str] = []
        self.finished_at: str | None = None
        self.had_error = False

    def start(self) -> bool:
        with self._lock:
            if self.running:
                return False
            self.running = True
            self.log_lines = []
            self.finished_at = None
            self.had_error = False
            return True

    def log(self, line: str) -> None:
        with self._lock:
            self.log_lines.append(line)

    def finish(self, had_error: bool) -> None:
        with self._lock:
            self.running = False
            self.had_error = had_error
            self.finished_at = dt.datetime.utcnow().isoformat()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "running": self.running,
                "log": list(self.log_lines),
                "finished_at": self.finished_at,
                "had_error": self.had_error,
            }


def _run_pipeline_in_background(cfg: Config, db_path, state: _PipelineRunState) -> None:
    try:
        with open_db(db_path) as conn:
            result = run_full_pipeline(conn, cfg, log=state.log)
        state.finish(had_error=result["had_error"] or result["data"] is None)
    except Exception as exc:  # noqa: BLE001 - background thread; must report rather than vanish silently
        logger.exception("Unexpected error during background pipeline run")
        state.log(f"Unexpected error: {exc}")
        state.finish(had_error=True)


def create_app(cfg: Config, config_path: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="LitDesk")
    # Permissive CORS: this only ever binds loopback, and the digest HTML is
    # sometimes opened directly as a file:// page (origin "null") rather than
    # served by this app, so same-origin alone isn't enough.
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"],
    )

    db_path = resolve_path(cfg.db_path)
    digest_dir = resolve_path(cfg.digest.output_dir)
    config_path = Path(config_path) if config_path else resolve_path("config/config.yaml")
    run_state = _PipelineRunState()
    tool_result: dict = {}

    jinja_env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=select_autoescape(),
        trim_blocks=True, lstrip_blocks=True,
    )

    def render(template_name: str, **context) -> HTMLResponse:
        return HTMLResponse(jinja_env.get_template(template_name).render(**context))

    # --- Digest + ratings (Phase 2/3) ---

    @app.get("/", response_class=HTMLResponse)
    def index():
        latest = digest_dir / "latest.html"
        if not latest.exists():
            return HTMLResponse("<p>No digest yet. Use the Control Panel's \"Run now\", or run <code>litdesk digest</code>.</p>")
        return HTMLResponse(latest.read_text())

    @app.post("/rate")
    def rate(rating: RatingIn):
        if rating.label not in (1, -1):
            raise HTTPException(400, "label must be +1 or -1")
        with open_db(db_path) as conn:
            paper = conn.execute("SELECT id FROM papers WHERE id = ?", (rating.paper_id,)).fetchone()
            if paper is None:
                raise HTTPException(404, "unknown paper_id")
            conn.execute(
                "INSERT INTO ratings (paper_id, label, rated_at) VALUES (?, ?, ?)",
                (rating.paper_id, rating.label, dt.datetime.utcnow().isoformat()),
            )
            conn.commit()
            n_rated = conn.execute(
                "SELECT COUNT(DISTINCT paper_id) AS n FROM ratings"
            ).fetchone()["n"]
        return {"status": "ok", "paper_id": rating.paper_id, "label": rating.label, "total_rated_papers": n_rated}

    # --- Settings ---

    @app.get("/settings", response_class=HTMLResponse)
    def settings_page(request: Request):
        current = load_config(config_path)
        return render(
            "settings.html.jinja",
            values=config_to_form_dict(current),
            saved=request.query_params.get("saved") == "1",
            smtp_password_set=bool(os.environ.get(SMTP_PASSWORD_ENV_VAR)),
        )

    @app.post("/settings")
    async def save_settings(request: Request):
        current = load_config(config_path)
        form = await request.form()
        plain_form = {k: v for k, v in form.multi_items() if isinstance(v, str)}

        upload = form.get("seeds_bibtex_file")
        if upload is not None and getattr(upload, "filename", ""):
            upload_dir = resolve_path("data/uploads")
            upload_dir.mkdir(parents=True, exist_ok=True)
            dest = upload_dir / "seeds.bib"
            dest.write_bytes(await upload.read())
            plain_form["seeds_bibtex_path_uploaded"] = str(dest)

        smtp_password = plain_form.pop("email_smtp_password_session_only", "").strip()
        if smtp_password:
            # Session-only by design — never written to config.yaml. See the
            # help text on the Settings page for the cron/persistent caveat.
            os.environ[SMTP_PASSWORD_ENV_VAR] = smtp_password

        yaml_dict = form_to_config_dict(plain_form, current)
        save_config_yaml(config_path, yaml_dict)
        return RedirectResponse(url="/settings?saved=1", status_code=303)

    # --- Control panel ---

    @app.get("/control", response_class=HTMLResponse)
    def control_page():
        with open_db(db_path) as conn:
            recent_runs = [dict(r) for r in conn.execute(
                "SELECT started_at, source, n_fetched, n_new, n_deduped, error "
                "FROM runs ORDER BY id DESC LIMIT 10"
            ).fetchall()]
            notifications = [dict(n) for n in conn.execute(
                "SELECT type, created_at, message, seen FROM notifications ORDER BY id DESC LIMIT 10"
            ).fetchall()]
            unseen_count = conn.execute("SELECT COUNT(*) AS n FROM notifications WHERE seen = 0").fetchone()["n"]
        return render(
            "control.html.jinja",
            recent_runs=recent_runs, notifications=notifications, unseen_count=unseen_count,
            result=tool_result if tool_result.get("tool") else None,
        )

    @app.post("/control/run")
    def control_start_run():
        started = run_state.start()
        if started:
            fresh_cfg = load_config(config_path)
            thread = threading.Thread(
                target=_run_pipeline_in_background, args=(fresh_cfg, db_path, run_state), daemon=True,
            )
            thread.start()
        return {"started": started}

    @app.get("/control/run-status")
    def control_run_status():
        return run_state.snapshot()

    @app.post("/control/notifications/mark-seen")
    def control_mark_notifications_seen():
        with open_db(db_path) as conn:
            conn.execute("UPDATE notifications SET seen = 1 WHERE seen = 0")
            conn.commit()
        return RedirectResponse(url="/control", status_code=303)

    @app.post("/control/retrain")
    def control_retrain():
        from litdesk import classifier

        fresh_cfg = load_config(config_path)
        with open_db(db_path) as conn:
            result = classifier.evaluate_holdout(conn, fresh_cfg)

        if result is None:
            tool_result.update(tool="Ranker accuracy", is_error=True, output=(
                f"Not enough ratings yet — need >= {fresh_cfg.ranker.min_ratings_for_classifier} total, "
                "with at least 2 thumbs-up and 2 thumbs-down. Digests use the seed-centroid ranking until then."
            ))
        else:
            tool_result.update(tool="Ranker accuracy", is_error=False, output=(
                f"Held-out accuracy: {result['accuracy']:.1%} "
                f"(trained on {result['n_train']}, tested on {result['n_test']})\n"
                f"precision={result['precision']:.2f}  recall={result['recall']:.2f}"
            ))
        return RedirectResponse(url="/control", status_code=303)

    @app.post("/control/export")
    def control_export():
        from litdesk.export import export_rated_up_bibtex

        with open_db(db_path) as conn:
            bibtex = export_rated_up_bibtex(conn)

        if not bibtex:
            tool_result.update(tool="Export BibTeX", is_error=False, output="Nothing thumbed up yet — nothing to export.")
        else:
            digest_dir.mkdir(parents=True, exist_ok=True)
            out_path = digest_dir / f"export-{dt.date.today().isoformat()}.bib"
            out_path.write_text(bibtex + "\n")
            tool_result.update(
                tool="Export BibTeX", is_error=False,
                output=f"Wrote {out_path.name} — download at /files/{out_path.name}\n\n{bibtex}",
            )
        return RedirectResponse(url="/control", status_code=303)

    @app.post("/control/check-retractions")
    def control_check_retractions():
        from litdesk.retractions import check_all_papers

        with open_db(db_path) as conn:
            flagged = check_all_papers(conn)

        if not flagged:
            tool_result.update(tool="Retraction check", is_error=False, output="No new retractions/corrections found.")
        else:
            lines = "\n".join(f"[{f['note']}] {f['title']}" for f in flagged)
            tool_result.update(
                tool="Retraction check", is_error=False,
                output=f"{len(flagged)} paper(s) newly flagged:\n{lines}",
            )
        return RedirectResponse(url="/control", status_code=303)

    @app.post("/control/synthesis")
    def control_synthesis():
        fresh_cfg = load_config(config_path)
        if not fresh_cfg.llm.enabled:
            tool_result.update(
                tool="Weekly synthesis", is_error=True,
                output="Claude summaries are off — turn them on in Settings first.",
            )
            return RedirectResponse(url="/control", status_code=303)

        from litdesk.llm.summarize import weekly_synthesis_for_recent_digests

        with open_db(db_path) as conn:
            text = weekly_synthesis_for_recent_digests(conn, fresh_cfg, days=7)

        if text is None:
            tool_result.update(
                tool="Weekly synthesis", is_error=True,
                output="No synthesis produced — either no digests in the last 7 days, or the LLM call failed.",
            )
        else:
            digest_dir.mkdir(parents=True, exist_ok=True)
            out_path = digest_dir / f"synthesis-{dt.date.today().isoformat()}.md"
            out_path.write_text(text)
            tool_result.update(
                tool="Weekly synthesis", is_error=False,
                output=f"Wrote {out_path.name}:\n\n{text}",
            )
        return RedirectResponse(url="/control", status_code=303)

    # --- Downloads (exports, synthesis notes) ---

    @app.get("/files/{filename}")
    def get_file(filename: str):
        # Path(...).name strips any directory components, so this can never
        # resolve outside digest_dir regardless of what's in `filename`.
        safe_name = Path(filename).name
        target = digest_dir / safe_name if safe_name else None
        if target is None or not target.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(target)

    return app

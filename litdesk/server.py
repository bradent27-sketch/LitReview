"""Phase 3: minimal local FastAPI server — serves the latest digest and
accepts thumbs up/down ratings. Bind to 127.0.0.1 only when running this
(see cli.cmd_serve): the /rate endpoint has no auth, and spec section 1 is
explicit that this tool has no business being reachable off your machine.
"""

from __future__ import annotations

import datetime as dt

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from litdesk.config import Config, resolve_path
from litdesk.db import open_db


class RatingIn(BaseModel):
    paper_id: int
    label: int  # +1 or -1


def create_app(cfg: Config) -> FastAPI:
    app = FastAPI(title="LitDesk")
    # Permissive CORS: this only ever binds loopback, and the digest HTML is
    # sometimes opened directly as a file:// page (origin "null") rather than
    # served by this app, so same-origin alone isn't enough.
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"],
    )

    db_path = resolve_path(cfg.db_path)
    digest_dir = resolve_path(cfg.digest.output_dir)

    @app.get("/", response_class=HTMLResponse)
    def index():
        latest = digest_dir / "latest.html"
        if not latest.exists():
            return HTMLResponse("<p>No digest yet. Run <code>litdesk digest</code> first.</p>")
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

    return app

"""litdesk CLI entrypoint."""

from __future__ import annotations

import argparse
import logging
import sys

from litdesk.config import load_config, resolve_path
from litdesk.db import open_db
from litdesk.embeddings import embed_missing
from litdesk.ingest import run_ingest
from litdesk.pipeline import build_and_render_digest, run_full_pipeline
from litdesk.seeds import load_seeds_from_config

logger = logging.getLogger("litdesk.cli")


def cmd_init(args: argparse.Namespace) -> int:
    example = resolve_path("config/config.example.yaml")
    target = resolve_path("config/config.yaml")
    if target.exists():
        print(f"{target} already exists, leaving it alone.")
    else:
        target.write_text(example.read_text())
        print(f"Wrote {target}")
    for d in ("data", "data/cache", "data/logs", "digests"):
        resolve_path(d).mkdir(parents=True, exist_ok=True)
    print(
        "\nReady. Run `litdesk serve` and open http://127.0.0.1:8000/settings to set your "
        "standing queries, seed papers, and everything else from a form (no file-editing needed) "
        "— or edit config/config.yaml directly if you prefer."
    )
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    db_path = resolve_path(cfg.db_path)
    with open_db(db_path) as conn:
        results = run_ingest(conn, cfg)

    print(f"{'source':<10} {'fetched':>8} {'new':>8} {'deduped':>8}  error")
    had_error = False
    for r in results:
        err = r["error"] or ""
        had_error = had_error or bool(err)
        print(f"{r['source']:<10} {r['fetched']:>8} {r['new']:>8} {r['deduped']:>8}  {err}")
    total_fetched = sum(r["fetched"] for r in results)
    total_new = sum(r["new"] for r in results)
    total_deduped = sum(r["deduped"] for r in results)
    print(f"{'TOTAL':<10} {total_fetched:>8} {total_new:>8} {total_deduped:>8}")

    if had_error:
        print(
            "\nOne or more sources errored (see above / `litdesk runs`). "
            "Other sources still completed and were saved.",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    db_path = resolve_path(cfg.db_path)
    with open_db(db_path) as conn:
        rows = conn.execute(
            "SELECT started_at, finished_at, source, n_fetched, n_new, n_deduped, error "
            "FROM runs ORDER BY id DESC LIMIT ?",
            (args.limit,),
        ).fetchall()

    if not rows:
        print("No runs logged yet. Run `litdesk ingest` first.")
        return 0
    for r in rows:
        flag = f"  ERROR: {r['error']}" if r["error"] else ""
        print(
            f"{r['started_at']}  {r['source']:<10} "
            f"fetched={r['n_fetched']:<5} new={r['n_new']:<5} deduped={r['n_deduped']:<5}{flag}"
        )
    return 0


def cmd_seeds(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    db_path = resolve_path(cfg.db_path)
    with open_db(db_path) as conn:
        results = load_seeds_from_config(conn, cfg)

    if not results:
        print("No seeds configured. Set seeds.dois (or seeds.bibtex_path) in config.yaml.")
        return 0
    for r in results:
        print(f"{r['status']:<16} {r['doi']}")
    n_ok = sum(1 for r in results if r["status"] != "not_found")
    print(f"\n{n_ok}/{len(results)} seeds loaded.")
    return 0 if n_ok == len(results) else 1


def cmd_embed(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    db_path = resolve_path(cfg.db_path)
    with open_db(db_path) as conn:
        n = embed_missing(conn, cfg)
    print(f"Embedded {n} paper(s) with {cfg.embeddings.model}.")
    return 0


def cmd_digest(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    db_path = resolve_path(cfg.db_path)
    with open_db(db_path) as conn:
        load_seeds_from_config(conn, cfg)
        embed_missing(conn, cfg)
        try:
            path, data = build_and_render_digest(conn, cfg)
        except ValueError as exc:
            print(f"Can't build a digest yet: {exc}", file=sys.stderr)
            return 1
    print(f"{len(data['entries'])} paper(s) -> {path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Ingest + seeds + embed + digest in one shot — the crontab entry."""
    cfg = load_config(args.config)
    db_path = resolve_path(cfg.db_path)
    with open_db(db_path) as conn:
        result = run_full_pipeline(conn, cfg, log=print)
    return 1 if result["data"] is None or result["had_error"] else 0


def cmd_retrain(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    db_path = resolve_path(cfg.db_path)
    from litdesk import classifier

    with open_db(db_path) as conn:
        result = classifier.evaluate_holdout(conn, cfg)

    if result is None:
        print(
            f"Not enough ratings yet to evaluate — need >= {cfg.ranker.min_ratings_for_classifier} total, "
            "with at least 2 thumbs-up and 2 thumbs-down. Digests use the seed-centroid ranking until then."
        )
        return 1

    print(f"Held-out accuracy: {result['accuracy']:.1%}  (trained on {result['n_train']}, tested on {result['n_test']})")
    print(f"  precision={result['precision']:.2f}  recall={result['recall']:.2f}")
    return 0


def cmd_synthesis(args: argparse.Namespace) -> int:
    """Phase 4: weekly cross-paper synthesis. Meant for a separate weekly
    cron entry, distinct from the daily `litdesk run`."""
    cfg = load_config(args.config)
    if not cfg.llm.enabled:
        print("llm.enabled is false in config.yaml — nothing to do.", file=sys.stderr)
        return 1

    import datetime as dt

    from litdesk.llm.summarize import weekly_synthesis_for_recent_digests

    with open_db(resolve_path(cfg.db_path)) as conn:
        text = weekly_synthesis_for_recent_digests(conn, cfg, days=args.days)

    if text is None:
        print(
            "No synthesis produced — either no digests in the lookback window, "
            "or the LLM call failed (check the log above).",
            file=sys.stderr,
        )
        return 1

    out_dir = resolve_path(cfg.digest.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"synthesis-{dt.date.today().isoformat()}.md"
    out_path.write_text(text)
    print(text)
    print(f"\nWritten to {out_path}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Phase 5: BibTeX of everything currently thumbed up."""
    import datetime as dt

    from litdesk.export import export_rated_up_bibtex

    cfg = load_config(args.config)
    with open_db(resolve_path(cfg.db_path)) as conn:
        bibtex = export_rated_up_bibtex(conn)

    if not bibtex:
        print("Nothing thumbed up yet — nothing to export.")
        return 0

    out_dir = resolve_path(cfg.digest.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"export-{dt.date.today().isoformat()}.bib"
    out_path.write_text(bibtex + "\n")
    print(bibtex)
    print(f"\nWritten to {out_path}")
    return 0


def cmd_check_retractions(args: argparse.Namespace) -> int:
    """Phase 5: re-scan stored metadata for retraction/correction flags."""
    from litdesk.retractions import check_all_papers

    cfg = load_config(args.config)
    with open_db(resolve_path(cfg.db_path)) as conn:
        flagged = check_all_papers(conn)

    if not flagged:
        print("No new retractions/corrections found.")
        return 0
    for f in flagged:
        print(f"[{f['note']}] {f['title']}")
    print(f"\n{len(flagged)} paper(s) newly flagged — see ‘litdesk notifications’.")
    return 0


def cmd_notifications(args: argparse.Namespace) -> int:
    """Surfaces preprint->published, scoop-alarm, and retraction notices."""
    cfg = load_config(args.config)
    with open_db(resolve_path(cfg.db_path)) as conn:
        query = "SELECT * FROM notifications" + ("" if args.all else " WHERE seen = 0") + " ORDER BY id DESC"
        rows = conn.execute(query).fetchall()
        if not args.all:
            conn.execute("UPDATE notifications SET seen = 1 WHERE seen = 0")
            conn.commit()

    if not rows:
        print("No notifications." if args.all else "No new notifications.")
        return 0
    for r in rows:
        print(f"[{r['type']}] {r['created_at']}  {r['message']}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    port = args.port or cfg.server.port
    import uvicorn

    from litdesk.server import create_app

    app = create_app(cfg, config_path=args.config)
    print(f"Serving on http://127.0.0.1:{port} — digest, Settings, and Control Panel. Ctrl+C to stop.")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="litdesk", description="Personal literature triage agent.")
    parser.add_argument(
        "--config",
        default=str(resolve_path("config/config.yaml")),
        help="Path to config.yaml (default: config/config.yaml; falls back to built-in defaults if missing)",
    )
    sub = parser.add_subparsers(dest="command")

    p_init = sub.add_parser("init", help="Write config/config.yaml from the example template and create data dirs")
    p_init.set_defaults(func=cmd_init)

    p_ingest = sub.add_parser("ingest", help="Phase 1: fetch new papers from configured sources, dedup, store")
    p_ingest.set_defaults(func=cmd_ingest)

    p_runs = sub.add_parser("runs", help="Show recent run log entries")
    p_runs.add_argument("--limit", type=int, default=20)
    p_runs.set_defaults(func=cmd_runs)

    p_seeds = sub.add_parser("seeds", help="Load seed papers from config.yaml (seeds.dois / seeds.bibtex_path)")
    p_seeds.set_defaults(func=cmd_seeds)

    p_embed = sub.add_parser("embed", help="Phase 2: embed every paper missing a vector under the configured model")
    p_embed.set_defaults(func=cmd_embed)

    p_digest = sub.add_parser("digest", help="Phase 2: rank candidates and render today's HTML digest")
    p_digest.set_defaults(func=cmd_digest)

    p_run = sub.add_parser("run", help="One command, today's digest end to end: ingest + seeds + embed + digest")
    p_run.set_defaults(func=cmd_run)

    p_retrain = sub.add_parser("retrain", help="Phase 3: report held-out classifier accuracy on your ratings so far")
    p_retrain.set_defaults(func=cmd_retrain)

    p_serve = sub.add_parser("serve", help="Phase 3: serve the digest locally with working thumbs up/down buttons")
    p_serve.add_argument("--port", type=int, default=None, help="Overrides server.port from config")
    p_serve.set_defaults(func=cmd_serve)

    p_synthesis = sub.add_parser("synthesis", help="Phase 4: weekly cross-paper synthesis (needs llm.enabled)")
    p_synthesis.add_argument("--days", type=int, default=7, help="Lookback window in days (default 7)")
    p_synthesis.set_defaults(func=cmd_synthesis)

    p_export = sub.add_parser("export", help="Phase 5: BibTeX of everything currently thumbed up")
    p_export.set_defaults(func=cmd_export)

    p_retractions = sub.add_parser(
        "check-retractions", help="Phase 5: re-scan stored metadata for retraction/correction flags"
    )
    p_retractions.set_defaults(func=cmd_check_retractions)

    p_notifications = sub.add_parser(
        "notifications", help="Phase 5: preprint->published, scoop-alarm, and retraction notices"
    )
    p_notifications.add_argument("--all", action="store_true", help="Show already-seen notifications too")
    p_notifications.set_defaults(func=cmd_notifications)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())

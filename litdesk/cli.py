"""litdesk CLI entrypoint."""

from __future__ import annotations

import argparse
import logging
import sys

from litdesk.config import load_config, resolve_path
from litdesk.db import open_db
from litdesk.ingest import run_ingest

logger = logging.getLogger("litdesk.cli")


def cmd_init(args: argparse.Namespace) -> int:
    example = resolve_path("config/config.example.yaml")
    target = resolve_path("config/config.yaml")
    if target.exists():
        print(f"{target} already exists, leaving it alone.")
    else:
        target.write_text(example.read_text())
        print(f"Wrote {target} — edit it with your standing queries, journals, and seed papers.")
    for d in ("data", "data/cache", "data/logs", "digests"):
        resolve_path(d).mkdir(parents=True, exist_ok=True)
    print("Ready. Try: litdesk ingest")
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

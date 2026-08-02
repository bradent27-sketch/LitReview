"""litdesk CLI entrypoint."""

from __future__ import annotations

import argparse
import logging
import sys

from litdesk.config import load_config, resolve_path
from litdesk.db import open_db
from litdesk.digest import generate_digest
from litdesk.embeddings import embed_missing
from litdesk.ingest import run_ingest
from litdesk.seeds import load_seeds_from_config

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
            path, data = generate_digest(conn, cfg)
        except ValueError as exc:
            print(f"Can't build a digest yet: {exc}", file=sys.stderr)
            return 1
    print(f"{len(data['entries'])} paper(s) -> {path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Ingest + seeds + embed + digest in one shot — the crontab entry."""
    cfg = load_config(args.config)
    db_path = resolve_path(cfg.db_path)
    had_error = False
    with open_db(db_path) as conn:
        for r in run_ingest(conn, cfg):
            err = f"  ERROR: {r['error']}" if r["error"] else ""
            had_error = had_error or bool(r["error"])
            print(f"ingest {r['source']:<10} fetched={r['fetched']:<5} new={r['new']:<5} deduped={r['deduped']:<5}{err}")

        load_seeds_from_config(conn, cfg)
        embed_missing(conn, cfg)
        try:
            path, data = generate_digest(conn, cfg)
        except ValueError as exc:
            print(f"Can't build a digest yet: {exc}", file=sys.stderr)
            return 1

    print(f"\n{len(data['entries'])} paper(s) -> {path}")
    if had_error:
        print("(one or more sources errored during ingest — see above / `litdesk runs`)", file=sys.stderr)
        return 1
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

"""Command line for pipeline 3, the drama writers' room.

    python -m drama_factory series            # pitch contest -> plan contest -> a new series bible
    python -m drama_factory write --count 3   # the next 3 episodes of the newest series
    python -m drama_factory status
    python -m drama_factory report [EPISODE_ID]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys

from story_factory import db
from story_factory.__main__ import apply_overrides
from story_factory.config import load_config
from story_factory.pipeline import Context
from story_factory.router import Router

from . import pipeline


def _context(cfg: dict) -> Context:
    conn = db.connect(cfg["database"])
    return Context(conn=conn, cfg=cfg, router=Router(cfg, conn), rng=random.Random())


def report(conn, row) -> str:
    script = json.loads(row["script"])
    review = json.loads(row["review"] or "{}")
    audience = json.loads(row["audience_report"] or "null") or {}
    packaging = json.loads(row["packaging"] or "null") or {}
    lines = [f"## Episode {row['number']}: {row['title']} [{row['status']}]", "",
             f"- **Editors' median scores:** {review.get('scores')}",
             f"- **Shots:** {len(script.get('shots', []))}; hook: \"{script.get('hook_line', '')}\"; "
             f"cliffhanger: \"{script.get('cliffhanger', '')}\""]
    if review.get("reason"):
        lines.append(f"- **Why not approved:** {review['reason']}")
    if audience:
        lines.append(f"- **Simulated viewers:** unlock the next episode {audience.get('unlock_next')}/10; "
                     f"would swipe at shots {audience.get('swipes')}")
    if packaging:
        lines.append(f"- **Title:** {packaging.get('title')} | **Caption:** {packaging.get('caption')}")
    stages = [e.get("stage") for e in json.loads(row["production_log"] or "[]")]
    lines += ["", "**Stages:** " + " → ".join(stages)]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="drama_factory", description="Pipeline 3: the drama writers' room")
    parser.add_argument("--config")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("series", help="pitch contest and plan contest: a new series bible")
    p = sub.add_parser("write", help="write the next episodes of a series")
    p.add_argument("--series", type=int)
    p.add_argument("--count", type=int, default=1)
    sub.add_parser("status", help="series and episodes")
    p = sub.add_parser("report", help="Markdown summary of an episode (default: the latest)")
    p.add_argument("id", type=int, nargs="?")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = apply_overrides(load_config(args.config), args.set)
    ctx = _context(cfg)

    if args.command == "series":
        series_id = pipeline.create_series(ctx)
        bible = json.loads(ctx.conn.execute("SELECT bible FROM series WHERE id = ?", (series_id,)).fetchone()[0])
        print(f"Series #{series_id}: {bible['title']}\n{bible['logline']}\nCast: "
              + ", ".join(f"{c['name']} ({c['role']})" for c in bible["cast"]))
        return 0
    if args.command == "write":
        row = ctx.conn.execute("SELECT id FROM series WHERE status = 'active' ORDER BY id DESC LIMIT 1").fetchone()
        series_id = args.series or (row["id"] if row else None)
        if series_id is None:
            print("No active series; run `python -m drama_factory series` first", file=sys.stderr)
            return 1
        approved = 0
        for _ in range(args.count):
            episode_id, status = pipeline.write_episode(ctx, series_id)
            print(report(ctx.conn, ctx.conn.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,)).fetchone()))
            approved += status == "approved"
        return 0 if approved else 1
    if args.command == "status":
        for s in ctx.conn.execute("SELECT * FROM series ORDER BY id"):
            print(f"Series #{s['id']} [{s['status']}] {s['title']} ({s['genre']})")
            for e in ctx.conn.execute("SELECT id, number, title, status FROM episodes WHERE series_id = ? "
                                      "ORDER BY number", (s["id"],)):
                print(f"  {e['number']:>3}. [{e['status']}] {e['title']} (episode id {e['id']})")
        return 0
    if args.command == "report":
        row = (ctx.conn.execute("SELECT * FROM episodes WHERE id = ?", (args.id,)).fetchone() if args.id else
               ctx.conn.execute("SELECT * FROM episodes ORDER BY updated_at DESC, id DESC LIMIT 1").fetchone())
        print(report(ctx.conn, row) if row else "No episodes yet.")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())

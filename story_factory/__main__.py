"""Command line: python -m story_factory <command>."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys

from . import catalog, db, learning, pipeline, research
from .config import load_config
from .llm import BudgetExceeded, LLMError, ModelSpec
from .router import Router


def _context(cfg: dict) -> pipeline.Context:
    conn = db.connect(cfg["database"])
    return pipeline.Context(conn=conn, cfg=cfg, router=Router(cfg, conn), rng=random.Random())


def _story_json(row) -> dict:
    story = dict(row)
    story.pop("synopsis_embedding", None)
    for key in ("inspiration_ids", *db.JSON_COLUMNS):
        if story.get(key):
            story[key] = json.loads(story[key])
    return story


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="story_factory", description="Hollow Lines story pipeline")
    parser.add_argument("--config", help="TOML config file (default: ./config.toml if present)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create the database and seed the story world")
    sub.add_parser("models", help="check every configured model against what your API keys can use")
    p = sub.add_parser("research", help="collect folklore and audience-demand signals")
    p.add_argument("--skip-wikipedia", action="store_true")
    p.add_argument("--skip-youtube", action="store_true")
    p = sub.add_parser("fill", help="generate stories until every series backlog is full")
    p.add_argument("--max", type=int, default=10, help="most stories to approve in this run")
    p = sub.add_parser("generate", help="generate one story for a series")
    p.add_argument("series", choices=sorted(catalog.SERIES))
    sub.add_parser("status", help="backlog, story counts and this month's spend")
    p = sub.add_parser("show", help="print one story as JSON")
    p.add_argument("id", type=int)
    sub.add_parser("world", help="list the shared story world's canon")
    p = sub.add_parser("claim", help="(production pipeline) take the next approved story")
    p.add_argument("--kind", choices=("long", "short"), required=True)
    p.add_argument("--worker", default="production")
    p.add_argument("--series", choices=sorted(catalog.SERIES))
    p = sub.add_parser("set-status", help="(production pipeline) update a story's status")
    p.add_argument("id", type=int)
    p.add_argument("status", choices=db.STATUSES)
    p.add_argument("--video-id")
    p = sub.add_parser("record-performance", help="(production pipeline) save a video's YouTube metrics")
    p.add_argument("id", type=int)
    p.add_argument("--views", type=int)
    p.add_argument("--avg-view-percentage", type=float)
    p.add_argument("--ctr", type=float)
    p.add_argument("--likes", type=int)
    p.add_argument("--comments", type=int)
    sub.add_parser("learn", help="turn real performance into lessons for future pitches")
    sub.add_parser("stats", help="cost per story, winning models and common rejection reasons")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)

    if args.command == "init":
        db.connect(cfg["database"])
        print(f"Database ready at {cfg['database']}")
    elif args.command == "models":
        router = Router(cfg, db.connect(cfg["database"]))
        available: dict[str, list[str] | None] = {}
        for role, specs in cfg["roles"].items():
            for text in specs:
                spec = ModelSpec.parse(text)
                if spec.provider not in available:
                    try:
                        available[spec.provider] = router._client(text).list_models()
                    except (LLMError, Exception) as exc:  # missing key or SDK: report, keep going
                        print(f"{spec.provider}: cannot list models ({exc})")
                        available[spec.provider] = None
                models = available[spec.provider]
                state = "UNVERIFIED" if models is None else "OK" if spec.model in models else "NOT FOUND"
                print(f"{role:<11} {text:<45} {state}")
    elif args.command == "research":
        conn = db.connect(cfg["database"])
        if not args.skip_wikipedia:
            print(f"Wikipedia legends saved: {research.research_wikipedia(conn, cfg)}")
        if not args.skip_youtube:
            print(f"YouTube trend titles saved: {research.research_youtube(conn, cfg)}")
    elif args.command in ("fill", "generate"):
        ctx = _context(cfg)
        try:
            added = research.ensure_legend_supply(ctx.conn, cfg)
            if added:
                print(f"Topped up legends from Wikipedia: {added} new")
        except Exception as exc:  # research is a bonus; stories can still be written without it
            print(f"Wikipedia top-up skipped: {exc}", file=sys.stderr)
        outcomes = (pipeline.fill_backlog(ctx, args.max) if args.command == "fill"
                    else [pipeline.generate_story(ctx, args.series)])
        for o in outcomes:
            print(f"[{o.status}] {o.series} #{o.story_id or '-'} {o.detail}")
        return 0 if any(o.status == "approved" for o in outcomes) or not outcomes else 1
    elif args.command == "status":
        conn = db.connect(cfg["database"])
        backlog = db.backlog_counts(conn)
        print("Approved backlog:")
        for name, target in cfg["backlog"].items():
            print(f"  {name:<16} {backlog.get(name, 0)}/{target}")
        print("All stories:", json.dumps(db.status_counts(conn)))
        cap = cfg["budget"]["monthly_cap_usd"]
        print(f"Spend this month: ${db.month_spend(conn):.2f}" + (f" of ${cap:.2f}" if cap else " (no cap)"))
        print("Shelved pitches waiting:", db.status_counts(conn).get("shelved", 0))
    elif args.command == "show":
        row = db.get_story(db.connect(cfg["database"]), args.id)
        if row is None:
            print("No such story", file=sys.stderr)
            return 1
        print(json.dumps(_story_json(row), indent=2, ensure_ascii=False))
    elif args.command == "world":
        for entity in db.world_entities(db.connect(cfg["database"]), limit=1000):
            print(f"- {entity['name']} ({entity['kind']}): {entity['description']}")
    elif args.command == "claim":
        row = db.claim_next(db.connect(cfg["database"]), args.kind, args.worker, args.series)
        if row is None:
            print("No approved story available", file=sys.stderr)
            return 1
        print(json.dumps(_story_json(row), indent=2, ensure_ascii=False))
    elif args.command == "set-status":
        conn = db.connect(cfg["database"])
        fields = {"status": args.status}
        if args.video_id:
            fields["video_id"] = args.video_id
        if args.status == "published":
            fields["published_at"] = db.now()
        db.update_story(conn, args.id, **fields)
        print(f"Story #{args.id} -> {args.status}")
    elif args.command == "record-performance":
        db.record_performance(db.connect(cfg["database"]), args.id, views=args.views,
                              avg_view_percentage=args.avg_view_percentage, ctr=args.ctr,
                              likes=args.likes, comments=args.comments)
        print(f"Metrics saved for story #{args.id}")
    elif args.command == "stats":
        print(json.dumps(db.stats(db.connect(cfg["database"])), indent=2))
    elif args.command == "learn":
        ctx = _context(cfg)
        insights = learning.learn(ctx.conn, ctx.router)
        print("\n".join(f"- {i}" for i in insights) or "Not enough published videos with metrics yet")
    return 0


def run() -> int:
    try:
        return main()
    except (LLMError, BudgetExceeded) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(run())

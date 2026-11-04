"""Command line: python -m story_factory <command>."""

from __future__ import annotations

import argparse
import json
import tomllib
import logging
import random
import sys
import threading

from . import bench, catalog, db, learning, pipeline, research
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


PING_SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"],
               "additionalProperties": False}


def apply_overrides(cfg: dict, items: list[str]) -> dict:
    """`section.key=value` pairs; the value is read as TOML (8, true, "text") or else as text."""
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"--set needs KEY=VALUE, got {item!r}")
        try:
            value = tomllib.loads(f"v = {raw.strip()}")["v"]
        except tomllib.TOMLDecodeError:
            value = raw.strip()
        *path, last = key.strip().split(".")
        node = cfg
        for part in path:
            if not isinstance(node.get(part), dict):
                raise SystemExit(f"--set: unknown config section {part!r}")
            node = node[part]
        if last not in node:
            raise SystemExit(f"--set: unknown config key {key!r}")
        node[last] = value
    return cfg


def report(conn, row) -> str:
    """A short Markdown summary of one story: verdict, scores, audience, packaging, cost."""
    story = _story_json(row)
    cost = conn.execute("SELECT COALESCE(SUM(usd), 0) FROM costs WHERE story_id = ?", (row["id"],)).fetchone()[0]
    lines = [f"## Story #{row['id']}: {row['title']}", "",
             f"- **Status:** {row['status']}" + (f" ({row['reject_reason']})" if row["reject_reason"] else ""),
             f"- **Series:** {row['series']} ({row['kind']}), {row['word_count'] or 0} words",
             f"- **Recipe:** {row['setting']} / {row['core_fear']} / {row['twist']} / {row['ending']}",
             f"- **Estimated cost of this story:** ${cost:.2f}"]
    if story.get("author_models"):
        lines.append("- **Models:** " + ", ".join(f"{k}: {v}" for k, v in story["author_models"].items()))
    bible = story.get("bible") or {}
    if bible.get("hero"):
        hero = bible["hero"]
        lines.append(f"- **Hero:** {hero.get('name', '')}, {hero.get('who', '')}; wants {hero.get('want', '')}; "
                     f"wound: {hero.get('wound', '')}")
        lines.append(f"- **Theme:** {bible.get('theme', '')} ({bible.get('theme_question', '')}); "
                     f"refrain: \"{bible.get('refrain', '')}\"; red accent: {bible.get('red_accent', '')}")
    review = story.get("review") or {}
    if review.get("scores"):
        lines.append("- **Editors' median scores:** " + ", ".join(f"{k} {v}" for k, v in review["scores"].items()))
    audience = story.get("audience_report") or {}
    if audience.get("chunk_interest"):
        lines.append(f"- **Simulated viewers:** interest by part {audience['chunk_interest']}; "
                     f"weak parts {audience.get('weak_chunks')}; rewrite {audience.get('rewrite', 'not needed')}")
    packaging = story.get("packaging") or {}
    if packaging.get("titles"):
        lines.append("- **Titles (best first):** " + " | ".join(packaging["titles"]))
        lines.append(f"- **Thumbnail:** {packaging.get('thumbnail_brief', '')} [{packaging.get('thumbnail_text', '')}]")
    if audience.get("surprise_test"):
        s = audience["surprise_test"]
        lines.append(f"- **Surprise test:** {s.get('guessed', 0)} of {s.get('readers', 0)} readers guessed the ending"
                     + (f"; misdirection revision {s['revision']}" if s.get("revision") else ""))
    if story.get("pauses") or story.get("pronunciations"):
        lines.append(f"- **Narration notes:** {len(story.get('pauses') or [])} pauses, "
                     f"{len(story.get('pronunciations') or [])} pronunciations")
    if row["status"] in ("rejected", "parked") and story.get("diagnosis"):
        lines.append(f"- **Diagnosis:** {story['diagnosis']}" + ("; best draft kept" if story.get("best_draft") else ""))
    if story.get("scenes"):
        lines.append(f"- **Scenes:** {len(story['scenes'])}, cast: "
                     + ", ".join(m["name"] for m in story.get("cast") or []))
    if row["narration"]:
        opening = " ".join(row["narration"].split()[:80])
        lines += ["", "**Opening:**", "", f"> {opening}..."]
    stages = [entry.get("stage") for entry in story.get("production_log") or []]
    if stages:
        lines += ["", "**Stages run:** " + " → ".join(stages)]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="story_factory", description="Hollow Lines story pipeline")
    parser.add_argument("--config", help="TOML config file (default: ./config.toml if present)")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override one config value for this run, e.g. --set quality.min_score=8")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create the database and seed the story world")
    p = sub.add_parser("models", help="check every configured model against what your API keys can use")
    p.add_argument("--list", action="store_true", help="also print every model ID each provider offers")
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
    sub.add_parser("recheck", help="run the legend recognition test on approved retellings that never had it")
    sub.add_parser("stats", help="cost per story, winning models and common rejection reasons")
    p = sub.add_parser("ping", help="send a tiny real request to every configured model (costs a fraction of a cent)")
    p.add_argument("--models", help="comma-separated provider:model specs to test instead of the configured roles")
    p.add_argument("--timeout", type=int, default=120, help="seconds to wait for each model")
    p = sub.add_parser("report", help="Markdown summary of a story (default: the latest)")
    p.add_argument("id", type=int, nargs="?")
    p = sub.add_parser("bench", help="blind-compare this config's writing against another config on fixed pitches")
    p.add_argument("--against", required=True, help="the other TOML config (setup B)")
    p.add_argument("--pitches", type=int, default=2, help=f"how many fixed pitches (1-{len(bench.PITCHES)})")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = apply_overrides(load_config(args.config), args.set)

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
        if args.list:
            for provider, models in available.items():
                print(f"\n{provider} offers {len(models or [])} models:")
                for model in models or []:
                    print(f"  {model}")
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
    elif args.command == "report":
        conn = db.connect(cfg["database"])
        row = db.get_story(conn, args.id) if args.id else conn.execute(
            # The story worked on last (a shelved idea developed today has an old id).
            "SELECT * FROM stories WHERE status != 'shelved' ORDER BY updated_at DESC, id DESC LIMIT 1").fetchone()
        print(report(conn, row) if row else "No stories yet.")
    elif args.command == "ping":
        router = Router(cfg, db.connect(cfg["database"]))
        failed = 0
        specs = ([t.strip() for t in args.models.split(",") if t.strip()] if args.models
                 else [t for specs in cfg["roles"].values() for t in specs])
        def check(text: str) -> str:
            try:
                client = router._client(text)
                if hasattr(client, "embed") and (text in cfg["roles"]["embedder"] or "embed" in text):
                    return f"OK ({len(client.embed('A lantern in the fog.'))} dimensions)"
                reply = client.generate_json("ping", "You answer with JSON only.", 'Reply {"ok": true}.', PING_SCHEMA)
                # Any well-formed reply proves the model is reachable and returns JSON.
                return "OK" if isinstance(reply.get("ok"), bool) else f"UNEXPECTED REPLY {reply}"
            except Exception as exc:  # report every model, whatever goes wrong
                return f"FAILED: {str(exc)[:400]}"

        for text in dict.fromkeys(specs):
            # A model that hangs gets a verdict too, instead of stalling the whole check.
            box: list[str] = []
            worker = threading.Thread(target=lambda t=text: box.append(check(t)), daemon=True)
            worker.start()
            worker.join(args.timeout)
            result = box[0] if box else f"FAILED: no answer within {args.timeout} seconds"
            failed += not result.startswith("OK")
            print(f"{text:<45} {result}", flush=True)
        return 1 if failed else 0
    elif args.command == "stats":
        print(json.dumps(db.stats(db.connect(cfg["database"])), indent=2))
    elif args.command == "bench":
        summary = bench.run(cfg, load_config(args.against), max(1, min(args.pitches, len(bench.PITCHES))))
        print(bench.markdown(summary, args.config or "default config", args.against))
    elif args.command == "recheck":
        ctx = _context(cfg)
        results = pipeline.recheck_legends(ctx)
        for r in results:
            verdict = "recognized" if r["recognized"] else "NOT recognized, moved back to rejected"
            print(f"#{r['id']} {r['title']}: {verdict}" + (f" ({r['issues'][0]})" if r["issues"] else ""))
        if not results:
            print("No approved retold legend needs a re-check")
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

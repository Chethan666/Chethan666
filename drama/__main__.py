"""Command line for the short drama series.

    python -m drama check                     # what each configured model expects; Boson voices
    python -m drama series                    # write a new series bible
    python -m drama write --count 3           # write the next 3 episodes of the newest series
    python -m drama produce                   # make the next approved episode's video
    python -m drama status
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

from story_factory import db
from story_factory.__main__ import apply_overrides
from story_factory.config import load_config
from story_factory.pipeline import Context
from story_factory.router import Router

from . import writing
from .media import BosonVoice, Fal, MediaError

log = logging.getLogger("drama")


def _context(cfg: dict) -> Context:
    conn = db.connect(cfg["database"])
    return Context(conn=conn, cfg=cfg, router=Router(cfg, conn), rng=random.Random())


def _latest_series(conn) -> int | None:
    row = conn.execute("SELECT id FROM series WHERE status = 'active' ORDER BY id DESC LIMIT 1").fetchone()
    return row["id"] if row else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="drama", description="Short drama series")
    parser.add_argument("--config")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="list each configured fal.ai model's inputs and the Boson voices")
    p = sub.add_parser("series", help="write a new series bible")
    p.add_argument("--genre")
    p.add_argument("--episodes", type=int)
    p = sub.add_parser("write", help="write the next episodes of a series")
    p.add_argument("--series", type=int)
    p.add_argument("--count", type=int, default=1)
    p = sub.add_parser("produce", help="make the video for an approved episode")
    p.add_argument("--episode", type=int, help="a chosen episode id (default: the next approved one)")
    p.add_argument("--out", default="out")
    p.add_argument("--voice", choices=("auto", "kokoro", "boson"), default="auto")
    sub.add_parser("status", help="series and episodes")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = apply_overrides(load_config(args.config), args.set)
    d = cfg["drama"]

    if args.command == "check":
        failed = 0
        print("FAL_KEY:", "set" if os.environ.get("FAL_KEY") else "MISSING (add it as a GitHub Actions secret)")
        print("BOSON_API_KEY:", "set" if os.environ.get("BOSON_API_KEY") else "MISSING (voices fall back to Kokoro)")
        try:
            fal = Fal(key=os.environ.get("FAL_KEY") or "not-needed-to-read-schemas")
            for role, endpoint in d["models"].items():
                try:
                    info = fal.schema(endpoint)
                    print(f"{role:<9} {endpoint}\n          inputs: {', '.join(info['fields'])}\n"
                          f"          required: {', '.join(info['required'])}")
                    unknown = set(d["model_args"].get(role, {})) - set(info["fields"]) if info["fields"] else set()
                    if unknown:
                        print(f"          WARNING: configured arguments the model doesn't list: {sorted(unknown)}")
                except MediaError as exc:
                    failed += 1
                    print(f"{role:<9} {endpoint}  NOT FOUND: {exc}")
        except MediaError as exc:
            failed += 1
            print(f"fal.ai: {exc}")
        try:
            print("Boson voices:", json.dumps(BosonVoice(d["boson_model"]).voices(), indent=2)[:6000])
        except MediaError as exc:
            print(f"Boson: {exc}")
        return 1 if failed else 0

    ctx = _context(cfg)
    if args.command == "series":
        series_id = writing.create_series(ctx, args.genre or d["genre"], args.episodes or d["episodes_per_series"])
        bible = json.loads(ctx.conn.execute("SELECT bible FROM series WHERE id = ?", (series_id,)).fetchone()[0])
        print(f"Series #{series_id}: {bible['title']}\n{bible['logline']}\nCast: "
              + ", ".join(f"{c['name']} ({c['role']})" for c in bible["cast"]))
        return 0
    if args.command == "write":
        series_id = args.series or _latest_series(ctx.conn)
        if series_id is None:
            print("No active series; run `python -m drama series` first", file=sys.stderr)
            return 1
        approved = 0
        for _ in range(args.count):
            episode_id, status = writing.write_episode(ctx, series_id)
            row = ctx.conn.execute("SELECT number, title, review FROM episodes WHERE id = ?", (episode_id,)).fetchone()
            print(f"[{status}] series #{series_id} episode {row['number']}: {row['title']} "
                  f"{json.loads(row['review'] or '{}').get('scores', '')}")
            approved += status == "approved"
        return 0 if approved else 1
    if args.command == "produce":
        from .produce import Producer  # video libraries are only needed here

        row = (ctx.conn.execute("SELECT * FROM episodes WHERE id = ?", (args.episode,)).fetchone() if args.episode
               else ctx.conn.execute("SELECT * FROM episodes WHERE status = 'approved' ORDER BY series_id, number "
                                     "LIMIT 1").fetchone())
        if row is None:
            print("No approved episode waiting", file=sys.stderr)
            return 1
        use_boson = args.voice == "boson" or (args.voice == "auto" and d.get("boson_voices")
                                              and os.environ.get("BOSON_API_KEY"))
        if use_boson:
            voices = BosonVoice(d["boson_model"])
            if d.get("boson_voices"):
                d["voice_pool"] = d["boson_voices"]
        else:
            from production.voice import KokoroVoice
            voices = KokoroVoice("am_michael", cfg["video"].get("lang", "a"), 1.0)
        assets = Path(cfg["database"]).parent / "series" / str(row["series_id"])
        try:
            meta = Producer(ctx.conn, cfg, assets, Fal(), voices).produce(row["id"], Path(args.out))
        except Exception as exc:  # record and report; the episode stays approved to retry
            log.exception("Episode production failed")
            db_info = {"error": str(exc)[:500], "run": os.environ.get("GITHUB_RUN_ID")}
            ctx.conn.execute("UPDATE episodes SET production = ?, updated_at = ? WHERE id = ?",
                             (json.dumps(db_info), db.now(), row["id"]))
            print(f"[failed] episode #{row['id']}: {exc}", file=sys.stderr)
            return 2
        meta["run"] = os.environ.get("GITHUB_RUN_ID")
        ctx.conn.execute("UPDATE episodes SET status = 'produced', production = ?, updated_at = ? WHERE id = ?",
                         (json.dumps(meta), db.now(), row["id"]))
        print(json.dumps(meta, indent=2, ensure_ascii=False))
        return 0
    if args.command == "status":
        for s in ctx.conn.execute("SELECT * FROM series ORDER BY id"):
            print(f"Series #{s['id']} [{s['status']}] {s['title']} ({s['genre']})")
            for e in ctx.conn.execute("SELECT number, title, status FROM episodes WHERE series_id = ? ORDER BY number",
                                      (s["id"],)):
                print(f"  {e['number']:>3}. [{e['status']}] {e['title']}")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())

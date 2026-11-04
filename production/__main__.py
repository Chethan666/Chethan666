"""Command line for pipeline 2.

    python -m production --config config.aws-open.toml produce            # next approved long story
    python -m production --config config.aws-open.toml produce --story 12 # a chosen story, status unchanged
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from story_factory import db
from story_factory.config import load_config
from story_factory.router import Router

from . import run, voice

log = logging.getLogger("production")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="production", description="Hollow Lines video pipeline")
    parser.add_argument("--config", help="TOML config file")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("produce", help="make the video for one approved story")
    p.add_argument("--story", type=int, help="a chosen story (its status is left unchanged)")
    p.add_argument("--out", default="out", help="output folder")
    p.add_argument("--voice", choices=("kokoro", "silent"), default="kokoro",
                   help="silent: a placeholder tone, for testing the rest of the pipeline")
    p.add_argument("--resolution", choices=sorted(run.SIZES), default="1080")
    p.add_argument("--no-direction-model", action="store_true",
                   help="direct scenes from keywords only (no model calls)")
    p.add_argument("--keep-work", action="store_true", help="keep the scene pictures and clips")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    conn = db.connect(cfg["database"])
    video_cfg = cfg.get("video", {})

    if args.story:
        row = db.get_story(conn, args.story)
        if row is None:
            print(f"No story #{args.story}", file=sys.stderr)
            return 1
        claimed = False
    else:
        row = db.claim_next(conn, "long", worker=f"production-{os.environ.get('GITHUB_RUN_ID', 'local')}")
        if row is None:
            print("No approved long story waiting", file=sys.stderr)
            return 1
        claimed = True
    story = dict(row)
    print(f"Producing story #{story['id']}: {story['title']}", flush=True)

    narrator: voice.Voice
    if args.voice == "silent":
        narrator = voice.SilentVoice()
    else:
        narrator = voice.KokoroVoice(video_cfg.get("voice", "am_michael"), video_cfg.get("lang", "a"),
                                     float(video_cfg.get("speed", 0.92)))
    router = None if args.no_direction_model else Router(cfg, conn)
    out_dir = Path(args.out) / f"story-{story['id']}"
    try:
        summary = run.produce(story, out_dir, narrator, router, args.resolution, args.keep_work)
    except Exception as exc:  # report, and hand a claimed story back to the queue
        log.exception("Production failed")
        info = {"error": str(exc)[:500], "run": os.environ.get("GITHUB_RUN_ID")}
        db.update_story(conn, story["id"], production=info, **({"status": "approved"} if claimed else {}))
        print(f"[failed] story #{story['id']}: {exc}", file=sys.stderr)
        return 2
    summary.update(run=os.environ.get("GITHUB_RUN_ID"), voice=args.voice)
    db.update_story(conn, story["id"], production=summary, **({"status": "produced"} if claimed else {}))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

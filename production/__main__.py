"""Command line for pipeline 2.

    python -m production --config config.aws-open.toml produce            # next approved long story
    python -m production --config config.aws-open.toml produce --story 12 # a chosen story, status unchanged
    python -m production --config config.aws-open.toml produce --script production/samples/one_minute.json
    python -m production --config config.aws-open.toml check              # what each paid model expects (no spend)
    python -m production --config config.aws-open.toml produce --script production/samples/one_minute.json \
        --bakeoff all                                                      # pictures only, by each setup
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from story_factory import db
from story_factory.__main__ import apply_overrides
from story_factory.config import load_config
from story_factory.router import Router

from . import voice

log = logging.getLogger("production")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="production", description="Hollow Lines video pipeline")
    parser.add_argument("--config", help="TOML config file")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="list what each paid model on fal.ai expects (no spend)")
    h = sub.add_parser("host", help="paint candidates for the Storyteller's fixed sheet and portrait (paid, once)")
    h.add_argument("--out", default="out/host_refs")
    h.add_argument("--candidates", type=int, default=2)
    h.add_argument("--max-usd", type=float, default=1.5)
    a = sub.add_parser("approve", help="a person has watched the video: approve it for publishing (or reject it)")
    a.add_argument("--story", type=int, required=True)
    a.add_argument("--reject", metavar="REASON", help="reject instead, with the reason")
    u = sub.add_parser("publish", help="upload an approved video to YouTube (private or scheduled)")
    u.add_argument("--story", type=int, required=True)
    u.add_argument("--dir", required=True, help="the video's folder (video.mp4, metadata.json, ...)")
    u.add_argument("--privacy", choices=("private", "unlisted"), default="private")
    u.add_argument("--at", help="schedule: an RFC 3339 time, e.g. 2026-11-20T17:00:00Z")
    sub.add_parser("stats", help="read every published video's YouTube stats into the performance table")
    p = sub.add_parser("produce", help="make the video for one approved story")
    p.add_argument("--story", type=int, help="a chosen story (its status is left unchanged)")
    p.add_argument("--script", help="a story JSON file instead of the database (stills mode; nothing is saved "
                                    "to the database); its optional \"stills\" object overrides settings")
    p.add_argument("--out", default="out", help="output folder")
    p.add_argument("--voice", choices=("kokoro", "silent"), default="kokoro",
                   help="silent: a placeholder tone, for testing the rest of the pipeline")
    p.add_argument("--no-direction-model", action="store_true",
                   help="no planner model calls (storyboard, casting and sound from the scene plan)")
    p.add_argument("--keep-work", action="store_true", help="keep the work folder (pictures, segments)")
    p.add_argument("--max-usd", type=float, help="spending cap for this video (default stills.max_usd)")
    p.add_argument("--bakeoff", metavar="SETUPS", help="no video: paint every picture with each setup in "
                   "stills.bakeoff.setups (comma-separated, or 'all') from the same references, side by side")
    lib = sub.add_parser("library", help="the sound library: build it from free sources, sync, or show it")
    lib.add_argument("action", choices=("build", "sync", "show"))
    lib.add_argument("--sources", default="sounds/sources.json")
    lib.add_argument("--out", default="build/library", help="where new files and library.json are written")
    lib.add_argument("--existing", help="the current library.json (default: downloaded from the release)")
    lib.add_argument("--only", default="", help="comma-separated source types to run (e.g. incompetech,opengameart)")
    lib.add_argument("--generate-music", action="store_true", help="also make the ACE-Step cues (paid, capped)")
    lib.add_argument("--max-usd", type=float, default=1.0, help="cap for --generate-music")
    lib.add_argument("--runs", nargs="*", default=[], help="sync: video output folders (each with sounds_used.json)")
    lib.add_argument("--fresh", action="store_true", help="build: ignore the current library and start over")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = apply_overrides(load_config(args.config), args.set)
    if args.command == "check":
        stills = cfg.get("stills", {})
        return check(stills.get("models", {}), stills.get("model_args", {}))
    if args.command == "library":
        return library(args, cfg)
    if args.command == "host":
        from production.media import Fal
        from .stills import host
        report = host.make(cfg.get("stills", {}), Fal(), Path(args.out), args.candidates, args.max_usd)
        print(json.dumps(report, indent=2))
        return 0 if any("portrait" in c for c in report["candidates"]) else 1
    conn = db.connect(cfg["database"])
    if args.command in ("approve", "publish", "stats"):
        return channel(args, conn)
    video_cfg = cfg.get("video", {})

    script_settings: dict = {}
    if args.command == "produce" and args.bakeoff and not (args.script or args.story):
        print("A bake-off needs --script or --story (it never takes a story from the queue)", file=sys.stderr)
        return 1
    if args.script:
        story = json.loads(Path(args.script).read_text())
        script_settings = story.pop("stills", {}) or {}
        story = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in story.items()}
        row, claimed = story, False
    elif args.story:
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
    stills_cfg = dict(cfg.get("stills", {}), **script_settings)
    if args.max_usd is not None:
        stills_cfg["max_usd"] = args.max_usd
    try:
        from .stills import pipeline as stills
        fal = None
        if os.environ.get("FAL_KEY"):
            from production.media import Fal
            fal = Fal()
        if args.bakeoff:
            from .stills import bakeoff
            stills_cfg["max_usd"] = float(args.max_usd if args.max_usd is not None
                                          else stills_cfg.get("bakeoff", {}).get("max_usd", 7.0))
            chosen = None if args.bakeoff == "all" else [x.strip() for x in args.bakeoff.split(",") if x.strip()]
            summary = bakeoff.run(story, out_dir, narrator, router, stills_cfg, fal, conn, chosen)
        else:
            summary = stills.produce(story, out_dir, narrator, router, stills_cfg, fal, conn, args.keep_work)
    except Exception as exc:  # report, and hand a claimed story back to the queue
        log.exception("Production failed")
        info = {"error": str(exc)[:500], "run": os.environ.get("GITHUB_RUN_ID")}
        if args.script:
            print(f"[failed] script {args.script}: {exc}", file=sys.stderr)
            return 2
        db.update_story(conn, story["id"], production=info, **({"status": "approved"} if claimed else {}))
        print(f"[failed] story #{story['id']}: {exc}", file=sys.stderr)
        return 2
    summary.update(run=os.environ.get("GITHUB_RUN_ID"), run_number=os.environ.get("GITHUB_RUN_NUMBER"),
                   narrator_fallback=args.voice)
    if not args.script and not args.bakeoff:
        db.update_story(conn, story["id"], production=summary, **({"status": "produced"} if claimed else {}))
    print(json.dumps(summary, indent=2, default=str))
    return 0


def channel(args: argparse.Namespace, conn) -> int:
    """The human review gate, publishing, and stats."""
    from . import youtube

    if args.command == "stats":
        rows = conn.execute("SELECT id, video_id FROM stories WHERE video_id IS NOT NULL AND video_id != ''").fetchall()
        if not rows:
            print("No published videos yet")
            return 0
        stats = youtube.YouTube().stats([r["video_id"] for r in rows])
        for r in rows:
            s = stats.get(r["video_id"])
            if s:
                db.record_performance(conn, r["id"], **s)
                print(f"story #{r['id']} {r['video_id']}: {s}")
        return 0
    row = db.get_story(conn, args.story)
    if row is None:
        print(f"No story #{args.story}", file=sys.stderr)
        return 1
    production = json.loads(row["production"] or "{}") if isinstance(row["production"], str) else {}
    if args.command == "approve":
        production["review"] = "rejected" if args.reject else "approved"
        if args.reject:
            production["review_note"] = args.reject
        db.update_story(conn, args.story, production=production)
        print(f"Story #{args.story}: {production['review']}")
        return 0
    if production.get("review") != "approved":
        print(f"Story #{args.story} is not approved yet (review: {production.get('review', 'none')}). "
              "Watch preview_60s.mp4 and the contact sheet, then run mode video-approve.", file=sys.stderr)
        return 1
    done = youtube.publish(youtube.YouTube(), Path(args.dir), args.privacy, args.at)
    production["youtube"] = done
    db.update_story(conn, args.story, production=production, video_id=done["video_id"], status="published",
                    published_at=db.now())
    print(json.dumps(done, indent=2))
    return 0


ALTERNATES = [  # listed alongside the configured models, to compare
    "fal-ai/nano-banana-pro", "fal-ai/nano-banana-pro/edit", "fal-ai/nano-banana-2", "fal-ai/nano-banana-2/edit",
    "fal-ai/bytedance/seedream/v4.5/text-to-image", "fal-ai/bytedance/seedream/v4.5/edit",
    "fal-ai/elevenlabs/tts/eleven-v3", "fal-ai/elevenlabs/sound-effects/v2", "fal-ai/ace-step-1.5",
    # candidates for the picture bake-off: FLUX.2 with references, and detail (upscale) passes
    "fal-ai/flux-2-pro/edit", "fal-ai/flux-2-flex/edit", "fal-ai/flux-2/edit", "fal-ai/flux-2-max/edit",
    "fal-ai/seedvr/upscale/image", "fal-ai/clarity-upscaler", "fal-ai/recraft/upscale/crisp",
    "fal-ai/topaz/upscale/image", "fal-ai/crystal-upscaler",
    # Nano Banana 2.1 (October 2026): its name on fal, to confirm
    "fal-ai/nano-banana-2.1", "fal-ai/nano-banana-2.1/edit", "google/nano-banana-2.1", "google/nano-banana-2.1/edit",
    "fal-ai/gemini-3.1-flash-image", "fal-ai/gemini-3.1-flash-image/edit",
    "fal-ai/flux-3/edit", "fal-ai/flux-3-pro/edit",
    # Stable Diffusion: which versions fal offers, and whether any take reference pictures
    "fal-ai/stable-diffusion-v35-large", "fal-ai/stable-diffusion-v35-large/image-to-image",
    "fal-ai/stable-diffusion-v35-medium", "fal-ai/stable-diffusion-v35-large/turbo", "fal-ai/ip-adapter-face-id",
    "fal-ai/fast-sdxl", "fal-ai/lora", "fal-ai/stable-diffusion-v3-medium", "fal-ai/sd35-large/ip-adapter",
    "fal-ai/stable-diffusion-v35-large/ip-adapter",
]


def library(args: argparse.Namespace, cfg: dict) -> int:
    """Build, sync or show the sound library (production/library.py)."""
    from . import library_build
    from .library import Library, fetch_release_file, read_catalogue, release_base

    lib_cfg = dict(cfg.get("stills", {}).get("sound_library") or {})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    existing = Path(args.existing) if args.existing else None
    if existing is None and not getattr(args, "fresh", False):
        base = release_base(lib_cfg)
        got = out.parent / "library-current.json"
        try:
            if not base:
                raise ValueError("no repository known (set GITHUB_REPOSITORY or stills.sound_library.repo)")
            got.write_bytes(fetch_release_file(f"{base}/library.json"))
            existing = got
        except Exception as exc:  # noqa: BLE001 - the first build starts an empty library
            print(f"No current library ({str(exc)[:120]}); starting a new one")
    if args.action == "show":
        sounds = read_catalogue(existing) if existing else []
        lib = Library(sounds, out, out)
        print(json.dumps({"total": len(lib.sounds), "by_kind": lib.summary()}, indent=2))
        for s in sorted(lib.sounds, key=lambda s: (s.kind, s.title))[:400]:
            print(f"{s.kind:8} {s.seconds:6.1f}s  {','.join(s.moods):14} {s.license:10} uses {s.uses:3}  {s.title}")
        return 0
    if args.action == "sync":
        print(json.dumps(library_build.sync([Path(r) for r in args.runs], existing, out), indent=2))
        return 0
    fal = None
    if args.generate_music and os.environ.get("FAL_KEY"):
        from .media import Fal
        fal = Fal()
    only = {x.strip() for x in args.only.split(",") if x.strip()} or None
    summary = library_build.build(Path(args.sources), out, existing, only, fal, args.max_usd)
    print(json.dumps(summary, indent=2))
    return 0


def check(models: dict, model_args: dict | None = None) -> int:
    """Every paid model's inputs and choices, from fal's public descriptions. No spend."""
    from production.media import Fal

    print("FAL_KEY:", "set" if os.environ.get("FAL_KEY") else "MISSING (add it as a GitHub Actions secret)")
    fal = Fal(key=os.environ.get("FAL_KEY") or "not-needed-to-read-schemas")
    failed = 0
    roles = {role: endpoint for role, endpoint in models.items() if endpoint}  # empty: switched off
    seen = set(roles.values())
    roles.update({f"(alt) {e}": e for e in ALTERNATES if e not in seen})
    for role, endpoint in roles.items():
        try:
            info = fal.schema(endpoint)
        except Exception as exc:  # noqa: BLE001 - a report, not a run
            failed += not role.startswith("(alt)")
            print(f"{role:<12} {endpoint}  NOT FOUND: {str(exc)[:160]}")
            continue
        price = fal.price(endpoint) if os.environ.get("FAL_KEY") else ""
        print(f"{role:<12} {endpoint}{'  [' + price + ']' if price else ''}\n    inputs: {', '.join(info['fields'])}\n"
              f"    required: {', '.join(info['required'])}")
        for field, choices in info.get("options", {}).items():
            print(f"    {field}: {json.dumps(choices)[:1500]}")
        args = (model_args or {}).get(role, {})
        unknown = set(args) - set(info["fields"]) if info["fields"] else set()
        if unknown:
            print(f"    WARNING: configured arguments the model doesn't list: {sorted(unknown)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

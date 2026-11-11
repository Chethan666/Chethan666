"""Command line for pipeline 2.

    python -m production --config config.aws-open.toml produce            # next approved long story
    python -m production --config config.aws-open.toml produce --story 12 # a chosen story, status unchanged
    python -m production --config config.aws-open.toml check              # what each studio model expects (no spend)
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

from . import run, voice

log = logging.getLogger("production")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="production", description="Hollow Lines video pipeline")
    parser.add_argument("--config", help="TOML config file")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="list what each studio model on fal.ai expects (no spend)")
    a = sub.add_parser("approve", help="a person has watched the video: approve it for publishing (or reject it)")
    a.add_argument("--story", type=int, required=True)
    a.add_argument("--reject", metavar="REASON", help="reject instead, with the reason")
    u = sub.add_parser("publish", help="upload an approved video to YouTube (private or scheduled)")
    u.add_argument("--story", type=int, required=True)
    u.add_argument("--dir", required=True, help="the video's folder (video.mp4, metadata.json, ...)")
    u.add_argument("--privacy", choices=("private", "unlisted"), default="private")
    u.add_argument("--at", help="schedule: an RFC 3339 time, e.g. 2026-11-20T17:00:00Z")
    sub.add_parser("stats", help="read every published video's YouTube stats into the performance table")
    v = sub.add_parser("vast-video", help="a 40-60 s piece made with open models on a rented Vast.ai GPU")
    v.add_argument("--story", type=int, help="a story id (default: the newest approved or produced long story)")
    v.add_argument("--out", default="out")
    v.add_argument("--max-usd", type=float, help="the GPU rental cap for this video (default vast.max_usd)")
    sub.add_parser("vast-check", help="Vast.ai balance and the best GPU offers (no spend)")
    sub.add_parser("vast-cleanup", help="destroy any Vast.ai machine this run left behind")
    b = sub.add_parser("bakeoff", help="the same shots through every candidate model, side by side")
    b.add_argument("--story", type=int, help="a story id (default: the newest approved or produced long story)")
    b.add_argument("--out", default="out")
    b.add_argument("--max-usd", type=float, default=10.0)
    p = sub.add_parser("produce", help="make the video for one approved story")
    p.add_argument("--story", type=int, help="a chosen story (its status is left unchanged)")
    p.add_argument("--out", default="out", help="output folder")
    p.add_argument("--voice", choices=("kokoro", "silent"), default="kokoro",
                   help="silent: a placeholder tone, for testing the rest of the pipeline")
    p.add_argument("--resolution", choices=sorted(run.SIZES), default="1080")
    p.add_argument("--no-direction-model", action="store_true",
                   help="direct scenes from keywords only (no model calls)")
    p.add_argument("--keep-work", action="store_true", help="keep the scene pictures and clips")
    p.add_argument("--no-motion", action="store_true", help="drawn scenes only (no paid moving clips)")
    p.add_argument("--drawn", action="store_true", help="the drawn pipeline even when FAL_KEY is set")
    p.add_argument("--max-usd", type=float, help="studio spending cap for this video")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = apply_overrides(load_config(args.config), args.set)
    if args.command == "check":
        return check(cfg.get("studio", {}))
    if args.command in ("vast-check", "vast-cleanup"):
        return vast_admin(args.command)
    conn = db.connect(cfg["database"])
    if args.command == "bakeoff":
        return bakeoff(args, cfg, conn)
    if args.command == "vast-video":
        return vast_video(args, cfg, conn)
    if args.command in ("approve", "publish", "stats"):
        return channel(args, conn)
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
    animator = None
    if not args.no_motion and int(video_cfg.get("motion_scenes", 0)) > 0 and os.environ.get("FAL_KEY"):
        from drama_production.media import Fal

        from .animate import Animator
        animator = Animator(Fal(), video_cfg["motion_model"], video_cfg.get("motion_args", {}),
                            float(video_cfg.get("motion_usd_per_second", 0.0)),
                            float(video_cfg.get("motion_max_usd", 0.0)), int(video_cfg["motion_scenes"]))
    out_dir = Path(args.out) / f"story-{story['id']}"
    studio_cfg = dict(cfg.get("studio", {}))
    if args.max_usd is not None:
        studio_cfg["max_usd"] = args.max_usd
    use_studio = (studio_cfg.get("enabled") and os.environ.get("FAL_KEY") and not args.drawn
                  and not args.no_motion)
    try:
        if use_studio:
            from drama_production.media import Fal

            from . import studio
            animator = None
            summary = studio.produce(story, out_dir, narrator, router, studio_cfg, Fal(), conn, args.keep_work)
        else:
            summary = run.produce(story, out_dir, narrator, router, args.resolution, args.keep_work, video_cfg,
                                  animator)
    except Exception as exc:  # report, and hand a claimed story back to the queue
        log.exception("Production failed")
        info = {"error": str(exc)[:500], "run": os.environ.get("GITHUB_RUN_ID")}
        db.update_story(conn, story["id"], production=info, **({"status": "approved"} if claimed else {}))
        print(f"[failed] story #{story['id']}: {exc}", file=sys.stderr)
        return 2
    if animator is not None and animator.spent:
        db.record_cost(conn, pipeline="video", task="motion", model=animator.model, input_units=0,
                       output_units=len(summary.get("moving_scenes", [])), usd=animator.spent)
    summary.update(run=os.environ.get("GITHUB_RUN_ID"), run_number=os.environ.get("GITHUB_RUN_NUMBER"),
                   narrator_fallback=args.voice)
    db.update_story(conn, story["id"], production=summary, **({"status": "produced"} if claimed else {}))
    print(json.dumps(summary, indent=2))
    return 0


def _label() -> str:
    return f"hollow-lines-{os.environ.get('GITHUB_RUN_ID', 'local')}"


def vast_admin(command: str) -> int:
    from .vast import QUERIES, Vast, download_usd, expected_cost, upload_usd

    vast = Vast(label=_label())
    if command == "vast-cleanup":
        print(f"Destroyed {vast.cleanup_label()} leftover machine(s) labelled {vast.label}")
        return 0
    print("Balance:", vast.balance())
    try:
        user = vast.cli("show", "user")
        print("Account:", {k: user.get(k) for k in ("credit", "balance", "balance_threshold", "has_billing",
                                                    "billing_creditonly", "can_pay") if isinstance(user, dict)})
        mine = vast.cli("show", "instances") or []
        print(f"Your machines ({len(mine)}):")
        for inst in mine:
            print(f"  id {inst.get('id')} {inst.get('gpu_name')} status {inst.get('actual_status')} "
                  f"label {inst.get('label')} ${float(inst.get('dph_total') or 0):.2f}/h")
    except Exception as exc:  # noqa: BLE001 - a report
        print("Account details unavailable:", exc)
    offers = vast.offers()
    print(f"{len(offers)} suitable offers; the cheapest:")
    for o in offers[:8]:
        print(f"  id {o.get('id')}  {o.get('gpu_name')}  ${float(o.get('dph_total') or 0):.2f}/h  "
              f"gpu_ram {o.get('gpu_ram')}  cpu_ram {o.get('cpu_ram')}  disk {o.get('disk_space')}  "
              f"down {o.get('inet_down')}  reliability {o.get('reliability2', o.get('reliability'))}  "
              f"cuda {o.get('cuda_max_good')}  traffic ~${download_usd(o) + upload_usd(o):.2f}  "
              f"typical run ~${expected_cost(o):.2f}")
    if offers:
        print("Offer fields:", ", ".join(sorted(offers[0])))
    # Why offers were left out: the cheapest 80 GB machines before the traffic rule, with their charges.
    for label, query in (("our filters", QUERIES[-1]),
                         ("any verified 80 GB", "num_gpus=1 gpu_ram>=79 verified=true rentable=true")):
        try:
            raw = vast.cli("search", "offers", query, "-o", "dph_total")
        except Exception as exc:  # noqa: BLE001 - a report
            print(f"{label}: search failed: {exc}")
            continue
        raw = raw if isinstance(raw, list) else []
        free = [o for o in raw if download_usd(o) == 0 and upload_usd(o) == 0]
        print(f"{label}: {len(raw)} offers, {len(free)} with no traffic charges; the cheapest:")
        for o in raw[:12]:
            print(f"  id {o.get('id')}  {o.get('gpu_name')}  ${float(o.get('dph_total') or 0):.2f}/h  "
                  f"disk {o.get('disk_space')}  down {o.get('inet_down')}  arch {o.get('cpu_arch')}  "
                  f"down $/TB {o.get('internet_down_cost_per_tb')} (per GB {o.get('inet_down_cost')})  "
                  f"up $/TB {o.get('internet_up_cost_per_tb')} (per GB {o.get('inet_up_cost')})  "
                  f"-> traffic for this job ~${download_usd(o) + upload_usd(o):.2f}")
    return 0


def vast_video(args: argparse.Namespace, cfg: dict, conn) -> int:
    from . import vastvideo
    from .vast import Vast

    row = db.get_story(conn, args.story) if args.story else conn.execute(
        "SELECT * FROM stories WHERE kind = 'long' AND status IN ('approved', 'produced') "
        "ORDER BY id DESC LIMIT 1").fetchone()
    if row is None:
        print("No story to make", file=sys.stderr)
        return 1
    story = dict(row)
    print(f"Open-model video on Vast.ai from story #{story['id']}: {story['title']}", flush=True)
    video_cfg = cfg.get("video", {})
    kokoro = voice.KokoroVoice(video_cfg.get("voice", "am_michael"), video_cfg.get("lang", "a"),
                               float(video_cfg.get("speed", 0.92)))
    vcfg = dict(cfg.get("vast", {}))
    if args.max_usd is not None:
        vcfg["max_usd"] = args.max_usd
    summary = vastvideo.produce(story, Path(args.out) / f"vast-story-{story['id']}", kokoro, Router(cfg, conn), vcfg,
                                Vast(label=_label()), Path(__file__).with_name("vast_worker.py"), conn)
    print(json.dumps(summary, indent=2))
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


def bakeoff(args: argparse.Namespace, cfg: dict, conn) -> int:
    from drama_production.media import Fal

    from . import bakeoff as bake
    row = db.get_story(conn, args.story) if args.story else conn.execute(
        "SELECT * FROM stories WHERE kind = 'long' AND status IN ('approved', 'produced') "
        "ORDER BY id DESC LIMIT 1").fetchone()
    if row is None:
        print("No story to test with", file=sys.stderr)
        return 1
    story = dict(row)
    print(f"Bake-off with story #{story['id']}: {story['title']}", flush=True)
    video_cfg = cfg.get("video", {})
    kokoro = voice.KokoroVoice(video_cfg.get("voice", "am_michael"), video_cfg.get("lang", "a"),
                               float(video_cfg.get("speed", 0.92)))
    studio_cfg = dict(cfg.get("studio", {}), max_usd=args.max_usd)
    report = bake.run(story, Path(args.out) / f"bakeoff-story-{story['id']}", kokoro, Router(cfg, conn), studio_cfg,
                      Fal(), conn)
    print(json.dumps(report, indent=2, ensure_ascii=False)[:20000])
    return 0


# Other names a model may go by, tried by `check` so a wrong ID can be corrected.
ALTERNATES = [
    "fal-ai/nano-banana-pro", "fal-ai/nano-banana-pro/edit", "fal-ai/nano-banana-2", "fal-ai/nano-banana-2/edit",
    "fal-ai/bytedance/seedream/v4.5/text-to-image", "fal-ai/bytedance/seedream/v4.5/edit",
    "bytedance/seedance-2.0/image-to-video", "bytedance/seedance-2.0/fast/image-to-video",
    "bytedance/seedance-2.0/image-to-video-fast", "fal-ai/bytedance/seedance/v2/image-to-video",
    "fal-ai/kling-video/v3/pro/image-to-video", "fal-ai/kling-video/v3/standard/image-to-video",
    "fal-ai/kling-video/v3/turbo/pro/image-to-video", "fal-ai/kling-video/o3/standard/image-to-video",
    "fal-ai/elevenlabs/tts/eleven-v3", "fal-ai/elevenlabs/tts/multilingual-v2",
    "fal-ai/mmaudio-v2", "fal-ai/mmaudio-v2/text-to-audio", "fal-ai/elevenlabs/sound-effects/v2",
    "fal-ai/elevenlabs/sound-effects", "fal-ai/elevenlabs/music", "fal-ai/stable-audio-25/text-to-audio",
    "fal-ai/minimax-music/v2", "fal-ai/lyria2", "beatoven/music-generation", "fal-ai/ace-step/prompt-to-audio",
    "openrouter/router/vision", "fal-ai/any-llm/vision", "fal-ai/imageutils/depth",
]


def check(studio: dict) -> int:
    """Every studio model's inputs and choices, from fal's public descriptions. No spend."""
    from drama_production.media import Fal

    print("FAL_KEY:", "set" if os.environ.get("FAL_KEY") else "MISSING (add it as a GitHub Actions secret)")
    fal = Fal(key=os.environ.get("FAL_KEY") or "not-needed-to-read-schemas")
    failed = 0
    roles = dict(studio.get("models", {}))
    seen = set(roles.values())
    roles.update({f"(alt) {e}": e for e in ALTERNATES if e not in seen})
    for role, endpoint in roles.items():
        try:
            info = fal.schema(endpoint)
        except Exception as exc:  # noqa: BLE001 - a report, not a run
            failed += not role.startswith("(alt)")
            print(f"{role:<12} {endpoint}  NOT FOUND: {str(exc)[:160]}")
            continue
        print(f"{role:<12} {endpoint}\n    inputs: {', '.join(info['fields'])}\n"
              f"    required: {', '.join(info['required'])}")
        for field, choices in info.get("options", {}).items():
            print(f"    {field}: {json.dumps(choices)[:1500]}")
        args = studio.get("model_args", {}).get(role, {})
        unknown = set(args) - set(info["fields"]) if info["fields"] else set()
        if unknown:
            print(f"    WARNING: configured arguments the model doesn't list: {sorted(unknown)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

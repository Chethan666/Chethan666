"""Command line for pipeline 4, the drama video pipeline.

    python -m drama_production check              # what each configured model expects; Boson voices
    python -m drama_production produce            # the next approved episode's video
    python -m drama_production produce --episode 7
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

from .media import BosonVoice, Fal, MediaError

log = logging.getLogger("drama_production")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="drama_production", description="Pipeline 4: drama videos")
    parser.add_argument("--config")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="list each configured fal.ai model's inputs and the Boson voices (no spend)")
    p = sub.add_parser("produce", help="make the video for an approved episode")
    p.add_argument("--episode", type=int, help="a chosen episode id (default: the next approved one)")
    p.add_argument("--out", default="out")
    p.add_argument("--voice", choices=("auto", "kokoro", "boson"), default="auto")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = apply_overrides(load_config(args.config), args.set)
    d = cfg["drama_video"]

    if args.command == "check":
        failed = 0
        print("FAL_KEY:", "set" if os.environ.get("FAL_KEY") else "MISSING (add it as a GitHub Actions secret)")
        print("BOSON_API_KEY:", "set" if os.environ.get("BOSON_API_KEY") else "MISSING (voices fall back to Kokoro)")
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
        try:
            boson = BosonVoice(d["boson_model"])
        except MediaError as exc:
            print(f"Boson: {exc}")
        else:
            for label, call in (("models", boson.models), ("voices", boson.voices)):
                try:
                    print(f"Boson {label}:", json.dumps(call(), indent=2)[:4000])
                except MediaError as exc:
                    print(f"Boson {label}: {exc}")
            print(f"Boson speech with {d['boson_model']}:")
            for name, outcome in boson.probe(list(d.get("boson_probe_voices", []))).items():
                print(f"  {name:<18} {outcome}")
        return 1 if failed else 0

    from .produce import Producer  # the video libraries are only needed here

    conn = db.connect(cfg["database"])
    row = (conn.execute("SELECT * FROM episodes WHERE id = ?", (args.episode,)).fetchone() if args.episode
           else conn.execute("SELECT * FROM episodes WHERE status = 'approved' ORDER BY series_id, number LIMIT 1").fetchone())
    if row is None:
        print("No approved episode waiting", file=sys.stderr)
        return 1
    from .voices import KokoroVoice

    voices = KokoroVoice("am_michael", cfg["video"].get("lang", "a"), 1.0)  # every character's fixed voice
    boson = None
    if args.voice == "boson" or (args.voice == "auto" and os.environ.get("BOSON_API_KEY")):
        boson = BosonVoice(d["boson_model"])  # expressive delivery, cloned from each fixed voice
    assets = Path(cfg["database"]).parent / "series" / str(row["series_id"])
    try:
        meta = Producer(conn, cfg, assets, Fal(), voices, boson=boson).produce(row["id"], Path(args.out))
    except Exception as exc:  # record and report; the episode stays approved to retry
        log.exception("Episode production failed")
        conn.execute("UPDATE episodes SET production = ?, updated_at = ? WHERE id = ?",
                     (json.dumps({"error": str(exc)[:500], "run": os.environ.get("GITHUB_RUN_ID")}), db.now(), row["id"]))
        print(f"[failed] episode #{row['id']}: {exc}", file=sys.stderr)
        return 2
    meta["run"] = os.environ.get("GITHUB_RUN_ID")
    status = {"status": "produced"} if row["status"] == "approved" else {}
    conn.execute("UPDATE episodes SET production = ?, updated_at = ?" + (", status = 'produced'" if status else "")
                 + " WHERE id = ?", (json.dumps(meta), db.now(), row["id"]))
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

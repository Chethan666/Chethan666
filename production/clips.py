"""Moving shots in tiers.

Hero shots (the moments the story turns on) go to the best image-to-video model, with Kling
as the second choice if it fails; medium shots go to the cheap model; the rest stay living
stills. Each clip is as long as its shot needs (the shortest length the model offers that
covers it). Frames from every clip are checked in code, and a hero clip's middle frame by the
vision model too, so a clip that has drifted from the chalk look (colour, realism, melting)
is dropped and the shot stays a still. Work stops at the budget, most important first.
"""

from __future__ import annotations

import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from production.media import MediaError, data_uri

from .look import chalk_check
from .paid import OverBudget, Studio, vision_check
from .shots import Shot

log = logging.getLogger(__name__)

MOTION = ("Animate this chalk drawing on a black chalkboard. Keep the exact drawing style in every frame: white "
          "chalk lines, simple stick figures, black background, the only colour is the existing red chalk. Do not "
          "add colour, shading, realistic detail, faces, text or new characters. Movement is simple and "
          "deliberate, like hand-drawn animation, with a faint chalk-line boil.")
CAMERA_WORDS = {"still": "Locked-off camera.", "push_in": "Slow push in.", "pull_out": "Slow pull back.",
                "pan_left": "Slow pan left.", "pan_right": "Slow pan right.", "tilt_up": "Slow tilt up.",
                "tilt_down": "Slow tilt down.", "handheld": "Slight handheld shake."}


def motion_prompt(shot: Shot) -> str:
    return f"{MOTION} Action: {shot.motion or 'subtle breathing and drifting chalk dust'}. {CAMERA_WORDS.get(shot.camera, '')}"


def length(seconds: float, allowed: list[int], stretch: float = 1.25) -> int:
    """The shortest clip that covers the shot (slowed by up to `stretch`)."""
    allowed = sorted(int(a) for a in allowed) or [5]
    return next((a for a in allowed if a * stretch >= seconds), allowed[-1])


def frames(video: Path, at: list[float], out_prefix: Path) -> list[Path]:
    paths = []
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                            str(video)], capture_output=True, text=True)
    try:
        total = float(probe.stdout.strip())
    except ValueError:
        return []
    for k, share in enumerate(at):
        path = out_prefix.with_name(f"{out_prefix.name}_f{k}.png")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{total * share:.2f}", "-i",
                        str(video), "-frames:v", "1", str(path)], capture_output=True)
        if path.exists():
            paths.append(path)
    return paths


class Clips:
    def __init__(self, studio: Studio, cfg: dict[str, Any], work: Path, parallel: int = 4):
        self.studio, self.cfg, self.work, self.parallel = studio, cfg, work, parallel
        self.report: dict[str, Any] = {"made": {}, "rejected": [], "failed": 0}
        work.mkdir(parents=True, exist_ok=True)

    def _args(self, role: str, shot: Shot, picture: Path, seconds: int) -> dict[str, Any]:
        image_key = "start_image_url" if "kling" in self.studio.models.get(role, "") else "image_url"
        return {image_key: data_uri(picture), "prompt": motion_prompt(shot), "duration": str(seconds)}

    def make(self, shot: Shot, picture: Path) -> Path | None:
        roles = ["hero", "hero_alt"] if shot.tier == "hero" else ["medium"]
        for role in roles:
            if role not in self.studio.models:
                continue
            seconds = length(shot.seconds, self.cfg.get("durations", {}).get(role, [5]))
            out = self.work / f"clip_{shot.index:03d}_{role}.mp4"
            try:
                self.studio.call(role, self._args(role, shot, picture, seconds), "video", out, f"clip_{role}",
                                 units=seconds, key=[role, shot.index, shot.motion, shot.camera, seconds,
                                                     picture.stat().st_size])
            except OverBudget:
                return None
            except MediaError as exc:
                self.report["failed"] += 1
                log.warning("Shot %d clip (%s) failed: %s", shot.index, role, str(exc)[:200])
                continue
            checks = frames(out, [0.3, 0.65, 0.95], self.work / f"check_{shot.index:03d}_{role}")
            bad = [why for ok, why in map(chalk_check, checks) if not ok]
            if not bad and shot.tier == "hero" and len(checks) > 1:
                ok, why = vision_check(self.studio, checks[1], shot.picture, task="vision_clip")
                bad = [] if ok else [why]
            if bad or not checks:
                self.report["rejected"].append(f"shot {shot.index} ({role}): {bad[0] if bad else 'unreadable'}")
                continue
            self.report["made"][shot.index] = role
            return out
        return None

    def run(self, shots: list[Shot], pictures: dict[int, Path]) -> dict[int, Path]:
        order = sorted((s for s in shots if s.tier in ("hero", "medium") and s.index in pictures),
                       key=lambda s: (s.tier != "hero", -s.importance, s.index))

        def one(shot: Shot) -> tuple[int, Path | None]:
            return shot.index, self.make(shot, pictures[shot.index])

        with ThreadPoolExecutor(max_workers=self.parallel) as pool:
            made = dict(pool.map(one, order))
        moving = {i: p for i, p in made.items() if p is not None}
        log.info("Clips: %d of %d planned (%d rejected by the checks)", len(moving), len(order),
                 len(self.report["rejected"]))
        return moving

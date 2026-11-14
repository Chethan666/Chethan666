"""Moving pictures: the stage that turns chosen stills into video clips (off until a model is set).

The channel is built as a film made of shots. Today every shot is a still under a camera move;
when an image-to-video model is good and cheap enough, this stage animates the shots that
matter most, and the rest of the pipeline does not change:

- each clip starts from the beat's finished picture (designed people and place, graded), so it
  matches the film, and is asked for only the movement the storyboard wrote for that beat
  (`Shot.motion`: one subtle movement, the camera mostly still);
- the beats chosen follow the same tiers as the pictures (`animation.beats`: "scares", "key" or
  "all"), most important first, within `animation.max_clips` and the video's spending cap;
- every clip is checked: in code (not frozen, not blank, not too bright) and by the vision model
  on its middle frame, against the beat's picture; a clip that fails, or a model that fails,
  leaves the beat a still;
- the edit plays a clip at the beat's exact length (slowed by up to `max_stretch`, then holding
  its last frame), with the same grain, atmosphere, vignette and colours as a still.

Turning it on: set `stills.models.clip` (and optionally `clip_alt`) to image-to-video models on
fal.ai, their prices per second in `stills.prices`, the lengths they accept in
`animation.durations`, and `animation.enabled = true`. `python -m production check` shows what each
model accepts.
"""

from __future__ import annotations

import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from production.media import MediaError, data_uri

from ..paid import OverBudget
from ..shots import Shot
from .board import Board
from .images import still_check

log = logging.getLogger(__name__)

STYLE = ("Animate this film still as a few seconds of a slow-burn horror film. Keep the picture exactly as it is: "
         "the same people, faces, clothes, place, light and colour grade in every frame. Only small, natural "
         "movement; the camera holds almost still. No new people or objects, no text, no cuts.")


def length(seconds: float, allowed: list[int], stretch: float = 1.25) -> int:
    """The shortest clip the model offers that covers the beat (slowed by up to `stretch`)."""
    allowed = sorted(int(a) for a in allowed) or [5]
    return next((a for a in allowed if a * stretch >= seconds), allowed[-1])


def frames(video: Path, shares: list[float], prefix: Path) -> list[Path]:
    """Frames at these shares of a clip's length (for the checks)."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                            str(video)], capture_output=True, text=True)
    try:
        total = float(probe.stdout.strip())
    except ValueError:
        return []
    out = []
    for k, share in enumerate(shares):
        path = prefix.with_name(f"{prefix.name}_f{k}.png")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{total * share:.2f}", "-i",
                        str(video), "-frames:v", "1", str(path)], capture_output=True)
        if path.exists():
            out.append(path)
    return out


def moves(paths: list[Path]) -> bool:
    """Something changes between the first and the last frame checked (a frozen clip is a still we paid for)."""
    if len(paths) < 2:
        return False
    with Image.open(paths[0]) as a, Image.open(paths[-1]) as b:
        x = np.asarray(a.convert("L").resize((160, 90)), dtype=np.float32)
        y = np.asarray(b.convert("L").resize((160, 90)), dtype=np.float32)
    return float(np.abs(x - y).mean()) > 0.5


class MotionAgent:
    name = "motion"
    critical = False  # a clip that fails leaves its beat a still

    def run(self, board: Board) -> dict[str, Any]:
        d, cfg = board.data, board.cfg
        mcfg = dict(cfg.get("animation") or {})
        studio = board.studio
        roles = [r for r in ("clip", "clip_alt") if studio is not None and studio.models.get(r)]
        if not mcfg.get("enabled") or not roles:
            d["clips"] = {}
            return {"enabled": False}
        beats: list[Shot] = d["beats"]
        pictures: dict[int, Path] = d["pictures"]
        host = set(d.get("host_beats") or ())
        tiers = d.get("tiers") or {}
        which = str(mcfg.get("beats", "scares"))
        wanted = {"scares": ("scare",), "key": ("scare", "key"), "all": ("scare", "key", "beat")}.get(which, ("scare",))
        chosen = [b for b in beats if b.index in pictures and b.index not in host and tiers.get(b.index, "beat") in wanted]
        chosen.sort(key=lambda b: (-(b.importance + (2 if b.scare else 0)), b.index))
        chosen = chosen[: int(mcfg.get("max_clips", 12))]
        folder = board.work / "clips"
        folder.mkdir(parents=True, exist_ok=True)
        report: dict[str, Any] = {"enabled": True, "planned": len(chosen), "made": {}, "rejected": [], "failed": 0}
        critic = d.get("critic")

        def make(beat: Shot) -> tuple[int, Path | None]:
            picture = pictures[beat.index]
            for role in roles:
                seconds = length(beat.seconds, list((mcfg.get("durations") or {}).get(role, [5])),
                                 float(mcfg.get("max_stretch", 1.25)))
                image_field = str((mcfg.get("image_field") or {}).get(role, "image_url"))
                jpeg = folder / f"start_{beat.index:03d}.jpg"
                with Image.open(picture) as im:
                    im.convert("RGB").save(jpeg, quality=93)
                args = {image_field: data_uri(jpeg), "prompt": f"{STYLE} Movement: "
                        f"{beat.motion or 'a slow breath of movement in the dark'}.", "duration": str(seconds)}
                out = folder / f"clip_{beat.index:03d}_{role}.mp4"
                try:
                    studio.call(role, args, "video", out, f"clip_{role}", units=seconds,
                                key=[role, beat.index, beat.motion, seconds, picture.stat().st_size])
                except OverBudget:
                    return beat.index, None
                except MediaError as exc:
                    report["failed"] += 1
                    log.warning("Beat %d clip (%s) failed: %s", beat.index, role, str(exc)[:200])
                    continue
                checks = frames(out, [0.15, 0.5, 0.95], folder / f"check_{beat.index:03d}_{role}")
                bad = [why for ok, why in map(still_check, checks) if not ok]
                if not bad and not moves(checks):
                    bad = ["it does not move"]
                if not bad and critic is not None and len(checks) > 1 and mcfg.get("check_clips", True):
                    ok, why = critic.look(checks[1], beat.picture)
                    bad = [] if ok else [why]
                if bad or not checks:
                    report["rejected"].append(f"beat {beat.index} ({role}): {bad[0] if bad else 'unreadable'}")
                    continue
                report["made"][beat.index] = role
                return beat.index, out
            return beat.index, None

        with ThreadPoolExecutor(max_workers=int(mcfg.get("parallel", 3))) as pool:
            made = dict(pool.map(make, chosen))
        d["clips"] = {i: p for i, p in made.items() if p is not None}
        report["clips"] = len(d["clips"])
        return report

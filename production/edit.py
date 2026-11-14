"""The edit's running order: a cold open, a title card, the story cut shot by shot, an end screen.

Shots cut hard on the narration's own sentence breaks; the only fades are into the story
after the title and out of it before the end screen. Segment lengths are counted in whole
frames from the programme's running time, so picture and sound never drift apart. The
stills edit (production/stills/post.py) renders each segment.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from . import draw
from .look import W, H
from .shots import Shot

log = logging.getLogger(__name__)

FPS = 24


def _ff(args: list[str], cwd: Path) -> None:
    result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], cwd=cwd,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[-800:]}")


def probe(path: Path) -> float:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                             str(path)], capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


@dataclass
class Segment:
    name: str  # output file inside the work folder
    start: float  # programme time
    end: float
    picture: Path | None = None
    camera: str = "none"  # the camera move over it (production/stills/camera.py)
    clip: Path | None = None  # a moving clip instead of the still (production/stills/motion.py)
    atmosphere: tuple[str, ...] = ()  # rain, fog, flicker
    fade_in: float = 0.0
    fade_out: float = 0.0

    def frames(self) -> int:
        return max(round(self.end * FPS) - round(self.start * FPS), 1)


def card(background: Path | None, lines: list[tuple[str, int, tuple[int, int, int]]], out: Path,
         left: bool = False, dim: float = 0.35, size: tuple[int, int] = (W, H)) -> Path:
    """A title or end card: the style frame dimmed and blurred, with bone-white text."""
    w, h = size
    if background and background.exists():
        with Image.open(background) as im:
            base = im.convert("RGB").resize((w, h), Image.LANCZOS)
        base = ImageEnhance.Brightness(base.filter(ImageFilter.GaussianBlur(3 * h / 1080))).enhance(dim)
    else:
        base = Image.new("RGB", (w, h), (8, 8, 8))
    d = ImageDraw.Draw(base)
    total = sum(int(points * 1.25) for _, points, _ in lines)
    y = (h - total) // 2
    for text, points, colour in lines:
        f = draw.font(points)
        width = d.textlength(text, font=f)
        x = int(w * 0.08) if left else int((w - width) / 2)
        d.text((x, y), text, font=f, fill=colour, stroke_width=max(points // 18, 2), stroke_fill=(0, 0, 0))
        y += int(points * 1.25)
    base.save(out)
    return out


def wrap(text: str, width: int = 26) -> list[str]:
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 > width and line:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        lines.append(line)
    return lines[:3]


def programme(shots: list[Shot], pictures: dict[int, Path], cold: list[int],
              cold_seconds: float, title_seconds: float, end_seconds: float, story_seconds: float,
              title_card: Path, end_card: Path) -> list[Segment]:
    """The running order. Programme time = cold open, then title, then the story, then the end screen."""
    segs: list[Segment] = []
    t = 0.0
    if cold:
        first = shots[cold[0]].t0
        for k, i in enumerate(cold):
            s = shots[i]
            a, b = s.t0 - first, (shots[cold[k + 1]].t0 - first) if k + 1 < len(cold) else cold_seconds
            segs.append(Segment(f"seg_cold_{k}.mp4", a, b, pictures.get(i),
                                fade_in=0.2 if k == 0 else 0.0))
        t = cold_seconds
    if title_seconds:
        segs.append(Segment("seg_title.mp4", t, t + title_seconds, title_card))
        t += title_seconds
    offset = t
    for k, s in enumerate(shots):
        segs.append(Segment(f"seg_{s.index:03d}.mp4", s.t0 + offset, s.t1 + offset, pictures.get(s.index), fade_in=0.4 if k == 0 else 0.0,
                            fade_out=1.0 if k == len(shots) - 1 else 0.0))
    t = offset + story_seconds
    if end_seconds:
        segs.append(Segment("seg_end.mp4", t, t + end_seconds, end_card, fade_in=0.5,
                            fade_out=1.5))
    return segs


def pick_cold_open(shots: list[Shot], scenes: list[dict[str, Any]], min_s: float = 6.0,
                   max_s: float = 14.0) -> list[int]:
    """The scariest moment, before the ending: a scare (or the most important shot) in a key scene,
    with the shots after it until 6-14 s are covered. Never from the final scene (no spoilers)."""
    if not shots:
        return []
    last_scene = shots[-1].scene
    pool = [s for s in shots if s.scene != last_scene and s.t0 > 20] or shots
    def rank(s: Shot) -> tuple:
        key = scenes[s.scene].get("key_moment") if s.scene < len(scenes) else False
        return (not s.scare, not key, -s.importance, s.index)
    start = sorted(pool, key=rank)[0]
    chosen = [start.index]
    covered = start.seconds
    i = start.index + 1
    while covered < min_s and i < len(shots) and shots[i].scene == start.scene:
        if covered + shots[i].seconds > max_s:
            break
        chosen.append(i)
        covered += shots[i].seconds
        i += 1
    return chosen

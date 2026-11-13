"""Packaging: thumbnails to test, Shorts that send viewers to the story, and the review kit.

- Three thumbnails (the story's thumbnail brief, framed three ways: the victim, the threat,
  the place), drawn by the image model in the channel's chalk style with big chalk lettering,
  for YouTube's Test & Compare. The code-drawn thumbnail is the fallback.
- Two or three vertical Shorts cut from the finished video at the scariest stretches (never
  the ending), 25-55 s, with the story's title on top, big burned-in captions and a line
  pointing to the full story.
- A contact sheet of every shot and a 60-second preview, so a person can approve the video in
  two minutes before anything is published.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from production.media import MediaError, data_uri

from . import draw, timing
from .edit import FPS
from .look import style
from .paid import OverBudget, Studio
from .shots import Shot

log = logging.getLogger(__name__)

ANGLES = (
    "the main character in close-up, frozen in terror, eyes wide, the threat looming just behind them",
    "the threat itself, closer than it should be: a dark shape with glowing red eyes filling half the frame",
    "a single tiny figure in the key place at night, something wrong in the darkness around them",
)


def thumbnails(studio: Studio | None, out_dir: Path, packaging: dict[str, Any], red: str, refs: list[Path],
               key_picture: Path, count: int = 3) -> list[Path]:
    text = str(packaging.get("thumbnail_text", "")).strip()
    brief = str(packaging.get("thumbnail_brief", "")).strip()
    made: list[Path] = []
    for k, angle in enumerate(ANGLES[:count]):
        out = out_dir / f"thumbnail_{k + 1}.jpg"
        raw = out_dir / f"thumbnail_{k + 1}_raw.png"
        if studio is not None and "thumbnail" in studio.models:
            prompt = (f"A YouTube thumbnail. {style(red)} Exception to the no-text rule: big, bold, hand-lettered chalk "
                      f"words \"{text.upper()}\" in the left third, the last word in blood-red chalk, readable at "
                      f"phone size. Picture: {angle}. Brief: {brief} Extreme contrast, one clear focal point, "
                      "lots of black, nothing small or busy.")
            try:
                studio.call("thumbnail", {"prompt": prompt, "image_urls": [data_uri(p) for p in refs if p.exists()][:4]},
                            "image", raw, "thumbnail", key=[prompt, k])
                with Image.open(raw) as im:
                    im.convert("RGB").resize((1280, 720), Image.LANCZOS).save(out, quality=90)
                raw.unlink(missing_ok=True)
                made.append(out)
                continue
            except (MediaError, OverBudget) as exc:
                log.warning("Thumbnail %d not generated: %s", k + 1, str(exc)[:200])
        with Image.open(key_picture) as im:
            draw.thumbnail(im.convert("RGB"), text).convert("RGB").save(out, quality=90)
        made.append(out)
    return made


def short_windows(shots: list[Shot], scenes: list[dict[str, Any]], offset: float, count: int = 3,
                  min_s: float = 25.0, max_s: float = 55.0) -> list[tuple[float, float]]:
    """Programme times of the best stretches: each starts at the start of a key scene (or a scene
    with a scare) and ends on a shot boundary; no overlaps; never the final scene."""
    if not shots or count <= 0:
        return []
    last = shots[-1].scene
    starts: dict[int, Shot] = {}
    for s in shots:
        starts.setdefault(s.scene, s)
    def rank(scene: int) -> tuple:
        sc = scenes[scene] if scene < len(scenes) else {}
        scares = sum(1 for s in shots if s.scene == scene and s.scare)
        weight = sum(s.importance for s in shots if s.scene == scene)
        return (not sc.get("key_moment"), -scares, -weight, scene)
    windows: list[tuple[float, float]] = []
    for scene in sorted((k for k in starts if k != last), key=rank):
        a = starts[scene].t0
        end = a
        for s in shots:
            if s.t0 >= a and s.scene != last and s.t1 - a <= max_s:
                end = s.t1
        if end - a < min_s:
            continue
        if any(not (end <= x or a >= y) for x, y in ((w0 - offset, w1 - offset) for w0, w1 in windows)):
            continue
        windows.append((a + offset, end + offset))
        if len(windows) >= count:
            break
    return sorted(windows)


def _frame(title: str, out: Path) -> Path:
    """The vertical frame around a Short: title on top, a pointer to the full story at the bottom."""
    W, H = 1080, 1920
    im = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(im)
    f = draw.font(78)
    y = 190
    words, line, lines = title.upper().split(), "", []
    for w in words:
        if d.textlength(f"{line} {w}".strip(), font=f) > W - 140 and line:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    lines.append(line)
    for k, text in enumerate(lines[:3]):
        colour = draw.RED if k == len(lines[:3]) - 1 and len(lines) > 1 else draw.CHALK
        d.text(((W - d.textlength(text, font=f)) / 2, y), text, font=f, fill=colour)
        y += 96
    small = draw.font(46)
    note = "FULL STORY ON THE CHANNEL"
    d.text(((W - d.textlength(note, font=small)) / 2, H - 230), note, font=small, fill=draw.CHALK)
    im.save(out)
    return out


def shorts(video: Path, windows: list[tuple[float, float]], cues: list[timing.Cue], title: str,
           out_dir: Path, work: Path) -> list[Path]:
    made = []
    frame = _frame(title, work / "short_frame.png")
    for k, (a, b) in enumerate(windows, 1):
        local = [timing.Cue(max(c.start - a, 0), min(c.end, b) - a, c.text) for c in cues if c.end > a and c.start < b]
        caps = work / f"short_{k}.ass"
        caps.write_text(timing.ass(local, 1080, 1920, size_ratio=0.036, margin_ratio=0.25))
        out = out_dir / f"short_{k}.mp4"
        # The frame is converted with the HD colour matrix like the film, and the Short keeps the
        # film's frame rate: 24 turned into 30 repeats every fourth frame, and a camera move stutters.
        graph = ("[0:v]scale=out_color_matrix=bt709:out_range=limited,format=yuv420p[bg];"
                 "[1:v]scale=1080:-2,setsar=1[v];[bg][v]overlay=0:(H-h)/2-60,"
                 f"subtitles={caps.name},format=yuv420p[out]")
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-loop", "1", "-framerate", str(FPS),
             "-i", str(frame), "-ss", f"{a:.2f}", "-t", f"{b - a:.2f}", "-i", str(video), "-filter_complex", graph,
             "-map", "[out]", "-map", "1:a", "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
             "-x264-params", "aq-mode=3", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
             "-color_range", "tv", "-c:a", "aac", "-b:a", "160k",
             "-t", f"{b - a:.2f}", "-r", str(FPS), "-movflags", "+faststart", str(out.resolve())],
            cwd=work, capture_output=True, text=True)
        if result.returncode != 0:
            log.warning("Short %d failed: %s", k, result.stderr[-400:])
            continue
        made.append(out)
    return made


def contact_sheet(shots: list[Shot], pictures: dict[int, Path], clips: dict[int, Path], out: Path,
                  columns: int = 8, thumb: tuple[int, int] = (240, 135)) -> Path:
    rows = -(-len(shots) // columns) or 1
    sheet = Image.new("RGB", (columns * (thumb[0] + 8) + 8, rows * (thumb[1] + 34) + 8), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    f = draw.font(16, bold=False)
    for k, s in enumerate(shots):
        x, y = 8 + (k % columns) * (thumb[0] + 8), 8 + (k // columns) * (thumb[1] + 34)
        path = pictures.get(s.index)
        if path and path.exists():
            with Image.open(path) as im:
                sheet.paste(im.convert("RGB").resize(thumb), (x, y))
        moving = "CLIP " + s.tier.upper() if s.index in clips else "still"
        d.text((x, y + thumb[1] + 4), f"#{s.index} {s.framing} {moving}{' SCARE' if s.scare else ''}", font=f,
               fill=(220, 60, 60) if s.index in clips else (200, 200, 200))
    sheet.save(out, quality=85)
    return out


def preview(video: Path, out: Path, seconds: float = 60.0) -> Path | None:
    result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-t",
                             f"{seconds:.0f}", "-c", "copy", str(out)], capture_output=True, text=True)
    return out if result.returncode == 0 else None

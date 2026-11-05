"""The edit: a cold open, a title card, the story cut shot by shot, and an end screen.

Shots cut hard on the narration's own sentence breaks; the only fades are into the story
after the title and out of it before the end screen. Each living still keeps a slow camera
move and the chalk boil; each clip plays from its first frame (slowed up to 1.25x) and holds
its last frame if the shot runs longer. Segment lengths are counted in whole frames from the
programme's running time, so picture and sound never drift apart.
"""

from __future__ import annotations

import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from . import assemble, draw
from .look import W, H, boil
from .shots import Shot

log = logging.getLogger(__name__)

FPS = 24
STILL_MOVES = {"still": "push_in", "handheld": "shake"}


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
    picture: Path | None = None  # a still (boiled, with a camera move)
    clip: Path | None = None  # or a moving clip
    camera: str = "push_in"
    fade_in: float = 0.0
    fade_out: float = 0.0

    def frames(self) -> int:
        return max(round(self.end * FPS) - round(self.start * FPS), 1)


ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", "-r", str(FPS), "-an"]


def render(seg: Segment, work: Path, index: int) -> str:
    n = seg.frames()
    seconds = n / FPS
    fades = ""
    if seg.fade_in:
        fades += f",fade=t=in:st=0:d={seg.fade_in:.2f}"
    if seg.fade_out:
        fades += f",fade=t=out:st={max(seconds - seg.fade_out, 0):.2f}:d={seg.fade_out:.2f}"
    if seg.clip is not None:
        length = probe(seg.clip) or seconds
        stretch = min(max(seconds / length, 1.0), 1.25)
        vf = (f"setpts={stretch:.4f}*PTS,scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
              f"fps={FPS},tpad=stop_mode=clone:stop_duration={seconds + 1:.2f}{fades},format=yuv420p")
        _ff(["-i", str(seg.clip), "-vf", vf, "-frames:v", str(n), *ENC, seg.name], work)
        return seg.name
    variants = boil(seg.picture, work / f"boil_{index:03d}", seed=index)
    camera = STILL_MOVES.get(seg.camera, seg.camera)
    vf = assemble.camera_filter(camera, seconds, (W, H), "dark", fades=False) + fades
    pattern = variants[0].name.replace("_0.png", "_%d.png")
    _ff(["-stream_loop", "-1", "-framerate", str(assemble.BOIL_FPS), "-i", pattern, "-vf", vf,
         "-frames:v", str(n), *ENC, seg.name], work)
    return seg.name


def card(background: Path | None, lines: list[tuple[str, int, tuple[int, int, int]]], out: Path,
         left: bool = False, dim: float = 0.35) -> Path:
    """A title or end card: the style frame dimmed and blurred, with chalk-white text."""
    if background and background.exists():
        with Image.open(background) as im:
            base = im.convert("RGB").resize((W, H))
        base = ImageEnhance.Brightness(base.filter(ImageFilter.GaussianBlur(3))).enhance(dim)
    else:
        base = Image.new("RGB", (W, H), (8, 8, 8))
    d = ImageDraw.Draw(base)
    total = sum(int(size * 1.25) for _, size, _ in lines)
    y = (H - total) // 2
    for text, size, colour in lines:
        f = draw.font(size)
        width = d.textlength(text, font=f)
        x = int(W * 0.08) if left else int((W - width) / 2)
        d.text((x, y), text, font=f, fill=colour, stroke_width=max(size // 18, 2), stroke_fill=(0, 0, 0))
        y += int(size * 1.25)
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


def build(work: Path, segments: list[Segment], audio: Path, captions: Path | None, output: Path,
          lufs: float = -14.0, workers: int = 3) -> Path:
    """Render every segment, join them, and add the sound (loudness-normalised for YouTube)."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        names = list(pool.map(lambda job: render(job[1], work, job[0]), enumerate(segments)))
    (work / "segments.txt").write_text("".join(f"file '{n}'\n" for n in names))
    _ff(["-f", "concat", "-safe", "0", "-i", "segments.txt", "-c", "copy", "picture.mp4"], work)
    vf = ["-vf", f"subtitles={captions.name}"] if captions is not None else []
    video_codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p"] if vf else ["-c:v", "copy"]
    _ff(["-i", "picture.mp4", "-i", str(audio), *vf, *video_codec,
         "-af", f"loudnorm=I={lufs}:TP=-1.5:LRA=11", "-ar", "48000", "-c:a", "aac", "-b:a", "192k",
         "-shortest", "-movflags", "+faststart", str(output)], work)
    return output


def programme(shots: list[Shot], pictures: dict[int, Path], clips: dict[int, Path], cold: list[int],
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
            segs.append(Segment(f"seg_cold_{k}.mp4", a, b, pictures.get(i), clips.get(i), s.camera,
                                fade_in=0.2 if k == 0 else 0.0))
        t = cold_seconds
    if title_seconds:
        segs.append(Segment("seg_title.mp4", t, t + title_seconds, title_card, None, "push_in"))
        t += title_seconds
    offset = t
    for k, s in enumerate(shots):
        segs.append(Segment(f"seg_{s.index:03d}.mp4", s.t0 + offset, s.t1 + offset, pictures.get(s.index),
                            clips.get(s.index), s.camera, fade_in=0.4 if k == 0 else 0.0,
                            fade_out=1.0 if k == len(shots) - 1 else 0.0))
    t = offset + story_seconds
    if end_seconds:
        segs.append(Segment("seg_end.mp4", t, t + end_seconds, end_card, None, "push_in", fade_in=0.5,
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

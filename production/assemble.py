"""Putting the video together with FFmpeg.

Each scene becomes a short clip: its three chalk pictures alternate (the boil), a camera
move plays over them (push in, pull out, pan or shake), flickering light dips the
brightness, and every scene fades through black into the next. The clips are joined,
the captions are burned in, and the mixed sound track is added.
"""

from __future__ import annotations

import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

FPS = 24
BOIL_FPS = 6


def _run(args: list[str], cwd: Path) -> None:
    result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], cwd=cwd,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[-800:]}")


def camera_filter(camera: str, seconds: float, size: tuple[int, int], light: str) -> str:
    W, H = size
    n = max(int(seconds * FPS), 1)
    centre = "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
    moves = {
        "push_in": f"z='1+0.08*on/{n}':{centre}",
        "pull_out": f"z='1.08-0.08*on/{n}':{centre}",
        "pan_left": f"z='1.1':x='(iw-iw/zoom)*(1-on/{n})':y='ih/2-(ih/zoom/2)'",
        "pan_right": f"z='1.1':x='(iw-iw/zoom)*on/{n}':y='ih/2-(ih/zoom/2)'",
        "shake": "z='1.05':x='(iw-iw/zoom)/2+5*sin(on*1.9)':y='(ih-ih/zoom)/2+4*cos(on*2.7)'",
    }
    parts = [f"fps={FPS}"]
    if camera in moves:
        # Upscale first so the slow zoom doesn't wobble on whole-pixel steps.
        parts += [f"scale={W * 2}:{H * 2}:flags=bicubic",
                  f"zoompan={moves[camera]}:d=1:s={W}x{H}:fps={FPS}"]
    if light == "flicker":
        parts.append("eq=eval=frame:brightness='-0.07*gt(random(1),0.82)'")
    fade = min(0.35, seconds / 4)
    parts += [f"fade=t=in:st=0:d={fade:.2f}", f"fade=t=out:st={max(seconds - fade, 0):.2f}:d={fade:.2f}",
              "format=yuv420p"]
    return ",".join(parts)


def scene_clip(work: Path, index: int, seconds: float, camera: str, light: str, size: tuple[int, int]) -> str:
    name = f"clip_{index:03d}.mp4"
    _run(["-stream_loop", "-1", "-framerate", str(BOIL_FPS), "-i", f"scene_{index:03d}_%d.png", "-t", f"{seconds:.3f}",
          "-vf", camera_filter(camera, seconds, size, light), "-c:v", "libx264", "-preset", "veryfast",
          "-crf", "20", "-threads", "2", "-r", str(FPS), name], work)
    return name


def _length(path: Path) -> float:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                             str(path)], capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def moving_clip(work: Path, index: int, seconds: float, camera: str, light: str, size: tuple[int, int],
                video: Path, max_stretch: float = 1.4) -> str:
    """A generated clip for the scene, slowed a little if needed; if the scene runs longer,
    the drawn scene (with its camera move) carries on after it."""
    W, H = size
    name = f"clip_{index:03d}.mp4"
    length = _length(video) or 5.0
    stretch = min(max(seconds / length, 1.0), max_stretch)
    played = min(seconds, length * stretch)
    fade = min(0.35, played / 4)
    fill = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}"
    flicker = ",eq=eval=frame:brightness='-0.07*gt(random(1),0.82)'" if light == "flicker" else ""
    end_fade = f",fade=t=out:st={max(played - fade, 0):.2f}:d={fade:.2f}" if played >= seconds - 0.05 else ""
    head = f"moving_{index:03d}.mp4"
    _run(["-i", str(video.resolve()), "-vf", f"setpts={stretch:.4f}*PTS,{fill},fps={FPS}{flicker},"
          f"fade=t=in:st=0:d={fade:.2f}{end_fade},format=yuv420p", "-t", f"{played:.3f}", "-an",
          "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-threads", "2", "-r", str(FPS), head], work)
    rest = seconds - played
    if rest < 0.05:
        (work / head).rename(work / name)
        return name
    tail = scene_clip(work, index, rest, camera, light, size)  # writes clip_<index>.mp4
    (work / tail).rename(work / f"rest_{index:03d}.mp4")
    (work / f"join_{index:03d}.txt").write_text(f"file '{head}'\nfile 'rest_{index:03d}.mp4'\n")
    _run(["-f", "concat", "-safe", "0", "-i", f"join_{index:03d}.txt", "-c", "copy", name], work)
    return name


def assemble(work: Path, scenes: list[dict[str, Any]], durations: list[float], size: tuple[int, int],
             audio: str, captions: str, output: str, workers: int = 2,
             moving: dict[int, Path] | None = None) -> None:
    """Render every scene clip, join them, burn in captions and add the sound."""
    moving = moving or {}

    def render(job: tuple) -> str:
        i = job[0]
        return moving_clip(work, *job, moving[i]) if i in moving else scene_clip(work, *job)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        clips = list(pool.map(render, [
            (i, d, str(s.get("camera", "still")), str(s.get("light", "dark")), size)
            for i, (s, d) in enumerate(zip(scenes, durations))]))
    (work / "clips.txt").write_text("".join(f"file '{c}'\n" for c in clips))
    _run(["-f", "concat", "-safe", "0", "-i", "clips.txt", "-i", audio, "-vf", f"subtitles={captions}",
          "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac",
          "-b:a", "192k", "-shortest", "-movflags", "+faststart", output], work)

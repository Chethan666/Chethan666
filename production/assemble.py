"""Putting the video together with FFmpeg.

Each scene's still picture becomes a short segment: its three chalk versions alternate (the
boil), a slow camera move plays over them (push in, pull out, pan or shake), flickering light
dips the brightness, and every scene fades through black into the next. The segments are
joined, the captions are burned in, and the mixed sound track is added.
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


def camera_filter(camera: str, seconds: float, size: tuple[int, int], light: str, fades: bool = True) -> str:
    W, H = size
    n = max(int(seconds * FPS), 1)
    centre = "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
    moves = {
        "push_in": f"z='1+0.08*on/{n}':{centre}",
        "pull_out": f"z='1.08-0.08*on/{n}':{centre}",
        "pan_left": f"z='1.1':x='(iw-iw/zoom)*(1-on/{n})':y='ih/2-(ih/zoom/2)'",
        "pan_right": f"z='1.1':x='(iw-iw/zoom)*on/{n}':y='ih/2-(ih/zoom/2)'",
        "shake": "z='1.05':x='(iw-iw/zoom)/2+5*sin(on*1.9)':y='(ih-ih/zoom)/2+4*cos(on*2.7)'",
        "tilt_up": f"z='1.1':x='iw/2-(iw/zoom/2)':y='(ih-ih/zoom)*(1-on/{n})'",
        "tilt_down": f"z='1.1':x='iw/2-(iw/zoom/2)':y='(ih-ih/zoom)*on/{n}'",
    }
    moves["handheld"] = moves["shake"]
    parts = [f"fps={FPS}"]
    if camera in moves:
        # Upscale first so the slow zoom doesn't wobble on whole-pixel steps.
        parts += [f"scale={W * 2}:{H * 2}:flags=bicubic",
                  f"zoompan={moves[camera]}:d=1:s={W}x{H}:fps={FPS}"]
    if light == "flicker":
        parts.append("eq=eval=frame:brightness='-0.07*gt(random(1),0.82)'")
    fade = min(0.35, seconds / 4)
    if fades:
        parts += [f"fade=t=in:st=0:d={fade:.2f}", f"fade=t=out:st={max(seconds - fade, 0):.2f}:d={fade:.2f}"]
    parts.append("format=yuv420p")
    return ",".join(parts)


def scene_clip(work: Path, index: int, seconds: float, camera: str, light: str, size: tuple[int, int]) -> str:
    name = f"clip_{index:03d}.mp4"
    _run(["-stream_loop", "-1", "-framerate", str(BOIL_FPS), "-i", f"scene_{index:03d}_%d.png", "-t", f"{seconds:.3f}",
          "-vf", camera_filter(camera, seconds, size, light), "-c:v", "libx264", "-preset", "veryfast",
          "-crf", "20", "-threads", "2", "-r", str(FPS), name], work)
    return name


def assemble(work: Path, scenes: list[dict[str, Any]], durations: list[float], size: tuple[int, int],
             audio: str, captions: str, output: str, workers: int = 2) -> None:
    """Render every scene clip, join them, burn in captions and add the sound."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        clips = list(pool.map(lambda job: scene_clip(work, *job), [
            (i, d, str(s.get("camera", "still")), str(s.get("light", "dark")), size)
            for i, (s, d) in enumerate(zip(scenes, durations))]))
    (work / "clips.txt").write_text("".join(f"file '{c}'\n" for c in clips))
    _run(["-f", "concat", "-safe", "0", "-i", "clips.txt", "-i", audio, "-vf", f"subtitles={captions}",
          "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac",
          "-b:a", "192k", "-shortest", "-movflags", "+faststart", output], work)

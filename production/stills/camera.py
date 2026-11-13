"""The camera operator and the film finish: how a still becomes a shot.

A still held for seven seconds is a slideshow; the same still under a slow, steady camera
move is a shot. Every story picture gets a move that suits its framing: wide shots pan,
close ones push in, a scare lunges and settles, a security camera never moves. A move runs
at an even speed from the first frame to the last, so the picture is already moving when
it cuts in and still moving when it cuts out, like footage.

The move is rendered with sub-pixel accuracy (FFmpeg's `perspective` filter, evaluated on
every frame). The usual `zoompan` filter rounds its window to whole pixels and judders by
about 0.2 px a frame on a slow zoom; this is ten times steadier at the same speed.

The film finish is the same on every picture, whichever painter made it:
- fine grain, laid into the picture at the size it is shown (not animated: grain that
  changes every frame triples the file and is the first thing YouTube's encoder smears);
- a vignette fixed to the frame, added after the move, so it stays put when a shot pans;
- colours converted for HD video (BT.709) and labelled as such. FFmpeg's default is the
  standard-definition matrix, which shows the channel's blood red as orange-red in any
  HD player.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from ..look import H, W
from ..shots import Shot

MOVES = ("none", "push_in", "pull_out", "pan_left", "pan_right", "push_in_fast")
# The stream's colour labels (what the conversion in `film_filter` produces).
COLOUR_TAGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
TO_VIDEO = "scale=out_color_matrix=bt709:out_range=limited,format=yuv420p"


def move_for(beat: Shot, k: int) -> str:
    """The move that suits a beat (k: its place in the running order, so neighbours differ)."""
    if beat.view == "cctv":
        return "none"  # a camera bolted to a wall
    if beat.scare:
        return "push_in_fast"
    if beat.framing in ("wide", "high_angle"):
        return "pan_left" if k % 2 else "pan_right"
    if beat.framing in ("insert", "extreme_close", "close", "pov"):
        return "push_in"
    return ("push_in", "pull_out", "push_in")[k % 3]


def path(move: str, seconds: float) -> tuple[float, float, float, float, str]:
    """A move as (zoom at the start, zoom at the end, x at the start, x at the end, curve).

    x is where the window sits in the room the zoom leaves: 0 left, 0.5 centre, 1 right. The
    amount grows with the shot's length so every move has about the same gentle speed."""
    if move in ("push_in", "pull_out"):
        amount = min(max(0.010 * seconds, 0.035), 0.08)
        a, b = (1.0, 1.0 + amount) if move == "push_in" else (1.0 + amount, 1.0)
        return a, b, 0.5, 0.5, "even"
    if move in ("pan_left", "pan_right"):
        room = 1.0 + min(max(0.012 * seconds, 0.04), 0.09)
        a, b = (1.0, 0.0) if move == "pan_left" else (0.0, 1.0)
        return room, room, a, b, "even"
    if move == "push_in_fast":
        amount = min(max(0.03 * seconds, 0.08), 0.14)
        return 1.0, 1.0 + amount, 0.5, 0.5, "settle"  # fast at the cut, slowing: a lunge
    return 1.0, 1.0, 0.5, 0.5, "even"


def grain_on(rgb: np.ndarray, strength: float, seed: int) -> np.ndarray:
    """Fine film grain on a 0-1 picture: a little larger than a pixel, strongest in the shadows
    and mid-tones, nearly absent from pure black (which stays clean) and from highlights."""
    if strength <= 0:
        return rgb
    h, w = rgb.shape[:2]
    rng = np.random.default_rng(seed)
    noise = np.clip(rng.normal(128.0, 42.0, (h, w)), 0, 255).astype("uint8")
    soft = np.asarray(Image.fromarray(noise).filter(ImageFilter.GaussianBlur(0.6)), dtype=np.float32) - 128.0
    soft /= max(float(soft.std()), 1e-6)
    luma = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    weight = (0.45 + 0.55 * (1.0 - luma)) * np.clip(luma / 0.06, 0.25, 1.0)
    return np.clip(rgb + (soft * strength * weight)[..., None], 0.0, 1.0)


def canvas(picture: Path, out: Path, zoom: float, grain: float, seed: int,
           size: tuple[int, int] = (W, H)) -> tuple[int, int]:
    """The picture at the largest size its move shows it (frame x zoom), with the film's grain.

    Pictures are kept larger than the frame until here, so a push-in ends on real detail
    rather than on an enlargement."""
    cw = int(math.ceil(size[0] * max(zoom, 1.0) / 2) * 2)
    ch = int(math.ceil(size[1] * max(zoom, 1.0) / 2) * 2)
    with Image.open(picture) as im:
        im = im.convert("RGB")
        scale = max(cw / im.width, ch / im.height)
        if abs(scale - 1.0) > 1e-6:
            im = im.resize((max(round(im.width * scale), cw), max(round(im.height * scale), ch)), Image.LANCZOS)
        x, y = (im.width - cw) // 2, (im.height - ch) // 2
        rgb = np.asarray(im.crop((x, y, x + cw, y + ch)), dtype=np.float32) / 255.0
    rgb = grain_on(rgb, grain, seed)
    Image.fromarray((rgb * 255.0 + 0.5).astype("uint8")).save(out, compress_level=1)
    return cw, ch


def film_filter(route: tuple[float, float, float, float, str], frames: int, canvas_size: tuple[int, int],
                vignette: float = 0.0, size: tuple[int, int] = (W, H)) -> str:
    """The FFmpeg filters that turn a canvas into a shot: HD colours, the move (`route`, from
    `path`), then the vignette. The canvas must be the frame times the route's largest zoom."""
    w, h = size
    cw, ch = canvas_size
    z0, z1, xa, xb, curve = route
    zmax = max(z0, z1, 1.0)
    parts = [TO_VIDEO]
    if (cw, ch) != (w, h):
        p = f"clip((on-1)/{max(frames - 1, 1)},0,1)"  # the filter counts frames from 1
        if curve == "settle":
            p = f"(1-(1-{p})*(1-{p}))"
        z = f"({z0:.5f}+({z1 - z0:.5f})*{p})"
        s = f"({z}/{zmax:.5f})"  # frame pixels per canvas pixel
        ww, hh = f"({w}/{s})", f"({h}/{s})"  # the window on the canvas that fills the frame
        x0 = f"(({cw}-{ww})*({xa:.3f}+({xb - xa:.3f})*{p}))"
        y0 = f"(({ch}-{hh})/2)"
        x1, y1 = f"({x0}+{cw}/{s})", f"({y0}+{ch}/{s})"
        parts.append(f"perspective=x0='{x0}':y0='{y0}':x1='{x1}':y1='{y0}':x2='{x0}':y2='{y1}':x3='{x1}':y3='{y1}':"
                     f"interpolation=cubic:sense=source:eval=frame,crop={w}:{h}:0:0")
    if vignette > 0:
        # FFmpeg darkens by cos^4(angle x distance): this angle gives `vignette` at the corners.
        angle = math.acos(max(1.0 - min(vignette, 0.9), 0.1) ** 0.25)
        parts.append(f"vignette=angle={angle:.4f}")
    return ",".join(parts)

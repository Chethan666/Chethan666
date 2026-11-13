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

Atmosphere: what moves in the air of a shot, when the storyboard says the picture has it.
- rain: streaks falling at a slant, in front of everything (only when rain falls between
  the camera and the scene, never over an interior);
- fog: soft banks of mist drifting slowly across the frame, thicker low down;
- flicker: a failing light, the picture dipping darker for a frame or two, irregularly.
Rain and fog are textures made once per video (tiled, so they can scroll for ever) and laid
over the moving picture, fixed to the frame like the vignette; flicker is a brightness curve.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..look import H, W
from ..shots import Shot

MOVES = ("none", "push_in", "pull_out", "pan_left", "pan_right", "push_in_fast")
ATMOSPHERES = ("rain", "fog", "flicker")
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


def move_filter(route: tuple[float, float, float, float, str], frames: int, canvas_size: tuple[int, int],
                size: tuple[int, int] = (W, H)) -> str:
    """HD colours, then the move (`route`, from `path`) over a canvas of the frame times the
    route's largest zoom."""
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
    return ",".join(parts)


def vignette_filter(vignette: float) -> str:
    """FFmpeg darkens by cos^4(angle x distance): this angle gives `vignette` at the corners."""
    if vignette <= 0:
        return ""
    angle = math.acos(max(1.0 - min(vignette, 0.9), 0.1) ** 0.25)
    return f"vignette=angle={angle:.4f}"


def film_filter(route: tuple[float, float, float, float, str], frames: int, canvas_size: tuple[int, int],
                vignette: float = 0.0, size: tuple[int, int] = (W, H)) -> str:
    """A shot without atmosphere, as one filter chain: HD colours, the move, the vignette."""
    return ",".join(x for x in (move_filter(route, frames, canvas_size, size), vignette_filter(vignette)) if x)


# --- atmosphere ---------------------------------------------------------------------------
RAIN_SPEED = (-0.10, 1.25)  # frames' widths and heights per second: falling fast, drifting left
FOG_SPEED = 0.025  # frame widths per second


def rain_texture(out: Path, size: tuple[int, int] = (W, H), seed: int = 7) -> Path:
    """Rain over a tile the size of the frame, wrapped at its edges so it can repeat in both
    directions; saved as a 2x2 tile (RGBA). Streaks are drawn along the direction they fall."""
    w, h = size
    rng = np.random.default_rng(seed)
    dx, dy = RAIN_SPEED[0] * w, RAIN_SPEED[1] * h
    norm = math.hypot(dx, dy)
    ux, uy = dx / norm, dy / norm
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    scale = h / 1080
    for _ in range(int(900 * w * h / (1920 * 1080))):
        near = rng.random() < 0.18  # a few big, bright drops close to the lens
        length = (rng.uniform(70, 130) if near else rng.uniform(25, 60)) * scale
        bright = int(rng.uniform(150, 230) if near else rng.uniform(60, 140))
        width = max(1, round((2.2 if near else 1.0) * scale))
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        for ox in (-w, 0, w):
            for oy in (-h, 0, h):
                draw.line((x + ox, y + oy, x + ox + ux * length, y + oy + uy * length), fill=bright, width=width)
    alpha = np.asarray(mask.filter(ImageFilter.GaussianBlur(0.7 * scale)), dtype=np.float32) * 0.45
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = 205  # neutral: the picture's own light colours it through the blend
    rgba[..., 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    Image.fromarray(np.tile(rgba, (2, 2, 1)), "RGBA").save(out, compress_level=1)
    return out


def fog_texture(out: Path, size: tuple[int, int] = (W, H), seed: int = 11) -> Path:
    """Banks of mist over a tile the size of the frame that wraps left to right (made in the
    frequency domain, so it is seamless), thicker towards the bottom; saved 2 tiles wide."""
    w, h = size
    rng = np.random.default_rng(seed)
    small_w, small_h = max(w // 8, 16), max(h // 8, 9)
    noise = rng.normal(0.0, 1.0, (small_h, small_w))
    fy = np.fft.fftfreq(small_h)[:, None] * small_h
    fx = np.fft.fftfreq(small_w)[None, :] * small_w
    soft = np.real(np.fft.ifft2(np.fft.fft2(noise) * np.exp(-(fx ** 2 + (fy * 2.6) ** 2) / (2 * 4.5 ** 2))))
    soft = (soft - soft.mean()) / (soft.std() + 1e-9)
    wrapped = np.concatenate([soft, soft[:, :1]], axis=1)  # resize sees the wrap at the right edge
    big = np.asarray(Image.fromarray(wrapped.astype(np.float32), "F").resize((w + w // small_w, h), Image.BICUBIC),
                     dtype=np.float32)[:, :w]
    density = np.clip((big + 1.2) / 3.0, 0.0, 1.0) ** 1.1  # long, low banks rather than one cloud
    ramp = 0.45 + 0.55 * np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None] ** 1.2  # mist lies low
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = 175
    rgba[..., 3] = np.clip(density * ramp * 0.36 * 255, 0, 255).astype(np.uint8)
    Image.fromarray(np.tile(rgba, (1, 2, 1)), "RGBA").save(out, compress_level=1)
    return out


FLICKER = "eq=eval=frame:brightness='-0.11*gt(random(1),0.93)-0.05*gt(random(2),0.85)+0.012*sin(t*9.7)'"


def atmosphere(kinds: Iterable[str], textures: dict[str, Path], first_input: int,
               size: tuple[int, int] = (W, H)) -> tuple[list[str], list[str]]:
    """The extra inputs and the filter steps that lay `kinds` over a moving picture labelled
    [shot]. Returns (FFmpeg input arguments, filter graph steps); the steps end at [shot]."""
    w, h = size
    inputs: list[str] = []
    steps: list[str] = []
    k = first_input
    wanted = [a for a in ATMOSPHERES if a in set(kinds)]  # flicker first: it is the scene's own light
    for kind in wanted:
        if kind == "flicker":
            steps.append(f"[shot]{FLICKER}[shot]")
            continue
        texture = textures.get(kind)
        if texture is None:
            continue
        inputs += ["-i", str(texture)]
        # converted once, then looped as the same frame: cheap
        steps.append(f"[{k}:v]format=yuva420p,loop=loop=-1:size=1:start=0,setpts=N/(24*TB)[{kind}]")
        if kind == "rain":
            vx, vy = abs(RAIN_SPEED[0]) * w, RAIN_SPEED[1] * h
            x, y = f"-mod(t*{vx:.1f},{w})", f"mod(t*{vy:.1f},{h})-{h}"
        else:
            x, y = f"-mod(t*{FOG_SPEED * w:.1f},{w})", "0"
        steps.append(f"[shot][{kind}]overlay=x='{x}':y='{y}':eval=frame:shortest=1:format=yuv420[shot]")
        k += 1
    return inputs, steps

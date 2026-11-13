"""The colorist: every picture graded so the film cuts together.

Each picture comes from a model that sees only its own prompt, so one frame can come back
twice as bright as its neighbours, or with a different colour cast (run 94: a bright grey
security-camera frame and a blue close-up among dark teal frames). A colorist doesn't make
every shot the same; they fix the ones that jump at the cut. So each picture is measured
(brightness, contrast, colour cast) against the film's median, and only what lies outside
a tolerance is corrected, most of the way back to the edge of that tolerance:

- brightness with a gamma curve, which keeps black black and white white (adding or
  subtracting light would turn shadows milky or crush them);
- contrast, only raised, when a picture is much flatter than the rest (a grey, washed-out
  frame); a harsh picture is left alone, since flattening it would lift its blacks;
- colour cast, only for film frames: a view through a security camera, a phone or an old
  photograph keeps its own colour and is matched in brightness alone.

Pictures inside the tolerance are passed on untouched.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..shots import Shot
from .board import Board

log = logging.getLogger(__name__)

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)
BRIGHTNESS_BAND = 14.0  # levels (0-255) a picture's mean may sit from the film's without being touched
CONTRAST_FLOOR = 0.72  # how much flatter than the film a picture may be
COLOUR_BAND = 5.0  # levels its colour cast (Cb, Cr) may sit from the film's
MAX_BRIGHTNESS_MOVE = 60.0


def stats(rgb: np.ndarray) -> tuple[float, float, float, float]:
    """Brightness (mean luma), contrast (its spread) and colour cast (mean Cb, Cr) of a 0-1 picture,
    in 0-255 levels."""
    y = rgb @ LUMA
    cb = 128.0 + 255.0 * float((-0.168736 * rgb[..., 0] - 0.331264 * rgb[..., 1] + 0.5 * rgb[..., 2]).mean())
    cr = 128.0 + 255.0 * float((0.5 * rgb[..., 0] - 0.418688 * rgb[..., 1] - 0.081312 * rgb[..., 2]).mean())
    return 255.0 * float(y.mean()), 255.0 * float(y.std()), cb, cr


def measure(path: Path) -> tuple[float, float, float, float]:
    with Image.open(path) as im:
        return stats(np.asarray(im.convert("RGB").resize((320, 180)), dtype=np.float32) / 255.0)


def _toward(value: float, low: float, high: float, strength: float) -> float:
    """`value` pulled `strength` of the way back to the band it lies outside of."""
    if value > high:
        return value - (value - high) * strength
    if value < low:
        return value + (low - value) * strength
    return value


def gamma_for(luma: np.ndarray, target: float) -> float:
    """The gamma that brings a 0-1 luma picture's mean to `target` (0-1)."""
    low, high = 0.35, 3.0
    y = np.clip(luma, 1e-4, 1.0)
    for _ in range(24):
        mid = (low + high) / 2
        if float(np.mean(y ** mid)) > target:
            low = mid  # still too bright: a higher gamma darkens
        else:
            high = mid
    return (low + high) / 2


def grade(rgb: np.ndarray, want: tuple[float, float, float, float], strength: float,
          colour: bool = True) -> tuple[np.ndarray, dict[str, float]]:
    """A 0-1 picture brought within the film's tolerances, and what was changed ({} = nothing).

    Each step is decided on the picture as the step before left it: a gamma change moves the
    contrast and the colour cast too."""
    M, S, CB, CR = want
    small = rgb[::8, ::8]
    changed: dict[str, float] = {}
    m = stats(small)[0]
    target = _toward(m, M - BRIGHTNESS_BAND, M + BRIGHTNESS_BAND, strength)
    target = min(max(target, m - MAX_BRIGHTNESS_MOVE), m + MAX_BRIGHTNESS_MOVE)
    if abs(target - m) >= 1.0:
        g = gamma_for(small @ LUMA, target / 255.0)
        rgb = np.clip(rgb, 1e-4, 1.0) ** g
        small = rgb[::8, ::8]
        changed["brightness"] = round(target - m, 1)
    m, s = stats(small)[:2]
    spread = _toward(s, S * CONTRAST_FLOOR, float("inf"), strength)
    if s > 1.0 and spread / s >= 1.03:
        factor = min(spread / s, 1.25)
        rgb = np.clip((rgb - m / 255.0) * factor + m / 255.0, 0.0, 1.0)
        small = rgb[::8, ::8]
        changed["contrast"] = round(factor, 2)
    if colour:
        cb, cr = stats(small)[2:]
        dcb = float(np.clip(_toward(cb, CB - COLOUR_BAND, CB + COLOUR_BAND, strength) - cb, -14, 14))
        dcr = float(np.clip(_toward(cr, CR - COLOUR_BAND, CR + COLOUR_BAND, strength) - cr, -14, 14))
        if abs(dcb) >= 1.0 or abs(dcr) >= 1.0:
            shift = np.array([1.402 * dcr, -0.344 * dcb - 0.714 * dcr, 1.772 * dcb], dtype=np.float32) / 255.0
            rgb = np.clip(rgb + shift, 0.0, 1.0)
            changed["colour"] = round(max(abs(dcb), abs(dcr)), 1)
    return rgb, changed


class ColoristAgent:
    name = "colorist"
    critical = False

    def run(self, board: Board) -> dict[str, Any]:
        d, cfg = board.data, board.cfg
        strength = min(max(float(cfg.get("grade_strength", 0.75)), 0.0), 1.0)
        beats: list[Shot] = d.get("beats", [])
        pictures: dict[int, Path] = d.get("pictures", {})
        if strength <= 0 or len(beats) < 3:
            return {"graded": 0, "pictures": len(beats)}
        measured = {b.index: measure(pictures[b.index]) for b in beats if pictures.get(b.index)}
        film = [measured[b.index] for b in beats if b.index in measured and b.view == "film"] or list(measured.values())
        want = tuple(float(np.median([m[k] for m in film])) for k in range(4))
        folder = board.work / "graded"
        folder.mkdir(parents=True, exist_ok=True)
        done: dict[Path, Path] = {}  # a picture shared by two beats is graded once
        changes: dict[int, dict[str, float]] = {}
        for b in beats:
            src = pictures.get(b.index)
            if src is None or b.index not in measured:
                continue
            if src not in done:
                with Image.open(src) as im:
                    rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
                out, changed = grade(rgb, want, strength, colour=b.view == "film")
                done[src] = src
                if changed:
                    done[src] = folder / f"beat_{b.index:03d}.png"
                    Image.fromarray((out * 255.0 + 0.5).astype("uint8")).save(done[src])
                    changes[b.index] = changed
            pictures[b.index] = done[src]
        return {"graded": len(changes), "pictures": len(measured), "film_brightness": round(want[0], 1),
                "film_contrast": round(want[1], 1), "changes": dict(list(changes.items())[:20])}

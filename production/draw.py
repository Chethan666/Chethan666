"""Chalk stickman pictures drawn in code: white chalk on black paper, one red accent.

Every line is a few jittered passes of a chalk stroke, and three versions of each picture
with different jitter make the classic hand-drawn "boil" when they alternate. Characters
keep the same features in every scene because their look is fixed per story.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

CHALK = (236, 234, 226)
RED = (200, 24, 30)
PAPER = (12, 12, 14)
Point = tuple[float, float]


class Canvas:
    """One picture being drawn: a chalk layer, a red layer and a glow layer."""

    def __init__(self, width: int, height: int, seed: int):
        self.w, self.h = width, height
        self.rng = random.Random(seed)
        self.chalk = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        self.red = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        self.glow = Image.new("L", (width, height), 0)
        self._chalk, self._red, self._glow = (ImageDraw.Draw(self.chalk), ImageDraw.Draw(self.red),
                                              ImageDraw.Draw(self.glow))
        self.unit = height / 1080  # line widths and jitter scale with the picture

    def stroke(self, points: list[Point], red: bool = False, width: float = 5, alpha: int = 230,
               passes: int = 2, jitter: float = 1.6, dashed: bool = False) -> None:
        if len(points) < 2:
            return
        layer = self._red if red else self._chalk
        color = RED if red else CHALK
        for p in range(passes):
            j = jitter * self.unit
            pts = [(x + self.rng.gauss(0, j), y + self.rng.gauss(0, j)) for x, y in _densify(points, 18 * self.unit)]
            w = max(1, round((width - p * 1.5) * self.unit))
            a = alpha if p == 0 else int(alpha * 0.55)
            if dashed:
                for k in range(0, len(pts) - 1, 2):
                    layer.line(pts[k:k + 2], fill=(*color, a), width=w)
            else:
                layer.line(pts, fill=(*color, a), width=w, joint="curve")

    def circle(self, cx: float, cy: float, r: float, red: bool = False, **kw: Any) -> None:
        n = max(28, int(r / (4 * self.unit)))
        pts = [(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n)) for k in range(n + 1)]
        self.stroke(pts, red=red, **kw)

    def fill(self, points: list[Point], shade: int = 0, red: bool = False) -> None:
        """A dark filled shape (silhouettes), or a red one."""
        if red:
            self._red.polygon(points, fill=(*RED, 210))
        else:
            self._chalk.polygon(points, fill=(shade, shade, shade, 255))

    def dot(self, x: float, y: float, r: float, red: bool = False) -> None:
        layer = self._red if red else self._chalk
        layer.ellipse((x - r, y - r, x + r, y + r), fill=(*(RED if red else CHALK), 235))

    def light(self, x: float, y: float, radius: float, strength: int = 70) -> None:
        self._glow.ellipse((x - radius, y - radius, x + radius, y + radius), fill=strength)

    def render(self, paper: Image.Image, grain: np.ndarray, darkness: float = 0.55) -> Image.Image:
        base = paper.copy().convert("RGB")
        glow = self.glow.filter(ImageFilter.GaussianBlur(self.h / 9))
        base = Image.composite(Image.new("RGB", base.size, (58, 56, 52)), base, glow)
        chalk = self.chalk.copy()
        alpha = np.asarray(chalk.split()[3], dtype=np.float32) * grain
        chalk.putalpha(Image.fromarray(alpha.clip(0, 255).astype(np.uint8)))
        base.paste(chalk, (0, 0), chalk)
        red_glow = self.red.split()[3].filter(ImageFilter.GaussianBlur(self.h / 70))
        base = Image.composite(Image.new("RGB", base.size, (120, 10, 14)), base,
                               red_glow.point(lambda v: int(v * 0.45)))
        base.paste(self.red, (0, 0), self.red)
        return vignette(base, darkness)


def _densify(points: list[Point], step: float) -> list[Point]:
    out = [points[0]]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
        out += [(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n) for k in range(1, n + 1)]
    return out


def make_paper(width: int, height: int, seed: int = 7) -> tuple[Image.Image, np.ndarray]:
    """Black paper with a faint texture, and a chalk grain mask (0.55-1.0)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, (height // 4, width // 4)).astype(np.float32)
    blotch = Image.fromarray(((noise - noise.min()) / (np.ptp(noise) + 1e-6) * 255).astype(np.uint8))
    blotch = np.asarray(blotch.resize((width, height), Image.BICUBIC).filter(ImageFilter.GaussianBlur(6)),
                        dtype=np.float32) / 255
    fine = rng.normal(0, 3.0, (height, width)).astype(np.float32)
    level = PAPER[0] + 8 * (blotch - 0.5) + fine
    paper = Image.fromarray(np.dstack([level, level, level + 2]).clip(0, 255).astype(np.uint8))
    grain = 0.55 + 0.45 * rng.random((height, width)).astype(np.float32) ** 0.6
    return paper, grain


def vignette(image: Image.Image, strength: float) -> Image.Image:
    w, h = image.size
    y, x = np.ogrid[:h, :w]
    d = np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2)
    mask = 1 - strength * np.clip(d - 0.35, 0, 1) ** 1.4
    arr = np.asarray(image, dtype=np.float32) * mask[..., None]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


# --- Stick figures ------------------------------------------------------------

# Angles in degrees from straight down; positive swings toward where the figure faces.
# (upper, lower) per limb: back arm, front arm, back leg, front leg.
POSE_ANGLES: dict[str, dict[str, Any]] = {
    "stand": {"lean": 0, "arms": [(8, 4), (-8, -4)], "legs": [(6, 0), (-6, 0)]},
    "walk": {"lean": 4, "arms": [(-28, -12), (30, 28)], "legs": [(-22, -18), (24, 0)]},
    "run": {"lean": 16, "arms": [(-50, 40), (60, 120)], "legs": [(-35, -80), (55, -10)]},
    "sit": {"lean": -4, "arms": [(25, 70), (30, 75)], "legs": [(88, 0), (92, 0)], "seated": True},
    "crouch": {"lean": 32, "arms": [(40, 80), (55, 95)], "legs": [(70, -60), (85, -45)], "drop": 0.42},
    "reach": {"lean": 10, "arms": [(10, 5), (92, 96)], "legs": [(-8, 0), (12, 0)]},
    "point": {"lean": 2, "arms": [(6, 3), (100, 100)], "legs": [(-6, 0), (8, 0)]},
    "look_back": {"lean": -3, "arms": [(-12, -6), (14, 8)], "legs": [(-8, 0), (10, 0)], "turn": True},
    "arms_up": {"lean": -2, "arms": [(165, 175), (-165, -175)], "legs": [(-10, 0), (10, 0)]},
    "hands_face": {"lean": 8, "arms": [(25, 155), (30, 160)], "legs": [(-6, 0), (8, 0)]},
    "kneel": {"lean": 6, "arms": [(10, 5), (40, 60)], "legs": [(0, -88), (85, 0)], "drop": 0.22},
    "hold": {"lean": 0, "arms": [(6, 3), (35, 95)], "legs": [(-6, 0), (8, 0)]},
    "lie": {"lean": 0, "arms": [(8, 4), (-8, -4)], "legs": [(6, 0), (-6, 0)], "lying": True},
}


def _limb(start: Point, angle: float, length: float, facing: int) -> Point:
    a = math.radians(angle)
    return start[0] + math.sin(a) * length * facing, start[1] + math.cos(a) * length


def figure(c: Canvas, x: float, ground: float, size: float, pose: str, facing: str, look: dict[str, Any],
           red_part: str = "", ghost: bool = False, silhouette: bool = False) -> Point:
    """Draws one stick figure with its look. Returns where its front hand ends up."""
    spec = POSE_ANGLES.get(pose, POSE_ANGLES["stand"])
    f = -1 if facing == "left" else 1
    L = size * look.get("height", 1.0) * (0.78 if look.get("child") else 1.0)
    kw: dict[str, Any] = {"alpha": 120, "dashed": True} if ghost else {}
    if spec.get("lying"):
        y = ground - L * 0.05
        head = (x - L * 0.42 * f, y - L * 0.05)
        c.circle(*head, L * 0.075, **kw)
        c.stroke([(head[0] + L * 0.08 * f, y), (x + L * 0.1 * f, y)], **kw)
        for dy in (-0.03, 0.03):
            c.stroke([(x + L * 0.1 * f, y), (x + L * 0.5 * f, y + L * dy)], **kw)
            c.stroke([(head[0] + L * 0.15 * f, y), (head[0] + L * 0.3 * f, y + L * (dy + 0.04))], **kw)
        return (x, y)
    thigh, shin = L * 0.24, L * 0.23
    hip_y = ground - thigh - shin
    if spec.get("seated"):
        hip_y = ground - shin * 1.02
    hip_y += L * spec.get("drop", 0) * 0.9
    hip = (x, hip_y)
    torso = L * 0.30
    lean = math.radians(spec["lean"])
    neck = (x + math.sin(lean) * torso * f, hip_y - math.cos(lean) * torso)
    head_r = L * 0.075
    head = (neck[0] + math.sin(lean) * head_r * f, neck[1] - head_r * 1.15)
    if silhouette:  # something not fully seen: a dark tall shape
        body = [(x - L * 0.14, ground), (x - L * 0.11, neck[1]), (head[0] - head_r, head[1]),
                (head[0], head[1] - head_r * 1.2), (head[0] + head_r, head[1]), (x + L * 0.11, neck[1]),
                (x + L * 0.14, ground)]
        c.fill(body, shade=4)
        c.stroke(body + [body[0]], width=3, alpha=110, **({"dashed": True} if ghost else {}))
        eyes_red = red_part == "eyes" or red_part == "body"
        for dx in (-0.35, 0.35):
            c.dot(head[0] + head_r * dx, head[1], max(2.5, head_r * 0.12), red=eyes_red)
        return (x, neck[1])
    red_body = red_part == "body"
    shoulder = (neck[0], neck[1] + L * 0.03)
    # outfit under the limbs
    outfit = look.get("outfit")
    knee_y = hip_y + thigh * 0.9
    if outfit in ("coat", "robe", "dress"):
        bottom = ground - L * 0.04 if outfit == "robe" else knee_y
        top_w, low_w = L * 0.07, (L * 0.16 if outfit == "dress" else L * 0.12)
        start_y = shoulder[1] if outfit != "dress" else hip_y - torso * 0.35
        c.stroke([(shoulder[0] - top_w, start_y), (x - low_w, bottom), (x + low_w, bottom),
                  (shoulder[0] + top_w, start_y)], red=red_part == "outfit", width=4, **kw)
    c.stroke([hip, neck], red=red_body, **kw)
    hands = []
    for upper, lower in spec["arms"]:
        elbow = _limb(shoulder, upper, L * 0.17, f)
        hand = _limb(elbow, upper + (lower - upper) * 0.5 if abs(lower) > 0 else upper, L * 0.16, f)
        c.stroke([shoulder, elbow, hand], red=red_body, **kw)
        hands.append(hand)
    for upper, lower in spec["legs"]:
        knee = _limb(hip, upper, thigh, f)
        foot = _limb(knee, upper + lower if not spec.get("seated") else 0, shin, f)
        c.stroke([hip, knee, foot], red=red_body, **kw)
    _head(c, head, head_r, f if not spec.get("turn") else -f, look, red_part, kw, ghost)
    _accessory(c, look, shoulder, hip, hands[-1], head, head_r, ground, L, f, red_part, kw)
    return hands[-1]


def _head(c: Canvas, head: Point, r: float, f: int, look: dict[str, Any], red_part: str, kw: dict, ghost: bool) -> None:
    hx, hy = head
    c.circle(hx, hy, r, red=red_part == "head", **kw)
    hair = look.get("hair")
    if hair == "long":
        for side in (-1, 1):
            c.stroke([(hx + side * r * 0.9, hy - r * 0.2), (hx + side * r * 1.1, hy + r * 2.1)], width=3, **kw)
    elif hair == "bun":
        c.circle(hx - f * r * 0.7, hy - r * 0.9, r * 0.35, width=3, **kw)
    elif hair == "short":
        c.stroke([(hx - r * 0.8, hy - r * 0.6), (hx, hy - r * 1.05), (hx + r * 0.8, hy - r * 0.6)], width=3, **kw)
    wear = look.get("headwear")
    red_hat = red_part == "hat"
    if wear == "cap":
        c.stroke([(hx - r, hy - r * 0.35), (hx, hy - r * 1.1), (hx + r, hy - r * 0.35), (hx + f * r * 1.8, hy - r * 0.3)],
                 red=red_hat, width=4, **kw)
    elif wear == "hat":
        c.stroke([(hx - r * 1.6, hy - r * 0.7), (hx + r * 1.6, hy - r * 0.7)], red=red_hat, width=4, **kw)
        c.stroke([(hx - r * 0.8, hy - r * 0.7), (hx - r * 0.7, hy - r * 1.7), (hx + r * 0.7, hy - r * 1.7),
                  (hx + r * 0.8, hy - r * 0.7)], red=red_hat, width=4, **kw)
    elif wear in ("hood", "scarf_head"):
        c.stroke([(hx - r * 1.2, hy + r * 1.3), (hx - r * 1.35, hy - r * 0.4), (hx, hy - r * 1.45),
                  (hx + r * 1.35, hy - r * 0.4), (hx + r * 1.2, hy + r * 1.3)], red=red_hat, width=4, **kw)
    if ghost:
        for dx in (-0.35, 0.35):
            c.circle(hx + r * dx + f * r * 0.15, hy - r * 0.05, r * 0.16, width=2, **kw)
        return
    for dx in (-0.12, 0.38):  # eyes toward where the figure looks
        c.dot(hx + f * r * dx, hy - r * 0.08, max(2.0, r * 0.08), red=red_part == "eyes")
    if look.get("accessory") == "glasses":
        for dx in (-0.12, 0.38):
            c.circle(hx + f * r * dx, hy - r * 0.08, r * 0.22, width=2)


def _accessory(c: Canvas, look: dict[str, Any], shoulder: Point, hip: Point, hand: Point, head: Point,
               r: float, ground: float, L: float, f: int, red_part: str, kw: dict) -> None:
    item = look.get("accessory")
    red_item = red_part in ("accessory", item)
    if item == "lantern":
        hx, hy = hand
        c.stroke([(hx, hy), (hx, hy + L * 0.03)], width=3, **kw)
        box = [(hx - L * 0.025, hy + L * 0.03), (hx + L * 0.025, hy + L * 0.03), (hx + L * 0.03, hy + L * 0.1),
               (hx - L * 0.03, hy + L * 0.1), (hx - L * 0.025, hy + L * 0.03)]
        c.stroke(box, width=3, **kw)
        c.dot(hx, hy + L * 0.065, L * 0.014, red=True)  # the Storyteller's flame is always red
        c.light(hx, hy + L * 0.065, L * 0.25, 60)
    elif item == "scarf":
        nx, ny = shoulder
        c.stroke([(nx - r * 0.8, ny), (nx + r * 0.8, ny), (nx + f * r * 0.4, ny + r * 1.8)], red=red_item, width=4, **kw)
    elif item == "bag":
        bx, by = hip
        c.stroke([(bx - f * L * 0.05, by - L * 0.02), (bx - f * L * 0.12, by - L * 0.02), (bx - f * L * 0.12, by + L * 0.08),
                  (bx - f * L * 0.05, by + L * 0.08), (bx - f * L * 0.05, by - L * 0.02)], red=red_item, width=3, **kw)
    elif item == "cane":
        c.stroke([hand, (hand[0] + f * L * 0.02, ground)], red=red_item, width=4, **kw)
    elif item == "headset":
        hx, hy = head
        c.stroke([(hx - r * 1.05, hy), (hx - r * 0.9, hy - r * 0.9), (hx, hy - r * 1.25), (hx + r * 0.9, hy - r * 0.9),
                  (hx + r * 1.05, hy), (hx + f * r * 0.8, hy + r * 0.8)], red=red_item, width=3, **kw)
    elif item == "badge":
        sx, sy = shoulder
        c.stroke([(sx + f * L * 0.02, sy + L * 0.05), (sx + f * L * 0.05, sy + L * 0.05)], red=red_item, width=5, **kw)


# --- Props --------------------------------------------------------------------

def _rect(c: Canvas, x0: float, y0: float, x1: float, y1: float, red: bool = False, **kw: Any) -> None:
    c.stroke([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], red=red, **kw)


def prop(c: Canvas, kind: str, x: float, ground: float, s: float, red: bool = False) -> None:
    """Draw one prop. `s` is a size unit (about a person's height / 2.4)."""
    fn = PROP_DRAWERS.get(kind)
    if fn:
        fn(c, x, ground, s, red)


def _door(c, x, g, s, red):
    _rect(c, x - s * 0.45, g - s * 1.9, x + s * 0.45, g, red=red, width=5)
    c.dot(x + s * 0.3, g - s * 0.95, s * 0.04, red=red)


def _window(c, x, g, s, red):
    y = g - s * 2.3
    _rect(c, x - s * 0.6, y, x + s * 0.6, y + s * 0.9, red=red, width=5)
    c.stroke([(x, y), (x, y + s * 0.9)], red=red, width=3)
    c.stroke([(x - s * 0.6, y + s * 0.45), (x + s * 0.6, y + s * 0.45)], red=red, width=3)


def _lamp(c, x, g, s, red):
    c.stroke([(x, g), (x, g - s * 1.3)], width=4)
    c.stroke([(x - s * 0.3, g - s * 1.3), (x - s * 0.15, g - s * 1.6), (x + s * 0.15, g - s * 1.6),
              (x + s * 0.3, g - s * 1.3), (x - s * 0.3, g - s * 1.3)], red=red, width=4)
    c.light(x, g - s * 1.3, s * 1.8, 80)


def _streetlight(c, x, g, s, red):
    c.stroke([(x, g), (x, g - s * 3.2), (x + s * 0.6, g - s * 3.4)], width=5)
    c.stroke([(x + s * 0.45, g - s * 3.4), (x + s * 0.75, g - s * 3.4), (x + s * 0.6, g - s * 3.2)], red=red, width=4)
    c.light(x + s * 0.6, g - s * 2.4, s * 2.0, 85)


def _tree(c, x, g, s, red):
    top = g - s * 3.4
    c.stroke([(x, g), (x - s * 0.05, top)], red=red, width=6)
    rng = c.rng
    for k in range(5):
        y = top + s * (0.3 + k * 0.45)
        side = 1 if k % 2 else -1
        c.stroke([(x, y), (x + side * s * (0.9 - k * 0.1), y - s * 0.5)], red=red, width=3)
        c.stroke([(x + side * s * 0.5, y - s * 0.25), (x + side * s * (0.6 + rng.random() * 0.2), y - s * 0.75)],
                 red=red, width=2)


def _car(c, x, g, s, red):
    body = [(x - s * 1.5, g - s * 0.25), (x - s * 1.5, g - s * 0.75), (x - s * 0.8, g - s * 0.8),
            (x - s * 0.45, g - s * 1.25), (x + s * 0.6, g - s * 1.25), (x + s * 1.0, g - s * 0.8),
            (x + s * 1.5, g - s * 0.7), (x + s * 1.5, g - s * 0.25)]
    c.stroke(body + [body[0]], red=red, width=5)
    for wx in (-0.9, 0.9):
        c.circle(x + s * wx, g - s * 0.22, s * 0.22, width=4)
    c.dot(x + s * 1.45, g - s * 0.55, s * 0.06, red=True)


def _bus(c, x, g, s, red):
    _rect(c, x - s * 2.2, g - s * 1.8, x + s * 2.2, g - s * 0.25, red=red, width=5)
    for k in range(5):
        _rect(c, x - s * 2.0 + k * s * 0.85, g - s * 1.6, x - s * 1.4 + k * s * 0.85, g - s * 1.05, width=3)
    for wx in (-1.4, 1.4):
        c.circle(x + s * wx, g - s * 0.22, s * 0.24, width=4)


def _train(c, x, g, s, red):
    _rect(c, x - s * 3.0, g - s * 1.9, x + s * 3.0, g - s * 0.3, red=red, width=5)
    for k in range(6):
        _rect(c, x - s * 2.8 + k * s, g - s * 1.7, x - s * 2.2 + k * s, g - s * 1.1, width=3)
    c.stroke([(x - s * 3.4, g - s * 0.1), (x + s * 3.4, g - s * 0.1)], width=3)


def _boat(c, x, g, s, red):
    c.stroke([(x - s * 1.6, g - s * 0.6), (x - s * 1.1, g), (x + s * 1.1, g), (x + s * 1.6, g - s * 0.6),
              (x - s * 1.6, g - s * 0.6)], red=red, width=5)
    c.stroke([(x, g - s * 0.6), (x, g - s * 2.6)], width=4)
    c.stroke([(x, g - s * 2.5), (x + s * 1.0, g - s * 0.8), (x, g - s * 0.8)], width=3)


def _bed(c, x, g, s, red):
    _rect(c, x - s * 1.3, g - s * 0.75, x + s * 1.3, g - s * 0.35, red=red, width=5)
    c.stroke([(x - s * 1.3, g), (x - s * 1.3, g - s * 1.3)], width=5)
    c.stroke([(x + s * 1.3, g), (x + s * 1.3, g - s * 0.9)], width=5)
    _rect(c, x - s * 1.2, g - s * 0.95, x - s * 0.7, g - s * 0.75, width=3)


def _chair(c, x, g, s, red):
    c.stroke([(x - s * 0.3, g), (x - s * 0.3, g - s * 0.7), (x + s * 0.3, g - s * 0.7), (x + s * 0.3, g)], red=red, width=4)
    c.stroke([(x - s * 0.3, g - s * 0.7), (x - s * 0.3, g - s * 1.5)], red=red, width=4)


def _table(c, x, g, s, red):
    c.stroke([(x - s * 0.9, g - s * 0.9), (x + s * 0.9, g - s * 0.9)], red=red, width=5)
    for dx in (-0.8, 0.8):
        c.stroke([(x + s * dx, g - s * 0.9), (x + s * dx, g)], width=4)


def _desk(c, x, g, s, red):
    _table(c, x, g, s, red)
    _rect(c, x - s * 0.5, g - s * 1.6, x + s * 0.3, g - s * 1.05, width=4)
    c.stroke([(x - s * 0.1, g - s * 1.05), (x - s * 0.1, g - s * 0.9)], width=3)


def _phone(c, x, g, s, red):
    y = g - s * 0.24
    _rect(c, x - s * 0.12, y - s * 0.22, x + s * 0.12, y + s * 0.22, red=red, width=4)
    c.light(x, y, s * 0.6, 50)


def _lantern(c, x, g, s, red):
    _rect(c, x - s * 0.15, g - s * 0.55, x + s * 0.15, g, width=4)
    c.stroke([(x - s * 0.1, g - s * 0.55), (x, g - s * 0.75), (x + s * 0.1, g - s * 0.55)], width=3)
    c.dot(x, g - s * 0.27, s * 0.07, red=True)
    c.light(x, g - s * 0.27, s * 1.4, 70)


def _candle(c, x, g, s, red):
    _rect(c, x - s * 0.07, g - s * 0.5, x + s * 0.07, g, width=3)
    c.dot(x, g - s * 0.6, s * 0.05, red=red or True)
    c.light(x, g - s * 0.6, s * 1.2, 70)


def _mirror(c, x, g, s, red):
    y = g - s * 2.0
    c.circle(x, y, s * 0.5, red=red, width=5)
    c.circle(x, y, s * 0.42, width=2, alpha=120)


def _clock(c, x, g, s, red):
    y = g - s * 2.6
    c.circle(x, y, s * 0.35, red=red, width=4)
    c.stroke([(x, y), (x, y - s * 0.25)], width=3)
    c.stroke([(x, y), (x + s * 0.18, y + s * 0.05)], width=3)


def _house(c, x, g, s, red):
    _rect(c, x - s * 1.4, g - s * 1.8, x + s * 1.4, g, width=5)
    c.stroke([(x - s * 1.6, g - s * 1.8), (x, g - s * 3.0), (x + s * 1.6, g - s * 1.8)], width=5)
    _rect(c, x - s * 0.3, g - s * 1.0, x + s * 0.3, g, width=4)
    _rect(c, x + s * 0.6, g - s * 1.5, x + s * 1.1, g - s * 1.0, red=red, width=4)


def _fence(c, x, g, s, red):
    for k in range(-3, 4):
        c.stroke([(x + k * s * 0.4, g), (x + k * s * 0.4, g - s * 0.9)], red=red, width=3)
    c.stroke([(x - s * 1.3, g - s * 0.65), (x + s * 1.3, g - s * 0.65)], width=3)


def _stairs(c, x, g, s, red):
    pts = [(x - s * 1.5, g)]
    for k in range(6):
        px, py = pts[-1]
        pts += [(px, py - s * 0.35), (px + s * 0.5, py - s * 0.35)]
    c.stroke(pts, red=red, width=4)


def _radio(c, x, g, s, red):
    y = g - s * 0.32
    _rect(c, x - s * 0.45, y - s * 0.3, x + s * 0.45, y + s * 0.3, red=red, width=4)
    c.circle(x - s * 0.2, y, s * 0.15, width=3)
    c.stroke([(x + s * 0.3, y - s * 0.3), (x + s * 0.55, y - s * 0.9)], width=3)


def _screen(c, x, g, s, red):
    y = g - s * 1.6
    _rect(c, x - s * 0.6, y - s * 0.4, x + s * 0.6, y + s * 0.4, red=red, width=5)
    c.stroke([(x, y + s * 0.4), (x, y + s * 0.65)], width=4)
    c.light(x, y, s * 1.3, 55)


def _moon(c, x, g, s, red):
    c.circle(x, c.h * 0.16, s * 0.45, red=red, width=4)
    c.light(x, c.h * 0.16, s * 1.5, 45)


def _grave(c, x, g, s, red):
    c.stroke([(x - s * 0.35, g), (x - s * 0.35, g - s * 0.8), (x - s * 0.2, g - s * 1.0), (x + s * 0.2, g - s * 1.0),
              (x + s * 0.35, g - s * 0.8), (x + s * 0.35, g)], red=red, width=4)


def _sign(c, x, g, s, red):
    c.stroke([(x, g), (x, g - s * 1.8)], width=4)
    _rect(c, x - s * 0.55, g - s * 2.3, x + s * 0.55, g - s * 1.8, red=red, width=4)


def _small(shape: Callable) -> Callable:
    """Small objects sit on a table-height surface so they read on screen."""
    def draw(c, x, g, s, red):
        shape(c, x, g - s * 0.9, s, red)
        c.stroke([(x - s * 0.5, g - s * 0.9), (x + s * 0.5, g - s * 0.9)], width=3, alpha=150)
    return draw


def _cup(c, x, g, s, red):
    _rect(c, x - s * 0.12, g - s * 0.3, x + s * 0.12, g, red=red, width=3)


def _bag(c, x, g, s, red):
    _rect(c, x - s * 0.3, g - s * 0.45, x + s * 0.3, g, red=red, width=4)
    c.stroke([(x - s * 0.15, g - s * 0.45), (x, g - s * 0.65), (x + s * 0.15, g - s * 0.45)], width=3)


def _book(c, x, g, s, red):
    _rect(c, x - s * 0.25, g - s * 0.12, x + s * 0.25, g, red=red, width=3)


def _photo(c, x, g, s, red):
    y = g - s * 2.2
    _rect(c, x - s * 0.3, y - s * 0.4, x + s * 0.3, y + s * 0.4, red=red, width=4)
    c.circle(x, y - s * 0.1, s * 0.1, width=2)


def _key(c, x, g, s, red):
    c.circle(x - s * 0.15, g - s * 0.08, s * 0.08, red=red, width=3)
    c.stroke([(x - s * 0.07, g - s * 0.08), (x + s * 0.2, g - s * 0.08), (x + s * 0.2, g)], red=red, width=3)


def _bell(c, x, g, s, red):
    y = g - s * 2.4
    c.stroke([(x - s * 0.3, y + s * 0.4), (x - s * 0.2, y), (x, y - s * 0.15), (x + s * 0.2, y), (x + s * 0.3, y + s * 0.4),
              (x - s * 0.3, y + s * 0.4)], red=red, width=4)


def _lighthouse(c, x, g, s, red):
    c.stroke([(x - s * 0.6, g), (x - s * 0.35, g - s * 3.6), (x + s * 0.35, g - s * 3.6), (x + s * 0.6, g)], width=5)
    _rect(c, x - s * 0.4, g - s * 4.2, x + s * 0.4, g - s * 3.6, red=red, width=5)
    c.light(x, g - s * 3.9, s * 3.0, 90)


def _elevator(c, x, g, s, red):
    _rect(c, x - s * 0.7, g - s * 2.0, x + s * 0.7, g, red=red, width=5)
    c.stroke([(x, g - s * 2.0), (x, g)], width=3)
    c.dot(x + s * 0.9, g - s * 1.0, s * 0.05, red=red)


def _shelf(c, x, g, s, red):
    _rect(c, x - s * 0.8, g - s * 2.2, x + s * 0.8, g, width=4)
    for k in range(1, 4):
        c.stroke([(x - s * 0.8, g - s * 0.55 * k), (x + s * 0.8, g - s * 0.55 * k)], red=red, width=3)


def _sink(c, x, g, s, red):
    c.stroke([(x - s * 0.6, g - s * 0.9), (x - s * 0.45, g - s * 0.6), (x + s * 0.45, g - s * 0.6),
              (x + s * 0.6, g - s * 0.9)], red=red, width=4)
    c.stroke([(x, g - s * 0.9), (x, g - s * 1.2), (x + s * 0.2, g - s * 1.2)], width=3)


def _jar(c, x, g, s, red):
    _rect(c, x - s * 0.18, g - s * 0.45, x + s * 0.18, g, red=red, width=3)


def _toy(c, x, g, s, red):
    c.circle(x, g - s * 0.35, s * 0.12, red=red, width=3)
    c.stroke([(x, g - s * 0.23), (x, g - s * 0.05)], red=red, width=3)
    c.stroke([(x - s * 0.12, g), (x, g - s * 0.08), (x + s * 0.12, g)], red=red, width=3)


def _letter(c, x, g, s, red):
    _rect(c, x - s * 0.25, g - s * 0.15, x + s * 0.25, g, red=red, width=3)
    c.stroke([(x - s * 0.25, g - s * 0.15), (x, g - s * 0.05), (x + s * 0.25, g - s * 0.15)], red=red, width=2)


def _rock(c, x, g, s, red):
    c.stroke([(x - s * 0.6, g), (x - s * 0.45, g - s * 0.45), (x + s * 0.1, g - s * 0.55), (x + s * 0.55, g - s * 0.3),
              (x + s * 0.6, g)], red=red, width=4)


PROP_DRAWERS: dict[str, Callable] = {
    "door": _door, "window": _window, "lamp": _lamp, "streetlight": _streetlight, "tree": _tree, "car": _car,
    "bus": _bus, "train": _train, "boat": _boat, "bed": _bed, "chair": _chair, "table": _table, "desk": _desk,
    "phone": _small(_phone), "lantern": _lantern, "candle": _small(_candle), "mirror": _mirror, "clock": _clock,
    "house": _house, "fence": _fence, "stairs": _stairs, "radio": _small(_radio), "screen": _screen, "moon": _moon,
    "grave": _grave, "sign": _sign, "cup": _small(_cup), "bag": _bag, "book": _small(_book), "photo": _photo,
    "key": _small(_key), "bell": _bell, "lighthouse": _lighthouse, "elevator": _elevator, "shelf": _shelf,
    "sink": _sink, "jar": _small(_jar), "toy": _toy, "letter": _small(_letter), "rock": _rock,
}


# --- Backgrounds --------------------------------------------------------------

def background(c: Canvas, kind: str, ground: float) -> None:
    W, H = c.w, c.h
    faint = {"alpha": 110, "width": 3}
    if kind in ("room", "office"):
        c.stroke([(0, ground), (W, ground)], **faint)
        for x in (W * 0.12, W * 0.88):
            c.stroke([(x, 0), (x, ground)], alpha=70, width=2)
        c.stroke([(0, ground - H * 0.03), (W * 0.12, ground - H * 0.03)], alpha=70, width=2)
        if kind == "office":
            for k in range(3):
                c.stroke([(W * (0.25 + k * 0.25) - W * 0.06, H * 0.08), (W * (0.25 + k * 0.25) + W * 0.06, H * 0.08)],
                         alpha=120, width=3)
                c.light(W * (0.25 + k * 0.25), H * 0.1, H * 0.25, 30)
    elif kind in ("hallway", "station"):
        vx, vy = W * 0.5, H * 0.45
        for corner in ((0, 0), (W, 0), (0, H), (W, H)):
            c.stroke([corner, (vx + (corner[0] - vx) * 0.18, vy + (corner[1] - vy) * 0.18)], alpha=90, width=3)
        _rect(c, vx - W * 0.09, vy - H * 0.1, vx + W * 0.09, vy + H * 0.1, alpha=90, width=3)
        for k in range(1, 4):
            t = 0.25 * k
            xl = vx + (0 - vx) * (1 - t) * 0.9
            c.stroke([(xl, vy - H * 0.35 * (1 - t)), (xl, vy + H * 0.4 * (1 - t))], alpha=60, width=2)
            xr = vx + (W - vx) * (1 - t) * 0.9
            c.stroke([(xr, vy - H * 0.35 * (1 - t)), (xr, vy + H * 0.4 * (1 - t))], alpha=60, width=2)
        if kind == "station":
            c.stroke([(0, ground + H * 0.04), (W, ground + H * 0.04)], alpha=140, width=4)
    elif kind == "road":
        vx, vy = W * 0.5, H * 0.48
        c.stroke([(0, vy), (W, vy)], alpha=70, width=2)
        c.stroke([(W * 0.05, H), (vx - W * 0.01, vy)], **faint)
        c.stroke([(W * 0.95, H), (vx + W * 0.01, vy)], **faint)
        for k in range(6):
            t0, t1 = k / 6, k / 6 + 0.06
            c.stroke([(vx, H - (H - vy) * t0), (vx, H - (H - vy) * t1)], alpha=120, width=3)
    elif kind == "forest":
        c.stroke([(0, ground), (W, ground)], **faint)
        for k in range(9):
            x = W * (0.03 + 0.12 * k) + c.rng.uniform(-30, 30) * c.unit
            depth = 0.5 + 0.5 * ((k * 7) % 3) / 2
            c.stroke([(x, ground - H * 0.02 * depth), (x + c.rng.uniform(-6, 6), H * (0.05 + 0.15 * (1 - depth)))],
                     alpha=int(60 + 60 * depth), width=int(3 + 4 * depth))
    elif kind in ("field", "sky"):
        horizon = H * (0.82 if kind == "sky" else 0.7)
        c.stroke([(0, horizon), (W * 0.4, horizon - H * 0.02), (W, horizon + H * 0.01)], **faint)
        for k in range(40 if kind == "sky" else 14):
            x, y = c.rng.uniform(0, W), c.rng.uniform(0, horizon * 0.8)
            c.dot(x, y, c.rng.uniform(1.2, 2.6) * c.unit)
    elif kind in ("shore", "water"):
        horizon = H * 0.5
        c.stroke([(0, horizon), (W, horizon)], alpha=120, width=3)
        for k in range(10 if kind == "water" else 5):
            y = horizon + (H - horizon) * (k + 1) / 12
            x0 = c.rng.uniform(0, W * 0.6)
            c.stroke([(x0 + W * 0.05 * j, y + (H * 0.01 if j % 2 else 0)) for j in range(6)], alpha=90, width=2)
        if kind == "shore":
            c.stroke([(0, ground), (W * 0.5, ground - H * 0.02), (W, ground)], **faint)
    elif kind == "street":
        c.stroke([(0, ground), (W, ground)], **faint)
        x = 0.0
        while x < W:
            bw, bh = W * c.rng.uniform(0.12, 0.2), H * c.rng.uniform(0.3, 0.55)
            _rect(c, x, ground - bh, x + bw, ground, alpha=80, width=3)
            for wy in range(2):
                wx = x + bw * 0.3
                c.stroke([(wx, ground - bh * (0.4 + 0.3 * wy)), (wx + bw * 0.2, ground - bh * (0.4 + 0.3 * wy))],
                         alpha=70, width=2)
            x += bw + W * 0.02
    elif kind == "stairwell":
        _stairs(c, W * 0.2, ground, H * 0.18, False)
        c.stroke([(W * 0.18, ground - H * 0.2), (W * 0.62, ground - H * 0.62)], alpha=110, width=3)
        c.stroke([(0, ground), (W, ground)], **faint)
    elif kind == "vehicle":
        c.stroke([(W * 0.1, H * 0.12), (W * 0.9, H * 0.12), (W * 0.97, H * 0.6), (W * 0.03, H * 0.6), (W * 0.1, H * 0.12)],
                 alpha=120, width=4)
        c.stroke([(0, H * 0.72), (W * 0.5, H * 0.66), (W, H * 0.72)], alpha=120, width=4)
        c.circle(W * 0.3, H * 0.86, H * 0.13, alpha=120, width=5)
    elif kind == "cave":
        c.stroke([(0, H * 0.9), (W * 0.1, H * 0.3), (W * 0.35, H * 0.08), (W * 0.65, H * 0.1), (W * 0.9, H * 0.3),
                  (W, H * 0.9)], alpha=110, width=4)
        for k in range(6):
            x = W * (0.2 + 0.12 * k)
            c.stroke([(x - W * 0.01, H * 0.1), (x, H * 0.1 + H * c.rng.uniform(0.05, 0.12)), (x + W * 0.01, H * 0.1)],
                     alpha=90, width=2)
        c.stroke([(0, ground), (W, ground)], **faint)
    else:
        c.stroke([(0, ground), (W, ground)], **faint)


# --- Whole scenes ---------------------------------------------------------------

WALL_PROPS = {"window", "mirror", "clock", "photo", "bell"}


def scene_frames(spec: dict[str, Any], looks: dict[str, dict[str, Any]], size: tuple[int, int],
                 paper: Image.Image, grain: np.ndarray, seed: int, variants: int = 3) -> list[Image.Image]:
    """The scene's picture in `variants` jittered versions (the hand-drawn boil)."""
    W, H = size
    red = str(spec.get("red", "none"))
    frames = []
    for v in range(variants):
        c = Canvas(W, H, seed * 31 + v)
        ground = H * 0.80
        background(c, spec.get("background", "room"), ground)
        props = list(spec.get("props") or [])
        if spec.get("light") == "moonlight" and spec.get("background") in ("road", "forest", "field", "shore", "street",
                                                                          "sky", "water") \
                and not any(p["kind"] == "moon" for p in props):
            props.append({"kind": "moon", "x": 0.8})
        unit = H * 0.42 / 2.4
        for p in props:
            prop(c, p["kind"], W * p["x"], ground, unit, red=red == f"prop:{p['kind']}")
        if spec.get("light") == "lamplight" and not any(p["kind"] in ("lamp", "candle", "lantern", "streetlight", "screen")
                                                      for p in props):
            c.light(W * 0.5, H * 0.15, H * 0.5, 45)
        for fig in spec.get("figures") or []:
            name = str(fig.get("name", "Figure"))
            look = looks.get(name) or looks.get(name.lower()) or {}
            silhouette = name.lower() in ("shadow", "the shadow", "it", "thing", "something") or (
                look.get("ghost") and "shadow" in name.lower())
            red_part = ""
            if red == f"figure:{name}":
                red_part = "eyes" if silhouette else ("hat" if look.get("headwear") not in (None, "none")
                                                     else "outfit" if look.get("outfit") not in (None, "none") else "body")
            figure(c, W * float(fig.get("x", 0.5)), ground, H * 0.42, str(fig.get("pose", "stand")),
                   str(fig.get("facing", "right")), look, red_part=red_part,
                   ghost=bool(fig.get("ghost") or look.get("ghost")), silhouette=silhouette)
        darkness = 0.75 if spec.get("light") == "dark" else 0.55
        frames.append(c.render(paper, grain, darkness))
    return frames


def font(size: int, bold: bool = True) -> ImageFont.ImageFont:
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def thumbnail(picture: Image.Image, text: str, size: tuple[int, int] = (1280, 720)) -> Image.Image:
    """The key moment's picture with up to three big words, the last one in red."""
    image = picture.resize(size, Image.LANCZOS)
    words = text.upper().split()[:3]
    if not words:
        return image
    draw = ImageDraw.Draw(image)
    f = font(int(size[1] * 0.16))
    x, y = int(size[0] * 0.05), int(size[1] * 0.07)
    for i, word in enumerate(words):
        color = RED if i == len(words) - 1 and len(words) > 1 else CHALK
        draw.text((x, y), word, font=f, fill=color, stroke_width=int(size[1] * 0.012), stroke_fill=(0, 0, 0))
        y += int(size[1] * 0.18)
    return image

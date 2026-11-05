"""The look: one chalk style for the whole channel, fixed characters, and a picture per shot.

Once per story an image model draws a style frame (the story's main place in the channel's
chalk style) and a character sheet for each cast member (front, side, walking). Every shot's
picture is then drawn from those references, so the style and the characters stay the same
from the first shot to the last. Each picture is checked twice: in code (mostly black, white
chalk, no colour except the red accent) and by a vision model (the right style and the right
moment). A failed picture is redrawn by a second model; if that fails too, the code renderer
draws the scene so a video is always made.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from drama_production.media import MediaError, data_uri

from . import director, draw
from .paid import OverBudget, Studio, vision_check
from .shots import FRAMING_TEXT, Shot

log = logging.getLogger(__name__)

STYLE = ("A chalk drawing on a black chalkboard: rough, hand-drawn white chalk lines with dusty chalk texture "
         "and faint smudges, deep black negative space, high contrast. Characters are simple stick figures: a "
         "round head, single-line body and limbs, no facial features except two small dot eyes when needed, "
         "each recognisable by one or two simple drawn details. Monochrome white chalk only; the single "
         "exception is blood-red chalk for {red}. No other colour anywhere, no grey gradients, no realistic "
         "shading, no 3D, no photo, no text, letters, numbers or signatures, no border, no panels, one single "
         "16:9 frame.")
W, H = 1920, 1080


def style(red: str) -> str:
    return STYLE.format(red=red or "one small detail")


def chalk_check(path: Path) -> tuple[bool, str]:
    """Code check of the channel look: mostly black, some white chalk, no colour except red."""
    with Image.open(path) as im:
        hsv = np.asarray(im.convert("RGB").resize((320, 180)).convert("HSV"), dtype=np.float32) / 255.0
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    dark = float(np.mean(v < 0.3))
    chalk = float(np.mean((v > 0.6) & (s < 0.25)))
    coloured = (s > 0.35) & (v > 0.3)
    red = (h < 0.06) | (h > 0.93)
    other = float(np.mean(coloured & ~red))
    if dark < 0.4:
        return False, f"too bright ({dark:.0%} dark)"
    if chalk < 0.003:
        return False, "nothing drawn"
    if other > 0.02:
        return False, f"colour other than red ({other:.1%})"
    return True, "ok"


def fit(src: Path, dst: Path, size: tuple[int, int] = (W, H)) -> Path:
    """Cover-crop any picture to the video frame."""
    with Image.open(src) as im:
        im = im.convert("RGB")
        scale = max(size[0] / im.width, size[1] / im.height)
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        x, y = (im.width - size[0]) // 2, (im.height - size[1]) // 2
        im.crop((x, y, x + size[0], y + size[1])).save(dst)
    return dst


def boil(picture: Path, prefix: Path, variants: int = 3, seed: int = 0) -> list[Path]:
    """The hand-drawn boil for a still: a few versions shifted by a pixel or two."""
    rng = np.random.default_rng(seed)
    out = []
    with Image.open(picture) as im:
        im = im.convert("RGB")
        for v in range(variants):
            dx, dy = rng.uniform(-1.6, 1.6, 2)
            angle = rng.uniform(-0.12, 0.12)
            frame = im.rotate(angle, resample=Image.BICUBIC, translate=(dx, dy), fillcolor=(0, 0, 0))
            path = prefix.with_name(f"{prefix.name}_{v}.png")
            frame.save(path)
            out.append(path)
    return out


class Look:
    def __init__(self, studio: Studio, work: Path, cast: list[dict[str, Any]], red: str, parallel: int = 4):
        self.studio, self.work, self.cast, self.red, self.parallel = studio, work, cast, red, parallel
        self.style_frame: Path | None = None
        self.sheets: dict[str, Path] = {}
        self.report: dict[str, Any] = {"redrawn": 0, "drawn_in_code": 0, "vision_failures": []}
        work.mkdir(parents=True, exist_ok=True)

    # --- once per story -----------------------------------------------------------
    def references(self, scenes: list[dict[str, Any]]) -> None:
        place = next((str(s.get("location")) for s in scenes if s.get("location")), "a lonely place at night")
        mood = next((str(s.get("visual")) for s in scenes if s.get("visual")), "")
        try:
            self.style_frame = self.studio.call(
                "sheet", {"prompt": f"{style(self.red)} A wide establishing view of {place}. {mood[:300]} "
                          "No figures in this picture."}, "image", self.work / "style_frame.png", "style_frame")
        except MediaError as exc:
            log.warning("No style frame (pictures will be drawn from the text alone): %s", exc)
        for c in self.cast[:5]:
            name, looks = str(c.get("name", "")), str(c.get("look", ""))
            if not name:
                continue
            refs = [data_uri(self.style_frame)] if self.style_frame else []
            prompt = (f"{style(self.red)} Using exactly the drawing style of the reference image (not its content), "
                      f"draw a character sheet of one stick figure, {name}: {looks}. The same figure three times "
                      "side by side on the black chalkboard: facing front, in profile, and walking. Make the "
                      "distinguishing details (hair, hat, coat, bag, glasses) bold and simple so they can be "
                      "drawn the same way in every scene.")
            role = "shot" if refs else "sheet"
            args = {"prompt": prompt, **({"image_urls": refs} if refs else {})}
            try:
                self.sheets[name] = self.studio.call(role, args, "image", self.work / f"sheet_{len(self.sheets)}.png",
                                                     "character_sheet", key=[prompt, bool(refs)])
            except MediaError as exc:
                log.warning("No character sheet for %s: %s", name, exc)

    # --- every shot ---------------------------------------------------------------
    def prompt(self, shot: Shot, refs: list[str]) -> str:
        notes = []
        if self.style_frame:
            notes.append("Reference image 1 shows the drawing style only: copy its chalk look, not its content.")
        for k, name in enumerate(refs, 2 if self.style_frame else 1):
            notes.append(f"Reference image {k} is the character sheet of {name}: draw {name} exactly like it.")
        return (f"{style(self.red)} {' '.join(notes)} Shot: {FRAMING_TEXT.get(shot.framing, shot.framing)}. "
                f"{shot.picture}")

    def picture(self, shot: Shot, scene: dict[str, Any], cast_names: list[str],
                looks: dict[str, Any]) -> Path:
        out = self.work / f"shot_{shot.index:03d}.png"
        names = [n for n in shot.characters if n in self.sheets][:3]
        images = ([data_uri(self.style_frame)] if self.style_frame else []) + [data_uri(self.sheets[n]) for n in names]
        prompt = self.prompt(shot, names)
        for attempt, role in enumerate(("shot", "shot_cheap")):
            if role not in self.studio.models:
                continue
            raw = self.work / f"raw_{shot.index:03d}_{attempt}.png"
            args: dict[str, Any] = {"prompt": prompt if attempt == 0 else prompt + " Strictly black and white chalk."}
            if images:
                args["image_urls"] = images
            elif role == "shot_cheap":
                continue  # the cheap model needs a reference picture
            try:
                self.studio.call(role if images else "sheet", args, "image", raw, f"shot_{role}",
                                 key=[args["prompt"], len(images), shot.index])
            except OverBudget:
                break
            except MediaError as exc:
                log.warning("Shot %d picture (%s) failed: %s", shot.index, role, str(exc)[:200])
                continue
            fit(raw, out)
            ok, why = chalk_check(out)
            if ok:
                ok, why = vision_check(self.studio, out, shot.picture, task="vision_picture")
            if ok:
                return out
            self.report["vision_failures"].append(f"shot {shot.index}: {why}")
            self.report["redrawn"] += 1
            log.info("Shot %d picture rejected (%s); redrawing", shot.index, why)
        # Last resort: the code renderer, from the scene's keywords.
        self.report["drawn_in_code"] += 1
        spec = director.guess(scene, cast_names)
        paper, grain = draw.make_paper(W, H)
        draw.scene_frames(spec, looks, (W, H), paper, grain, seed=shot.index, variants=1)[0].save(out)
        return out

    def pictures(self, shots: list[Shot], scenes: list[dict[str, Any]], cast_names: list[str],
                 looks: dict[str, Any]) -> dict[int, Path]:
        def one(shot: Shot) -> tuple[int, Path]:
            scene = scenes[shot.scene] if shot.scene < len(scenes) else {}
            return shot.index, self.picture(shot, scene, cast_names, looks)

        with ThreadPoolExecutor(max_workers=self.parallel) as pool:
            return dict(pool.map(one, shots))

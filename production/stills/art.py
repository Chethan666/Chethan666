"""The art director and the illustrator.

The art director keeps one house look for the channel (`stills.look`: cinematic film stills
by default, or painted illustrations), turns each cast member's look into a design that can
be repeated exactly in every picture, and paints the references:
a style frame (the story's main place, no people) and a character sheet per character, The
Storyteller included. The illustrator then paints every beat from those references, most
important beats first, and the critic passes or rejects each picture.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter

from story_factory.llm import LLMError

from ..look import W, H
from ..shots import Shot
from .board import Board
from .images import Critic, Painter, finish, house_look
from .storyboard import HOST

log = logging.getLogger(__name__)

HOST_DESIGN = ("a tall, gaunt man of indeterminate age in a long, worn charcoal coat and a wide-brimmed hat, his "
               "face mostly in shadow except for pale, kind, tired eyes; he always carries an old iron lantern "
               "whose flame burns blood red")
DESIGN_SCHEMA = {"type": "object", "required": ["characters"], "properties": {"characters": {"type": "array", "items": {
    "type": "object", "required": ["name", "design"],
    "properties": {"name": {"type": "string"}, "design": {"type": "string"}}}}}}
FRAMING = {
    "wide": "Wide shot: the whole place, the people small in it, lots of dark negative space",
    "medium": "Medium shot: a person from head to knees, body language readable",
    "close": "Close-up: a face and shoulders fill the frame, the expression clear",
    "extreme_close": "Extreme close-up on one detail (a hand, an eye, a keyhole, a screen)",
    "insert": "Insert shot of the important object, no people",
    "pov": "Point of view: what the character sees, framed by a doorway, window or flashlight beam",
    "over_shoulder": "Over the shoulder of one person, looking at what they face",
    "low_angle": "Low angle looking up, the subject looming",
    "high_angle": "High angle looking down, the person small and exposed",
}
STICK = re.compile(r"\b(stick ?figures?|stickm[ae]n|chalk|single[- ]line)\b", re.I)


def designs(router: Any, cast: list[dict[str, Any]], style: str, people: str = "") -> dict[str, str]:
    """A repeatable design per character (older stories describe stick figures).

    Every feature is fixed for the whole story: a pair of glasses that is "sometimes" there
    comes and goes between pictures, so the design says always or never for each accessory.
    """
    plain = {str(c.get("name")): STICK.sub("", str(c.get("look", ""))).strip(" ,.") for c in cast if c.get("name")}
    if router is None or not plain:
        return plain
    listing = "\n".join(f"- {n}: {look}" for n, look in plain.items())
    try:
        result = router.one("planner").generate_json(
            "character_design", "You are the character designer and casting director of a horror channel. "
            "JSON only.",
            f"House style: {style}\n\nCharacters and their notes (older notes may describe stick figures: "
            f"ignore that and design real people):\n{listing}\n\nFor each character write one design that "
            f"can be repeated exactly in every picture, in 40-70 words, for {people or 'real people'}: age, "
            "ethnicity or skin tone, build, face shape and one facial feature, hair (colour, length, how it is "
            "worn), one outfit with colours that never changes, and one distinctive detail. Accessories are "
            "fixed: write 'always wears ...' or 'never wears glasses'; nothing 'sometimes' or 'often'. Keep "
            "what the notes say; do not add a feature the notes contradict. Keep each name exactly as written.",
            DESIGN_SCHEMA)
    except (LLMError, ValueError, TypeError) as exc:
        log.warning("Character designs from the cast notes: %s", exc)
        return plain
    for item in result.get("characters", []) if isinstance(result, dict) else []:
        if isinstance(item, dict) and item.get("name") in plain and str(item.get("design", "")).strip():
            plain[item["name"]] = str(item["design"]).strip()[:500]
    return plain


class ArtDirectorAgent:
    name = "art"
    critical = False  # without references, every picture is painted from words alone

    def run(self, board: Board) -> dict[str, Any]:
        cfg, d = board.cfg, board.data
        look = house_look(cfg)
        style = look["style"]
        red = str((d.get("bible") or {}).get("red_accent", "")).strip()
        house = style + (f" The only vivid colour is the blood red of {red}." if red else "")
        cast = [dict(c) for c in d["cast"] if c.get("name")]
        looks = designs(board.router, cast, style, look["people"])
        for c in cast:
            c["design"] = looks.get(str(c["name"]), c.get("look", ""))
        if any(b.characters == [HOST] for b in d["beats"]) or any(
                HOST in b.characters for b in d["beats"]):
            cast.append({"name": HOST, "design": HOST_DESIGN})
        d.update(cast=cast, house=house, look=look)
        critic = Critic(board.studio, house, look["wrong"])
        painter = Painter(board.studio, cfg, board.work / "art", critic)
        d.update(critic=critic, painter=painter)
        if not cfg.get("references", True):
            return {"style_frame": False, "sheets": [], "designs": looks}
        scenes = d["scenes"]
        place = next((str(s.get("location")) for s in scenes if s.get("location")), "a lonely place at night")
        mood = next((str(s.get("visual")) for s in scenes if s.get("visual")), "")
        frame = board.work / "art" / "style_frame.png"
        got, source = painter.paint(f"{house} A wide establishing view of {place}. {mood[:300]} No people. "
                                    f"{look['avoid']}".strip(), [],
                                    frame, "style_frame", cfg["chains"]["reference"])
        d["style_frame"] = got
        sheets: dict[str, Path] = {}
        people = [c for c in cast if c["name"] in {n for b in d["beats"] for n in b.characters}][:8]

        def sheet(c: dict[str, Any]) -> tuple[str, Path | None]:
            out = board.work / "art" / f"sheet_{re.sub(r'[^a-z0-9]+', '_', c['name'].lower())}.png"
            refs = [got] if got else []
            match = ("Match the lighting, colour grade and texture of the reference image (not its content). "
                     if refs else "")
            text = (f"{house} {match}A character reference sheet of one person, {c['name']}: {c['design']}. "
                    "The same person three times side by side on a plain dark background, lit by one soft "
                    "side light: full body from the front, full body in three-quarter view, and a "
                    "head-and-shoulders close-up with a neutral, tired expression. Identical face, hair, "
                    f"clothes and accessories in all three. No text, no labels. {look['avoid']}").strip()
            path, _ = painter.paint(text, refs, out, ["sheet", c["name"], c["design"]], cfg["chains"]["sheet"])
            return c["name"], path

        with ThreadPoolExecutor(max_workers=int(cfg.get("parallel", 4))) as pool:
            for name, path in pool.map(sheet, people):
                if path is not None:
                    sheets[name] = path
        d["sheets"] = sheets
        return {"style_frame": bool(got), "style_frame_by": source, "sheets": sorted(sheets),
                "designs": {c["name"]: c["design"] for c in cast}}


def placeholder(out: Path, seed: int) -> Path:
    """The last resort when no painter answers: a dark, grained, vignetted frame."""
    import numpy as np

    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:H // 4, 0:W // 4]
    r = np.hypot((x - W / 8) / (W / 8), (y - H / 8) / (H / 8))
    v = np.clip(0.16 - 0.12 * r + rng.normal(0, 0.012, r.shape), 0, 1)
    rgb = np.stack([v * 0.8, v * 0.85, v], axis=-1)
    im = Image.fromarray((rgb * 255).astype("uint8")).resize((W, H), Image.BICUBIC).filter(ImageFilter.GaussianBlur(2))
    ImageDraw.Draw(im)
    im.save(out)
    return out


class IllustratorAgent:
    name = "illustrator"
    critical = True

    def prompt(self, beat: Shot, house: str, cast: dict[str, str], refs: list[str], style_frame: bool,
               look: dict[str, Any] | None = None, camera: str = "") -> str:
        look = look or {}
        notes = []
        if style_frame:
            notes.append("Reference image 1 shows the look only (lighting, colour grade, texture): copy its "
                         "look, not its content.")
        for k, name in enumerate(refs, 2 if style_frame else 1):
            notes.append(f"Reference image {k} is the character sheet of {name}: {name} must be the same person, "
                         "with an identical face, age, hair, clothes and accessories (glasses only if the sheet "
                         "has them).")
        people = "; ".join(f"{n}: {cast[n]}" for n in beat.characters if n in cast)
        empty = "" if beat.characters else " No people in this picture."
        picture = STICK.sub(lambda m: '' if m.group(0).lower() in ('chalk', 'single-line', 'single line')
                            else 'person', beat.picture)
        lens = f" {camera}." if camera else ""
        avoid = f" {look['avoid']}" if look.get("avoid") else ""
        return (f"{house} {' '.join(notes)} One single 16:9 {look.get('noun', 'frame')}.{lens} "
                f"{FRAMING.get(beat.framing, beat.framing)}. {picture}"
                f"{(' Characters: ' + people + '.') if people else ''}{empty} No text, letters, "
                f"captions, timestamps, borders or panels.{avoid}")

    @staticmethod
    def check_text(beat: Shot, cast: dict[str, str]) -> str:
        """What the critic compares the picture with: the moment, and each named person's design."""
        people = "; ".join(f"{n}: {cast[n]}" for n in beat.characters if cast.get(n))
        return beat.picture + (f" People in it: {people}" if people else "")

    def run(self, board: Board) -> dict[str, Any]:
        cfg, d = board.cfg, board.data
        beats: list[Shot] = d["beats"]
        painter: Painter = d["painter"]
        sheets: dict[str, Path] = d.get("sheets", {})
        style_frame: Path | None = d.get("style_frame")
        cast = {c["name"]: c.get("design", "") for c in d["cast"]}
        look = d.get("look") or house_look(cfg)
        cameras = cfg.get("cameras") or {}
        use_cameras = look.get("name") == "cinematic"
        folder = board.work / "pictures"
        folder.mkdir(parents=True, exist_ok=True)
        sources: dict[int, str] = {}

        def one(beat: Shot, fix: str = "") -> tuple[int, Path | None]:
            names = [n for n in beat.characters if n in sheets][:3]
            refs = ([style_frame] if style_frame else []) + [sheets[n] for n in names]
            camera = str(cameras.get(beat.view, "")) if use_cameras else ""
            text = self.prompt(beat, d["house"], cast, names, bool(style_frame), look, camera)
            if fix:  # the art review's note on the first version
                text += f" Correct this from the previous attempt: {fix[:300]}"
            tag = f"_fix{abs(hash(fix)) % 10_000}" if fix else ""
            path, source = painter.paint(text, refs, folder / f"beat_{beat.index:03d}{tag}.png",
                                         ["beat", beat.index, text], cfg["chains"]["beat"],
                                         check=self.check_text(beat, cast))
            sources[beat.index] = source
            if path is not None and (look.get("grain") or look.get("vignette")):
                path = finish(path, folder / f"beat_{beat.index:03d}{tag}_film.png", float(look.get("grain", 0)),
                              float(look.get("vignette", 0)), beat.index)
            return beat.index, path

        d["repaint"] = lambda beat, fix: one(beat, fix)[1]  # for the art review

        order = sorted(beats, key=lambda b: (-(b.importance + (2 if b.scare else 0)), b.index))
        with ThreadPoolExecutor(max_workers=int(cfg.get("parallel", 4))) as pool:
            made = dict(pool.map(one, order))
        pictures: dict[int, Path] = {}
        borrowed = placeholders = 0
        for b in beats:  # a beat no painter could paint borrows its nearest neighbour in the same scene
            if made.get(b.index):
                pictures[b.index] = made[b.index]
                continue
            near = sorted((x for x in beats if made.get(x.index)), key=lambda x: (x.scene != b.scene,
                                                                                  abs(x.index - b.index)))
            if near:
                pictures[b.index] = made[near[0].index]
                borrowed += 1
            else:
                pictures[b.index] = placeholder(folder / f"beat_{b.index:03d}.png", b.index)
                placeholders += 1
        d["pictures"] = pictures
        return {"look": look.get("name"), "cameras": {b.index: b.view for b in beats if b.view != "film"},
                "pictures": len(beats) - borrowed - placeholders, "borrowed_neighbour": borrowed,
                "placeholders": placeholders, "by_painter": dict(painter.counts),
                "rejected": painter.rejections[:30]}

"""The art director and the illustrator.

The art director keeps one house look for the channel (`stills.look`: cinematic film stills
by default, or painted illustrations) and designs everything that has to stay the same from
picture to picture:
- the people: each cast member's look becomes a design that can be repeated exactly, and a
  character sheet (The Storyteller included);
- the places: each place the story returns to (a bus, a house, a street) gets a design that
  fixes what the script leaves open (what it is, its colours, its fixtures, its lights) and
  a location sheet. Without it every picture invents the place again (run 94: a city bus in
  one shot, a yellow school bus in the next);
- the look: a style frame (the story's main place, no people).

The illustrator then paints every beat from those references, most important beats first,
and the critic passes or rejects each picture.
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
from .checks import sheet_ok
from . import lessons
from .images import Critic, Painter, house_look
from .storyboard import HOST

log = logging.getLogger(__name__)

HOST_DESIGN = ("a tall, gaunt man of indeterminate age in a long, worn charcoal greatcoat and a shapeless old black "
               "felt hat with a wide, drooping brim (never a cowboy hat), his face always mostly in shadow under "
               "the brim except for pale, kind, tired eyes; he always carries a plain old iron railway lantern with "
               "glass panes (no faces, masks or carvings on it) whose flame burns blood red")
# How the host is shown in his sheet and portrait: a face seen in full there is copied into every picture
# (run 103: his portrait showed his whole face, and a cowboy).
HOST_FACE = ("His face stays mostly in shadow under the drooping brim: only his pale, tired eyes and the line of "
             "his jaw catch the lantern's red light. Never a cowboy hat, never his whole face lit.")
NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
           "ten": 10, "eleven": 11, "twelve": 12}
AGE = re.compile(r"\b(\d{1,2}|" + "|".join(NUMBERS) + r")[- ](?:year|yr)s?[- ]old\b|\baged? (\d{1,2})\b", re.I)


def child_age(text: str) -> int | None:
    """A child's age from a design ("a four-year-old girl", "aged 7"), or None for an adult."""
    m = AGE.search(text or "")
    if not m:
        return None
    word = (m.group(1) or m.group(2) or "").lower()
    age = NUMBERS.get(word) or (int(word) if word.isdigit() else None)
    return age if age is not None and age <= 12 else None


def age_lock(age: int) -> str:
    """Words that hold a child at their age: image models paint a four-year-old as about eight
    (run 103: Lily, four, looked eight or nine in two painters out of three)."""
    if age <= 6:
        return (f"Exactly {age} years old, a small child, never older: toddler-like proportions with a large head "
                "for the body, round full cheeks, a small button nose, short arms and legs, small chubby hands, "
                "as tall as a kitchen table")
    return f"Exactly {age} years old, a child, never a teenager: a child's face and proportions"


LIGHT = r"(night[- ]?lights?|lamps?|lights?|bulbs?|torch(?:es)?|candles?|lanterns?|screens?)"
OFF = re.compile(LIGHT + r"\b[^.!?\"]{0,25}?\b(?:was|were|is|are|went|had gone|had been|stayed)\s+(?:off|out|dark)\b"
                 r"|\b(?:turned|switched|clicked|blew)\s+(?:off|out)\s+(?:the\s+|her\s+|his\s+|my\s+)?" + LIGHT, re.I)
ON = re.compile(r"\b(?:turned|switched|clicked|flicked)\s+(?:it\s+|them\s+|the\s+\w+\s+)?on\b"
                r"|\b(?:came|flickered|blinked)\s+(?:back\s+)?on\b", re.I)


ROOT = Path(__file__).resolve().parents[2]


def host_references(cfg: dict[str, Any]) -> tuple[Path, Path] | None:
    """The Storyteller's fixed sheet and portrait (`host_references`), made once and kept in the
    repository: the host is the same man in every episode, and no painter kept his face in shadow
    when asked again each time (run 104: six refused sheets). None until both files exist."""
    refs = cfg.get("host_references") or {}
    sheet, face = (ROOT / str(refs.get(k, "")) if refs.get(k) else None for k in ("sheet", "portrait"))
    return (sheet, face) if sheet and face and sheet.is_file() and face.is_file() else None


def lights_off(beats: list[Shot], host_beats: set[int] | frozenset[int] = frozenset()) -> dict[int, list[str]]:
    """Lights the story has said are off, per story beat from the one that says so until one is
    switched on again. A location sheet shows its lamps lit, and every picture copies it: run 104's
    night-light glowed in the very pictures whose narration says it was off."""
    off: list[str] = []
    out: dict[int, list[str]] = {}
    for b in sorted(beats, key=lambda b: b.index):
        if b.index in host_beats:
            continue
        text = str(b.text or "")
        if ON.search(text):
            off = []
        for m in OFF.finditer(text):
            name = (m.group(1) or m.group(2) or "light").lower().replace("nightlight", "night-light")
            name = re.sub(r"night[- ]light", "night-light", name)
            if name not in off:
                off.append(name)
        if off:
            out[b.index] = list(off)
    return out

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
            f"ignore that and design real people):\n{listing}\n\nThe notes are fixed by the script and are "
            "kept word for word. For each character write ONLY what the notes leave out, so the person can be "
            f"repeated exactly in every picture, in 25-50 words, for {people or 'real people'}: age if not "
            "given, ethnicity or skin tone, build, face shape and one facial feature, hair (colour, length, how "
            "it is worn), and the colours of any clothing the notes do not describe. Never change, replace or "
            "contradict anything in the notes (their clothes, colours, age, a hidden face). Accessories are "
            "fixed: write 'always wears ...' or 'never wears glasses'; nothing 'sometimes' or 'often'. Keep each "
            "name exactly as written.",
            DESIGN_SCHEMA)
    except (LLMError, ValueError, TypeError) as exc:
        log.warning("Character designs from the cast notes: %s", exc)
        return plain
    for item in result.get("characters", []) if isinstance(result, dict) else []:
        if isinstance(item, dict) and item.get("name") in plain and str(item.get("design", "")).strip():
            notes = plain[item["name"]]
            added = str(item["design"]).strip()[:400]
            # The script's own notes come first and always stay: the designer only adds to them.
            plain[item["name"]] = f"{notes}; {added}" if notes and notes.lower() not in added.lower() else added
    return plain


PLACE_SYSTEM = ("You are the production designer of Hollow Lines, a horror storytelling channel told in film "
                "stills. You design each place so it can be rebuilt identically in every picture. JSON only.")
PLACE_SCHEMA = {"type": "object", "required": ["places"], "properties": {"places": {"type": "array", "items": {
    "type": "object", "required": ["name", "scenes", "design", "views"],
    "properties": {"name": {"type": "string"}, "scenes": {"type": "array", "items": {"type": "integer"}},
                   "design": {"type": "string"}, "views": {"type": "array", "items": {"type": "string"}}}}}}}


def places(router: Any, scenes: list[dict[str, Any]], beats: list[Shot], host_beats: set[int],
           style: str) -> list[dict[str, Any]]:
    """The places the story returns to, each with a design that can be repeated in every picture:
    [{"name", "scenes": [scene numbers], "design", "views": [one or two views for its sheet]}],
    the place with the most pictures first. Empty without a model (pictures then follow their
    own descriptions, as before)."""
    story = [b for b in beats if b.index not in host_beats]
    if router is None or not story:
        return []
    by_scene: dict[int, list[Shot]] = {}
    for b in story:
        by_scene.setdefault(b.scene, []).append(b)
    blocks = []
    for k, group in sorted(by_scene.items()):
        sc = scenes[k] if k < len(scenes) else {}
        shown = " | ".join(str(b.picture)[:180] for b in group[:4])
        blocks.append(f"SCENE {k} ({len(group)} pictures) - location: {sc.get('location', '')}; plan: "
                      f"{str(sc.get('visual', ''))[:200]}\n  pictures: {shown}")
    try:
        result = router.one("planner").generate_json(
            "set_design", PLACE_SYSTEM,
            f"House style: {style}\n\n" + "\n".join(blocks) + "\n\n"
            "Group the scenes into the story's PLACES. Scenes in the same place share one entry even when the "
            "script words the location differently; the inside and the outside of one building or vehicle are "
            "ONE place. For each place with two or more pictures give:\n"
            "- name: two to four plain words (\"the night bus\")\n"
            "- scenes: the scene numbers set there\n"
            "- design: 45-80 words that fix what the script and the pictures leave open, so the place is "
            "identical in every picture: exactly what it is (type, age, condition), its main colours and "
            "materials, three telling objects or fixtures and where they are, its small light sources at night "
            "and their colour (a horror story's places are dark: name what little light there is, never "
            "daylight or an evenly lit room), the weather. If pictures are both inside and outside it, describe both "
            "(\"Outside: ... Inside: ...\"). Never contradict the script or the pictures. No people. Nothing "
            "'sometimes'.\n"
            "- views: the views for its reference sheet, one sentence each: one view, or two when pictures are "
            "both inside and outside it (\"outside: the whole bus from the front left, on the wet road\", "
            "\"inside: from the driver's cab down the aisle to the back seat\").\n"
            "Leave out places with a single picture.",
            PLACE_SCHEMA)
    except (LLMError, ValueError, TypeError) as exc:
        log.warning("No set designs (pictures follow their own descriptions): %s", exc)
        return []
    out: list[dict[str, Any]] = []
    taken: set[int] = set()
    for item in result.get("places", []) if isinstance(result, dict) else []:
        if not isinstance(item, dict):
            continue
        name, design = str(item.get("name", "")).strip()[:60], str(item.get("design", "")).strip()[:700]
        numbers = []
        for n in item.get("scenes") or []:
            try:
                n = int(n)
            except (TypeError, ValueError):
                continue
            if n in by_scene and n not in taken and n not in numbers:
                numbers.append(n)
        if not name or len(design) < 40 or not numbers:
            continue
        taken.update(numbers)
        views = [str(v).strip()[:200] for v in (item.get("views") or []) if str(v).strip()][:2]
        out.append({"name": name, "scenes": numbers, "design": design,
                    "views": views or ["one wide view of the whole place"],
                    "pictures": sum(len(by_scene[n]) for n in numbers)})
    return sorted(out, key=lambda x: -x["pictures"])


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
            age = child_age(c["design"])
            if age is not None:  # the age leads, so it is read first and kept in every picture
                c["design"] = f"{age_lock(age)}. {c['design']}"
        if any(b.characters == [HOST] for b in d["beats"]) or any(
                HOST in b.characters for b in d["beats"]):
            cast.append({"name": HOST, "design": HOST_DESIGN})
        d.update(cast=cast, house=house, look=look)
        critic = Critic(board.studio, house, look["wrong"])
        painter = Painter(board.studio, cfg, board.work / "art", critic)
        d.update(critic=critic, painter=painter)
        scenes = d["scenes"]
        sets = places(board.router, scenes, d["beats"], set(d.get("host_beats") or ()), style) \
            if cfg.get("set_design", True) else []
        d["sets"] = sets
        if not cfg.get("references", True):
            return {"style_frame": False, "sheets": [], "designs": looks,
                    "places": [{"name": x["name"], "scenes": x["scenes"], "design": x["design"], "sheet": False}
                               for x in sets]}
        # The style frame sets the look of every picture and is the title card's background: the
        # story's main place (the one with the most pictures), not the host's.
        main = sets[0] if sets else None
        if main:
            subject = f"{main['name']}: {main['design']}"
            shows = f"{main['name']}, with no people. Any one view of the place is right: {main['design'][:400]}"
        else:
            place = next((str(s.get("location")) for s in scenes if s.get("location")), "a lonely place at night")
            mood = next((str(s.get("visual")) for s in scenes if s.get("visual")), "")
            subject = f"{place}. {mood[:300]}"
            shows = f"{place}, with no people."
        frame = board.work / "art" / "style_frame.png"
        got, source = painter.paint(f"{house} A wide establishing view of {subject} No people. "
                                    f"{look['avoid']}".strip(), [], frame, "style_frame", cfg["chains"]["reference"],
                                    check=f"A wide establishing view of {shows}")
        d["style_frame"] = got
        sheets: dict[str, Path] = {}
        people = [c for c in cast if c["name"] in {n for b in d["beats"] for n in b.characters}][:8]
        fixed = host_references(cfg)
        if fixed:
            people = [c for c in people if c["name"] != HOST]  # painted once, kept in the repository
        checks: list[dict[str, Any]] = []

        def checked(kind: str, name: str, design: str):
            """The check a sheet must pass before pictures are painted from it (and its record)."""
            def check(image: Path) -> tuple[bool, str]:
                ok, why = sheet_ok(board.studio, image, kind, name, design)
                checks.append({"sheet": name, "kind": kind, "ok": ok, "why": why})
                return ok, why
            return check

        def location_sheet(place: dict[str, Any]) -> tuple[str, Path | None]:
            out = board.work / "art" / f"place_{re.sub(r'[^a-z0-9]+', '_', place['name'].lower())}.png"
            refs = [got] if got else []
            match = ""
            if refs and place is main:  # the style frame was painted of this very place
                match = "The reference image shows this same place: keep it identical, and keep its look. "
            elif refs:
                match = "Match the lighting, colour grade and texture of the reference image (not its content). "
            views = place["views"]
            if len(views) > 1:
                layout = (f"Two views of the same place side by side in one image, split down the middle: on "
                          f"the left, {views[0]}; on the right, {views[1]}. Identical colours, materials, "
                          "fixtures and lighting in both.")
            else:
                layout = f"One wide view: {views[0]}."
            text = (f"{house} {match}A location reference sheet of one place, {place['name']}: {place['design']} "
                    f"{layout} At night, in its darkest light: lit only by its own small light sources, most of "
                    f"it in deep shadow. Empty: no people. No text, no labels. {look['avoid']}").strip()
            path, step = painter.paint(text, refs, out, ["place", place["name"], place["design"], views],
                                       cfg["chains"]["sheet"], check=checked("location", place["name"], place["design"]))
            if path is not None and out.stem in painter.doubted and not step.startswith("fal:"):
                return place["name"], None
            return place["name"], path

        def sheet(c: dict[str, Any]) -> tuple[str, Path | None]:
            out = board.work / "art" / f"sheet_{re.sub(r'[^a-z0-9]+', '_', c['name'].lower())}.png"
            refs = [got] if got else []
            match = ("Match the lighting, colour grade and texture of the reference image (not its content). "
                     if refs else "")
            # A photoreal face can drift toward a famous one; the channel's people are nobody's likeness.
            unknown = ("An unknown, ordinary person: the face resembles no actor, celebrity or public figure. "
                       if look.get("synthetic_media") else "")
            host = f" {HOST_FACE}" if c["name"] == HOST else ""
            text = (f"{house} {match}A character reference sheet of one person, {c['name']}: {c['design']}.{host} {unknown}"
                    "The same person three times side by side on a plain dark background, lit by one soft "
                    "side light: full body from the front, full body in three-quarter view, and a "
                    "head-and-shoulders close-up with a neutral, tired expression. Identical face, hair, "
                    f"clothes and accessories in all three. No text, no labels. {look['avoid']}").strip()
            path, step = painter.paint(text, refs, out, ["sheet", c["name"], c["design"]], cfg["chains"]["sheet"],
                                       check=checked("character", c["name"], c["design"]))
            if path is not None and out.stem in painter.doubted and not step.startswith("fal:"):
                return c["name"], None  # a free painter's refused sheet (run 96: an empty room for Lily) misleads
            return c["name"], path

        with ThreadPoolExecutor(max_workers=int(cfg.get("parallel", 4))) as pool:
            place_jobs = pool.map(location_sheet, sets[: int(cfg.get("location_sheets", 3))])
            sheet_jobs = pool.map(sheet, people)  # both kinds are painted side by side
            painted = list(place_jobs)
            for name, path in sheet_jobs:
                if path is not None:
                    sheets[name] = path
        for place in sets:
            place["sheet"] = dict(painted).get(place["name"])
        d["sheets"] = sheets

        def portrait(c: dict[str, Any]) -> tuple[str, Path | None]:
            """A large, sharp face painted from the person's own sheet: on the sheet the face is a
            small part of one view, and a painter given only that drifts most in the face."""
            out = board.work / "art" / f"face_{re.sub(r'[^a-z0-9]+', '_', c['name'].lower())}.png"
            text = (f"{house} A close head-and-shoulders portrait of the person in the reference image, "
                    f"{c['name']}: {c['design']}. Exactly the same person: identical face, age, skin, hair and "
                    "clothes. The face fills the middle of the frame, sharp focus on the eyes, a neutral, tired "
                    "expression, lit by one soft side light against a plain dark background. One single frame. "
                    f"No text, no labels. {look['avoid']}").strip()
            if c["name"] == HOST:  # the host's face is never shown in full, so neither is his portrait
                text = (f"{house} A close head-and-shoulders portrait of the man in the reference image, {c['name']}: "
                        f"{c['design']}. Exactly the same man, hat and coat. {HOST_FACE} Lit only by his lantern's "
                        f"red glow from below, against a plain dark background. One single frame. No text, no "
                        f"labels. {look['avoid']}").strip()
            path, step = painter.paint(text, [sheets[c["name"]]], out, ["portrait", c["name"], c["design"]],
                                       cfg["chains"]["sheet"], check=checked("portrait", c["name"], c["design"]))
            if path is not None and out.stem in painter.doubted and not step.startswith("fal:"):
                return c["name"], None
            return c["name"], path

        portraits: dict[str, Path] = {}
        if fixed and HOST in {n for b in d["beats"] for n in b.characters}:
            sheets[HOST], portraits[HOST] = fixed
        if cfg.get("portraits", True):
            with ThreadPoolExecutor(max_workers=int(cfg.get("parallel", 4))) as pool:
                for name, path in pool.map(portrait, [c for c in people if c["name"] in sheets
                                                      and c["name"] not in portraits]):
                    if path is not None:
                        portraits[name] = path
        d["portraits"] = portraits
        last = {(c["kind"], c["sheet"]): c for c in checks}  # a sheet's last check is the one that stands
        d["reference_issues"] = [f"{c['kind']} sheet of {c['sheet']}: {c['why']}" for c in last.values() if not c["ok"]]
        return {"style_frame": bool(got), "style_frame_by": source, "sheets": sorted(sheets),
                "portraits": sorted(portraits), "host_references": bool(fixed),
                "designs": {c["name"]: c["design"] for c in cast},
                "places": [{"name": x["name"], "scenes": x["scenes"], "design": x["design"],
                            "sheet": x.get("sheet") is not None} for x in sets],
                "reference_checks": checks}


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


def short(text: str, words: int = 14) -> str:
    """The first words of a design: enough to find the person or place in their sheet."""
    first = text.split(";")[0].split(". ")[0].strip()
    parts = first.split()
    return " ".join(parts[:words]).rstrip(",") + ("..." if len(parts) > words else "")


class IllustratorAgent:
    name = "illustrator"
    critical = True

    def prompt(self, beat: Shot, house: str, cast: dict[str, str], refs: list[str], style_frame: bool,
               look: dict[str, Any] | None = None, camera: str = "", place: dict[str, Any] | None = None,
               lessons: str = "", faces: set[str] | frozenset[str] = frozenset(), dark: list[str] | None = None) -> str:
        """`style_frame`: reference image 1 is there (the place's sheet when `place` has one, else
        the style frame). `place`: the set this beat is in, {"name", "design", "sheet"}. `faces`:
        the people in `refs` sent as two images, a close portrait and then the sheet."""
        look = look or {}
        notes = []
        through = bool(camera) and beat.view != "film"  # seen through a security camera, a phone, an old photo
        if style_frame and place and place.get("sheet"):
            its_look = "" if through else " and has its look (lighting, colour grade, texture)"
            notes.append(f"Reference image 1 is the location sheet of {place['name']}: this picture is set in "
                         "that exact place (the same build, colours, materials, fixtures and light sources)"
                         f"{its_look}. Shoot one new frame there; never copy the sheet's layout or its split.")
        elif style_frame and through:
            notes.append("Reference image 1 is a frame of the same film: this picture is seen through the camera "
                         "described above, so keep that camera's look, not the reference's colours.")
        elif style_frame:
            notes.append("Reference image 1 shows the look only (lighting, colour grade, texture): copy its "
                         "look, not its content.")
        k = 2 if style_frame else 1
        for name in refs:
            if name in faces:
                notes.append(f"Reference image {k} is a close portrait of {name}: match this face exactly (eyes, "
                             f"nose, mouth, jaw, skin, hairline). Reference image {k + 1} is the character sheet "
                             f"of {name}: the same person's build, hair and clothes.")
                k += 2
            else:
                notes.append(f"Reference image {k} is the character sheet of {name}: the same person, identical "
                             "face, age, hair and clothes.")
                k += 1
        # A person or place with a sheet attached needs only a few words to be found in it: the full
        # design is in the picture, and a shorter prompt is followed better (450 words in run 96).
        people = "; ".join(f"{n}: {short(cast[n]) if n in refs else cast[n]}" for n in beat.characters if n in cast)
        empty = "" if beat.characters else " No people in this picture."
        picture = STICK.sub(lambda m: '' if m.group(0).lower() in ('chalk', 'single-line', 'single line')
                            else 'person', beat.picture)
        lens = f" {camera}." if camera else ""
        avoid = f" {look['avoid']}" if look.get("avoid") else ""
        where = ""
        if place and place.get("design"):
            where = f" Place: {place['name']}: {short(place['design'], 24) if place.get('sheet') else place['design']}"
        # The shot comes first: what a model reads first it follows best, and run 94's refusals were
        # nearly all pictures that ignored their description (it sat behind 190 words of style).
        # Then what is fixed (place, people), what each reference is for, the look, and what to avoid.
        warned = f" Avoid what went wrong before: {lessons}." if lessons else ""
        if dark:  # the story switched these off: the sheet's lit lamps must not win
            things = " and ".join(f"the {x}" for x in dark)
            picture += (f" {things[0].upper() + things[1:]} {'is' if len(dark) == 1 else 'are'} switched OFF and dark "
                        "in this picture, even where a reference image shows it lit.")
        return (f"One single 16:9 {look.get('noun', 'frame')}, one frame, never split into panels.{lens} "
                f"{FRAMING.get(beat.framing, beat.framing)}. "
                f"{picture}{empty}{where}{(' Characters: ' + people + '.') if people else ''} "
                f"{' '.join(notes)} The look: {house} No text, letters, captions, timestamps, borders or "
                f"panels.{warned}{avoid}").replace("  ", " ")

    @staticmethod
    def check_text(beat: Shot, cast: dict[str, str]) -> str:
        """What the critic compares the picture with: the moment, and each named person's design."""
        people = "; ".join(f"{n}: {cast[n]}" for n in beat.characters if cast.get(n))
        return (beat.picture + (f" People in it: {people}" if people else "")
                + f" | The narrator says over it: \"{beat.text[:200]}\" - fail the picture if it shows a "
                "different action, pose or place than these words.")

    def run(self, board: Board) -> dict[str, Any]:
        cfg, d = board.cfg, board.data
        beats: list[Shot] = d["beats"]
        painter: Painter = d["painter"]
        sheets: dict[str, Path] = d.get("sheets", {})
        portraits: dict[str, Path] = d.get("portraits", {})
        style_frame: Path | None = d.get("style_frame")
        cast = {c["name"]: c.get("design", "") for c in d["cast"]}
        look = d.get("look") or house_look(cfg)
        cameras = cfg.get("cameras") or {}
        use_cameras = look.get("name") == "cinematic"
        folder = board.work / "pictures"
        folder.mkdir(parents=True, exist_ok=True)
        sources: dict[int, str] = {}

        host_beats = set(d.get("host_beats") or ())
        place_of = {n: place for place in d.get("sets") or [] for n in place["scenes"]}
        scares = sorted((b for b in beats if b.scare), key=lambda b: (-b.importance, b.index))
        strongest = {b.index for b in scares[: int(cfg.get("scare_shots", 4))]}
        chains = cfg["chains"]
        # Key beats: the most important of the rest (at least `key_importance`, at most `key_share` of
        # the story) get the best corrector when refused; every other beat keeps a picture refused
        # only for a detail rather than paying to correct it.
        rest = sorted((b for b in beats if b.index not in strongest and b.index not in host_beats
                       and b.importance >= int(cfg.get("key_importance", 3))),
                      key=lambda b: (-b.importance, b.index))
        key_beats = {b.index for b in rest[: max(1, round(float(cfg.get("key_share", 0.25)) * len(beats)))]}

        def tier(beat: Shot) -> str:
            # the host's pictures open and close every video: they get the corrector like a key beat
            return ("scare" if beat.index in strongest else
                    "key" if beat.index in key_beats or beat.index in host_beats else "beat")

        warned = lessons.load(board.conn) if cfg.get("lessons", True) else ""
        d["tiers"] = {b.index: tier(b) for b in beats}
        dark = lights_off(beats, host_beats)

        def one(beat: Shot, fix: str = "") -> tuple[int, Path | None]:
            names = [n for n in beat.characters if n in sheets][:3]
            place = place_of.get(beat.scene) if beat.index not in host_beats else None  # the host has his own place
            first = (place or {}).get("sheet") or style_frame
            refs = ([first] if first else []) + [p for n in names for p in
                                                 ([portraits[n]] if n in portraits else []) + [sheets[n]]]
            camera = str(cameras.get(beat.view, "")) if use_cameras else ""
            text = self.prompt(beat, d["house"], cast, names, bool(first), look, camera, place, warned,
                               {n for n in names if n in portraits}, dark.get(beat.index))
            faces = [(n, portraits[n]) for n in names if n in portraits] if cfg.get("face_check", True) else []
            if float(cfg.get("letterbox", 0) or 0) > 1.78:  # cinema bars will cover the top and bottom
                text = text.replace("never split into panels.", "never split into panels. Composed for a wide "
                                    "cinema frame: heads and the subject in the middle band, nothing important "
                                    "near the top or bottom edge.", 1)
            if fix:  # the art review's note on the first version leads, so it is not lost at the end
                text = f"Correct this from the previous attempt: {fix[:300].rstrip('. ')}. {text}"
            tag = f"_fix{abs(hash(fix)) % 10_000}" if fix else ""
            path, source = painter.paint(text, refs, folder / f"beat_{beat.index:03d}{tag}.png",
                                         ["beat", beat.index, text], chains.get(tier(beat), chains["beat"]),
                                         check=self.check_text(beat, cast) + "".join(
                                             f" The {x} is off and dark." for x in dark.get(beat.index, [])),
                                         accept_minor=tier(beat) == "beat",
                                         capped=True, redraw=bool(fix), faces=faces)
            sources[beat.index] = source
            return beat.index, path  # clean: the colorist grades it, the edit adds grain and vignette

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
        refused = [r.split("): ", 1)[1] for r in painter.rejections if r.startswith("beat_") and "): " in r]
        learned = lessons.record(board.conn, refused) if cfg.get("lessons", True) else {}
        return {"look": look.get("name"), "warned": warned, "faults_seen": learned,
                "retry_usd": round(painter.retry_spent, 3), "retries_stopped": painter.retries_stopped, "cameras": {b.index: b.view for b in beats if b.view != "film"},
                "pictures": len(beats) - borrowed - placeholders, "borrowed_neighbour": borrowed,
                "placeholders": placeholders, "scare_shots": sorted(strongest), "key_beats": sorted(key_beats),
                "by_painter": dict(painter.counts),
                "doubted_beats": sorted(b.index for b in beats if f"beat_{b.index:03d}" in painter.doubted),
                "painter_sizes": dict(painter.sizes), "rejected": painter.rejections[:30]}

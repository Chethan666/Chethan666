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
from .images import Critic, Painter, house_look
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
            text = (f"{house} {match}A character reference sheet of one person, {c['name']}: {c['design']}. {unknown}"
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
        last = {c["sheet"]: c for c in checks}  # a sheet's last check is the one that stands
        d["reference_issues"] = [f"{c['kind']} sheet of {c['sheet']}: {c['why']}" for c in last.values() if not c["ok"]]
        return {"style_frame": bool(got), "style_frame_by": source, "sheets": sorted(sheets),
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


class IllustratorAgent:
    name = "illustrator"
    critical = True

    def prompt(self, beat: Shot, house: str, cast: dict[str, str], refs: list[str], style_frame: bool,
               look: dict[str, Any] | None = None, camera: str = "", place: dict[str, Any] | None = None) -> str:
        """`style_frame`: reference image 1 is there (the place's sheet when `place` has one, else
        the style frame). `place`: the set this beat is in, {"name", "design", "sheet"}."""
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
        where = f" Place: {place['name']}: {place['design']}" if place and place.get("design") else ""
        # The shot comes first: what a model reads first it follows best, and run 94's refusals were
        # nearly all pictures that ignored their description (it sat behind 190 words of style).
        # Then what is fixed (place, people), what each reference is for, the look, and what to avoid.
        return (f"One single 16:9 {look.get('noun', 'frame')}.{lens} {FRAMING.get(beat.framing, beat.framing)}. "
                f"{picture}{empty}{where}{(' Characters: ' + people + '.') if people else ''} "
                f"{' '.join(notes)} The look: {house} No text, letters, captions, timestamps, borders or "
                f"panels.{avoid}").replace("  ", " ")

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

        def one(beat: Shot, fix: str = "") -> tuple[int, Path | None]:
            names = [n for n in beat.characters if n in sheets][:3]
            place = place_of.get(beat.scene) if beat.index not in host_beats else None  # the host has his own place
            first = (place or {}).get("sheet") or style_frame
            refs = ([first] if first else []) + [sheets[n] for n in names]
            camera = str(cameras.get(beat.view, "")) if use_cameras else ""
            text = self.prompt(beat, d["house"], cast, names, bool(first), look, camera, place)
            if fix:  # the art review's note on the first version leads, so it is not lost at the end
                text = f"Correct this from the previous attempt: {fix[:300].rstrip('. ')}. {text}"
            tag = f"_fix{abs(hash(fix)) % 10_000}" if fix else ""
            path, source = painter.paint(text, refs, folder / f"beat_{beat.index:03d}{tag}.png",
                                         ["beat", beat.index, text],
                                         chains.get("scare", chains["beat"]) if beat.index in strongest else chains["beat"],
                                         check=self.check_text(beat, cast))
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
        return {"look": look.get("name"), "cameras": {b.index: b.view for b in beats if b.view != "film"},
                "pictures": len(beats) - borrowed - placeholders, "borrowed_neighbour": borrowed,
                "placeholders": placeholders, "scare_shots": sorted(strongest), "by_painter": dict(painter.counts),
                "doubted_beats": sorted(b.index for b in beats if f"beat_{b.index:03d}" in painter.doubted),
                "painter_sizes": dict(painter.sizes), "rejected": painter.rejections[:30]}

"""Art direction: what each scene's picture shows.

An open model reads the scene plan (what to draw, where, who, the mood) and picks from a
fixed vocabulary the renderer can draw: a background, the light, a camera move, figures
with a pose and place, props, and the one element drawn in red. A fixed vocabulary keeps
every picture drawable and every character consistent. If the model fails, keywords in
the scene plan decide, so a video is always produced.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from story_factory.llm import LLMError

log = logging.getLogger(__name__)

BACKGROUNDS = ("room", "hallway", "road", "forest", "field", "shore", "street", "stairwell", "vehicle",
               "office", "sky", "water", "cave", "station")
LIGHTS = ("dark", "moonlight", "lamplight", "flicker")
CAMERAS = ("still", "push_in", "pull_out", "pan_left", "pan_right", "shake")
POSES = ("stand", "walk", "run", "sit", "crouch", "reach", "point", "look_back", "arms_up", "hands_face",
         "lie", "kneel", "hold")
PROPS = ("door", "window", "lamp", "streetlight", "tree", "car", "bus", "train", "boat", "bed", "chair",
         "table", "desk", "phone", "lantern", "candle", "mirror", "clock", "house", "fence", "stairs",
         "radio", "screen", "moon", "grave", "sign", "cup", "bag", "book", "photo", "key", "bell",
         "lighthouse", "elevator", "shelf", "sink", "jar", "toy", "letter", "rock")

KEYWORDS = {  # word in the scene plan -> (kind, value)
    "room": ("bg", "room"), "kitchen": ("bg", "room"), "bedroom": ("bg", "room"), "apartment": ("bg", "room"),
    "flat": ("bg", "room"), "cabin": ("bg", "room"), "house": ("bg", "room"), "lobby": ("bg", "room"),
    "hallway": ("bg", "hallway"), "corridor": ("bg", "hallway"), "hall": ("bg", "hallway"),
    "ward": ("bg", "hallway"), "road": ("bg", "road"), "highway": ("bg", "road"), "forest": ("bg", "forest"),
    "woods": ("bg", "forest"), "trees": ("bg", "forest"), "field": ("bg", "field"), "beach": ("bg", "shore"),
    "shore": ("bg", "shore"), "coast": ("bg", "shore"), "sea": ("bg", "water"), "lake": ("bg", "water"),
    "river": ("bg", "water"), "street": ("bg", "street"), "city": ("bg", "street"), "stairs": ("bg", "stairwell"),
    "stairwell": ("bg", "stairwell"), "car": ("bg", "vehicle"), "cab": ("bg", "vehicle"), "bus": ("bg", "vehicle"),
    "train": ("bg", "station"), "platform": ("bg", "station"), "station": ("bg", "station"),
    "office": ("bg", "office"), "desk": ("bg", "office"), "call center": ("bg", "office"),
    "museum": ("bg", "hallway"), "cave": ("bg", "cave"), "sky": ("bg", "sky"), "night sky": ("bg", "sky"),
}
PROP_WORDS = {p: p for p in PROPS} | {
    "doorway": "door", "streetlamp": "streetlight", "headlights": "car", "taxi": "car", "ship": "boat",
    "ferry": "boat", "telephone": "phone", "cellphone": "phone", "monitor": "screen", "tv": "screen",
    "television": "screen", "computer": "screen", "gravestone": "grave", "tombstone": "grave", "watch": "clock",
    "lamp post": "streetlight", "bunk": "bed", "cot": "bed", "photograph": "photo", "picture": "photo",
    "envelope": "letter", "note": "letter", "keys": "key", "doll": "toy", "lift": "elevator", "bookcase": "shelf",
    "stone": "rock", "boulder": "rock",
}
MOOD_CAMERA = {"tense": "push_in", "terror": "shake", "dread": "push_in", "calm": "still", "sad": "pull_out",
               "uneasy": "pan_right", "eerie": "push_in"}
OUTDOOR = {"road", "forest", "field", "shore", "street", "sky", "water", "station"}

SCENE_SPEC = {
    "type": "object", "additionalProperties": False,
    "required": ["index", "background", "light", "camera", "figures", "props", "red"],
    "properties": {
        "index": {"type": "integer"},
        "background": {"type": "string", "enum": list(BACKGROUNDS)},
        "light": {"type": "string", "enum": list(LIGHTS)},
        "camera": {"type": "string", "enum": list(CAMERAS)},
        "figures": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["name", "pose", "x", "facing", "ghost"],
            "properties": {"name": {"type": "string"}, "pose": {"type": "string", "enum": list(POSES)},
                           "x": {"type": "number"}, "facing": {"type": "string", "enum": ["left", "right"]},
                           "ghost": {"type": "boolean"}}}},
        "props": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["kind", "x"],
            "properties": {"kind": {"type": "string", "enum": list(PROPS)}, "x": {"type": "number"}}}},
        "red": {"type": "string"},
    },
}
DIRECTION = {"type": "object", "additionalProperties": False, "required": ["scenes"],
             "properties": {"scenes": {"type": "array", "items": SCENE_SPEC}}}

SYSTEM = ("You are the art director of Hollow Lines, an illustrated horror channel: still pictures drawn as "
          "white chalk stick figures on black paper with exactly one blood-red accent. You turn each scene's "
          "description into simple, readable compositions using only the given vocabulary.")


def prompt(scenes: list[dict[str, Any]], cast: list[dict[str, Any]], red_accent: str, first_index: int) -> str:
    listed = "\n".join(
        f"[{first_index + i}] narration: {s.get('text', '')[:300]}\n    visual: {s.get('visual', '')}\n"
        f"    location: {s.get('location', '')}; characters: {', '.join(map(str, s.get('characters') or []))}; "
        f"mood: {s.get('mood', '')}; key moment: {bool(s.get('key_moment'))}"
        for i, s in enumerate(scenes))
    names = ", ".join(str(c.get("name")) for c in cast) or "none"
    return (f"Cast (use these exact names): {names}\nThe story's red accent: {red_accent or 'none'}\n\n"
            f"Scenes:\n{listed}\n\nFor every scene give: background, light, camera (push_in for building "
            "dread, shake only for a shock, pans to reveal, still for quiet), figures (cast names, or "
            "'Figure' for an unnamed person and 'Shadow' for something not fully seen; pose; x from 0.1 "
            "to 0.9; facing; ghost true for a spirit or memory), at most three props with x positions, "
            "and red: the one element drawn in red, written as 'prop:<kind>' or 'figure:<name>', using "
            "the story's red accent whenever it is on screen, otherwise 'none'. Keep pictures simple: "
            "one to three figures, clear silhouettes, lots of darkness.")


def _clamp(value: Any, lo: float = 0.08, hi: float = 0.92, default: float = 0.5) -> float:
    try:
        return min(max(float(value), lo), hi)
    except (TypeError, ValueError):
        return default


def clean(spec: dict[str, Any], scene: dict[str, Any], cast_names: list[str]) -> dict[str, Any]:
    """A model's answer made safe to draw: unknown values replaced, positions clamped."""
    fallback = guess(scene, cast_names)
    out = {
        "background": spec.get("background") if spec.get("background") in BACKGROUNDS else fallback["background"],
        "light": spec.get("light") if spec.get("light") in LIGHTS else fallback["light"],
        "camera": spec.get("camera") if spec.get("camera") in CAMERAS else fallback["camera"],
        "figures": [], "props": [], "red": str(spec.get("red") or "none"),
    }
    for f in (spec.get("figures") or [])[:3]:
        if not isinstance(f, dict):
            continue
        out["figures"].append({"name": str(f.get("name") or "Figure"),
                               "pose": f.get("pose") if f.get("pose") in POSES else "stand",
                               "x": _clamp(f.get("x")), "facing": "left" if f.get("facing") == "left" else "right",
                               "ghost": f.get("ghost") is True})
    for p in (spec.get("props") or [])[:3]:
        if isinstance(p, dict) and p.get("kind") in PROPS:
            out["props"].append({"kind": p["kind"], "x": _clamp(p.get("x"))})
    if not out["figures"] and not out["props"]:
        out["figures"], out["props"] = fallback["figures"], fallback["props"]
    return out


def guess(scene: dict[str, Any], cast_names: list[str]) -> dict[str, Any]:
    """Direction from keywords alone, used when the model can't answer."""
    text = f"{scene.get('location', '')} {scene.get('visual', '')}".lower()
    background = next((v for k, (kind, v) in KEYWORDS.items() if kind == "bg" and re.search(rf"\b{k}\b", text)),
                      "room")
    props = []
    for word, kind in PROP_WORDS.items():
        if re.search(rf"\b{re.escape(word)}s?\b", text) and kind not in [p["kind"] for p in props]:
            props.append({"kind": kind, "x": 0.25 + 0.25 * len(props)})
        if len(props) == 3:
            break
    names = [str(c) for c in scene.get("characters") or []][:3] or (cast_names[:1] if not props else [])
    figures = [{"name": n, "pose": "stand", "x": round(0.35 + 0.3 * i, 2), "facing": "left" if i % 2 else "right",
                "ghost": any(w in n.lower() for w in ("ghost", "shadow", "spirit", "memory"))}
               for i, n in enumerate(names)]
    mood = str(scene.get("mood", "")).lower()
    camera = next((c for m, c in MOOD_CAMERA.items() if m in mood), "push_in" if scene.get("key_moment") else "still")
    light = "moonlight" if background in OUTDOOR else ("flicker" if "flicker" in text else "lamplight")
    return {"background": background, "light": light, "camera": camera, "figures": figures, "props": props,
            "red": "none"}


def direct(router: Any, scenes: list[dict[str, Any]], cast: list[dict[str, Any]], red_accent: str,
           batch: int = 12) -> tuple[list[dict[str, Any]], int]:
    """Direction for every scene. Returns (specs, how many scenes fell back to keywords)."""
    cast_names = [str(c.get("name")) for c in cast]
    specs: list[dict[str, Any]] = []
    fallbacks = 0
    for start in range(0, len(scenes), batch):
        group = scenes[start:start + batch]
        answer: dict[int, dict[str, Any]] = {}
        if router is not None:
            try:
                result = router.one("planner").generate_json(
                    "art_direction", SYSTEM, prompt(group, cast, red_accent, start + 1), DIRECTION)
                answer = {int(s.get("index", 0)): s for s in result.get("scenes", []) if isinstance(s, dict)}
            except (LLMError, ValueError, TypeError) as exc:
                log.warning("Art direction for scenes %d-%d fell back to keywords: %s",
                            start + 1, start + len(group), exc)
        for i, scene in enumerate(group, start + 1):
            if i in answer:
                specs.append(clean(answer[i], scene, cast_names))
            else:
                fallbacks += 1
                specs.append(guess(scene, cast_names))
    return specs, fallbacks


# --- Character looks ------------------------------------------------------------

LOOK_WORDS = {
    "headwear": [("hood", "hood"), ("cap", "cap"), ("beanie", "cap"), ("hat", "hat"), ("helmet", "cap"),
                 ("headscarf", "scarf_head"), ("hijab", "scarf_head"), ("veil", "scarf_head")],
    "hair": [("long hair", "long"), ("braid", "long"), ("ponytail", "long"), ("bun", "bun"), ("bald", "none"),
             ("short hair", "short"), ("curly", "short"), ("hair", "short")],
    "outfit": [("dress", "dress"), ("gown", "dress"), ("skirt", "dress"), ("robe", "robe"), ("cloak", "robe"),
               ("coat", "coat"), ("jacket", "coat"), ("uniform", "coat"), ("scrubs", "coat"), ("parka", "coat")],
    "accessory": [("lantern", "lantern"), ("glasses", "glasses"), ("spectacles", "glasses"), ("scarf", "scarf"),
                  ("bag", "bag"), ("backpack", "bag"), ("cane", "cane"), ("walking stick", "cane"),
                  ("headset", "headset"), ("badge", "badge")],
}


def features(name: str, look: str) -> dict[str, Any]:
    """Drawable features from a cast member's fixed look, read from its words."""
    text = f" {look.lower()} "
    out: dict[str, Any] = {}
    for part, options in LOOK_WORDS.items():
        out[part] = next((value for word, value in options if re.search(rf"\b{word}s?\b", text)), "none")
    out["height"] = 1.12 if re.search(r"\btall\b", text) else 0.82 if re.search(r"\b(short|small|child|boy|girl|kid)\b", text) else 1.0
    out["child"] = bool(re.search(r"\b(child|boy|girl|kid|daughter|son)\b", text + name.lower()))
    out["ghost"] = bool(re.search(r"\b(ghost|spirit|shadow|faceless|translucent|memory)\b", text + name.lower()))
    if name.lower() == "the storyteller":
        out.update(outfit="coat", accessory="lantern", height=1.15)
    return out


def describe(direction: list[dict[str, Any]]) -> str:
    return json.dumps(direction, ensure_ascii=False)

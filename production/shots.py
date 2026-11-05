"""The shot list: every scene cut into shots of a few seconds, staged like a film.

The narration is already timed chunk by chunk, so shots are cut from those chunks: a shot
covers one or more consecutive chunks and lasts 3-6.5 s (a long sentence is split across
two shots). An open model then directs each shot like a storyboard artist: the framing
(wide, medium, close, insert, point of view), what the picture shows, how the stick figures
act with their whole bodies, the camera, what moves if it becomes a clip, one sound, whether
it is a scare, and how much it matters. The most important shots become real clips. If the
model fails, the scene's own plan fills the shots in, cycling the framings.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from story_factory.llm import LLMError

log = logging.getLogger(__name__)

FRAMINGS = ("wide", "medium", "close", "extreme_close", "insert", "pov", "over_shoulder", "low_angle", "high_angle")
CAMERAS = ("still", "push_in", "pull_out", "pan_left", "pan_right", "tilt_up", "tilt_down", "handheld")
FRAMING_TEXT = {
    "wide": "wide shot: the whole place, the figures small in it, lots of dark negative space",
    "medium": "medium shot: the figure from head to knees, body language readable",
    "close": "close-up: the stick figure's head and shoulders fill the frame",
    "extreme_close": "extreme close-up on one detail (a hand, an eye, a keyhole, a screen)",
    "insert": "insert shot of the important object, no figures",
    "pov": "point of view: what the character sees, framed by a doorway, window or flashlight beam",
    "over_shoulder": "over the shoulder of one figure, looking at what they face",
    "low_angle": "low angle looking up, the subject looming",
    "high_angle": "high angle looking down, the figure small and exposed",
}
SYSTEM = ("You are the director and storyboard artist of Hollow Lines, an animated horror channel drawn "
          "as white chalk stick figures on black paper with exactly one blood-red accent. You stage every "
          "shot for maximum dread and clarity: one idea per shot, strong silhouettes, exaggerated body "
          "language, consistent screen direction, darkness used as space. Answer with JSON only.")
SCHEMA = {
    "type": "object", "required": ["shots"],
    "properties": {"shots": {"type": "array", "items": {
        "type": "object",
        "required": ["index", "framing", "picture", "characters", "camera", "motion", "sound", "scare",
                     "importance"],
        "properties": {
            "index": {"type": "integer"},
            "framing": {"type": "string", "enum": list(FRAMINGS)},
            "picture": {"type": "string"},
            "characters": {"type": "array", "items": {"type": "string"}},
            "camera": {"type": "string", "enum": list(CAMERAS)},
            "motion": {"type": "string"},
            "sound": {"type": "string"},
            "scare": {"type": "boolean"},
            "importance": {"type": "integer"},
        }}}},
}
CHAPTERS = {"type": "object", "required": ["chapters"], "properties": {"chapters": {"type": "array", "items": {
    "type": "object", "required": ["scene", "title"],
    "properties": {"scene": {"type": "integer"}, "title": {"type": "string"}}}}}}


@dataclass
class Shot:
    index: int
    scene: int
    t0: float  # seconds on the narration timeline
    t1: float
    text: str  # the words spoken over it
    framing: str = "medium"
    picture: str = ""
    characters: list[str] = field(default_factory=list)
    camera: str = "still"
    motion: str = ""
    sound: str = ""
    scare: bool = False
    importance: int = 2
    tier: str = "still"  # hero, medium or still (a living still)
    directed: bool = False

    @property
    def seconds(self) -> float:
        return self.t1 - self.t0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def cut(chunks: list[Any], scene_of: list[int], duration: float, min_s: float = 3.0,
        max_s: float = 6.5) -> list[Shot]:
    """Shots from timed chunks: consecutive chunks of one scene, 3-6.5 s each, the first from 0 s
    and each running until the next begins (so the pictures cover every moment of the audio)."""
    pieces: list[tuple[int, float, float, str]] = []  # scene, start, end, text
    for chunk, scene in zip(chunks, scene_of):
        length = chunk.t1 - chunk.t0
        parts = max(1, int(length // max_s) + (1 if length % max_s > 0.01 and length > max_s else 0))
        words = chunk.text.split()
        for k in range(parts):
            a = chunk.t0 + length * k / parts
            b = chunk.t0 + length * (k + 1) / parts
            text = " ".join(words[len(words) * k // parts: len(words) * (k + 1) // parts])
            pieces.append((scene, a, b, text))
    shots: list[Shot] = []
    for scene, a, b, text in pieces:
        last = shots[-1] if shots else None
        if last and last.scene == scene and (last.t1 - last.t0) < min_s and (b - last.t0) <= max_s + 0.5:
            last.t1, last.text = b, f"{last.text} {text}".strip()
        else:
            shots.append(Shot(len(shots), scene, a, b, text))
    # A short last shot in a scene joins the one before it.
    merged: list[Shot] = []
    for shot in shots:
        prev = merged[-1] if merged else None
        if prev and prev.scene == shot.scene and shot.seconds < min_s * 0.6 and prev.seconds + shot.seconds <= max_s + 1.5:
            prev.t1, prev.text = shot.t1, f"{prev.text} {shot.text}".strip()
        else:
            merged.append(shot)
    for i, shot in enumerate(merged):  # continuous cover: each shot runs until the next starts
        shot.index = i
        shot.t0 = 0.0 if i == 0 else shot.t0
        shot.t1 = merged[i + 1].t0 if i + 1 < len(merged) else duration
    return merged


def scene_index(chunks: list[Any], spans: list[tuple[int, int]]) -> list[int]:
    out = []
    for chunk in chunks:
        i = 0
        for k, (start, end) in enumerate(spans):
            if start <= chunk.start < end or chunk.start >= start:
                i = k
        out.append(i)
    return out


def _prompt(shots: list[Shot], scenes: list[dict[str, Any]], cast: list[dict[str, Any]], red: str) -> str:
    cast_text = "; ".join(f"{c.get('name')}: {c.get('look', '')}" for c in cast) or "none"
    by_scene: dict[int, list[Shot]] = {}
    for s in shots:
        by_scene.setdefault(s.scene, []).append(s)
    blocks = []
    for k, group in by_scene.items():
        sc = scenes[k] if k < len(scenes) else {}
        lines = "\n".join(f"  shot {s.index} ({s.seconds:.1f} s): \"{s.text}\"" for s in group)
        blocks.append(f"SCENE {k} - location: {sc.get('location', '')}; characters: "
                      f"{', '.join(map(str, sc.get('characters') or []))}; mood: {sc.get('mood', '')}; "
                      f"key moment: {bool(sc.get('key_moment'))}\n  plan: {sc.get('visual', '')}\n{lines}")
    return (f"Cast (fixed looks): {cast_text}\nThe story's red accent: {red or 'none'}\n\n"
            + "\n\n".join(blocks) +
            "\n\nDirect every shot listed above (use its number as index). For each give:\n"
            "- framing: vary it within a scene (open wide, then medium, close, insert); never the same "
            "framing three times in a row\n"
            "- picture: one or two concrete sentences a chalk artist can draw: who is where, doing what, "
            "with what body language (hunched, frozen mid-step, head turned too far, hands over mouth), "
            "the setting and its few key objects, and where the red accent is if it is in frame. Show "
            "the exact moment the words describe; the unseen threat as a dark shape or a pair of eyes, "
            "placed where the eye lands second, not in the centre\n"
            "- characters: cast names in the shot (exact spelling), [] if none\n"
            "- camera: still, push_in (building dread), pull_out (isolation), pan_left/pan_right (reveal), "
            "tilt_up/tilt_down, handheld (panic only)\n"
            "- motion: what moves if this shot is animated, in one short sentence (a slow head turn, a "
            "door creaking open, a flashlight beam sweeping, breath fogging)\n"
            "- sound: one short sound effect for this shot (\"a door creaks open slowly\"), or \"\"\n"
            "- scare: true only for the few shots where something frightening is revealed\n"
            "- importance: 1 (connective) to 5 (the moment the story turns on)\n"
            "Keep screen direction consistent: going towards danger is left to right, fleeing is right to left.")


def direct(router: Any, shots: list[Shot], scenes: list[dict[str, Any]], cast: list[dict[str, Any]],
           red: str, batch_scenes: int = 5) -> int:
    """Fill in every shot's staging. Returns how many came from the model."""
    names = [str(c.get("name")) for c in cast if c.get("name")]
    for shot in shots:
        fallback(shot, scenes[shot.scene] if shot.scene < len(scenes) else {}, shots)
    if router is None:
        return 0
    got = 0
    scene_ids = sorted({s.scene for s in shots})
    for start in range(0, len(scene_ids), batch_scenes):
        wanted = set(scene_ids[start:start + batch_scenes])
        group = [s for s in shots if s.scene in wanted]
        try:
            result = router.one("planner").generate_json("shot_list", SYSTEM, _prompt(group, scenes, cast, red), SCHEMA)
        except (LLMError, ValueError, TypeError) as exc:
            log.warning("Shot list for scenes %s from the scene plan: %s", sorted(wanted), exc)
            continue
        by_index = {s.index: s for s in group}
        for item in result.get("shots", []) if isinstance(result, dict) else []:
            shot = by_index.get(item.get("index")) if isinstance(item, dict) else None
            if shot is None or not str(item.get("picture", "")).strip():
                continue
            shot.framing = item.get("framing") if item.get("framing") in FRAMINGS else shot.framing
            shot.picture = str(item["picture"]).strip()[:600]
            shot.characters = [_canonical(str(n), names) for n in item.get("characters") or [] if str(n).strip()][:3]
            shot.camera = item.get("camera") if item.get("camera") in CAMERAS else shot.camera
            shot.motion = str(item.get("motion", "")).strip()[:300] or shot.motion
            shot.sound = str(item.get("sound", "")).strip()[:120]
            shot.scare = bool(item.get("scare"))
            try:
                shot.importance = min(max(int(item.get("importance", 2)), 1), 5)
            except (TypeError, ValueError):
                pass
            shot.directed = True
            got += 1
    return got


def _canonical(name: str, names: list[str]) -> str:
    low = name.strip().lower()
    for n in names:
        if low == n.lower() or low == n.lower().split()[0]:
            return n
    return name.strip()


CYCLE = ("wide", "medium", "close", "insert", "medium", "pov")


def fallback(shot: Shot, scene: dict[str, Any], shots: list[Shot]) -> None:
    """Staging from the scene's own plan, when the model gives nothing."""
    position = sum(1 for s in shots if s.scene == shot.scene and s.index < shot.index)
    shot.framing = CYCLE[position % len(CYCLE)]
    shot.picture = f"{scene.get('visual', '')} {('Moment: ' + shot.text) if shot.text else ''}".strip()[:600]
    shot.characters = [str(c) for c in scene.get("characters") or []][:3]
    mood = str(scene.get("mood", "")).lower()
    shot.camera = "push_in" if any(m in mood for m in ("tense", "dread", "eerie")) else (
        "handheld" if "terror" in mood or "panic" in mood else ("pan_right" if position % 2 else "push_in"))
    shot.motion = "slow subtle movement: the figure breathes and shifts, chalk dust drifts"
    shot.sound = str((scene.get("sfx") or [""])[0]) if position == 0 else ""
    shot.importance = 4 if scene.get("key_moment") and position == 0 else 2
    shot.scare = bool(scene.get("key_moment")) and position == 0 and any(
        m in mood for m in ("terror", "shock", "horror", "panic"))


def assign_tiers(shots: list[Shot], scenes: list[dict[str, Any]], hero: int, medium: int) -> None:
    """The most important shots become hero clips, the next ones medium clips, spread through the
    video (never more than two hero shots in a row); the rest stay living stills."""
    def score(s: Shot) -> tuple:
        key = bool(scenes[s.scene].get("key_moment")) if s.scene < len(scenes) else False
        return (-(s.importance + (2 if s.scare else 0) + (1 if key else 0)), -s.seconds, s.index)

    for s in shots:
        s.tier = "still"
    chosen: set[int] = set()
    for s in sorted(shots, key=score):
        if len(chosen) >= hero:
            break
        if {s.index - 1, s.index - 2} <= chosen:
            continue
        chosen.add(s.index)
    for i in chosen:
        shots[i].tier = "hero"
    rest = [s for s in sorted(shots, key=score) if s.tier == "still"]
    # Medium clips also go where a run of stills would be longest.
    for s in rest[:medium]:
        s.tier = "medium"


def chapters(router: Any, scenes: list[dict[str, Any]], shots: list[Shot], offset: float,
             min_gap: float = 45.0) -> list[tuple[float, str]]:
    """YouTube chapters: (time, title), the first at 0:00, at least `min_gap` apart."""
    starts: dict[int, float] = {}
    for s in shots:
        starts.setdefault(s.scene, s.t0 + offset)
    titles: dict[int, str] = {}
    if router is not None:
        listing = "\n".join(f"{i}: {sc.get('location', '')} - {str(sc.get('visual', ''))[:140]}"
                            for i, sc in enumerate(scenes))
        try:
            result = router.one("planner").generate_json(
                "chapters", "You name YouTube chapters for a horror story video. JSON only.",
                f"Scenes:\n{listing}\n\nGroup the scenes into 5 to 9 chapters in story order. For each, give "
                "the first scene's number and a 2-5 word title that intrigues without spoiling the ending.",
                CHAPTERS)
            for item in result.get("chapters", []):
                if isinstance(item, dict) and isinstance(item.get("scene"), int) and str(item.get("title", "")).strip():
                    titles[item["scene"]] = str(item["title"]).strip()[:60]
        except (LLMError, ValueError, TypeError) as exc:
            log.warning("Chapters from the locations: %s", exc)
    if not titles:
        last = None
        for i, sc in enumerate(scenes):
            place = str(sc.get("location", "")).strip()
            if place and place != last:
                titles[i] = place[:60].capitalize()
                last = place
    out: list[tuple[float, str]] = [(0.0, "The beginning")]
    for i in sorted(titles):
        t = starts.get(i)
        if t is None:
            continue
        if t - out[-1][0] >= min_gap:
            out.append((t, titles[i]))
        elif len(out) == 1 and i == min(titles):
            out[0] = (0.0, titles[i])
    return out if len(out) >= 3 else []


def stamp(seconds: float) -> str:
    seconds = int(seconds)
    h, m, s = seconds // 3600, seconds // 60 % 60, seconds % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def save(shots: list[Shot]) -> str:
    return json.dumps([s.to_dict() for s in shots], indent=2, ensure_ascii=False)

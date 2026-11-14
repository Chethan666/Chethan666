"""The beats: the timed narration cut into pictures, and what is kept about each one.

The narration is already timed chunk by chunk, so beats are cut from those chunks: a beat
covers one or more consecutive chunks of one scene and lasts between a minimum and a maximum
length (a long sentence is split across two). The storyboard agent (production/stills/
storyboard.py) stages each beat; if its model fails, the scene's own plan fills the beat in
here, cycling the framings. Also: YouTube chapters, and timestamps.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from story_factory.llm import LLMError

log = logging.getLogger(__name__)

FRAMINGS = ("wide", "medium", "close", "extreme_close", "insert", "pov", "over_shoulder", "low_angle", "high_angle")
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
    sound: str = ""
    motion: str = ""  # what moves if the beat becomes a clip (production/stills/motion.py)
    scare: bool = False
    importance: int = 2
    directed: bool = False
    view: str = "film"  # stills: film, or seen through a cctv, phone or polaroid camera
    atmosphere: list[str] = field(default_factory=list)  # what moves in the air: rain, fog, flicker

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
    shot.sound = str((scene.get("sfx") or [""])[0]) if position == 0 else ""
    shot.importance = 4 if scene.get("key_moment") and position == 0 else 2
    shot.scare = bool(scene.get("key_moment")) and position == 0 and any(
        m in mood for m in ("terror", "shock", "horror", "panic"))
    shot.atmosphere = atmosphere_in(shot.picture)


INSIDE = ("inside", "interior", "indoors", "window", "windscreen", "windshield", "through the glass", "room")


def atmosphere_in(picture: str) -> list[str]:
    """What moves in the air of a picture, from its own words: the storyboard's fallback.
    Rain and fog only when nothing says the camera is indoors or behind glass (fog seen through
    a window must not drift across the room)."""
    text = picture.lower()
    found = []
    outdoors = not any(w in text for w in INSIDE)
    if any(w in text for w in ("rain", "downpour", "storm", "drizzle")) and outdoors:
        found.append("rain")
    if any(w in text for w in ("fog", "mist", "haze")) and outdoors:
        found.append("fog")
    if any(w in text for w in ("flicker", "failing light", "buzzing light", "guttering", "candle")):
        found.append("flicker")
    return found


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

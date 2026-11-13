"""The storyboard: the narration timeline cut into beats, each staged as one still picture.

A beat covers one or more spoken chunks of one scene and lasts 5-11 s, so a picture holds
long enough to be looked at but changes before it goes stale. An open model (the story
factory's planner role) stages each beat like a storyboard artist: the framing, exactly
what the picture shows, who is in it, one sound, whether it is a scare and how much it
matters, and whether it is seen through a security camera, a phone or an old photo when
the story itself looks through one. The host's intro and outro show The Storyteller. If the model fails, the scene's
own plan fills the beat in.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from story_factory.llm import LLMError

from .. import shots as shotlist
from ..shots import FRAMINGS, Shot
from .board import Board

log = logging.getLogger(__name__)

HOST = "The Storyteller"
ATMOSPHERES = ("rain", "fog", "flicker")
VIEWS = ("film", "cctv", "phone", "polaroid")
SYSTEM = ("You are the storyboard artist of Hollow Lines, a horror storytelling channel told in still "
          "pictures, like frames of a dark horror film, over a narrator's voice. You stage every beat for dread and clarity: one "
          "idea per picture, a strong composition, faces and body language that read at phone size, darkness "
          "used as space, the threat mostly unseen. Answer with JSON only.")
SCHEMA = {
    "type": "object", "required": ["beats"],
    "properties": {"beats": {"type": "array", "items": {
        "type": "object", "required": ["index", "line", "framing", "picture", "characters", "sound", "scare",
                                       "importance"],
        "properties": {
            "index": {"type": "integer"},
            "line": {"type": "string"},
            "framing": {"type": "string", "enum": list(FRAMINGS)},
            "picture": {"type": "string"},
            "characters": {"type": "array", "items": {"type": "string"}},
            "sound": {"type": "string"},
            "scare": {"type": "boolean"},
            "importance": {"type": "integer"},
            "view": {"type": "string", "enum": list(VIEWS)},
            "atmosphere": {"type": "array", "items": {"type": "string", "enum": list(ATMOSPHERES)}},
        }}}},
}


def prompt(beats: list[Shot], scenes: list[dict[str, Any]], cast: list[dict[str, Any]], red: str,
           host_beats: set[int]) -> str:
    cast_text = "; ".join(f"{c.get('name')}: {c.get('design') or c.get('look', '')}" for c in cast) or "none"
    by_scene: dict[int, list[Shot]] = {}
    for b in beats:
        by_scene.setdefault(b.scene, []).append(b)
    blocks = []
    for k, group in by_scene.items():
        sc = scenes[k] if k < len(scenes) else {}
        lines = "\n".join(f"  beat {b.index} ({b.seconds:.1f} s){' [HOST]' if b.index in host_beats else ''}: "
                          f"\"{b.text}\"" for b in group)
        blocks.append(f"SCENE {k} - location: {sc.get('location', '')}; characters: "
                      f"{', '.join(map(str, sc.get('characters') or []))}; mood: {sc.get('mood', '')}; "
                      f"key moment: {bool(sc.get('key_moment'))}\n  plan: {sc.get('visual', '')}\n{lines}")
    return (f"Cast (fixed designs): {cast_text}\nThe story's red accent: {red or 'none'}\n\n" + "\n\n".join(blocks) +
            "\n\nStage every beat listed above as ONE still picture, one entry per beat, in order:\n"
            "- index: the beat's number; line: the beat's narrated words, copied exactly (this is how your "
            "picture is matched to its words: a picture belongs to its own line, never the next one)\n"
            "- framing: vary it within a scene (establish wide, then medium, close, insert); never the same "
            "framing three times in a row\n"
            "- picture: two concrete sentences a cinematographer can shoot: who is where, doing what, with what "
            "expression and body language, the setting with its few telling objects, the light source, and "
            "where the red accent is if it is in frame. Show exactly what THIS beat's words describe (the same "
            "action, pose and place: 'standing behind' is standing, not leaning), nothing from a later line. Keep the "
            "threat half-seen: a silhouette, a shape in a doorway, eyes in the dark, placed off-centre. "
            f"Beats marked [HOST] show {HOST}, the channel's host, in his place between the worlds, speaking "
            "to us; describe him with the moment's mood\n"
            "- characters: cast names in the picture (exact spelling), [] if none\n"
            "- sound: one short sound effect for this beat (\"a door creaks open slowly\"), or \"\"\n"
            "- scare: true only for the few beats where something frightening is revealed\n"
            "- importance: 1 (connective) to 5 (the moment the story turns on)\n"
            "- view: \"film\" for almost every beat; \"cctv\", \"phone\" or \"polaroid\" only when the words "
            "say we are looking at a security-camera feed, a phone photo or video, or an old photograph, and "
            "then the picture shows what that camera sees. Never for [HOST] beats\n"
            "- atmosphere: what visibly moves in the air of this picture, for the edit to animate: \"rain\" only "
            "when rain falls between the camera and the scene (outdoors, never over an interior or behind "
            "glass), \"fog\" for mist or fog in the air, \"flicker\" when the picture's light is a failing lamp, "
            "a guttering candle or a stuttering screen; [] for most beats. Keep it consistent within a scene\n"
            "No text, signs or captions to read in any picture.")


class StoryboardAgent:
    name = "storyboard"
    critical = True

    def run(self, board: Board) -> dict[str, Any]:
        cfg, d = board.cfg, board.data
        timeline, spans, scenes = d["timeline"], d["spans"], d["scenes"]
        beats = shotlist.cut(timeline.chunks, shotlist.scene_index(timeline.chunks, spans), timeline.duration,
                             float(cfg.get("beat_min_seconds", 5.0)), float(cfg.get("beat_max_seconds", 11.0)))
        host_beats = self._host_beats(beats, timeline, d)
        for b in beats:
            shotlist.fallback(b, scenes[b.scene] if b.scene < len(scenes) else {}, beats)
            if b.index in host_beats:
                b.characters = [HOST]
                b.picture = (f"{HOST} sits by his lantern's red flame in the dark, looking at us as he speaks: "
                             f"\"{b.text[:160]}\"")
        names = [str(c.get("name")) for c in d["cast"] if c.get("name")] + [HOST]
        directed = self._direct(board, beats, scenes, names, host_beats)
        left = [b for b in beats if not b.directed]
        if left and directed:  # the model skipped some: ask again for just those
            directed += self._direct(board, left, scenes, names, host_beats)
        checked = check(board, beats, scenes, host_beats)
        d.update(beats=beats, host_beats=host_beats)
        return {"beats": len(beats), "directed_by_model": directed, "host_beats": len(host_beats),
                "scares": sum(1 for b in beats if b.scare), "check": checked,
                "average_seconds": round(timeline.duration / max(len(beats), 1), 1)}

    def _host_beats(self, beats: list[Shot], timeline: Any, d: dict[str, Any]) -> set[int]:
        frames = d.get("frames") or []  # the intro and outro, whoever tells the story
        if not frames:
            return set()
        times = [(timeline.time_at(a), timeline.time_at(b)) for a, b in frames]
        out = set()
        for b in beats:
            mid = (b.t0 + b.t1) / 2
            if any(t0 - 0.3 <= mid <= t1 + 0.3 for t0, t1 in times):
                out.add(b.index)
        return out

    def _direct(self, board: Board, beats: list[Shot], scenes: list[dict[str, Any]], names: list[str],
                host_beats: set[int], batch_scenes: int = 5) -> int:
        if board.router is None:
            return 0
        red = str((board.data.get("bible") or {}).get("red_accent", ""))
        got = 0
        scene_ids = sorted({b.scene for b in beats})
        for start in range(0, len(scene_ids), batch_scenes):
            wanted = set(scene_ids[start:start + batch_scenes])
            group = [b for b in beats if b.scene in wanted]
            try:
                result = board.router.one("planner").generate_json(
                    "storyboard", SYSTEM, prompt(group, scenes, board.data["cast"], red, host_beats), SCHEMA)
            except (LLMError, ValueError, TypeError) as exc:
                log.warning("Storyboard for scenes %s from the scene plan: %s", sorted(wanted), exc)
                continue
            for item, beat in match(result.get("beats", []) if isinstance(result, dict) else [], group):
                if beat.directed or not str(item.get("picture", "")).strip():
                    continue
                beat.framing = item.get("framing") if item.get("framing") in FRAMINGS else beat.framing
                beat.picture = str(item["picture"]).strip()[:700]
                chars = [shotlist._canonical(str(n), names) for n in item.get("characters") or [] if str(n).strip()]
                beat.characters = ([HOST] if beat.index in host_beats else chars)[:3]
                beat.sound = str(item.get("sound", "")).strip()[:120]
                beat.scare = bool(item.get("scare")) and beat.index not in host_beats
                try:
                    beat.importance = min(max(int(item.get("importance", 2)), 1), 5)
                except (TypeError, ValueError):
                    pass
                view = str(item.get("view") or "film")
                beat.view = view if view in VIEWS and beat.index not in host_beats else "film"
                air = item.get("atmosphere")
                beat.atmosphere = ([str(a) for a in air if str(a) in ATMOSPHERES][:2] if isinstance(air, list)
                                   else shotlist.atmosphere_in(beat.picture))
                beat.directed = True
                got += 1
        return got


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def match(items: list[Any], group: list[Shot]) -> list[tuple[dict[str, Any], Shot]]:
    """Each staged picture with the beat whose words it quotes. Run 94: the model numbered its
    pictures one beat late, so matching by number put every picture under the line before it."""
    out = []
    taken: set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        quoted = _words(str(item.get("line", "")))
        best, score = None, 0.0
        for b in group:
            if b.index in taken:
                continue
            words = _words(b.text)
            if not words or not quoted:
                continue
            same = sum(1 for w in quoted if w in words) / max(len(quoted), len(words))
            if same > score:
                best, score = b, same
        if best is None or score < 0.6:  # no clear quote: fall back to the number
            best = next((b for b in group if b.index == item.get("index") and b.index not in taken), None)
            if best is None:
                continue
        taken.add(best.index)
        out.append((item, best))
    return out


CHECK_SYSTEM = ("You are the script supervisor of Hollow Lines, a horror channel told in still pictures over a "
                "narrator. You make sure every picture shows exactly what is being said while it is on screen. "
                "Answer with JSON only.")
CHECK_SCHEMA = {"type": "object", "required": ["fix"], "properties": {"fix": {"type": "array", "items": {
    "type": "object", "required": ["index", "problem", "picture"],
    "properties": {"index": {"type": "integer"}, "problem": {"type": "string"}, "picture": {"type": "string"},
                   "characters": {"type": "array", "items": {"type": "string"}}}}}}}


def check(board: Board, beats: list[Shot], scenes: list[dict[str, Any]], host_beats: set[int]) -> dict[str, Any]:
    """The storyboard gate, before any picture is paid for: every picture description is held
    against its narrated line, the lines before it and the cast's fixed looks; the ones that
    don't match are rewritten. A description costs nothing to fix; a picture does."""
    if board.router is None or not beats:
        return {"checked": False}
    cast = "; ".join(f"{c.get('name')}: {c.get('look', '')}" for c in board.data.get("cast", []) if c.get("name"))
    rows = "\n".join(f"{b.index}. LINE: \"{b.text}\"\n   PICTURE: {b.picture}\n   PEOPLE: "
                     f"{', '.join(b.characters) or 'none'}" for b in beats if b.index not in host_beats)
    text = (f"Cast (fixed looks): {cast or 'none'}\n\n{rows}\n\nEach PICTURE is on screen while its LINE is "
            "narrated. Find every picture that does not show what its own line says: a different action or pose "
            "(the line says standing, the picture says leaning or only a hand), a different place, something from "
            "an earlier or later line instead of this one, a person the line does not involve, or a person's fixed "
            "look broken (a face the cast notes hide must stay hidden). For each, give the problem and a new "
            "PICTURE in the same style (two concrete sentences: who is where doing what, the light, the threat "
            "half-seen) that shows exactly this line, and its PEOPLE. Leave pictures that match out. "
            '{"fix": [{"index": N, "problem": "...", "picture": "...", "characters": [...]}]}')
    try:
        result = board.router.one("planner").generate_json("storyboard_check", CHECK_SYSTEM, text, CHECK_SCHEMA)
    except (LLMError, ValueError, TypeError) as exc:
        log.warning("Storyboard check unavailable: %s", exc)
        return {"checked": False}
    names = [str(c.get("name")) for c in board.data.get("cast", []) if c.get("name")]
    by_index = {b.index: b for b in beats if b.index not in host_beats}
    fixed = []
    for f in result.get("fix", []) if isinstance(result, dict) else []:
        beat = by_index.get(f.get("index")) if isinstance(f, dict) else None
        picture = str((f or {}).get("picture", "")).strip() if isinstance(f, dict) else ""
        if beat is None or len(picture) < 20:
            continue
        beat.picture = picture[:700]
        if "rain" in beat.atmosphere and any(w in picture.lower() for w in shotlist.INSIDE):
            beat.atmosphere = [a for a in beat.atmosphere if a != "rain"]  # the new picture is indoors
        if isinstance(f.get("characters"), list):
            beat.characters = [shotlist._canonical(str(n), names + [HOST]) for n in f["characters"] if str(n).strip()][:3]
        fixed.append({"beat": beat.index, "problem": str(f.get("problem", ""))[:200]})
    return {"checked": True, "fixed": fixed}

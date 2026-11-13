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
from typing import Any

from story_factory.llm import LLMError

from .. import shots as shotlist
from ..shots import FRAMINGS, Shot
from .board import Board

log = logging.getLogger(__name__)

HOST = "The Storyteller"
VIEWS = ("film", "cctv", "phone", "polaroid")
SYSTEM = ("You are the storyboard artist of Hollow Lines, a horror storytelling channel told in still "
          "pictures, like frames of a dark horror film, over a narrator's voice. You stage every beat for dread and clarity: one "
          "idea per picture, a strong composition, faces and body language that read at phone size, darkness "
          "used as space, the threat mostly unseen. Answer with JSON only.")
SCHEMA = {
    "type": "object", "required": ["beats"],
    "properties": {"beats": {"type": "array", "items": {
        "type": "object", "required": ["index", "framing", "picture", "characters", "sound", "scare", "importance"],
        "properties": {
            "index": {"type": "integer"},
            "framing": {"type": "string", "enum": list(FRAMINGS)},
            "picture": {"type": "string"},
            "characters": {"type": "array", "items": {"type": "string"}},
            "sound": {"type": "string"},
            "scare": {"type": "boolean"},
            "importance": {"type": "integer"},
            "view": {"type": "string", "enum": list(VIEWS)},
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
            "\n\nStage every beat listed above as ONE still picture (use its number as index):\n"
            "- framing: vary it within a scene (establish wide, then medium, close, insert); never the same "
            "framing three times in a row\n"
            "- picture: two concrete sentences a cinematographer can shoot: who is where, doing what, with what "
            "expression and body language, the setting with its few telling objects, the light source, and "
            "where the red accent is if it is in frame. Show the exact moment the words describe. Keep the "
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
            b.camera = "still"
            if b.index in host_beats:
                b.characters = [HOST]
                b.picture = (f"{HOST} sits by his lantern's red flame in the dark, looking at us as he speaks: "
                             f"\"{b.text[:160]}\"")
        names = [str(c.get("name")) for c in d["cast"] if c.get("name")] + [HOST]
        directed = self._direct(board, beats, scenes, names, host_beats)
        for b in beats:
            b.tier = "still"
        d.update(beats=beats, host_beats=host_beats)
        return {"beats": len(beats), "directed_by_model": directed, "host_beats": len(host_beats),
                "scares": sum(1 for b in beats if b.scare),
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
            by_index = {b.index: b for b in group}
            for item in result.get("beats", []) if isinstance(result, dict) else []:
                beat = by_index.get(item.get("index")) if isinstance(item, dict) else None
                if beat is None or not str(item.get("picture", "")).strip():
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
                beat.directed = True
                got += 1
        return got

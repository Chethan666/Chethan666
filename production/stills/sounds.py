"""The sound director's picks from the channel's sound library.

For every sound the video needs (a score cue per mood, a room tone per place, an effect per
beat that asks for one, the scare sting), the library is searched for a short list of
candidates. The planner model sees the needs with their candidates (title, description,
length, how often used) and picks one or none for each; without a model, the best candidate
is taken when it matches well enough. Picks cost nothing. Needs left open are generated as
before, and every generated sound is copied to `library_new/` so `library sync` can add it
to the library after the run.
"""

from __future__ import annotations

import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from story_factory.llm import LLMError

from ..library import Library, Sound
from ..paid import digest
from ..score import CUES, tension

log = logging.getLogger(__name__)

SYSTEM = ("You are the sound director of Hollow Lines, a horror storytelling channel. You choose music and "
          "sounds from the channel's library for one video. Pick what truly fits the moment; an honest \"none\" "
          "is better than a wrong sound. Answer with JSON only.")
SCHEMA = {"type": "object", "required": ["picks"], "properties": {"picks": {"type": "array", "items": {
    "type": "object", "required": ["need", "id"],
    "properties": {"need": {"type": "string"}, "id": {"type": "string"}}}}}}
MOOD_TEXT = {"unease": "quiet unease: sparse, eerie, patient", "dread": "building dread: tense, pulsing, creeping",
             "terror": "terror: intense, frantic, the climax"}


def needs(board_data: dict[str, Any], cue_names: list[str], max_effects: int) -> list[dict[str, Any]]:
    """What the video needs, each with the Score name it fills."""
    out: list[dict[str, Any]] = []
    scenes, beats = board_data["scenes"], board_data["beats"]
    moods_present = {tension(str(sc.get("mood", ""))) for sc in scenes} or {0}
    for k in cue_names:
        if k in CUES:
            out.append({"need": f"music_{k}", "kind": "music", "mood": k, "text": MOOD_TEXT.get(k, k),
                        "intensity": {"unease": 2, "dread": 3, "terror": 5}.get(k), "priority": k != "terror"
                        or 2 in moods_present})
    places: list[str] = []
    for b in beats:
        sc = scenes[b.scene] if b.scene < len(scenes) else {}
        place = str(sc.get("location", "")).strip()
        if place and place not in places and len(places) < 8:
            places.append(place)
            out.append({"need": f"room_{digest(place)}", "kind": "ambience", "mood": "",
                        "text": f"{place} night ambience room tone", "place": place})
    out.append({"need": "sting", "kind": "sting", "mood": "terror", "text": "horror jump scare stinger hit"})
    for b in [b for b in beats if b.sound][:max_effects]:
        out.append({"need": f"fx_{b.index:03d}", "kind": "sfx", "mood": "", "text": b.sound,
                    "moment": b.text[:120]})
    return out


def shortlist(library: Library, need: dict[str, Any], k: int = 5) -> list[tuple[Sound, float]]:
    return library.search(need["kind"], need["text"], need.get("mood", ""), k, need.get("intensity"))


def prompt(story_title: str, items: list[tuple[dict[str, Any], list[tuple[Sound, float]]]]) -> str:
    blocks = []
    for need, cands in items:
        head = f"NEED {need['need']} ({need['kind']}): {need['text'][:160]}"
        if need.get("moment"):
            head += f" | narrated: \"{need['moment']}\""
        lines = [f"  - {s.id}: {s.title[:70]} | {s.description[:110]} | {s.seconds:.0f} s"
                 f"{' | loops' if s.loop else ''} | used {s.uses}x" for s, _ in cands]
        blocks.append(head + "\n" + ("\n".join(lines) if lines else "  (no candidates)"))
    return (f"Video: {story_title}\n\n" + "\n\n".join(blocks) +
            "\n\nFor each NEED, pick the id of the candidate that fits best, or \"\" if none truly fits (a door "
            "creak is not a footstep; a cheerful track is never right). For music, prefer tracks without a "
            "strong melody, that sit quietly under a voice, and different tracks for different moods. Use only "
            "ids listed under that NEED. Return every NEED once.")


def pick(board: Any, library: Library, cue_names: list[str], max_effects: int,
         min_score: float = 0.55) -> tuple[dict[str, Sound], dict[str, Any]]:
    """The library sound chosen for each need (by Score name), and a report."""
    all_needs = needs(board.data, cue_names, max_effects)
    items = [(n, shortlist(library, n)) for n in all_needs]
    chosen: dict[str, Sound] = {}
    by_model = 0
    router = board.router
    if router is not None and any(c for _, c in items):
        try:
            result = router.one("planner").generate_json(
                "sound_pick", SYSTEM, prompt(str(board.story.get("title", "")), items), SCHEMA)
            allowed = {n["need"]: {s.id: s for s, _ in c} for n, c in items}
            for p in result.get("picks", []) if isinstance(result, dict) else []:
                if not isinstance(p, dict):
                    continue
                need, sid = str(p.get("need", "")), str(p.get("id", "")).strip()
                if sid and sid in allowed.get(need, {}):
                    chosen[need] = allowed[need][sid]
                    by_model += 1
        except (LLMError, ValueError, TypeError) as exc:
            log.warning("Sound picks by score alone: %s", exc)
    asked_model = by_model > 0
    for n, cands in items:
        if n["need"] in chosen or not cands or asked_model and n["kind"] not in ("music", "sting"):
            continue  # the model's "none" stands for effects; music and the sting always get the best match
        best, score = cands[0]
        good = score >= {"music": 0.9, "sting": 0.0}.get(n["kind"], min_score)  # any sting will do
        if good:
            chosen[n["need"]] = best
    report = {"library_sounds": len(library.sounds), "needs": len(all_needs), "picked": len(chosen),
              "picked_by_model": by_model,
              "picked_by_kind": {k: sum(1 for n, s in chosen.items() if s.kind == k)
                                 for k in ("music", "ambience", "sfx", "sting")},
              "picks": {n: s.id for n, s in chosen.items()}}
    return chosen, report


def fetch_all(library: Library, chosen: dict[str, Sound], parallel: int = 4) -> dict[str, Path]:
    def one(item: tuple[str, Sound]) -> tuple[str, Path | None]:
        return item[0], library.fetch(item[1])
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        return {n: p for n, p in pool.map(one, list(chosen.items())) if p is not None}


def save_generated(generated: list[dict[str, Any]], out: Path, story: dict[str, Any]) -> int:
    """Copy each paid-for sound with its catalogue facts into out/library_new/ for `library sync`."""
    if not generated:
        return 0
    folder = out / "library_new"
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for g in generated:
        path = Path(g["path"])
        if not path.exists():
            continue
        name = g["name"]
        kind = ("music" if name.startswith("music_") else "ambience" if name.startswith("room_")
                else "sting" if name == "sting" else "sfx")
        mood = name.removeprefix("music_") if kind == "music" else ""
        file = f"gen-{kind}-{digest(g['model'], g['text'])}{path.suffix or '.mp3'}"
        shutil.copyfile(path, folder / file)
        entries.append({"file": file, "kind": kind, "text": g["text"], "mood": mood, "loop": g.get("loop", False),
                        "model": g["model"], "story": story.get("id")})
    (folder / "entries.json").write_text(json.dumps(entries, indent=1))
    return len(entries)

"""The sound director's picks from the channel's sound library.

For every sound the video needs (a score cue per mood, a room tone per place, an effect per
beat that asks for one, the scare sting), the library is searched for a short list of
candidates. The planner model is the sound director: for each need it either picks a
candidate (free), or, when none satisfies it, asks for a new one in its own words (music is
made by ACE-Step 1.5 on fal.ai, other sounds by the sound-effects model), or leaves an effect
out. Without a model, the best candidate is taken when it matches well enough, and the rest is
made from the default descriptions. Every sound made is copied to `library_new/` so `library
sync` adds it to the library after the run: each is paid for once.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from story_factory.llm import LLMError

from ..library import Library, Sound
from ..paid import digest
from ..score import CUES, tension

log = logging.getLogger(__name__)

SYSTEM = ("You are the sound director of Hollow Lines, a horror storytelling channel. You decide the music "
          "and sounds of one video: take them from the channel's library when they truly fit, have new ones "
          "made when they don't, and leave an effect out when silence is scarier. Answer with JSON only.")
SCHEMA = {"type": "object", "required": ["picks"], "properties": {"picks": {"type": "array", "items": {
    "type": "object", "required": ["need", "id"],
    "properties": {"need": {"type": "string"}, "id": {"type": "string"}, "generate": {"type": "string"}}}}}}
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
    k = 8 if need["kind"] == "music" else k  # the score matters most: a wider choice
    return library.search(need["kind"], need["text"], need.get("mood", ""), k, need.get("intensity"))


def overview(library: Library | None) -> str:
    """What the whole library holds, so the director knows how much choice it has."""
    if library is None or not library.sounds:
        return "The channel's sound library is not available for this video."
    kinds: dict[str, int] = {}
    moods: dict[str, int] = {}
    for s in library.sounds:
        kinds[s.kind] = kinds.get(s.kind, 0) + 1
        if s.kind == "music":
            for m in s.moods:
                moods[m] = moods.get(m, 0) + 1
    return (f"The channel's sound library holds {len(library.sounds)} sounds: "
            + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))
            + (" (music by mood: " + ", ".join(f"{m} {n}" for m, n in sorted(moods.items())) + ")" if moods else "")
            + ". Under each NEED you see its best matches.")


def _line(s: Sound) -> str:
    facts = [s.title[:70], s.description[:110], f"{s.seconds:.0f} s"]
    if s.kind == "music":
        facts += [f"moods {'/'.join(s.moods) or '-'}", f"intensity {s.intensity}/5" if s.intensity else ""]
    facts += ["loops" if s.loop else "", f"used {s.uses}x"]
    return f"  - {s.id}: " + " | ".join(f for f in facts if f)


def prompt(story_title: str, items: list[tuple[dict[str, Any], list[tuple[Sound, float]]]],
           library: Library | None = None, can_make: bool = True) -> str:
    blocks = []
    for need, cands in items:
        head = f"NEED {need['need']} ({need['kind']}): {need['text'][:160]}"
        if need.get("moment"):
            head += f" | narrated: \"{need['moment']}\""
        lines = [_line(s) for s, _ in cands]
        blocks.append(head + "\n" + ("\n".join(lines) if lines else "  (no candidates)"))
    intro = f"Video: {story_title}\n{overview(library)}\n\n"
    if not can_make:
        return (intro + "\n\n".join(blocks) +
                "\n\nThis video uses the library only: nothing new can be made. For each NEED decide:\n"
                "1. Use a library sound: \"id\" = one of the ids listed under that NEED. A door creak is not a "
                "footstep; a cheerful or melodic track is never right under a horror story. For music, prefer "
                "tracks without a strong melody that sit quietly under a voice, an intensity that fits the mood, "
                "and different tracks for different moods; prefer sounds used less often.\n"
                "2. Leave it out: \"id\" = \"\" (an effect that does not fit is worse than silence; music, "
                "room tone and the sting always take the closest listed sound).\n"
                "Return every NEED once.")
    return (intro + "\n\n".join(blocks) +
            "\n\nFor each NEED decide one of three things:\n"
            "1. Use a library sound: \"id\" = one of the ids listed under that NEED (free). A door creak is not "
            "a footstep; a cheerful or melodic track is never right under a horror story. For music, prefer "
            "tracks without a strong melody that sit quietly under a voice, and different tracks for different "
            "moods.\n"
            "2. Have a new one made: \"id\" = \"\" and \"generate\" = a description of it. Music is made by "
            "ACE-Step 1.5 (about $0.05 a cue): describe genre, instruments, tempo and mood, always "
            "instrumental, e.g. \"dark ambient horror score, low cello drone, distant detuned piano, slow, "
            "sparse, instrumental\". Other sounds cost about $0.01: describe exactly what is heard.\n"
            "3. Leave it out (effects only): \"id\" = \"\" and \"generate\" = \"\".\n"
            "Prefer the library when a candidate truly fits; make a new one when it doesn't. Return every NEED "
            "once.")


@dataclass
class Decisions:
    chosen: dict[str, Sound] = field(default_factory=dict)  # need -> library sound (free)
    make: dict[str, str] = field(default_factory=dict)  # need -> the director's description of a new sound
    skip: set[str] = field(default_factory=set)  # effects the director left out
    report: dict[str, Any] = field(default_factory=dict)


def pick(board: Any, library: Library | None, cue_names: list[str], max_effects: int,
         min_score: float = 0.55, can_make: bool = True) -> Decisions:
    """For each need: a library sound, a new sound described by the director, or nothing."""
    all_needs = needs(board.data, cue_names, max_effects)
    items = [(n, shortlist(library, n) if library is not None else []) for n in all_needs]
    out = Decisions()
    decided: set[str] = set()
    router = board.router
    if router is not None:
        try:
            result = router.one("planner").generate_json(
                "sound_pick", SYSTEM, prompt(str(board.story.get("title", "")), items, library, can_make), SCHEMA)
            allowed = {n["need"]: {s.id: s for s, _ in c} for n, c in items}
            kinds = {n["need"]: n["kind"] for n, _ in items}
            for p in result.get("picks", []) if isinstance(result, dict) else []:
                if not isinstance(p, dict) or str(p.get("need", "")) not in kinds:
                    continue
                need, sid = str(p["need"]), str(p.get("id", "")).strip()
                text = str(p.get("generate") or "").strip() if can_make else ""
                if sid in allowed[need]:
                    out.chosen[need] = allowed[need][sid]
                elif sid:
                    continue  # an id not on its list: decided below as if unanswered
                elif text:
                    out.make[need] = text[:500]
                elif kinds[need] == "sfx":
                    out.skip.add(need)
                else:
                    continue  # music, room tone and the sting are never left out
                decided.add(need)
        except (LLMError, ValueError, TypeError) as exc:
            log.warning("Sound picks by score alone: %s", exc)
    for n, cands in items:
        if n["need"] in decided or not cands:
            continue  # undecided and nothing in the library: made from the default description
        best, score = cands[0]
        # Any sting will do; with nothing to be made, the closest music and room tone too.
        floor = {"music": 0.0 if not can_make else 0.9, "sting": 0.0,
                 "ambience": 0.0 if not can_make else min_score}.get(n["kind"], min_score)
        if score >= floor:
            out.chosen[n["need"]] = best
    out.report = {
        "library_sounds": len(library.sounds) if library is not None else 0, "needs": len(all_needs),
        "decided_by_model": len(decided), "from_library": len(out.chosen), "made_new": len(out.make),
        "left_out": len(out.skip),
        "from_library_by_kind": {k: sum(1 for s in out.chosen.values() if s.kind == k)
                                 for k in ("music", "ambience", "sfx", "sting")},
        "picks": {n: s.id for n, s in out.chosen.items()}, "made": out.make}
    return out


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

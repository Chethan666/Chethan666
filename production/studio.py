"""Studio mode: one approved story in, a finished, packaged video out.

    voices (ElevenLabs v3, every chunk timed)  ->  shot list (3-6 s shots, staged)
    ->  look (style frame, character sheets, a checked picture per shot)
    ->  clips (hero and medium tiers, checked)  ->  sound design (score, room tone, effects, stingers)
    ->  edit (cold open, title, hard cuts, end screen)  ->  thumbnails, Shorts, review kit

Every paid step has a free fallback (Kokoro, the code renderer, living stills, synthesized
sound), and one spending cap covers the whole video, most important work first.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile

from . import dialogue, director, draw, edit, package, shots as shotlist, timing, voice
from .clips import Clips
from .look import Look
from .paid import Budget, Studio
from .run import STORYTELLER_LOOK, _json
from .score import Score
from .speech import ElevenVoice

log = logging.getLogger(__name__)

CHANNEL = "HOLLOW LINES"


def voices(studio: Studio | None, cfg: dict[str, Any], kokoro: Any, router: Any, narration: str,
           cast: list[dict[str, Any]], work: Path) -> tuple[Any, dict[str, Any], list[Any], dict[str, str]]:
    """The narrator and each speaking character's voice (ElevenLabs, with Kokoro behind each)."""
    narrator = kokoro
    if studio is not None and "voice" in studio.models:
        narrator = ElevenVoice(studio, cfg.get("narrator_voice", "Brian"), kokoro, work / "voice",
                               int(cfg.get("parallel", 4)))
    lines, kinds = dialogue.attribute(router, narration, cast)
    kokoro_names = voice.pick_voices(kinds, {k: list(v) for k, v in voice.KOKORO_CAST.items()},
                                     str(getattr(kokoro, "voice", "")))
    if isinstance(narrator, ElevenVoice):
        names = voice.pick_voices(kinds, {k: list(v) for k, v in (cfg.get("cast_voices") or {}).items()},
                                  narrator.voice)
        cast_voices = {who: narrator.speaker(name, kokoro.speaker(kokoro_names.get(who, "am_fenrir")))
                       for who, name in names.items()}
    else:
        names = kokoro_names
        cast_voices = {who: kokoro.speaker(name) for who, name in names.items()}
    return narrator, cast_voices, lines, names


def produce(story: dict[str, Any], out_dir: Path, kokoro: Any, router: Any, cfg: dict[str, Any],
            fal: Any = None, conn: Any = None, keep_work: bool = False) -> dict[str, Any]:
    started = time.time()
    out_dir = out_dir.resolve()
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)

    narration = str(story.get("narration") or "").strip()
    scenes = _json(story.get("scenes"), [])
    if not narration or not scenes:
        raise ValueError(f"story #{story.get('id')} has no narration or scene plan")
    cast = _json(story.get("cast"), [])
    bible = _json(story.get("bible"), {})
    packaging = _json(story.get("packaging"), {})
    red = str(bible.get("red_accent", "")).strip()
    budget = Budget(float(cfg.get("max_usd", 0.0)), cfg.get("prices", {}), conn)
    studio = Studio(fal, budget, work / "cache", cfg) if fal is not None else None
    parallel = int(cfg.get("parallel", 4))

    # 1. Voices, timed chunk by chunk.
    narrator, cast_voices, lines, voice_names = voices(studio, cfg, kokoro, router, narration, cast, work)
    spans = timing.scene_spans(narration, scenes)
    audio, timeline = voice.narrate(narrator, narration, _json(story.get("pauses"), []),
                                    _json(story.get("pronunciations"), []), scenes, spans, lines, cast_voices)
    rate = narrator.sample_rate
    story_seconds = timeline.duration
    log.info("Narration: %.1f s in %d chunks", story_seconds, len(timeline.chunks))

    # 2. The shot list.
    shots = shotlist.cut(timeline.chunks, shotlist.scene_index(timeline.chunks, spans), story_seconds,
                         float(cfg.get("shot_min_seconds", 3.0)), float(cfg.get("shot_max_seconds", 6.5)))
    directed = shotlist.direct(router, shots, scenes, cast, red)
    shotlist.assign_tiers(shots, scenes, int(cfg.get("hero_shots", 20)), int(cfg.get("medium_shots", 40)))
    log.info("Shot list: %d shots (%d directed by the model)", len(shots), directed)

    # 3. Pictures.
    cast_names = [str(c.get("name")) for c in cast if c.get("name")]
    looks = {str(c.get("name")): director.features(str(c.get("name")), str(c.get("look", ""))) for c in cast}
    looks.setdefault("The Storyteller", director.features("The Storyteller", STORYTELLER_LOOK))
    look = None
    if studio is not None:
        look = Look(studio, work / "look", list(cast), red, parallel)
        look.extra_cast(shots)
        look.references(scenes)
        pictures = look.pictures(shots, scenes, cast_names, looks)
    else:
        pictures = {}
        paper, grain = draw.make_paper(1920, 1080)
        for s in shots:
            spec = director.guess(scenes[s.scene] if s.scene < len(scenes) else {}, cast_names)
            path = work / f"shot_{s.index:03d}.png"
            draw.scene_frames(spec, looks, (1920, 1080), paper, grain, seed=s.index, variants=1)[0].save(path)
            pictures[s.index] = path

    # 4. Clips.
    clipper = Clips(studio, cfg, work / "clips", parallel) if studio is not None else None
    clips = clipper.run(shots, pictures) if clipper is not None else {}

    # 5. The programme: cold open, title, story, end screen.
    cold = edit.pick_cold_open(shots, scenes) if cfg.get("cold_open", True) else []
    cold_a = shots[cold[0]].t0 if cold else 0.0
    cold_b = shots[cold[-1]].t1 if cold else 0.0
    cold_seconds = cold_b - cold_a
    title_seconds = float(cfg.get("title_seconds", 2.0))
    end_seconds = float(cfg.get("end_screen_seconds", 20.0))
    offset = cold_seconds + title_seconds
    total = offset + story_seconds + end_seconds

    track = np.zeros(int(round(total * rate)) + rate, dtype=np.float32)
    if cold:
        part = audio[int(cold_a * rate): int(cold_b * rate)]
        track[: part.size] = part
    track[int(offset * rate): int(offset * rate) + audio.size] = audio
    track = track[: int(round(total * rate))]
    speech = [(c.t0 + offset, c.t1 + offset) for c in timeline.chunks]
    speech += [(c.t0 - cold_a, c.t1 - cold_a) for c in timeline.chunks if c.t0 >= cold_a and c.t1 <= cold_b + 0.01]

    # 6. Sound design.
    score = Score(studio, work / "sound", rate, parallel)
    mixed = score.mix(track, speech, shots, offset, scenes, clips,
                      stings=[cold_seconds] if title_seconds and cold else [], music_until=total)
    soundfile.write(work / "mix.wav", mixed, rate)
    soundfile.write(out_dir / "narration.wav", audio, rate)

    # 7. Captions (programme times).
    cues = timing.caption_cues(timeline)
    prog_cues = [timing.Cue(c.start - cold_a, c.end - cold_a, c.text) for c in cues
                 if cold and c.start >= cold_a and c.end <= cold_b + 0.01]
    prog_cues += [timing.Cue(c.start + offset, c.end + offset, c.text) for c in cues]
    (out_dir / "captions.srt").write_text(timing.srt(prog_cues))
    burn = None
    if cfg.get("burn_captions"):
        burn = work / "captions.ass"
        burn.write_text(timing.ass(prog_cues, 1920, 1080))

    # 8. Cards and the edit.
    titles = packaging.get("titles") or [story.get("title", "")]
    style_frame = look.style_frame if look is not None else None
    title_card = edit.card(style_frame, [(CHANNEL, 40, draw.RED)] +
                           [(line, 92, draw.CHALK) for line in edit.wrap(str(titles[0]), 24)],
                           work / "title_card.png")
    end_line = str(packaging.get("end_screen_line", "")) or "Another story is waiting."
    end_card = edit.card(style_frame, [(line, 60, draw.CHALK) for line in edit.wrap(end_line, 30)] +
                         [("", 30, draw.CHALK), ("NEXT STORY  >", 54, draw.RED)], work / "end_card.png",
                         left=True, dim=0.25)
    segments = edit.programme(shots, pictures, clips, cold, cold_seconds, title_seconds, end_seconds,
                              story_seconds, title_card, end_card)
    video = edit.build(work, segments, work / "mix.wav", burn, out_dir / "video.mp4",
                       float(cfg.get("loudness_lufs", -14.0)), workers=3)

    # 9. Packaging and the review kit.
    key = next((s for s in shots if s.scare), None) or max(shots, key=lambda s: s.importance)
    refs = ([style_frame] if style_frame else []) + [pictures[key.index]] + (list(look.sheets.values())[:2] if look else [])
    thumbs = package.thumbnails(studio, out_dir, packaging, red, refs, pictures[key.index], int(cfg.get("thumbnails", 3)))
    windows = package.short_windows(shots, scenes, offset, int(cfg.get("shorts", 3)), 25.0,
                                    float(cfg.get("short_max_seconds", 55.0)))
    shorts_made = package.shorts(video, windows, prog_cues, str(titles[0]), out_dir, work)
    package.contact_sheet(shots, pictures, clips, out_dir / "contact_sheet.jpg")
    package.preview(video, out_dir / "preview_60s.mp4")
    chapters = shotlist.chapters(router, scenes, shots, offset)

    description = str(packaging.get("description", ""))
    if chapters:
        description += "\n\n" + "\n".join(f"{shotlist.stamp(t)} {name}" for t, name in chapters)
    metadata = {
        "story_id": story.get("id"), "title": titles[0], "other_titles": titles[1:],
        "description": description, "tags": packaging.get("tags", []),
        "pinned_comment": packaging.get("pinned_comment", "") or story.get("comment_question", ""),
        "end_screen_line": end_line, "credit": story.get("description_credit", ""), "made_for_kids": False,
        "duration_seconds": round(total, 1), "chapters": [[round(t, 1), n] for t, n in chapters],
        "thumbnails": [p.name for p in thumbs], "shorts": [p.name for p in shorts_made],
        "shorts_windows": [[round(a, 1), round(b, 1)] for a, b in windows],
        "end_screen_starts": round(offset + story_seconds, 1),
        # YouTube's label for realistic altered or synthetic media: not needed for a stylised
        # chalk animation, recorded so the uploader states it explicitly.
        "synthetic_media": False,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    (out_dir / "shots.json").write_text(shotlist.save(shots))
    by_tier = {t: sum(1 for s in shots if s.tier == t) for t in ("hero", "medium", "still")}
    summary = {
        "mode": "studio", "video": str(video), "seconds": round(total, 1), "story_seconds": round(story_seconds, 1),
        "scenes": len(scenes), "shots": len(shots), "directed_shots": directed, "tiers": by_tier,
        "clips": len(clips), "clips_by_model": clipper.report["made"] if clipper else {},
        "clips_rejected": clipper.report["rejected"][:20] if clipper else [],
        "pictures": look.report if look else {"drawn_in_code": len(shots)},
        "voice": {"engine": "elevenlabs-v3" if isinstance(narrator, ElevenVoice) else "kokoro",
                  "narrator": getattr(narrator, "voice", ""), "cast": voice_names,
                  "lines_fell_back": narrator.shared["failed"] if isinstance(narrator, ElevenVoice) else 0},
        "sound": score.report, "cold_open_shots": cold, "chapters": len(chapters),
        "thumbnails": len(thumbs), "shorts": len(shorts_made),
        "usd": round(budget.spent, 2), "usd_by_task": {k: round(v, 2) for k, v in sorted(budget.by_task.items())},
        "review": "pending", "render_minutes": round((time.time() - started) / 60, 1),
    }
    (out_dir / "report.json").write_text(json.dumps(summary, indent=2))
    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)
    log.info("Made %s", summary)
    return summary

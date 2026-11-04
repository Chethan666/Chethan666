"""One story in, one finished video out."""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import soundfile

from . import assemble, dialogue, director, draw, sound, timing, voice

log = logging.getLogger(__name__)

SIZES = {"1080": (1920, 1080), "720": (1280, 720), "360": (640, 360)}
STORYTELLER_LOOK = "Tall stick figure in a long coat, carrying a lantern with a red flame"


def _json(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def scene_times(timeline: timing.Timeline, spans: list[tuple[int, int]]) -> list[tuple[float, float]]:
    starts = [0.0] + [timeline.time_at(start) for start, _ in spans[1:]]
    ends = starts[1:] + [timeline.duration]
    return [(a, max(b, a + 0.5)) for a, b in zip(starts, ends)]


def produce(story: dict[str, Any], out_dir: Path, narrator: voice.Voice, router: Any = None,
            resolution: str = "1080", keep_work: bool = False, options: dict[str, Any] | None = None,
            animator: Any = None) -> dict[str, Any]:
    """Make the video for one story. Returns a summary of what was made.
    `options` is the config's [video] section; `animator` turns chosen scenes into moving clips."""
    options = options or {}
    started = time.time()
    out_dir = out_dir.resolve()  # FFmpeg runs inside the work folder: every path must be absolute
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "work"
    work.mkdir(exist_ok=True)
    size = SIZES[resolution]

    narration = str(story.get("narration") or "").strip()
    scenes = _json(story.get("scenes"), [])
    if not narration or not scenes:
        raise ValueError(f"story #{story.get('id')} has no narration or scene plan")
    cast = _json(story.get("cast"), [])
    bible = _json(story.get("bible"), {})
    packaging = _json(story.get("packaging"), {})
    pauses = _json(story.get("pauses"), [])
    pronunciations = _json(story.get("pronunciations"), [])

    # 1. Voices, timed chunk by chunk: the narrator, and each character for their own lines.
    spans = timing.scene_spans(narration, scenes)
    lines, cast_voices, voice_names = [], {}, {}
    if options.get("character_voices", True) and hasattr(narrator, "speaker"):
        lines, kinds = dialogue.attribute(router, narration, cast)
        pools = {k: list(v) for k, v in (options.get("cast_voices") or voice.KOKORO_CAST).items()}
        voice_names = voice.pick_voices(kinds, pools, str(getattr(narrator, "voice", options.get("voice", ""))))
        cast_voices = {who: narrator.speaker(name) for who, name in voice_names.items()}
        log.info("Dialogue: %d quoted lines; voices %s", len(lines), voice_names)
    audio, timeline = voice.narrate(narrator, narration, pauses, pronunciations, scenes, spans, lines, cast_voices)
    times = scene_times(timeline, spans)
    log.info("Narration: %.1f s in %d chunks", timeline.duration, len(timeline.chunks))

    # 2. Art direction and pictures.
    looks = {str(c.get("name")): director.features(str(c.get("name")), str(c.get("look", ""))) for c in cast}
    looks.setdefault("The Storyteller", director.features("The Storyteller", STORYTELLER_LOOK))
    direction, fallbacks = director.direct(router, scenes, cast, str(bible.get("red_accent", "")))
    paper, grain = draw.make_paper(*size)
    for i, spec in enumerate(direction):
        for v, frame in enumerate(draw.scene_frames(spec, looks, size, paper, grain, seed=i)):
            frame.save(work / f"scene_{i:03d}_{v}.png")
    log.info("Drew %d scenes (%d directed by keywords)", len(direction), fallbacks)
    moving = animator.run(work, scenes, direction) if animator is not None else {}

    # 3. Sound and captions.
    rate = narrator.sample_rate
    mixed = sound.mix(audio, rate, scenes, times, [(c.t0, c.t1) for c in timeline.chunks])
    soundfile.write(work / "mix.wav", mixed, rate)
    soundfile.write(out_dir / "narration.wav", audio, rate)
    cues = timing.caption_cues(timeline)
    (out_dir / "captions.srt").write_text(timing.srt(cues))
    (work / "captions.ass").write_text(timing.ass(cues, *size))

    # 4. Video.
    durations = [b - a for a, b in times]
    assemble.assemble(work, direction, durations, size, "mix.wav", "captions.ass", "video.mp4", moving=moving)
    shutil.move(str(work / "video.mp4"), out_dir / "video.mp4")

    # 5. Thumbnail and upload details.
    key = next((i for i, s in enumerate(scenes) if s.get("key_moment")), len(scenes) // 2)
    picture = draw.scene_frames(direction[key], looks, size, paper, grain, seed=key, variants=1)[0]
    draw.thumbnail(picture, str(packaging.get("thumbnail_text", ""))).save(out_dir / "thumbnail.png")
    titles = packaging.get("titles") or [story.get("title", "")]
    metadata = {
        "story_id": story.get("id"), "title": titles[0], "other_titles": titles[1:],
        "description": packaging.get("description", ""), "tags": packaging.get("tags", []),
        "pinned_comment": packaging.get("pinned_comment", "") or story.get("comment_question", ""),
        "end_screen_line": packaging.get("end_screen_line", ""),
        "credit": story.get("description_credit", ""), "made_for_kids": False,
        "duration_seconds": round(timeline.duration, 1),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    (out_dir / "direction.json").write_text(json.dumps(direction, indent=2, ensure_ascii=False))
    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)
    summary = {"video": str(out_dir / "video.mp4"), "seconds": round(timeline.duration, 1), "scenes": len(scenes),
               "chunks": len(timeline.chunks), "directed_by_keywords": fallbacks,
               "quoted_lines": len(lines), "character_voices": voice_names,
               "moving_scenes": sorted(moving), "motion_usd": round(getattr(animator, "spent", 0.0), 2),
               "render_minutes": round((time.time() - started) / 60, 1)}
    log.info("Made %s", summary)
    return summary


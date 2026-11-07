"""A 40-60 s chalk-horror piece made entirely with open models on a rented Vast.ai GPU.

The excerpt is the stretch of a story that builds up to its first key scare (never the
ending), so it plays as a self-contained teaser that ends on the hook. Narration (Kokoro)
and the shot list (the open planner model) are made here; the GPU draws every shot from a
style frame and character sheets (Qwen-Image-Edit-2509), animates every shot (Wan 2.2 I2V
A14B, 720p), and makes foley from each clip plus room tone, score beds and a stinger
(MMAudio). The edit, captions, cards, a vertical version and the review kit are made here.
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

from drama_production.media import MediaError

from . import director, draw, edit, package, shots as shotlist, timing, voice
from .clips import frames as clip_frames, motion_prompt
from .look import FRAMING_TEXT, chalk_check, fit, style
from .paid import digest
from .run import STORYTELLER_LOOK, _json
from .score import CUES, STING, Score
from .studio import CHANNEL

log = logging.getLogger(__name__)

PICTURE_MODEL = "Qwen/Qwen-Image-Edit-2509"
CLIP_MODEL = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"


def excerpt(scenes: list[dict[str, Any]], words: tuple[int, int] = (75, 118)) -> list[int]:
    """Consecutive scenes ending on the first key moment that is not the last scene, reaching back
    until the excerpt is long enough (about 40-60 s read aloud)."""
    if not scenes:
        return []
    count = [len(str(s.get("text", "")).split()) for s in scenes]
    last = len(scenes) - 1
    keys = [i for i, s in enumerate(scenes) if s.get("key_moment") and i != last] or [min(last, len(scenes) // 2)]
    best: list[int] = []
    for key in keys:
        chosen = [key]
        total = count[key]
        i = key - 1
        while i >= 0 and total + count[i] <= words[1]:
            chosen.insert(0, i)
            total += count[i]
            i -= 1
        if total < words[0] and key + 1 < last and total + count[key + 1] <= words[1] + 10:
            chosen.append(key + 1)
            total += count[key + 1]
        if words[0] <= total <= words[1] + 10:
            return chosen
        if not best or abs(sum(count[j] for j in chosen) - sum(words) / 2) < abs(
                sum(count[j] for j in best) - sum(words) / 2):
            best = chosen
    return best


class Provided:
    """Stands in for the paid studio inside the sound designer: hands over the sounds the GPU
    made, by name, and reports a miss so the synthesized fallback plays instead."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.models = {"music": "mmaudio", "sfx": "mmaudio", "foley": "mmaudio"}

    def call(self, role: str, arguments: dict[str, Any], kind: str, out: Path, task: str, units: float = 1.0,
             key: Any = None, tries: int = 2) -> Path:
        src = self.folder / f"{out.stem}.flac"
        if not src.exists():
            raise MediaError(f"{out.stem}: not made on the GPU")
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, out)
        return out


def build_job(shots: list[shotlist.Shot], scenes: list[dict[str, Any]], cast: list[dict[str, Any]], red: str,
              cfg: dict[str, Any]) -> dict[str, Any]:
    names = sorted({n for s in shots for n in s.characters})
    looks = {str(c.get("name")): str(c.get("look", "")) for c in cast}
    sheets = []
    for k, name in enumerate(names[:5]):
        first = next(s for s in shots if name in s.characters)
        look = looks.get(name) or f"as in this moment: {first.picture[:200]}"
        sheets.append({"name": name, "file": f"{k}", "prompt": (
            f"{style(red)} Using exactly the chalk drawing style of image 1 (not its content), draw a character "
            f"sheet of one stick figure, {name}: {look}. The same figure three times side by side on the black "
            "chalkboard: facing front, in profile, and walking. Bold, simple distinguishing details.")})
    sheet_names = {s["name"] for s in sheets}
    place = next((str(s.get("location")) for s in scenes if s.get("location")), "a lonely place at night")
    job: dict[str, Any] = {
        "picture_model": cfg.get("picture_model", PICTURE_MODEL), "clip_model": cfg.get("clip_model", CLIP_MODEL),
        "picture_steps": int(cfg.get("picture_steps", 40)), "picture_tries": int(cfg.get("picture_tries", 3)),
        "clip_steps": int(cfg.get("clip_steps", 30)), "clip_size": cfg.get("clip_size", [1280, 720]),
        "style_frame_prompt": (f"Turn this blank black chalkboard into a chalk drawing. {style(red)} A wide "
                               f"establishing view of {place}. No figures in this picture."),
        "sheets": sheets, "shots": [], "audio": [],
    }
    for s in shots:
        chars = [n for n in s.characters if n in sheet_names][:2]
        refs = "Image 1 shows the drawing style only: copy its chalk look, not its content. " + " ".join(
            f"{{sheet:{n}}} is the character sheet of {n}: draw {n} exactly like it." for n in chars)
        empty = "" if s.characters else " No people or stick figures anywhere in this shot."
        job["shots"].append({
            "index": s.index, "characters": chars, "frames": 81,
            "picture_prompt": f"{style(red)} {refs} Shot: {FRAMING_TEXT.get(s.framing, s.framing)}. {s.picture}{empty}",
            "motion_prompt": motion_prompt(s),
        })
        job["audio"].append({"name": f"foley_{s.index:03d}", "video": f"clip_{s.index:03d}.mp4", "duration": 5,
                             "prompt": f"{s.sound or s.motion}. Quiet, eerie, close foley, night."})
    for key, prompt in CUES.items():
        job["audio"].append({"name": f"music_{key}", "duration": 10, "prompt": prompt, "negative": "speech, voices"})
    for p in sorted({str(sc.get("location", "")).strip() for sc in scenes if sc.get("location")})[:6]:
        job["audio"].append({"name": f"room_{digest(p)}", "duration": 10,
                             "prompt": f"Night ambience of {p}: quiet, eerie room tone, no music, no voices."})
    job["audio"].append({"name": "sting", "duration": 3, "prompt": STING, "negative": "speech, voices"})
    job["clip_order"] = [s.index for s in sorted(shots, key=lambda s: (-(s.importance + 2 * s.scare), s.index))]
    return job


def produce(story: dict[str, Any], out_dir: Path, kokoro: Any, router: Any, cfg: dict[str, Any], vast: Any,
            worker: Path, conn: Any = None, run_job=None, keep_work: bool = False) -> dict[str, Any]:
    from . import vast as vastmod

    started = time.time()
    run_job = run_job or vastmod.run_job
    out_dir = out_dir.resolve()
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    scenes_all = _json(story.get("scenes"), [])
    cast = list(_json(story.get("cast"), []))
    bible = _json(story.get("bible"), {})
    packaging = _json(story.get("packaging"), {})
    red = str(bible.get("red_accent", "")).strip()

    # 1. The excerpt, read aloud.
    picked = excerpt(scenes_all)
    if not picked:
        raise ValueError(f"story #{story.get('id')} has no scene plan")
    scenes = [scenes_all[i] for i in picked]
    narration = "\n\n".join(str(s.get("text", "")).strip() for s in scenes)
    spans = timing.scene_spans(narration, scenes)
    lines, kinds = [], {"narrator": "narrator"}
    try:
        from . import dialogue
        lines, kinds = dialogue.attribute(router, narration, cast)
    except Exception as exc:  # noqa: BLE001 - voices for lines are a bonus
        log.warning("Dialogue attribution skipped: %s", exc)
    names = voice.pick_voices(kinds, {k: list(v) for k, v in voice.KOKORO_CAST.items()}, str(getattr(kokoro, "voice", "")))
    cast_voices = {who: kokoro.speaker(name) for who, name in names.items()}
    audio, timeline = voice.narrate(kokoro, narration, _json(story.get("pauses"), []),
                                    _json(story.get("pronunciations"), []), scenes, spans, lines, cast_voices)
    rate = kokoro.sample_rate
    story_seconds = timeline.duration
    log.info("Excerpt: scenes %s, %.1f s", picked, story_seconds)

    # 2. Shots: every one becomes a clip.
    shots = shotlist.cut(timeline.chunks, shotlist.scene_index(timeline.chunks, spans), story_seconds,
                         float(cfg.get("shot_min_seconds", 3.0)), float(cfg.get("shot_max_seconds", 6.0)))
    directed = shotlist.direct(router, shots, scenes, cast, red)
    for s in shots:
        s.tier = "hero"

    # 3. The GPU.
    job = build_job(shots, scenes, cast, red, cfg)
    for s in shots:
        s.sound = ""  # each clip's own foley carries its sound
    gpu = run_job(vast, job, worker, work / "gpu", float(cfg.get("max_usd", 12.0)),
                  max_hours=float(cfg.get("max_hours", 3.4)))
    made = work / "gpu" / "out"

    # 4. Pictures and clips, checked; any gap filled by a neighbour or, last, the code renderer.
    pictures: dict[int, Path] = {}
    for s in shots:
        src = made / f"shot_{s.index:03d}.png"
        if src.exists():
            pictures[s.index] = fit(src, work / f"pic_{s.index:03d}.png")
    cast_names = [str(c.get("name")) for c in cast if c.get("name")]
    looks = {str(c.get("name")): director.features(str(c.get("name")), str(c.get("look", ""))) for c in cast}
    looks.setdefault("The Storyteller", director.features("The Storyteller", STORYTELLER_LOOK))
    drawn = 0
    for s in shots:
        if s.index in pictures:
            continue
        near = sorted((i for i in pictures), key=lambda i: abs(i - s.index))
        if near:
            pictures[s.index] = pictures[near[0]]
            continue
        paper, grain = draw.make_paper(1920, 1080)
        path = work / f"pic_{s.index:03d}.png"
        draw.scene_frames(director.guess(scenes[s.scene], cast_names), looks, (1920, 1080), paper, grain,
                          seed=s.index, variants=1)[0].save(path)
        pictures[s.index] = path
        drawn += 1
    clips: dict[int, Path] = {}
    rejected = []
    for s in shots:
        clip = made / f"clip_{s.index:03d}.mp4"
        if not clip.exists():
            continue
        checks = clip_frames(clip, [0.3, 0.65, 0.95], work / f"check_{s.index:03d}")
        bad = [why for ok, why in map(chalk_check, checks) if not ok]
        if bad or not checks:
            rejected.append(f"shot {s.index}: {bad[0] if bad else 'unreadable'}")
            continue
        clips[s.index] = clip

    # 5. Sound and captions.
    title_seconds = float(cfg.get("title_seconds", 2.0))
    end_seconds = float(cfg.get("end_seconds", 5.0))
    offset = title_seconds
    total = offset + story_seconds + end_seconds
    track = np.zeros(int(round(total * rate)), dtype=np.float32)
    track[int(offset * rate): int(offset * rate) + audio.size] = audio[: track.size - int(offset * rate)]
    speech = [(c.t0 + offset, c.t1 + offset) for c in timeline.chunks]
    score = Score(Provided(made), work / "sound", rate)
    mixed = score.mix(track, speech, shots, offset, scenes, clips, stings=[title_seconds], music_until=total)
    soundfile.write(work / "mix.wav", mixed, rate)
    cues = [timing.Cue(c.start + offset, c.end + offset, c.text) for c in timing.caption_cues(timeline)]
    (out_dir / "captions.srt").write_text(timing.srt(cues))
    burn = None
    if cfg.get("burn_captions", True):
        burn = work / "captions.ass"
        burn.write_text(timing.ass(cues, 1920, 1080, size_ratio=0.046, margin_ratio=0.06))

    # 6. The edit.
    titles = packaging.get("titles") or [story.get("title", "")]
    style_frame = made / "style_frame.png" if (made / "style_frame.png").exists() else None
    title_card = edit.card(style_frame, [(CHANNEL, 40, draw.RED)] +
                           [(line, 92, draw.CHALK) for line in edit.wrap(str(titles[0]), 24)], work / "title_card.png")
    end_card = edit.card(style_frame, [("WHAT HAPPENED NEXT?", 76, draw.CHALK), ("", 30, draw.CHALK),
                                       ("THE FULL STORY IS ON THE CHANNEL", 46, draw.RED)], work / "end_card.png",
                         dim=0.25)
    segments = edit.programme(shots, pictures, clips, [], 0.0, title_seconds, end_seconds, story_seconds,
                              title_card, end_card)
    video = edit.build(work, segments, work / "mix.wav", burn, out_dir / "video.mp4",
                       float(cfg.get("loudness_lufs", -14.0)), workers=3)
    vertical = package.shorts(video, [(0.0, total)], cues, str(titles[0]), out_dir, work)
    package.contact_sheet(shots, pictures, clips, out_dir / "contact_sheet.jpg")
    for name in ("style_frame.png",):
        if (made / name).exists():
            shutil.copy(made / name, out_dir / name)
    for sheet in made.glob("sheet_*.png"):
        shutil.copy(sheet, out_dir / sheet.name)

    summary = {
        "mode": "vast", "video": str(video), "vertical": [p.name for p in vertical], "seconds": round(total, 1),
        "story": story.get("id"), "scenes": picked, "shots": len(shots), "directed_shots": directed,
        "pictures_from_gpu": sum(1 for s in shots if (made / f"shot_{s.index:03d}.png").exists()),
        "pictures_drawn_in_code": drawn, "clips": len(clips), "clips_rejected": rejected,
        "sound": score.report, "gpu": {k: v for k, v in gpu.items() if k != "log"},
        "usd": gpu.get("usd", 0.0), "render_minutes": round((time.time() - started) / 60, 1),
        "models": {"pictures": job["picture_model"], "clips": job["clip_model"], "sound": "MMAudio",
                   "voice": "Kokoro"},
    }
    (out_dir / "report.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "gpu.log").write_text(str(gpu.get("log", "")))
    (out_dir / "shots.json").write_text(shotlist.save(shots))
    if conn is not None and gpu.get("usd"):
        from story_factory import db
        db.record_cost(conn, pipeline="video", task="vast_gpu", model=f"{gpu.get('gpu')}", input_units=0,
                       output_units=len(clips), usd=float(gpu["usd"]))
    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)
    return summary

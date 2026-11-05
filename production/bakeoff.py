"""The bake-off: the same shots through each candidate model, side by side, before a whole video.

Pictures: four of the story's most important shots, each drawn by every picture model (from
the same style frame and character sheets). Clips: the two most important shots, each through
every motion model at 5 s. Voices: the opening of the narration in two ElevenLabs voices and
in Kokoro. Sound: one music cue and a stinger. Everything is checked and costed, and lands in
one folder with a comparison sheet, so the defaults are chosen from real output.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import soundfile
from PIL import Image, ImageDraw

from drama_production.media import MediaError, data_uri

from . import draw, shots as shotlist, timing, voice
from .clips import frames, motion_prompt
from .look import Look, chalk_check, fit
from .paid import Budget, OverBudget, Studio, vision_check
from .run import _json
from .score import CUES, STING
from .speech import ElevenVoice, decode

log = logging.getLogger(__name__)

PICTURE_ROLES = ("shot", "shot_alt", "shot_cheap")
CLIP_ROLES = ("hero", "hero_alt", "medium")


def run(story: dict[str, Any], out: Path, kokoro: Any, router: Any, cfg: dict[str, Any], fal: Any,
        conn: Any = None, voices: tuple[str, ...] = ("Brian", "George")) -> dict[str, Any]:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    budget = Budget(float(cfg.get("max_usd", 10.0)), cfg.get("prices", {}), conn)
    studio = Studio(fal, budget, work / "cache", cfg)
    narration = str(story.get("narration") or "")
    scenes = _json(story.get("scenes"), [])
    cast = _json(story.get("cast"), [])
    red = str(_json(story.get("bible"), {}).get("red_accent", ""))
    report: dict[str, Any] = {"story": story.get("id"), "pictures": {}, "clips": {}, "voices": {}, "sound": {}}

    # Timing from a silent read (free): the shots are cut exactly as in a real video.
    spans = timing.scene_spans(narration, scenes)
    _, timeline = voice.narrate(voice.SilentVoice(), narration, [], [], scenes, spans)
    shots = shotlist.cut(timeline.chunks, shotlist.scene_index(timeline.chunks, spans), timeline.duration)
    key = next((i for i, s in enumerate(scenes) if s.get("key_moment")), len(scenes) // 2)
    wanted = {key - 1, key, key + 1}
    near = [s for s in shots if s.scene in wanted]
    shotlist.direct(router, near, scenes, cast, red)
    best = sorted(near, key=lambda s: (-(s.importance + 2 * s.scare), s.index))[:4]

    # Pictures.
    look = Look(studio, work / "look", cast, red, parallel=4)
    look.references(scenes)
    for name, path in [("style_frame", look.style_frame)] + [(f"sheet {n}", p) for n, p in look.sheets.items()]:
        if path:
            shutil.copy(path, out / (name.replace(" ", "_") + ".png"))
    images = ([data_uri(look.style_frame)] if look.style_frame else [])
    pictures: dict[tuple[int, str], Path] = {}
    for s in best:
        names = [n for n in s.characters if n in look.sheets][:3]
        refs = images + [data_uri(look.sheets[n]) for n in names]
        prompt = look.prompt(s, names)
        for role in PICTURE_ROLES:
            if role not in studio.models:
                continue
            raw = work / f"raw_{s.index}_{role}.png"
            path = out / f"picture_shot{s.index:03d}_{role}.png"
            entry: dict[str, Any] = {"model": studio.models[role], "picture": s.picture}
            try:
                studio.call(role if refs else "sheet", {"prompt": prompt, **({"image_urls": refs} if refs else {})},
                            "image", raw, f"bakeoff_{role}")
                fit(raw, path)
                entry["chalk_check"] = chalk_check(path)
                entry["vision_check"] = vision_check(studio, path, s.picture)
                pictures[(s.index, role)] = path
            except (MediaError, OverBudget) as exc:
                entry["error"] = str(exc)[:300]
            report["pictures"][f"shot {s.index} {role}"] = entry
    sheet(best, pictures, out / "compare_pictures.jpg")

    # Clips: the two most important shots through every motion model.
    for s in best[:2]:
        picture = pictures.get((s.index, "shot")) or next((p for (i, _), p in pictures.items() if i == s.index), None)
        if picture is None:
            continue
        for role in CLIP_ROLES:
            if role not in studio.models:
                continue
            image_key = "start_image_url" if "kling" in studio.models[role] else "image_url"
            path = out / f"clip_shot{s.index:03d}_{role}.mp4"
            entry = {"model": studio.models[role], "motion": s.motion}
            try:
                studio.call(role, {image_key: data_uri(picture), "prompt": motion_prompt(s), "duration": "5"},
                            "video", path, f"bakeoff_{role}", units=5)
                checks = frames(path, [0.3, 0.65, 0.95], work / f"frames_{s.index}_{role}")
                entry["chalk_checks"] = [chalk_check(f) for f in checks]
                if len(checks) > 1:
                    shutil.copy(checks[1], out / f"clip_shot{s.index:03d}_{role}_middle.png")
            except (MediaError, OverBudget) as exc:
                entry["error"] = str(exc)[:300]
            report["clips"][f"shot {s.index} {role}"] = entry

    # Voices: the opening of the story.
    opening = " ".join(narration.split()[:90])
    for name in voices:
        reader = ElevenVoice(studio, name, kokoro, work / "voice")
        try:
            path = studio.call("voice", {"text": opening, "voice": name}, "audio", out / f"voice_eleven_{name}.mp3",
                               "bakeoff_voice", units=len(opening) / 1000)
            report["voices"][name] = {"seconds": round(decode(path, 44_100).size / 44_100, 1)}
        except (MediaError, OverBudget) as exc:
            report["voices"][name] = {"error": str(exc)[:300]}
        del reader
    try:
        audio = kokoro.say(opening, 1.0)
        soundfile.write(out / "voice_kokoro.wav", audio, kokoro.sample_rate)
        report["voices"]["kokoro"] = {"seconds": round(audio.size / kokoro.sample_rate, 1)}
    except Exception as exc:  # noqa: BLE001 - a comparison, not a run
        report["voices"]["kokoro"] = {"error": str(exc)[:200]}

    # Sound.
    for name, role, args in (("music_dread", "music", {"prompt": CUES["dread"], "music_length_ms": 30000,
                                                       "force_instrumental": True}),
                             ("sting", "sfx", {"text": STING, "duration_seconds": 3.0})):
        try:
            studio.call(role, args, "audio", out / f"{name}.mp3", f"bakeoff_{role}", units=30 if role == "music" else 3)
            report["sound"][name] = "ok"
        except (MediaError, OverBudget) as exc:
            report["sound"][name] = str(exc)[:300]

    report["usd"] = round(budget.spent, 2)
    report["usd_by_task"] = {k: round(v, 2) for k, v in budget.by_task.items()}
    report["shots"] = [s.to_dict() for s in best]
    (out / "bakeoff.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    shutil.rmtree(work, ignore_errors=True)
    return report


def sheet(best: list[Any], pictures: dict[tuple[int, str], Path], out: Path) -> None:
    """Rows: shots; columns: picture models."""
    w, h = 640, 360
    im = Image.new("RGB", (len(PICTURE_ROLES) * (w + 10) + 10, len(best) * (h + 40) + 50), (30, 30, 30))
    d = ImageDraw.Draw(im)
    f = draw.font(22)
    for c, role in enumerate(PICTURE_ROLES):
        d.text((10 + c * (w + 10), 12), role, font=f, fill=(230, 230, 230))
    for r, s in enumerate(best):
        for c, role in enumerate(PICTURE_ROLES):
            path = pictures.get((s.index, role))
            x, y = 10 + c * (w + 10), 50 + r * (h + 40)
            if path:
                with Image.open(path) as p:
                    im.paste(p.convert("RGB").resize((w, h)), (x, y))
            d.text((x, y + h + 6), f"shot {s.index}", font=f, fill=(200, 200, 200))
    im.save(out, quality=88)

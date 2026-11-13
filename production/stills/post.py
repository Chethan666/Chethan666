"""Post-production agents: the edit plan, sound design, the render, packaging and QA."""

from __future__ import annotations

import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import soundfile

from .. import assemble, draw, edit, package, shots as shotlist, timing
from ..edit import FPS, Segment, _ff, probe
from ..look import W, H
from ..library import Library
from ..score import Score
from ..shots import Shot
from .board import Board
from .images import Painter
from . import sounds

log = logging.getLogger(__name__)

CHANNEL = "HOLLOW LINES"
DRIFT = {"still": "push_in", "handheld": "push_in"}


class EditPlanAgent:
    """The running order: cold open, title card, the story, end screen; the speech track and captions."""

    name = "edit_plan"
    critical = True

    def run(self, board: Board) -> dict[str, Any]:
        cfg, d = board.cfg, board.data
        beats: list[Shot] = d["beats"]
        audio, rate, timeline = d["audio"], d["rate"], d["timeline"]
        story_seconds = timeline.duration
        cold = edit.pick_cold_open(beats, d["scenes"], 6.0, 16.0) if cfg.get("cold_open", True) else []
        cold = [i for i in cold if i not in d["host_beats"]]
        cold_a = beats[cold[0]].t0 if cold else 0.0
        cold_b = beats[cold[-1]].t1 if cold else 0.0
        cold_seconds = cold_b - cold_a
        title_seconds = float(cfg.get("title_seconds", 3.0))
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
        speech += [(c.t0 - cold_a, c.t1 - cold_a) for c in timeline.chunks if cold and c.t0 >= cold_a and c.t1 <= cold_b + 0.01]
        cues = timing.caption_cues(timeline)
        prog = [timing.Cue(c.start - cold_a, c.end - cold_a, c.text) for c in cues
                if cold and c.start >= cold_a and c.end <= cold_b + 0.01]
        prog += [timing.Cue(c.start + offset, c.end + offset, c.text) for c in cues]
        (board.out / "captions.srt").write_text(timing.srt(prog))
        d.update(cold=cold, cold_seconds=cold_seconds, title_seconds=title_seconds, end_seconds=end_seconds,
                 offset=offset, total=total, track=track, speech=speech, cues=prog, story_seconds=story_seconds)
        return {"cold_open_beats": cold, "cold_open_seconds": round(cold_seconds, 1),
                "total_seconds": round(total, 1)}


class SoundAgent:
    """Score cues that follow the tension, room tone per place, effects, a stinger per scare.

    Sounds come from the channel's sound library first (picked by the planner from a short
    list per need); only what the library cannot cover is generated, and that is saved for
    the library afterwards."""

    name = "sound"
    critical = False

    def run(self, board: Board) -> dict[str, Any]:
        d = board.data
        score = Score(board.studio, board.work / "sound", d["rate"], int(board.cfg.get("parallel", 4)))
        score.cue_names = list(board.cfg.get("music_cues") or score.cue_names)
        score.min_music = float(board.cfg.get("music_min_seconds", 60.0))
        lib_cfg = dict(board.cfg.get("sound_library") or {})
        library = Library.load(lib_cfg, board.work / "library") if lib_cfg.get("enabled", True) else None
        if library is not None and not library.sounds:
            library = None
        # The sound director decides each sound: from the library, newly made, or left out.
        decided = sounds.pick(board, library, score.cue_names, int(board.cfg.get("max_effects", 30)),
                              float(lib_cfg.get("min_score", 0.55)))
        library_report: dict[str, Any] = dict(decided.report, library=library is not None)
        score.prompts, score.skip = dict(decided.make), set(decided.skip)
        if library is not None:
            score.chosen = sounds.fetch_all(library, decided.chosen, int(board.cfg.get("parallel", 4)))
            used = [decided.chosen[n].id for n in score.chosen]
            d["sounds_used"] = used
            d["credits"] = library.credits(used)
            library_report.update(fetched=len(score.chosen), credits=len(d["credits"]))
        score.generate_missing = bool(lib_cfg.get("generate_missing", True))
        stings = [d["cold_seconds"]] if d["cold"] and d["title_seconds"] else []
        try:
            mixed = score.mix(d["track"], d["speech"], d["beats"], d["offset"], d["scenes"], {}, stings=stings,
                              music_until=d["total"], max_effects=int(board.cfg.get("max_effects", 30)))
        except Exception:
            board.data["mix"] = board.work / "mix.wav"
            soundfile.write(board.data["mix"], d["track"], d["rate"])  # the voices alone, if all else fails
            raise
        board.data["mix"] = board.work / "mix.wav"
        soundfile.write(board.data["mix"], mixed, d["rate"])
        soundfile.write(board.out / "narration.wav", d["audio"], d["rate"])
        saved = sounds.save_generated(score.generated, board.out, board.story) \
            if lib_cfg.get("save_generated", True) else 0
        (board.out / "sounds_used.json").write_text(json.dumps(d.get("sounds_used", [])))
        return dict(score.report, library=library_report, generated=len(score.generated), saved_for_library=saved)


def render(seg: Segment, work: Path, motion: str) -> str:
    """One still for its frames: static, or with a slow drift; fades where the segment asks."""
    n = seg.frames()
    seconds = n / FPS
    fades = ""
    if seg.fade_in:
        fades += f",fade=t=in:st=0:d={seg.fade_in:.2f}"
    if seg.fade_out:
        fades += f",fade=t=out:st={max(seconds - seg.fade_out, 0):.2f}:d={seg.fade_out:.2f}"
    if motion == "drift":
        camera = DRIFT.get(seg.camera, seg.camera)
        vf = assemble.camera_filter(camera, seconds, (W, H), "dark", fades=False).replace(",format=yuv420p", "")
    else:
        vf = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={FPS}"
    _ff(["-loop", "1", "-framerate", str(FPS), "-i", str(seg.picture), "-vf", f"{vf}{fades},format=yuv420p",
         "-frames:v", str(n), "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-tune", "stillimage",
         "-pix_fmt", "yuv420p", "-r", str(FPS), "-an", seg.name], work)
    return seg.name


class EditAgent:
    """Renders the programme: every still for exactly its frames, dips to black between scenes."""

    name = "edit"
    critical = True

    def run(self, board: Board) -> dict[str, Any]:
        cfg, d = board.cfg, board.data
        beats: list[Shot] = d["beats"]
        titles = d["packaging"].get("titles") or [board.story.get("title", "")]
        frame = d.get("style_frame")
        title_card = edit.card(frame, [(CHANNEL, 40, draw.RED)] +
                               [(line, 92, draw.CHALK) for line in edit.wrap(str(titles[0]), 24)],
                               board.work / "title_card.png")
        end_line = str(d["packaging"].get("end_screen_line", "")) or "Another story is waiting."
        end_card = edit.card(frame, [(line, 60, draw.CHALK) for line in edit.wrap(end_line, 30)] +
                             [("", 30, draw.CHALK), ("NEXT STORY  >", 54, draw.RED)], board.work / "end_card.png",
                             left=True, dim=0.25)
        segs = edit.programme(beats, d["pictures"], {}, d["cold"], d["cold_seconds"], d["title_seconds"],
                              d["end_seconds"], d["story_seconds"], title_card, end_card)
        first = len(d["cold"]) + (1 if d["title_seconds"] else 0)
        story = segs[first:first + len(beats)]
        dip = float(cfg.get("scene_dip_seconds", 0.35))
        for k in range(len(story) - 1):  # a breath of black where the scene changes
            if beats[k].scene != beats[k + 1].scene and dip:
                story[k].fade_out = max(story[k].fade_out, dip)
                story[k + 1].fade_in = max(story[k + 1].fade_in, dip)
        motion = str(cfg.get("motion", "none"))
        work = board.work
        with ThreadPoolExecutor(max_workers=int(cfg.get("render_workers", 3))) as pool:
            names = list(pool.map(lambda s: render(s, work, motion), segs))
        (work / "segments.txt").write_text("".join(f"file '{n}'\n" for n in names))
        _ff(["-f", "concat", "-safe", "0", "-i", "segments.txt", "-c", "copy", "picture.mp4"], work)
        burn = None
        if cfg.get("burn_captions"):
            burn = work / "captions.ass"
            burn.write_text(timing.ass(d["cues"], W, H))
        vf = ["-vf", f"subtitles={burn.name}"] if burn else []
        codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p"] if vf else ["-c:v", "copy"]
        video = board.out / "video.mp4"
        _ff(["-i", "picture.mp4", "-i", str(d["mix"]), *vf, *codec,
             "-af", f"loudnorm=I={float(cfg.get('loudness_lufs', -14.0))}:TP=-1.5:LRA=11", "-ar", "48000",
             "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(video)], work)
        d.update(video=video, end_line=end_line, titles=titles)
        return {"segments": len(segs), "motion": motion, "seconds": round(probe(video), 1)}


THUMB_ANGLES = (
    "the main character in close-up, frozen in fear, eyes wide, the threat looming in the darkness just behind them",
    "the threat itself, closer than it should be: a dark shape with one terrible detail catching the light",
    "a single small figure in the story's key place at night, something wrong in the darkness around them",
)


class PackagerAgent:
    """Thumbnails to test, Shorts, chapters, the upload details and the review kit."""

    name = "packager"
    critical = False

    def thumbnails(self, board: Board) -> list[Path]:
        cfg, d = board.cfg, board.data
        painter: Painter = d["painter"]
        text = str(d["packaging"].get("thumbnail_text", "")).strip()
        brief = str(d["packaging"].get("thumbnail_brief", "")).strip()
        beats: list[Shot] = d["beats"]
        key = next((b for b in beats if b.scare), None) or max(beats, key=lambda b: b.importance)
        refs = [p for p in [d.get("style_frame"), d["pictures"].get(key.index), *list(d.get("sheets", {}).values())[:2]]
                if p is not None]
        made = []
        for k, angle in enumerate(THUMB_ANGLES[: int(cfg.get("thumbnails", 3))]):
            out = board.out / f"thumbnail_{k + 1}.jpg"
            lettering = str((d.get("look") or {}).get("lettering") or "big bold hand-lettered words")
            words = (f" The words \"{text.upper()}\" in {lettering}, in the left third, the last word in blood "
                     "red, readable at phone size; nothing else written." if text else " No text.")
            prompt = (f"A YouTube thumbnail. {d['house']} Match the look of the reference images and keep each "
                      f"person's face exactly as in them.{words} Picture: "
                      f"{angle}. Brief: {brief} Extreme contrast, one clear focal point, lots of darkness, nothing "
                      "small or busy.")
            raw = board.work / f"thumb_{k + 1}.png"
            got, _ = painter.paint(prompt, refs, raw, ["thumbnail", k, prompt], cfg["chains"]["thumbnail"])
            source = got or d["pictures"][key.index]
            from PIL import Image
            with Image.open(source) as im:
                picture = im.convert("RGB")
                if got is None:
                    picture = draw.thumbnail(picture, text).convert("RGB")
                picture.resize((1280, 720), Image.LANCZOS).save(out, quality=90)
            made.append(out)
        return made

    def run(self, board: Board) -> dict[str, Any]:
        cfg, d = board.cfg, board.data
        beats: list[Shot] = d["beats"]
        thumbs = self.thumbnails(board)
        windows = package.short_windows(beats, d["scenes"], d["offset"], int(cfg.get("shorts", 2)), 25.0,
                                        float(cfg.get("short_max_seconds", 55.0)))
        shorts = package.shorts(d["video"], windows, d["cues"], str(d["titles"][0]), board.out, board.work)
        package.contact_sheet(beats, d["pictures"], {}, board.out / "contact_sheet.jpg")
        package.preview(d["video"], board.out / "preview_60s.mp4")
        chapters = shotlist.chapters(board.router, d["scenes"], beats, d["offset"])
        description = str(d["packaging"].get("description", ""))
        if d.get("credits"):  # CC BY music and sounds: the credit lines their licenses ask for
            description += "\n\nMusic and sound:\n" + "\n".join(d["credits"])
        if chapters:
            description += "\n\n" + "\n".join(f"{shotlist.stamp(t)} {name}" for t, name in chapters)
        story = board.story
        metadata = {
            "story_id": story.get("id"), "title": d["titles"][0], "other_titles": d["titles"][1:],
            "description": description, "tags": d["packaging"].get("tags", []),
            "pinned_comment": d["packaging"].get("pinned_comment", "") or story.get("comment_question", ""),
            "end_screen_line": d["end_line"], "credit": story.get("description_credit", ""), "made_for_kids": False,
            "duration_seconds": round(d["total"], 1), "chapters": [[round(t, 1), n] for t, n in chapters],
            "sound_credits": d.get("credits", []), "sounds_used": d.get("sounds_used", []),
            "thumbnails": [p.name for p in thumbs], "shorts": [p.name for p in shorts],
            "shorts_windows": [[round(a, 1), round(b, 1)] for a, b in windows],
            "end_screen_starts": round(d["offset"] + d["story_seconds"], 1),
            # YouTube's "altered or synthetic content" label is for realistic made-up people and events:
            # on for the cinematic look, off for illustrations; stills.synthetic_media true/false overrides.
            "synthetic_media": bool((d.get("look") or {}).get("synthetic_media", False)
                                    if cfg.get("synthetic_media") is None else cfg.get("synthetic_media")),
        }
        (board.out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        (board.out / "beats.json").write_text(shotlist.save(beats))
        return {"thumbnails": len(thumbs), "shorts": len(shorts), "chapters": len(chapters)}


def streams(path: Path) -> list[str]:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                             str(path)], capture_output=True, text=True)
    return [s.strip() for s in result.stdout.split() if s.strip()]


class QAAgent:
    """Checks the finished files before a person reviews them; never blocks, only reports."""

    name = "qa"
    critical = False

    def run(self, board: Board) -> dict[str, Any]:
        d, out = board.data, board.out
        issues = []
        video = out / "video.mp4"
        if not video.exists():
            return {"status": "fail", "issues": ["no video.mp4"]}
        seconds = probe(video)
        if abs(seconds - d["total"]) > 1.0:
            issues.append(f"video is {seconds:.1f} s, the programme {d['total']:.1f} s")
        kinds = streams(video)
        if "video" not in kinds or "audio" not in kinds:
            issues.append(f"streams: {kinds}")
        art = board.reports.get("illustrator", {})
        painted = art.get("pictures", 0)
        if d["beats"] and painted < 0.8 * len(d["beats"]):
            issues.append(f"only {painted} of {len(d['beats'])} beats have their own picture")
        if art.get("placeholders"):
            issues.append(f"{art['placeholders']} placeholder pictures")
        voice = board.reports.get("voice", {})
        if voice.get("lines_read_by_kokoro"):
            issues.append(f"{voice['lines_read_by_kokoro']} lines fell back to Kokoro")
        for name in ("metadata.json", "captions.srt", "thumbnail_1.jpg", "contact_sheet.jpg"):
            if not (out / name).exists():
                issues.append(f"missing {name}")
        for name, report in board.reports.items():
            if isinstance(report, dict) and report.get("error"):
                issues.append(f"{name} failed: {report['error'][:120]}")
        return {"status": "pass" if not issues else "check", "issues": issues, "video_seconds": round(seconds, 1)}

"""One episode script in, one vertical TV-style episode out.

Consistency comes from references: every character gets one reference portrait and every
location one establishing plate, made once per series and kept with the database. Every
shot's picture is generated from those references, so faces, clothes and places stay the
same across episodes. A few key beats become real moving clips. Each character has one
fixed voice. The edit cuts on every line like TV, with camera moves, a film look, big
subtitles and a sound bed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import soundfile

from story_factory import db

from . import sound
from .media import BosonVoice, Fal, MediaError, data_uri, download, first_url

log = logging.getLogger(__name__)

W, H, FPS = 1080, 1920, 30
STYLE = ("cinematic film still from a prestige TV drama, shot on 35mm film, natural skin texture, "
         "motivated dramatic lighting, shallow depth of field, muted cinematic color grade, "
         "vertical 9:16 composition, no text, no captions, no watermark")
FRAMING = {
    "establishing": "wide establishing shot of the location",
    "wide": "wide shot, full bodies in the setting",
    "medium": "medium shot from the waist up",
    "close_up": "close-up on the face, emotion readable",
    "extreme_close_up": "extreme close-up on the eyes or a detail",
    "over_shoulder": "over-the-shoulder shot, the listener's shoulder in the foreground, the speaker facing camera",
    "insert": "insert shot of the important object, no faces",
    "two_shot": "two-shot, both characters in frame facing each other",
}
KOKORO_VOICES = {"female": ["af_heart", "bf_emma", "af_nicole", "af_sarah", "bf_isabella", "af_bella"],
                 "male": ["am_michael", "bm_george", "am_fenrir", "bm_lewis", "am_puck", "am_echo"]}
REFERENCE_TEXT = ("I have lived in this city my whole life, and I have never once seen the river this still. "
                  "Tell me exactly what you saw, and don't leave anything out.")
EMOTION_SPEED = {"whisper": 0.9, "sad": 0.92, "cold": 0.95, "calm": 0.97, "warm": 0.97, "pleading": 1.0,
                 "tense": 1.02, "mocking": 1.0, "afraid": 1.06, "angry": 1.08, "shocked": 1.05}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "x"


def _ffmpeg(args: list[str], cwd: Path) -> None:
    result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], cwd=cwd,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[-800:]}")


class Costs:
    """Estimated spend on paid media, logged to the shared costs table, with a cap per episode."""

    def __init__(self, conn: Any, cfg: dict, cap: float):
        self.conn, self.prices, self.cap, self.spent, self.held = conn, cfg.get("prices", {}), cap, 0.0, 0.0
        self.lock = threading.Lock()

    def charge(self, task: str, model: str, units: float = 1.0) -> None:
        usd = float(self.prices.get(model, 0.0)) * units
        with self.lock:  # pictures are made on worker threads
            self.spent += usd
            db.record_cost(self.conn, pipeline="drama", task=task, model=model, input_units=0,
                           output_units=int(units), usd=usd)

    def credit(self, model: str, units: float = 1.0) -> None:
        """Count media already paid for (kept from a failed run) against this episode's cap."""
        with self.lock:
            self.spent += float(self.prices.get(model, 0.0)) * units

    def reserve(self, model: str, units: float = 1.0) -> bool:
        """Hold budget for a job before it starts (jobs run in parallel); `release` gives it back."""
        usd = float(self.prices.get(model, 0.0)) * units
        with self.lock:
            if self.cap and self.held + self.spent + usd > self.cap:
                return False
            self.held += usd
            return True

    def release(self, model: str, units: float = 1.0) -> None:
        with self.lock:
            self.held = max(0.0, self.held - float(self.prices.get(model, 0.0)) * units)

    def allows(self, model: str, units: float = 1.0) -> bool:
        return not self.cap or self.spent + float(self.prices.get(model, 0.0)) * units <= self.cap


def character_prompt(c: dict[str, Any], genre: str) -> str:
    return (f"{STYLE}, {genre} drama. Character reference portrait: medium close-up, facing the camera, "
            f"plain dark background, even soft light. {c['name']}, a {c.get('age', '')}-year-old "
            f"{c.get('gender', '')}: {c.get('look', '')}. Wearing {c.get('wardrobe', '')}.")


def location_prompt(loc: dict[str, Any], genre: str) -> str:
    return f"{STYLE}, {genre} drama. Establishing shot, no people: {loc.get('name', '')} - {loc.get('look', '')}."


def shot_prompt(shot: dict[str, Any], bible: dict[str, Any]) -> str:
    cast = {c["name"]: c for c in bible.get("cast", [])}
    places = {loc["name"]: loc for loc in bible.get("locations", [])}
    people = [cast[n] for n in shot.get("characters", []) if n in cast]
    who = "; ".join(f"{c['name']} ({c.get('look', '')}, wearing {c.get('wardrobe', '')})" for c in people)
    place = places.get(shot.get("location", ""), {})
    feeling = ""
    if shot.get("speaker") and shot.get("line"):
        feeling = f" {shot['speaker']} is speaking, {shot.get('emotion', 'tense')} expression."
    return (f"{STYLE}, {bible.get('genre', 'drama')}. {FRAMING.get(shot.get('shot', 'medium'), 'medium shot')}. "
            f"Location: {place.get('name', shot.get('location', ''))}, {place.get('look', '')}. "
            f"{shot.get('action', '')}.{feeling} Characters: {who or 'none'}. Keep every character's face, "
            "hair, skin tone and clothes exactly as in the reference images, and the place as in its reference.")


class Producer:
    def __init__(self, conn: Any, cfg: dict, assets: Path, fal: Fal | None, voices: Any,
                 size: tuple[int, int] = (W, H), boson: BosonVoice | None = None):
        self.conn, self.cfg = conn, cfg
        self.w, self.h = size
        self.d = cfg.get("drama_video", {})
        self.assets = assets.resolve()  # FFmpeg runs inside the work folder: every path must be absolute
        self.fal = fal
        self.voices = voices  # Kokoro (or the silent stand-in): every character's fixed voice
        self.boson = boson  # when set, speaks every line, cloned from that fixed voice
        self.boson_failures = 0
        self.models = self.d.get("models", {})
        self.args = self.d.get("model_args", {})

    # --- references -----------------------------------------------------------
    def references(self, series_id: int, bible: dict[str, Any], costs: Costs) -> dict[str, Path]:
        """A portrait per character and a plate per location, made once per series."""
        genre = bible.get("genre", "drama")
        jobs: list[tuple[str, Path, str]] = []
        for c in bible.get("cast", []):
            jobs.append((c["name"], self.assets / "cast" / f"{slug(c['name'])}.png", character_prompt(c, genre)))
        for loc in bible.get("locations", []):
            jobs.append((loc["name"], self.assets / "locations" / f"{slug(loc['name'])}.png", location_prompt(loc, genre)))
        missing = [j for j in jobs if not j[1].exists()]
        model = self.models["portrait"]

        def make(job: tuple[str, Path, str]) -> None:
            name, path, prompt = job
            result = self.fal.run(model, dict(self.args.get("portrait", {}), prompt=prompt))
            download(first_url(result, "image"), path)
            costs.charge("reference", model)
            log.info("Reference made: %s", name)

        if missing:
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(make, missing))
        return {name: path for name, path, _ in jobs}

    # --- shots ----------------------------------------------------------------
    def picture(self, shot: dict[str, Any], bible: dict[str, Any], refs: dict[str, Path], path: Path,
                costs: Costs) -> Path:
        images = [refs[n] for n in shot.get("characters", []) if n in refs][:3]
        if shot.get("location") in refs:
            images.append(refs[shot["location"]])
        model = self.models["shot"] if images else self.models["portrait"]
        args = dict(self.args.get("shot" if images else "portrait", {}), prompt=shot_prompt(shot, bible))
        if images:
            args["image_urls"] = [data_uri(p) for p in images if p.exists()]
        result = self.fal.run(model, args)
        costs.charge("shot", model)
        return download(first_url(result, "image"), path)

    def clip_seconds(self, needed: float) -> int:
        """The shortest clip length the model offers that covers the shot (else its longest)."""
        offered = sorted(int(x) for x in self.d.get("motion_durations") or
                         [self.args.get("motion", {}).get("duration", 5)])
        return next((x for x in offered if x >= needed - 0.5), offered[-1])

    def motion(self, shot: dict[str, Any], picture: Path, path: Path, costs: Costs,
               needed: float = 5.0) -> Path | None:
        model = self.models.get("motion")
        seconds = self.clip_seconds(needed)
        if not model or not costs.reserve(model, seconds):
            return None
        args = dict(self.args.get("motion", {}), image_url=data_uri(picture), prompt=motion_prompt(shot),
                    duration=str(seconds))
        try:
            result = self.fal.run(model, args)
            download(first_url(result, "video"), path)
        except MediaError as exc:
            log.warning("Motion clip skipped: %s", exc)
            return None
        finally:
            costs.release(model, seconds)
        costs.charge("motion", model, seconds)
        return path

    # --- voices ---------------------------------------------------------------
    def cast_voices(self, bible: dict[str, Any]) -> dict[str, str]:
        """One fixed voice per character: configured ones first, then a pool by gender."""
        chosen = dict(self.d.get("voices", {}))
        pools = {k: list(v) for k, v in KOKORO_VOICES.items()}
        used = set(chosen.values())
        for c in bible.get("cast", []):
            if c["name"] in chosen:
                continue
            pool = pools.get(c.get("gender", "male"), []) or pools.get("male", [])
            pick = next((v for v in pool if v not in used), pool[0] if pool else "")
            chosen[c["name"]] = pick
            used.add(pick)
        return chosen

    def reference(self, voice: str) -> Path:
        """A short clip of a character's fixed voice, made once per series, for Boson to clone."""
        path = self.assets / "voices" / f"{slug(voice)}.wav"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            self.voices.voice = voice
            soundfile.write(path, self.voices.say(REFERENCE_TEXT, 1.0), self.voices.sample_rate)
        return path

    def line_audio(self, shot: dict[str, Any], voice: str, path: Path) -> tuple[np.ndarray, int]:
        text = str(shot.get("line", "")).strip()
        audio = None
        if self.boson is not None:
            try:
                self.boson.speak(text, None, path, reference=(self.reference(voice), REFERENCE_TEXT))
                audio, rate = soundfile.read(path, dtype="float32", always_2d=False)
            except (MediaError, RuntimeError) as exc:  # one refused line is read by the fixed voice instead
                self.boson_failures += 1
                log.warning("Boson refused a line; the fixed voice reads it: %s", exc)
                audio = None
        if audio is None:
            self.voices.voice = voice
            audio = self.voices.say(text, EMOTION_SPEED.get(str(shot.get("emotion", "")), 1.0))
            rate = self.voices.sample_rate
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if shot.get("inner_voice"):  # a thought: softer, with a little room
            echo = np.zeros_like(audio)
            lag = int(0.06 * rate)
            echo[lag:] = audio[:-lag] * 0.35 if audio.size > lag else 0
            audio = (audio + echo) * 0.75
        return audio.astype(np.float32), int(rate)

    # --- edit -----------------------------------------------------------------
    def clip(self, work: Path, index: int, shot: dict[str, Any], picture: Path, moving: Path | None,
             seconds: float) -> str:
        """One shot as a 1080x1920 clip with a camera move (or the generated motion) and a film look."""
        name = f"shot_{index:03d}.mp4"
        look = ("eq=contrast=1.06:saturation=0.9:gamma=0.97,vignette=PI/5,noise=alls=5:allf=t,"
                f"format=yuv420p")
        frames = max(int(seconds * FPS), 1)
        w, h = self.w, self.h
        fill = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
        if moving is not None:
            length = _length(moving) or seconds
            stretch = min(max(seconds / length, 1.0), 1.3)  # slow a short clip a little, then hold its end
            _ffmpeg(["-i", str(moving), "-vf", f"setpts={stretch:.4f}*PTS,{fill},fps={FPS},"
                     f"tpad=stop_mode=clone:stop_duration={seconds},trim=duration={seconds:.3f},{look}", "-an",
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", name], work)
            return name
        kind = shot.get("shot", "medium")
        if kind in ("close_up", "extreme_close_up", "insert"):
            move = f"z='1+0.07*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        elif kind in ("establishing", "wide"):
            move = f"z='1.08':x='(iw-iw/zoom)*on/{frames}':y='ih/2-(ih/zoom/2)'"
        elif kind == "over_shoulder":
            move = f"z='1.06-0.04*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        else:
            move = f"z='1.02+0.04*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        _ffmpeg(["-loop", "1", "-i", str(picture), "-t", f"{seconds:.3f}", "-vf",
                 f"{fill},scale={w * 3 // 2}:{h * 3 // 2},zoompan={move}:d=1:s={w}x{h}:fps={FPS},{look}",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-r", str(FPS), name], work)
        return name

    def produce(self, episode_id: int, out_root: Path) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,)).fetchone()
        series = self.conn.execute("SELECT * FROM series WHERE id = ?", (row["series_id"],)).fetchone()
        bible, script = json.loads(series["bible"]), json.loads(row["script"])
        shots = script.get("shots") or []
        out = out_root.resolve() / f"series-{series['id']}-ep-{row['number']:02d}"
        work = out / "work"
        work.mkdir(parents=True, exist_ok=True)
        # Paid media for this episode is kept with the series until the episode is made, so a
        # failed run never pays for the same picture or clip twice. Names carry a hash of what
        # was asked for, so a rewritten shot gets a new picture.
        cache = self.assets / "episodes" / f"{row['number']:02d}"
        cache.mkdir(parents=True, exist_ok=True)
        costs = Costs(self.conn, self.d, float(self.d.get("max_usd_per_episode", 3.0)))
        # References are made once for the whole series, so they are logged but don't use this episode's cap.
        refs = self.references(series["id"], bible, Costs(self.conn, self.d, 0.0))

        # Pictures for every shot, in parallel.
        def picture(job: tuple[int, dict[str, Any]]) -> Path:
            i, shot = job
            path = cache / f"shot_{i:03d}_{_digest(shot_prompt(shot, bible), shot.get('characters'))}.png"
            if path.exists():
                costs.credit(self.models["shot"])
                return path
            return self.picture(shot, bible, refs, path, costs)

        with ThreadPoolExecutor(max_workers=int(self.d.get("parallel_images", 4))) as pool:
            pictures = list(pool.map(picture, enumerate(shots)))
        # Voices first: each shot lasts as long as its line, and its clip is made that long.
        voices = self.cast_voices(bible)
        rate = 24_000
        lines: list[np.ndarray] = []
        durations: list[float] = []
        for i, shot in enumerate(shots):
            if str(shot.get("line", "")).strip() and shot.get("speaker"):
                audio, r = self.line_audio(shot, voices.get(shot["speaker"], ""), work / f"line_{i:03d}.wav")
                if r != rate:
                    audio = np.interp(np.linspace(0, audio.size - 1, int(audio.size * rate / r)),
                                      np.arange(audio.size), audio).astype(np.float32)
                lead = np.zeros(int(0.15 * rate), dtype=np.float32)
                tail = np.zeros(int(0.45 * rate), dtype=np.float32)
                piece = np.concatenate([lead, audio, tail])
                least = int(2.0 * rate)  # a cut never comes before the viewer has taken the shot in
                if piece.size < least:
                    piece = np.concatenate([piece, np.zeros(least - piece.size, dtype=np.float32)])
                lines.append(piece)
            else:
                hold = {"establishing": 3.0, "insert": 2.0, "wide": 2.8}.get(shot.get("shot", ""), 2.5)
                lines.append(np.zeros(int(hold * rate), dtype=np.float32))
            durations.append(lines[-1].size / rate)

        # Every shot moves: real clips, the most important first, in parallel, within the budget.
        order = motion_order(shots)[: int(self.d.get("max_motion_shots", len(shots)))]

        def target(i: int) -> Path:
            key = _digest(motion_prompt(shots[i]), pictures[i].name, self.clip_seconds(durations[i]))
            return cache / f"motion_{i:03d}_{key}.mp4"

        moving = {i: target(i) for i in order if target(i).exists()}  # kept from a failed run
        for i in moving:
            costs.credit(self.models.get("motion", ""), self.clip_seconds(durations[i]))

        def film(i: int) -> tuple[int, Path | None]:
            return i, self.motion(shots[i], pictures[i], target(i), costs, durations[i])

        with ThreadPoolExecutor(max_workers=int(self.d.get("parallel_motion", 4))) as pool:
            moving.update({i: clip for i, clip in pool.map(film, [i for i in order if i not in moving]) if clip})
        log.info("Moving shots: %d of %d", len(moving), len(shots))

        track = np.concatenate(lines)
        starts = np.cumsum([0.0] + durations[:-1]).tolist()
        speech = [(s + 0.12, s + d - 0.28) for s, d, shot in zip(starts, durations, shots) if shot.get("line")]
        scenes = [{"mood": shot.get("emotion", ""), "sfx": []} for shot in shots]
        mixed = sound.mix(track, rate, scenes, list(zip(starts, [s + d for s, d in zip(starts, durations)])), speech)
        soundfile.write(work / "mix.wav", mixed, rate)

        # Subtitles: big, centered low, thoughts in italics.
        (work / "subs.ass").write_text(subtitles(shots, starts, durations, (self.w, self.h)))
        clips = [self.clip(work, i, shot, pictures[i], moving.get(i), durations[i]) for i, shot in enumerate(shots)]
        (work / "clips.txt").write_text("".join(f"file '{c}'\n" for c in clips))
        _ffmpeg(["-f", "concat", "-safe", "0", "-i", "clips.txt", "-i", "mix.wav", "-vf", "subtitles=subs.ass",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", "-c:a", "aac",
                 "-b:a", "192k", "-shortest", "-movflags", "+faststart", "episode.mp4"], work)
        shutil.move(str(work / "episode.mp4"), out / "episode.mp4")
        shutil.copy(pictures[0], out / "thumbnail.png")
        meta = {"series": series["title"], "episode": row["number"], "title": script.get("title"),
                "hook_line": script.get("hook_line"), "cliffhanger": script.get("cliffhanger"),
                "seconds": round(sum(durations), 1), "shots": len(shots), "moving_shots": len(moving),
                "voices": voices, "voice_engine": "boson (cloned)" if self.boson else "kokoro",
                "boson_lines_refused": self.boson_failures, "estimated_media_usd": round(costs.spent, 2)}
        (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        if not self.d.get("keep_work"):
            shutil.rmtree(work, ignore_errors=True)
            shutil.rmtree(cache, ignore_errors=True)  # made: the video holds it now
        return meta


def motion_prompt(shot: dict[str, Any]) -> str:
    """What the image-to-video model is asked for: the shot's movement, and a speaker who talks."""
    talk = ""
    if shot.get("speaker") and str(shot.get("line", "")).strip() and not shot.get("inner_voice"):
        talk = (f" {shot['speaker']} is speaking, with natural mouth, jaw and face movement and a "
                f"{shot.get('emotion', 'tense')} expression; small natural gestures.")
    elif shot.get("inner_voice"):
        talk = " The character is silent, lost in thought; only the eyes and breath move."
    return (f"{shot.get('action', '')}.{talk} Realistic, cinematic, natural human motion, gentle camera "
            "movement, keep every face, outfit and the location exactly as in the image, no cuts, no text.")


def motion_order(shots: list[dict[str, Any]]) -> list[int]:
    """Which shots get their clip first if the budget runs short: marked motion, the hook and the
    cliffhanger, spoken close shots, then the rest in story order."""
    last = len(shots) - 1
    close = ("close_up", "extreme_close_up", "over_shoulder", "medium")

    def rank(i: int) -> tuple:
        s = shots[i]
        return (not s.get("motion"), i not in (0, last), not (s.get("line") and s.get("shot") in close), i)

    return sorted(range(len(shots)), key=rank)


def _length(path: Path) -> float:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                             str(path)], capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _digest(*parts: Any) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:10]


def subtitles(shots: list[dict[str, Any]], starts: list[float], durations: list[float],
              frame: tuple[int, int] = (W, H)) -> str:
    fw, fh = frame
    size = round(fh * 0.0385)
    margin = round(fh * 0.22)
    outline = max(2, size // 15)

    def t(seconds: float) -> str:
        cs = int(round(seconds * 100))
        h, cs = divmod(cs, 360_000)
        m, cs = divmod(cs, 6_000)
        s, cs = divmod(cs, 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    head = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {fw}", f"PlayResY: {fh}", "WrapStyle: 0", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Line,DejaVu Sans,{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,{outline},2,2,70,70,{margin},1",
        f"Style: Thought,DejaVu Sans,{size},&H00D8F0FF,&H00D8F0FF,&H00000000,&H64000000,1,1,0,0,100,100,0,0,1,{outline},2,2,70,70,{margin},1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    events = []
    for shot, start, dur in zip(shots, starts, durations):
        line = str(shot.get("line", "")).strip().replace("{", "(").replace("}", ")")
        if line:
            style = "Thought" if shot.get("inner_voice") else "Line"
            events.append(f"Dialogue: 0,{t(start + 0.1)},{t(start + dur - 0.15)},{style},,0,0,0,,{line}")
    return "\n".join(head + events) + "\n"

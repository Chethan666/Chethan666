"""Sound design: half of every scare.

Layers, from the bottom: a room tone for each place (generated, looped), a music score that
follows the story's tension (three generated cues: unease, dread, terror, crossfaded as the
story moves between them), a sound effect for every shot that asks for one, and a stinger on
every scare. Before each scare the music and room tone drop away for a beat, so the hit lands
in silence. Everything dips under the voice. If a generated sound fails, the synthesized
drone and cues from `sound.py` fill in.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .media import MediaError

from . import sound
from .paid import OverBudget, Studio, digest
from .shots import Shot
from .speech import decode

log = logging.getLogger(__name__)

# An Indian folk-horror colour: the drones, bells, drums and bowed strings of Indian music.
CUES = {
    "unease": "Dark ambient Indian folk-horror score, slow and sparse: a low, slightly detuned tanpura drone, a "
              "distant temple bell, soft breathy bansuri notes bending out of tune, quiet and patient, "
              "instrumental, no drums, no vocals.",
    "dread": "Tense Indian folk-horror score building dread: a pulsing low tanpura drone, a slow muffled dhol "
             "like a heartbeat, scraping sarangi, faint anklet bells, rising dissonance, creeping and "
             "suspenseful, instrumental, no vocals.",
    "terror": "Intense Indian folk-horror climax: aggressive dissonant sarangi and strings, pounding low dhol and "
              "tabla, a shrill shehnai cry, frantic and terrifying, instrumental, no vocals.",
}
STING = ("A sudden horror jump-scare stinger: a heavy temple bell struck once, hard, with a fast reverse swell into "
         "it and a sharp dissonant hit.")
CALM, UNEASY, TENSE = 0, 1, 2


def tension(mood: str) -> int:
    mood = mood.lower()
    if any(w in mood for w in ("terror", "panic", "horror", "shock", "frantic", "violent")):
        return TENSE
    if any(w in mood for w in ("tense", "dread", "fear", "menac", "urgent", "eerie", "ominous", "uneas")):
        return UNEASY
    return CALM


class Score:
    def __init__(self, studio: Studio | None, work: Path, rate: int, parallel: int = 4):
        self.studio, self.work, self.rate, self.parallel = studio, work, rate, parallel
        self.report: dict[str, Any] = {"music": [], "ambience": 0, "effects": 0, "synthesized": []}
        work.mkdir(parents=True, exist_ok=True)

    # --- generated sounds -----------------------------------------------------------
    def _gen(self, role: str, args: dict[str, Any], name: str, units: float = 1.0) -> np.ndarray | None:
        if self.studio is None or role not in self.studio.models:
            return None
        try:
            path = self.studio.call(role, args, "audio", self.work / f"{name}.mp3", f"sound_{role}", units=units)
            return decode(path, self.rate)
        except (MediaError, OverBudget) as exc:
            log.warning("Sound %s not generated: %s", name, str(exc)[:200])
            return None

    def music(self, seconds: float) -> dict[str, np.ndarray]:
        length = int(min(max(seconds, 60), 150))
        def one(item: tuple[str, str]) -> tuple[str, np.ndarray | None]:
            key, prompt = item
            return key, self._gen("music", {"prompt": prompt, "music_length_ms": length * 1000,
                                            "force_instrumental": True}, f"music_{key}", units=length)
        with ThreadPoolExecutor(max_workers=3) as pool:
            got = {k: v for k, v in pool.map(one, CUES.items()) if v is not None and v.size}
        self.report["music"] = sorted(got)
        return got

    def effect(self, text: str, name: str, seconds: float | None = None, loop: bool = False) -> np.ndarray | None:
        args: dict[str, Any] = {"text": text}
        if seconds:
            args["duration_seconds"] = round(min(max(seconds, 0.5), 22.0), 1)
        if loop:
            args["loop"] = True
        return self._gen("sfx", args, name, units=seconds or 3.0)

    # --- the mix ----------------------------------------------------------------------
    def mix(self, voice: np.ndarray, speech: list[tuple[float, float]], shots: list[Shot], offset: float,
            scenes: list[dict[str, Any]], stings: list[float], music_until: float,
            max_effects: int = 45) -> np.ndarray:
        """`voice` is the whole programme's speech track; shot times are on the story timeline, so
        `offset` (cold open + title) moves them onto the programme. `stings` are extra stinger times
        (the title card); every scare shot gets one as well."""
        rate, n = self.rate, voice.size
        total = n / rate

        # Speech ducking envelope.
        duck = np.ones(n, dtype=np.float32)
        for t0, t1 in speech:
            duck[int(max(t0 - 0.2, 0) * rate): int(min(t1 + 0.25, total) * rate)] = 0.0
        duck = _smooth(duck, int(0.3 * rate))  # 0 under speech, 1 between lines

        # Tension over time, per shot.
        level = np.zeros(n, dtype=np.int8)
        for s in shots:
            mood = str(scenes[s.scene].get("mood", "")) if s.scene < len(scenes) else ""
            level[int((s.t0 + offset) * rate): int((s.t1 + offset) * rate)] = tension(mood)
        level[: int(offset * rate)] = TENSE if offset > 0 else 0  # the cold open is the scariest moment

        out = voice.astype(np.float32).copy()

        # Music: each tension run plays its cue, looped, crossfaded at the joins.
        cues = self.music(total)
        bed = np.zeros(n, dtype=np.float32)
        if cues:
            names = {CALM: "unease", UNEASY: "dread", TENSE: "terror"}
            for a, b, lv in _runs(level, rate, min_seconds=12.0):
                cue = cues[names[lv]] if names[lv] in cues else next(iter(cues.values()))
                bed[a:b] += _crossfaded(_loop(cue, b - a, start=a % max(cue.size, 1)), rate)
        else:
            self.report["synthesized"].append("music")
            bed = sound.drone(n, rate) * 0.5
        music_gain = 0.10 + 0.14 * duck
        end_i = int(min(music_until, total) * rate)
        bed[end_i:] = 0
        tail = int(3 * rate)
        bed[max(end_i - tail, 0):end_i] *= np.linspace(1, 0, end_i - max(end_i - tail, 0))

        # Room tone per place.
        room = np.zeros(n, dtype=np.float32)
        places: dict[str, np.ndarray | None] = {}
        for s in shots:
            place = str(scenes[s.scene].get("location", "")).strip() if s.scene < len(scenes) else ""
            if place and place not in places and len(places) < 8:
                places[place] = None
        def amb(place: str) -> tuple[str, np.ndarray | None]:
            return place, self.effect(f"Loopable night ambience of {place}: quiet, eerie room tone, no music, "
                                      "no voices.", f"room_{digest(place)}", 20.0, loop=True)
        with ThreadPoolExecutor(max_workers=self.parallel) as pool:
            places.update(dict(pool.map(amb, list(places))))
        self.report["ambience"] = sum(1 for v in places.values() if v is not None)
        rng = np.random.default_rng(3)
        for s in shots:
            place = str(scenes[s.scene].get("location", "")).strip() if s.scene < len(scenes) else ""
            a, b = int((s.t0 + offset) * rate), min(int((s.t1 + offset) * rate), n)
            sample = places.get(place)
            if sample is None or not sample.size:
                sample = sound._smooth_noise(max(b - a, 1), rate, 500, rng) * 0.3
            room[a:b] += _loop(sample, b - a, start=a % max(sample.size, 1))
        room = _smooth(room, int(0.5 * rate))
        room_gain = 0.10 + 0.08 * duck

        # Silence before each scare: music and room tone drop away for a beat.
        scare_times = [s.t0 + offset for s in shots if s.scare] + list(stings)
        hush = np.ones(n, dtype=np.float32)
        for t in scare_times:
            a, b = int(max(t - 0.9, 0) * rate), int(t * rate)
            hush[a:b] = 0.04
        hush = _smooth(hush, int(0.12 * rate))
        out += bed * music_gain * hush + room * room_gain * hush

        # Stingers.
        sting = self.effect(STING, "sting", 3.0)
        if sting is None:
            self.report["synthesized"].append("stinger")
            sting = sound.cue("thunder", rate, rng)
        for t in scare_times:
            _place(out, sting * 0.55, int(t * rate))

        # One effect per shot that asks for one.
        wanted = [s for s in shots if s.sound][:max_effects]
        def fx(s: Shot) -> tuple[Shot, np.ndarray | None]:
            return s, self.effect(s.sound, f"fx_{s.index:03d}", None)
        with ThreadPoolExecutor(max_workers=self.parallel) as pool:
            for s, sample in pool.map(fx, wanted):
                if sample is None:
                    names = sound.cues_for([s.sound])
                    sample = sound.cue(names[0], rate, rng) if names else None
                if sample is not None and sample.size:
                    self.report["effects"] += 1
                    a = int((s.t0 + offset + 0.15) * rate)
                    _place(out, sample * 0.4 * (0.55 + 0.45 * duck[a] if a < n else 1.0), a)

        fade = int(1.0 * rate)
        out[:fade] *= np.linspace(0.2, 1, min(fade, n))[: out[:fade].size]
        out[-fade:] *= np.linspace(1, 0, min(fade, n))[: out[-fade:].size]
        peak = float(np.max(np.abs(out))) or 1.0
        log.info("Sound: %s", self.report)
        return (out * min(1.0, 0.95 / peak)).astype(np.float32)


def _smooth(x: np.ndarray, width: int) -> np.ndarray:
    if width < 2:
        return x
    kernel = np.ones(width, dtype=np.float32) / width
    return np.convolve(x, kernel, mode="same").astype(np.float32)


def _loop(sample: np.ndarray, length: int, start: int = 0) -> np.ndarray:
    if sample.size == 0 or length <= 0:
        return np.zeros(max(length, 0), dtype=np.float32)
    reps = (start + length) // sample.size + 1
    return np.tile(sample, reps)[start:start + length].astype(np.float32)


def _crossfaded(part: np.ndarray, rate: int, seconds: float = 2.0) -> np.ndarray:
    k = min(int(seconds * rate), part.size // 2)
    if k > 0:
        part[:k] *= np.linspace(0, 1, k)
        part[-k:] *= np.linspace(1, 0, k)
    return part


def _runs(level: np.ndarray, rate: int, min_seconds: float) -> list[tuple[int, int, int]]:
    """Stretches of one tension level, short ones merged into their neighbour; overlapping by 2 s
    so the crossfades meet."""
    runs: list[list[int]] = []
    i = 0
    while i < level.size:
        j = i
        lv = int(level[i])
        step = rate // 4
        while j < level.size and int(level[j]) == lv:
            j += step
        j = min(j, level.size)
        if runs and (j - i) < min_seconds * rate:
            runs[-1][1] = j
        else:
            runs.append([i, j, lv])
        i = j
    overlap = 2 * rate
    return [(max(a - overlap, 0), min(b + overlap, level.size), lv) for a, b, lv in runs]


def _place(out: np.ndarray, sample: np.ndarray, at: int) -> None:
    if at >= out.size or sample.size == 0:
        return
    at = max(at, 0)
    b = min(at + sample.size, out.size)
    out[at:b] += sample[: b - at]

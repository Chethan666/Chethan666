"""The narrator's voice.

Kokoro (82M parameters, Apache-2.0) reads the narration on the CPU, chunk by chunk, so
every chunk's timing is exact and the story's dramatic pauses are real silences. Each
scene's delivery cue (slow, hushed, tense...) sets the reading speed. Pronunciation
respellings replace hard names in what the voice reads; captions keep the real spelling.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Protocol

import numpy as np

from .timing import Chunk, Timeline, split_chunks, split_lines

log = logging.getLogger(__name__)

# Reading speed for each scene delivery cue, relative to the narrator's base speed.
DELIVERY_SPEED = {"steady": 1.0, "slow": 0.9, "hushed": 0.93, "tense": 1.04, "urgent": 1.1, "tender": 0.94}


class Voice(Protocol):
    sample_rate: int

    def say(self, text: str, speed: float) -> np.ndarray: ...


class KokoroVoice:
    """Kokoro on the CPU. Weights download from Hugging Face on first use and are cached."""

    sample_rate = 24_000

    def __init__(self, voice: str, lang: str, base_speed: float):
        try:
            from kokoro import KPipeline
        except ImportError as exc:  # pragma: no cover - depends on the runner
            raise RuntimeError("Install the video extras: pip install -r requirements-video.txt") from exc
        self.pipeline = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M")
        self.voice = voice
        self.base_speed = base_speed

    def speaker(self, voice: str, base_speed: float = 1.0) -> "KokoroVoice":
        """Another voice sharing the loaded model (for a character's lines)."""
        other = object.__new__(KokoroVoice)
        other.pipeline, other.voice, other.base_speed = self.pipeline, voice, base_speed
        return other

    def say(self, text: str, speed: float) -> np.ndarray:  # pragma: no cover - needs the model
        parts = []
        for _, _, audio in self.pipeline(text, voice=self.voice, speed=self.base_speed * speed):
            if audio is not None:
                parts.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


class SilentVoice:
    """A stand-in for tests and dry runs: a quiet tone for as long as the words would take."""

    sample_rate = 24_000

    def __init__(self, words_per_minute: float = 140):
        self.wpm = words_per_minute

    def speaker(self, voice: str, base_speed: float = 1.0) -> "SilentVoice":
        other = SilentVoice(self.wpm)
        other.voice = voice
        return other

    def say(self, text: str, speed: float) -> np.ndarray:
        seconds = max(len(text.split()), 1) * 60 / (self.wpm * speed)
        t = np.arange(int(seconds * self.sample_rate)) / self.sample_rate
        return (0.02 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


def respell(text: str, pronunciations: list[dict[str, Any]]) -> str:
    """What the voice reads: each listed word swapped for its simple respelling."""
    for item in pronunciations or []:
        word, say_as = str(item.get("word", "")).strip(), str(item.get("say_as", "")).strip()
        if word and say_as:
            text = re.sub(rf"(?<!\w){re.escape(word)}(?!\w)", say_as.replace("-", " "), text, flags=re.IGNORECASE)
    return text


def speed_for(offset: int, spans: list[tuple[int, int]], scenes: list[dict[str, Any]]) -> float:
    for (start, end), scene in zip(spans, scenes):
        if start <= offset < end:
            return DELIVERY_SPEED.get(str(scene.get("delivery", "steady")), 1.0)
    return 1.0


def narrate(voice: Voice, narration: str, pauses: list[dict[str, Any]], pronunciations: list[dict[str, Any]],
            scenes: list[dict[str, Any]], spans: list[tuple[int, int]], lines: list[Any] | None = None,
            cast_voices: dict[str, Voice] | None = None) -> tuple[np.ndarray, Timeline]:
    """Read the whole narration. Returns (mono audio, timeline with every chunk's times).
    With `lines` and `cast_voices`, each quoted line is read in its speaker's own voice."""
    chunks: list[Chunk] = split_lines(narration, split_chunks(narration, pauses), lines or [])
    cast_voices = cast_voices or {}
    rate = voice.sample_rate
    pieces: list[np.ndarray] = []
    t = 0.0
    lead_in = np.zeros(int(0.6 * rate), dtype=np.float32)  # a breath before the first word
    pieces.append(lead_in)
    t += len(lead_in) / rate
    for chunk in chunks:
        reader = cast_voices.get(chunk.speaker, voice) if chunk.speaker != "narrator" else voice
        text = respell(chunk.text, pronunciations)
        if reader is not voice:
            text = text.strip("“”\"").strip() or text
            audio = _resample(reader.say(text, 1.0), reader.sample_rate, rate)
        else:
            audio = voice.say(text, speed_for(chunk.start, spans, scenes))
        audio = _trim(audio, rate)
        if reader is not voice:  # the cast sit a little behind the narrator in the mix
            audio = _match_level(audio, 0.85)
        chunk.t0, chunk.t1 = t, t + len(audio) / rate
        pieces.append(audio)
        gap = np.zeros(int(chunk.gap_ms / 1000 * rate), dtype=np.float32)
        pieces.append(gap)
        t = chunk.t1 + len(gap) / rate
    tail = np.zeros(int(1.5 * rate), dtype=np.float32)  # let the last line breathe
    pieces.append(tail)
    audio = np.concatenate(pieces)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio * (0.89 / peak)
    return audio.astype(np.float32), Timeline(narration, chunks, len(audio) / rate)


def _trim(audio: np.ndarray, rate: int, threshold: float = 0.004) -> np.ndarray:
    """Cut leading and trailing near-silence so our own gaps decide the rhythm."""
    if audio.size == 0:
        return audio
    loud = np.flatnonzero(np.abs(audio) > threshold)
    if loud.size == 0:
        return audio
    pad = int(0.03 * rate)
    return audio[max(loud[0] - pad, 0): min(loud[-1] + pad, audio.size)]


def _resample(audio: np.ndarray, source: int, target: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1) if np.ndim(audio) == 1 else np.asarray(audio).mean(axis=1)
    if source == target or audio.size == 0:
        return audio.astype(np.float32)
    n = int(audio.size * target / source)
    return np.interp(np.linspace(0, audio.size - 1, n), np.arange(audio.size), audio).astype(np.float32)


def _match_level(audio: np.ndarray, peak: float) -> np.ndarray:
    top = float(np.max(np.abs(audio))) if audio.size else 0.0
    return (audio * (peak / top)).astype(np.float32) if top > 0 else audio


# One voice per kind of speaker, from Kokoro's English voices (narrator excluded).
KOKORO_CAST = {
    "man": ["am_fenrir", "bm_george", "am_echo", "am_liam"],
    "woman": ["af_heart", "bf_emma", "af_nicole", "af_sarah"],
    "old_man": ["bm_lewis", "bm_daniel"],
    "old_woman": ["bf_isabella", "bf_alice"],
    "boy": ["am_puck"],
    "girl": ["af_sky", "af_bella"],
}


def pick_voices(kinds: dict[str, str], pools: dict[str, list[str]], narrator: str) -> dict[str, str]:
    """A distinct voice name for every speaker, by kind; never the narrator's."""
    chosen: dict[str, str] = {}
    used = {narrator}
    for speaker, kind in sorted(kinds.items()):
        if speaker == "narrator" or kind == "narrator":
            continue
        pool = pools.get(kind) or pools.get("man") or []
        pick = next((v for v in pool if v not in used), None) or (pool[len(chosen) % len(pool)] if pool else narrator)
        chosen[speaker] = pick
        used.add(pick)
    return chosen


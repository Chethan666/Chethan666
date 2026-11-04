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

from .timing import Chunk, Timeline, split_chunks

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
            scenes: list[dict[str, Any]], spans: list[tuple[int, int]]) -> tuple[np.ndarray, Timeline]:
    """Read the whole narration. Returns (mono audio, timeline with every chunk's times)."""
    chunks: list[Chunk] = split_chunks(narration, pauses)
    rate = voice.sample_rate
    pieces: list[np.ndarray] = []
    t = 0.0
    lead_in = np.zeros(int(0.6 * rate), dtype=np.float32)  # a breath before the first word
    pieces.append(lead_in)
    t += len(lead_in) / rate
    for chunk in chunks:
        audio = voice.say(respell(chunk.text, pronunciations), speed_for(chunk.start, spans, scenes))
        audio = _trim(audio, rate)
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

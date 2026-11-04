"""Voices for pipeline 4: Kokoro (free, on the CPU) and a silent stand-in for tests.

A copy of pipeline 2's narrator classes, kept here so the drama pipeline can change
without touching the stickman videos.
"""

from __future__ import annotations

import numpy as np


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

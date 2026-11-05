"""An expressive narrator and cast: ElevenLabs v3 on fal.

The narration is still read chunk by chunk (so every chunk's timing is exact), but every
chunk is sent at once, in parallel, before the reading starts. Each scene's delivery cue
becomes a v3 audio tag ([whispers], [nervously]...), so the narrator performs the story
instead of reading it. Any chunk that fails is read by Kokoro instead, at the same level.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import soundfile

from drama_production.media import MediaError

from .paid import OverBudget, Studio, digest

log = logging.getLogger(__name__)

TAGS = {"steady": "", "slow": "[slowly]", "hushed": "[whispers]", "tense": "[nervously]",
        "urgent": "[urgently]", "tender": "[softly]"}


def decode(path: Path, rate: int) -> np.ndarray:
    result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-ac", "1",
                             "-ar", str(rate), "-f", "f32le", "-"], capture_output=True)
    if result.returncode != 0:
        raise MediaError(f"could not decode {path.name}: {result.stderr.decode(errors='replace')[-200:]}")
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


class ElevenVoice:
    """Same interface as the Kokoro voice (say, speaker), plus `prepare` for parallel reading."""

    sample_rate = 44_100
    expressive = True

    def __init__(self, studio: Studio, voice: str, fallback: Any, work: Path, parallel: int = 4,
                 shared: dict[str, Any] | None = None):
        self.studio, self.voice, self.fallback, self.work, self.parallel = studio, voice, fallback, work, parallel
        self.shared = shared if shared is not None else {"failed": 0, "lines": 0, "lock": threading.Lock()}
        work.mkdir(parents=True, exist_ok=True)

    def speaker(self, voice: str, fallback: Any = None) -> "ElevenVoice":
        return ElevenVoice(self.studio, voice, fallback or self.fallback, self.work, self.parallel, self.shared)

    def _path(self, text: str) -> Path:
        return self.work / f"say_{digest(self.voice, text)}.mp3"

    def _make(self, text: str) -> Path | None:
        path = self._path(text)
        if path.exists() and path.stat().st_size:
            return path
        try:
            return self.studio.call("voice", {"text": text, "voice": self.voice}, "audio", path, "voice",
                                    units=max(len(text), 1) / 1000.0)
        except (MediaError, OverBudget) as exc:
            log.warning("ElevenLabs could not read %r (%s); Kokoro reads it", text[:60], str(exc)[:200])
            return None

    def prepare(self, texts: list[str]) -> None:
        todo = [t for t in dict.fromkeys(texts) if t.strip()]
        with ThreadPoolExecutor(max_workers=self.parallel) as pool:
            list(pool.map(self._make, todo))

    def say(self, text: str, speed: float = 1.0) -> np.ndarray:
        with self.shared["lock"]:
            self.shared["lines"] += 1
        path = self._make(text)
        if path is not None:
            try:
                return decode(path, self.sample_rate)
            except MediaError as exc:
                log.warning("%s", exc)
        with self.shared["lock"]:
            self.shared["failed"] += 1
        plain = " ".join(w for w in text.split() if not (w.startswith("[") and w.endswith("]")))
        from .voice import _resample

        audio = self.fallback.say(plain, speed)
        return _resample(audio, self.fallback.sample_rate, self.sample_rate)


def tagged(text: str, delivery: str) -> str:
    tag = TAGS.get(delivery, "")
    return f"{tag} {text}".strip() if tag else text


def write_wav(path: Path, audio: np.ndarray, rate: int) -> None:
    soundfile.write(path, audio, rate)

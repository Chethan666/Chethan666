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

from production.media import MediaError

from .paid import OverBudget, Studio, digest

log = logging.getLogger(__name__)

# A scene's delivery as a v3 audio tag, used when the performance director gave a sentence none.
# "slow" has no tag: [slowly] made whole scenes drag; pace comes from tempo and pauses instead.
TAGS = {"steady": "", "slow": "", "hushed": "[quietly]", "tense": "[nervously]",
        "urgent": "[urgently]", "tender": "[softly]"}


def decode(path: Path, rate: int, tempo: float = 1.0) -> np.ndarray:
    """Mono float samples; `tempo` speeds speech up (or slows it) without changing its pitch."""
    tempo = min(max(tempo, 0.5), 2.0)
    speed = ["-af", f"atempo={tempo:.3f}"] if abs(tempo - 1.0) > 0.005 else []
    result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), *speed, "-ac", "1",
                             "-ar", str(rate), "-f", "f32le", "-"], capture_output=True)
    if result.returncode != 0:
        raise MediaError(f"could not decode {path.name}: {result.stderr.decode(errors='replace')[-200:]}")
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


class ElevenVoice:
    """Same interface as the Kokoro voice (say, speaker), plus `prepare` for parallel reading."""

    sample_rate = 44_100
    expressive = True

    def __init__(self, studio: Studio, voice: str, fallback: Any, work: Path, parallel: int = 4,
                 shared: dict[str, Any] | None = None, pace: float = 1.0, takes: bool = False):
        self.studio, self.voice, self.fallback, self.work, self.parallel = studio, voice, fallback, work, parallel
        self.shared = shared if shared is not None else {"failed": 0, "lines": 0, "lock": threading.Lock()}
        self.pace = pace  # tempo applied to everything this voice reads (1.08 = 8% quicker, same pitch)
        self.takes = takes  # read whole passages in one take (voice.narrate)
        work.mkdir(parents=True, exist_ok=True)

    def speaker(self, voice: str, fallback: Any = None) -> "ElevenVoice":
        return ElevenVoice(self.studio, voice, fallback or self.fallback, self.work, self.parallel, self.shared,
                           self.pace, self.takes)

    def tempo(self, speed: float) -> float:
        """The scene's reading speed, half as strong as Kokoro's (v3 already acts the delivery)."""
        return self.pace * (1.0 + (speed - 1.0) * 0.5)

    def _path(self, text: str, attempt: int = 0) -> Path:
        return self.work / f"say_{digest(self.voice, text, attempt) if attempt else digest(self.voice, text)}.mp3"

    def _make(self, text: str, attempt: int = 0) -> Path | None:
        path = self._path(text, attempt)
        if path.exists() and path.stat().st_size:
            return path
        args = {"text": text, "voice": self.voice}
        try:  # a retry is a new request, not the cached first reading
            return self.studio.call("voice", args, "audio", path, "voice", units=max(len(text), 1) / 1000.0,
                                    key={"args": args, "attempt": attempt} if attempt else None)
        except (MediaError, OverBudget) as exc:
            log.warning("ElevenLabs could not read %r (%s); Kokoro reads it", text[:60], str(exc)[:200])
            return None

    def prepare(self, texts: list[str]) -> None:
        todo = [t for t in dict.fromkeys(texts) if t.strip()]
        with ThreadPoolExecutor(max_workers=self.parallel) as pool:
            list(pool.map(self._make, todo))

    def say(self, text: str, speed: float = 1.0, attempt: int = 0) -> np.ndarray:
        with self.shared["lock"]:
            self.shared["lines"] += 1
        path = self._make(text, attempt)
        if path is not None:
            try:
                return decode(path, self.sample_rate, self.tempo(speed))
            except MediaError as exc:
                log.warning("%s", exc)
        with self.shared["lock"]:
            self.shared["failed"] += 1
        plain = " ".join(w for w in text.split() if not (w.startswith("[") and w.endswith("]")))
        from .voice import _resample

        audio = self.fallback.say(plain, speed * self.pace)
        return _resample(audio, self.fallback.sample_rate, self.sample_rate)


def tagged(text: str, delivery: str) -> str:
    tag = TAGS.get(delivery, "")
    return f"{tag} {text}".strip() if tag else text


def write_wav(path: Path, audio: np.ndarray, rate: int) -> None:
    soundfile.write(path, audio, rate)

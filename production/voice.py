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


def delivery_for(offset: int, spans: list[tuple[int, int]], scenes: list[dict[str, Any]]) -> str:
    for (start, end), scene in zip(spans, scenes):
        if start <= offset < end:
            return str(scene.get("delivery", "steady"))
    return "steady"


def narrate(voice: Voice, narration: str, pauses: list[dict[str, Any]], pronunciations: list[dict[str, Any]],
            scenes: list[dict[str, Any]], spans: list[tuple[int, int]], lines: list[Any] | None = None,
            cast_voices: dict[str, Voice] | None = None, marks: dict[int, str] | None = None,
            gap_scale: float = 1.0) -> tuple[np.ndarray, Timeline]:
    """Read the whole narration. Returns (mono audio, timeline with every chunk's times).
    With `lines` and `cast_voices`, each quoted line is read in its speaker's own voice.

    An expressive voice with `takes` reads consecutive sentences of a paragraph in one take
    (up to TAKE_CHARS), so its intonation runs on from sentence to sentence like a real
    reader's instead of restarting flat at every full stop; the take is cut back into its
    sentences at its own pauses. `marks` (chunk start -> a v3 audio tag such as [whispers])
    is the performance director's delivery for each sentence; otherwise the scene's delivery
    is used. `gap_scale` shortens or lengthens every silence we add."""
    chunks: list[Chunk] = split_lines(narration, split_chunks(narration, pauses), lines or [])
    cast_voices = cast_voices or {}
    marks = marks or {}
    rate = voice.sample_rate
    expressive = getattr(voice, "expressive", False)
    plan = []  # (chunk, reader, tag, text, speed): what each voice reads
    for chunk in chunks:
        reader = cast_voices.get(chunk.speaker, voice) if chunk.speaker != "narrator" else voice
        text = respell(chunk.text, pronunciations)
        tag, speed = "", 1.0
        if reader is not voice:
            text = text.strip("\u201c\u201d\"").strip() or text
        else:
            speed = speed_for(chunk.start, spans, scenes)
            if expressive:  # an expressive voice performs the delivery
                from .speech import TAGS
                tag = marks.get(chunk.start, TAGS.get(delivery_for(chunk.start, spans, scenes), ""))
        plan.append((chunk, reader, tag, text, speed))
    # Group the narrator's sentences into takes: same paragraph, no story pause between, one speed.
    units: list[tuple[list[Chunk], Any, str, float]] = []
    group: list[tuple[Chunk, str, str]] = []
    group_speed = 1.0
    def close() -> None:
        if group:
            text, last_tag = [], None
            for _, tag, t in group:
                text.append(f"{tag} {t}".strip() if tag and tag != last_tag else t)
                last_tag = tag or last_tag
            units.append(([c for c, _, _ in group], voice, " ".join(text), group_speed))
            group.clear()
    takes = expressive and getattr(voice, "takes", False)
    for chunk, reader, tag, text, speed in plan:
        if reader is not voice:
            close()
            prev = units[-1] if units else None
            if (prev is not None and prev[1] is reader and getattr(reader, "takes", False)
                    and prev[0][-1].speaker == chunk.speaker and prev[0][-1].gap_ms <= TAKE_JOIN_MS
                    and len(prev[2]) + len(text) <= TAKE_CHARS):  # the host's intro flows too
                units[-1] = (prev[0] + [chunk], reader, f"{prev[2]} {text}", 1.0)
            else:
                units.append(([chunk], reader, text, 1.0))
            continue
        joinable = (takes and group and speed == group_speed and group[-1][0].gap_ms <= TAKE_JOIN_MS
                    and sum(len(t) for _, _, t in group) + len(text) <= TAKE_CHARS)
        if not joinable:
            close()
            group_speed = speed
        group.append((chunk, tag, text))
        if not takes:
            close()
    close()
    readers: dict[int, tuple[Any, list[str]]] = {}
    for _, reader, text, _ in units:
        readers.setdefault(id(reader), (reader, []))[1].append(text)
    for reader, texts in readers.values():  # paid voices read every take at once, in parallel
        if hasattr(reader, "prepare"):
            reader.prepare(texts)
    pieces: list[np.ndarray] = []
    lead_in = np.zeros(int(0.6 * rate), dtype=np.float32)  # a breath before the first word
    pieces.append(lead_in)
    t = len(lead_in) / rate
    for group_chunks, reader, text, speed in units:
        if reader is not voice:
            audio = _resample(reader.say(text, 1.0), reader.sample_rate, rate)
        else:
            audio = voice.say(text, speed)
        audio = _trim(audio, rate)
        if reader is not voice:  # the cast sit a little behind the narrator in the mix
            audio = _match_level(audio, 0.85)
        cuts = split_take(audio, rate, [len(c.text) for c in group_chunks])
        for chunk, (a, b) in zip(group_chunks, zip([0] + cuts, cuts + [audio.size])):
            chunk.t0, chunk.t1 = t + a / rate, t + b / rate
        pieces.append(audio)
        last = group_chunks[-1]
        gap = np.zeros(int(last.gap_ms * gap_scale / 1000 * rate), dtype=np.float32)
        pieces.append(gap)
        t = last.t1 + len(gap) / rate
    tail = np.zeros(int(1.5 * rate), dtype=np.float32)  # let the last line breathe
    pieces.append(tail)
    audio = np.concatenate(pieces)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio * (0.89 / peak)
    return audio.astype(np.float32), Timeline(narration, chunks, len(audio) / rate)


TAKE_CHARS = 600  # longest take: a few sentences, well inside the voice model's limit
TAKE_JOIN_MS = 300  # sentences separated by no more than an ordinary sentence gap share a take


def split_take(audio: np.ndarray, rate: int, weights: list[int]) -> list[int]:
    """Where one take's sentences meet: the pause nearest each boundary the text predicts.
    Returns len(weights) - 1 sample positions, in order."""
    n = len(weights)
    if n <= 1 or audio.size == 0:
        return []
    hop = max(int(0.01 * rate), 1)
    frames = audio[: audio.size // hop * hop].reshape(-1, hop)
    energy = np.sqrt(np.mean(frames ** 2, axis=1)) if frames.size else np.zeros(0)
    quiet = energy < max(0.004, 0.08 * float(np.percentile(energy, 90)) if energy.size else 0.0)
    runs: list[tuple[int, int]] = []  # (centre sample, length in frames) of every pause of 80 ms or more
    i = 0
    while i < quiet.size:
        if quiet[i]:
            j = i
            while j < quiet.size and quiet[j]:
                j += 1
            if j - i >= 8:
                runs.append(((i + j) // 2 * hop, j - i))
            i = j
        else:
            i += 1
    total = float(sum(max(w, 1) for w in weights))
    cuts: list[int] = []
    acc, previous = 0.0, 0
    for k, w in enumerate(weights[:-1]):
        acc += max(w, 1)
        expected = int(audio.size * acc / total)
        window = 0.35 * audio.size / n
        near = [(abs(c - expected) / (1 + 0.15 * length), c) for c, length in runs
                if c > previous and abs(c - expected) <= window]
        cut = min(near)[1] if near else max(expected, previous + 1)
        cuts.append(min(cut, audio.size - (n - 1 - k)))
        previous = cuts[-1]
    return cuts


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


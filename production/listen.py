"""The listener: checks that the narrator said what the script says, and when.

An expressive voice model can skip a word, repeat a phrase or invent one, more often in long
takes. The listener transcribes each take with Whisper (faster-whisper, MIT, on the CPU, free)
and compares it word by word with the text. A take that matches well enough is cut into its
sentences at the exact word times; one that doesn't is read again sentence by sentence.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

WORD = re.compile(r"[a-z0-9']+")


def tokens(text: str) -> list[str]:
    """Spoken words, lower case, without punctuation or v3 tags ([whispers])."""
    return WORD.findall(re.sub(r"\[[^\]]*\]", " ", text.lower()).replace("’", "'"))


class Listener:
    """Whisper on the CPU. `words(audio, rate)` -> [(word, start s, end s)]."""

    def __init__(self, model: str = "base.en"):
        self.model_name = model
        self._model: Any = None
        self.report: dict[str, Any] = {"model": model, "takes": 0, "reread": 0, "worst_match": 1.0}

    def _load(self) -> Any:  # pragma: no cover - needs the model download
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
        return self._model

    def words(self, audio: np.ndarray, rate: int) -> list[tuple[str, float, float]]:  # pragma: no cover
        from .voice import _resample
        clip = _resample(audio, rate, 16_000)
        segments, _ = self._load().transcribe(clip, language="en", beam_size=1, word_timestamps=True,
                                              condition_on_previous_text=False, vad_filter=False)
        return [(w.word, float(w.start), float(w.end)) for seg in segments for w in (seg.words or [])]

    def check(self, audio: np.ndarray, rate: int, sentences: list[str]) -> tuple[float, list[int] | None]:
        """How well the take matches its sentences (0-1), and where the sentences meet (samples),
        or None when a boundary cannot be placed from the words heard."""
        if self.report.get("unavailable"):
            return 1.0, None
        try:
            heard = self.words(audio, rate)
        except Exception as exc:  # no model download, a broken install: the video goes on unchecked
            log.warning("The listener is unavailable (%s): the narration is not checked by ear", str(exc)[:200])
            self.report["unavailable"] = str(exc)[:200]
            return 1.0, None
        score, cuts = align(sentences, heard, rate, audio.size)
        self.report["takes"] += 1
        self.report["worst_match"] = round(min(self.report["worst_match"], score), 2)
        return score, cuts


def _close(a: str, b: str) -> bool:
    """One letter added, dropped or changed, in words of four letters or more ("mummy's", "mommy's")."""
    if min(len(a), len(b)) < 4 or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    short, long = (a, b) if len(a) < len(b) else (b, a)
    return any(long[:k] + long[k + 1:] == short for k in range(len(long)))


def align(sentences: list[str], heard: list[tuple[str, float, float]], rate: int,
          size: int) -> tuple[float, list[int] | None]:
    expected: list[tuple[str, int]] = [(t, i) for i, s in enumerate(sentences) for t in tokens(s)]
    vocabulary = {t for t, _ in expected}
    said: list[tuple[str, float, float]] = []
    for word, a, b in heard:
        for t in tokens(word):
            if t not in vocabulary:  # a spelling the script doesn't use: the script's word if it is that close
                t = next((v for v in vocabulary if _close(t, v)), t)
            said.append((t, a, b))
    if not expected:
        return 1.0, None
    matcher = SequenceMatcher(None, [t for t, _ in expected], [t for t, _, _ in said], autojunk=False)
    pairs: dict[int, int] = {}
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            pairs[block.a + k] = block.b + k
    # How much of the script was heard, and how much heard was not in the script.
    recall = len(pairs) / len(expected)
    precision = len(pairs) / max(len(said), 1)
    score = min(recall, precision + 0.1)
    cuts: list[int] = []
    for i in range(len(sentences) - 1):
        last = [pairs[k] for k, (_, s) in enumerate(expected) if s == i and k in pairs]
        first = [pairs[k] for k, (_, s) in enumerate(expected) if s == i + 1 and k in pairs]
        if not last or not first:
            return score, None
        end, start = said[max(last)][2], said[min(first)][1]
        cut = int((end + max(start, end)) / 2 * rate)
        if cuts and cut <= cuts[-1]:
            return score, None
        cuts.append(min(max(cut, 1), size - 1))
    return score, cuts


def make(cfg: dict[str, Any]) -> Listener | None:
    """A listener when the config asks for one and faster-whisper is installed."""
    name = str(cfg.get("listener", "base.en") or "")
    if not name:
        return None
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        log.warning("faster-whisper is not installed: the narration is not checked by ear")
        return None
    return Listener(name)

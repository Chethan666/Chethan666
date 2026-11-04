"""Splitting the narration into spoken chunks, and timing scenes and captions.

The narration is read chunk by chunk (a sentence, or part of one where the story asks
for a dramatic pause), so every chunk's start and end time is known exactly. Scenes and
captions are then timed from the words they contain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

SENTENCE = re.compile(r"[^.!?…]+(?:[.!?…]+[\"'”’)]*|$)", re.UNICODE)
WORD = re.compile(r"\S+")


@dataclass
class Chunk:
    text: str
    start: int  # character offsets in the narration
    end: int
    gap_ms: int  # silence after the chunk
    t0: float = 0.0  # seconds, filled in after the voice has read it
    t1: float = 0.0
    speaker: str = "narrator"  # who reads it: the narrator, or the character whose line it is


@dataclass
class Timeline:
    narration: str
    chunks: list[Chunk] = field(default_factory=list)
    duration: float = 0.0

    def time_at(self, offset: int) -> float:
        """Time at which the narrator reaches a character offset (linear within a chunk)."""
        for chunk in self.chunks:
            if offset < chunk.end:
                if offset <= chunk.start:
                    return chunk.t0
                share = (offset - chunk.start) / max(chunk.end - chunk.start, 1)
                return chunk.t0 + share * (chunk.t1 - chunk.t0)
        return self.duration


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def split_chunks(narration: str, pauses: list[dict[str, Any]] | None = None, *, sentence_gap: int = 280,
                 paragraph_gap: int = 700, section_gap: int = 1100, max_words: int = 40) -> list[Chunk]:
    """Spoken chunks with the silence after each.

    A story pause ("after this phrase, stop N ms") cuts a chunk at that phrase. Paragraph
    and section breaks (the Storyteller's intro and outro) get longer silences. Very long
    sentences are split at a comma so the voice model never reads a wall of text.
    """
    chunks: list[Chunk] = []
    paragraphs = [(m.start(), m.end()) for m in re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", narration)]
    for p_index, (p_start, p_end) in enumerate(paragraphs):
        text = narration[p_start:p_end]
        for m in SENTENCE.finditer(text):
            s, e = p_start + m.start(), p_start + m.end()
            while s < e and narration[s].isspace():
                s += 1
            while e > s and narration[e - 1].isspace():
                e -= 1
            if s >= e:
                continue
            for a, b in _split_long(narration, s, e, max_words):
                chunks.append(Chunk(narration[a:b], a, b, sentence_gap))
        if chunks:
            last_paragraph = p_index == len(paragraphs) - 1
            chunks[-1].gap_ms = 0 if last_paragraph else paragraph_gap
    # Section breaks: the narration joins intro, body and outro with blank lines; the
    # first and last paragraphs are the Storyteller's when there are more than two.
    if len(paragraphs) > 2:
        for boundary in (paragraphs[0][1], paragraphs[-2][1]):
            for chunk in chunks:
                if chunk.end == boundary:
                    chunk.gap_ms = section_gap
    for pause in pauses or []:
        _apply_pause(narration, chunks, str(pause.get("after", "")), int(pause.get("ms") or 800))
    return chunks


def _split_long(narration: str, s: int, e: int, max_words: int) -> list[tuple[int, int]]:
    words = narration[s:e].split()
    if len(words) <= max_words:
        return [(s, e)]
    middle = s + (e - s) // 2
    commas = [m.end() for m in re.finditer(r"[,;:]\s", narration[s:e])]
    cut = min((s + c for c in commas), key=lambda c: abs(c - middle), default=None)
    if cut is None or cut <= s or cut >= e:
        cut = narration.rfind(" ", s, middle) + 1 or middle
    left_end = cut
    while left_end > s and narration[left_end - 1].isspace():
        left_end -= 1
    return _split_long(narration, s, left_end, max_words) + _split_long(narration, cut, e, max_words)


def _apply_pause(narration: str, chunks: list[Chunk], phrase: str, ms: int) -> None:
    target = _norm(phrase)
    if not target:
        return
    for i, chunk in enumerate(chunks):
        if target not in _norm(chunk.text):
            continue
        # Find where the phrase ends inside the chunk, in raw characters.
        words = list(WORD.finditer(chunk.text))
        phrase_words = target.split()
        for w in range(len(words) - len(phrase_words) + 1):
            window = _norm(" ".join(x.group() for x in words[w:w + len(phrase_words)]))
            if window == target:
                end = chunk.start + words[w + len(phrase_words) - 1].end()
                tail = narration[end:chunk.end]
                trailing = re.match(r"^[^\w\s]*", tail).group()  # keep closing punctuation
                end += len(trailing)
                if end >= chunk.end - 1:
                    chunk.gap_ms = max(chunk.gap_ms, ms)
                else:
                    rest_start = end
                    while rest_start < chunk.end and narration[rest_start].isspace():
                        rest_start += 1
                    second = Chunk(narration[rest_start:chunk.end], rest_start, chunk.end, chunk.gap_ms)
                    chunk.text, chunk.end, chunk.gap_ms = narration[chunk.start:end], end, ms
                    chunks.insert(i + 1, second)
                return
    return


def scene_spans(narration: str, scenes: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Character span of each scene in the narration.

    Scenes quote the narration exactly and in order; a scene whose text can't be found is
    placed by word count after the previous one, so timing never breaks.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    words = [(m.start(), m.end()) for m in WORD.finditer(narration)]
    for scene in scenes:
        text = str(scene.get("text", "")).strip()
        at = narration.find(text, cursor) if text else -1
        if at < 0 and text:
            at = _fuzzy_find(narration, text, cursor)
        if at < 0:
            count = max(len(text.split()), 1)
            first = next((k for k, (ws, _) in enumerate(words) if ws >= cursor), len(words))
            last = min(first + count, len(words)) - 1
            at = words[first][0] if first < len(words) else cursor
            end = words[last][1] if last >= first and last < len(words) else at
        else:
            end = at + len(text)
        spans.append((at, end))
        cursor = end
    if spans:  # scenes together cover the whole narration
        spans[0] = (0, spans[0][1])
        spans[-1] = (spans[-1][0], len(narration))
    return spans


def _fuzzy_find(narration: str, text: str, cursor: int) -> int:
    """Start of `text` in the narration ignoring punctuation and spacing, or -1."""
    head = " ".join(text.split()[:6])
    pattern = r"\W+".join(re.escape(w) for w in re.findall(r"\w+", head))
    if not pattern:
        return -1
    m = re.compile(pattern, re.IGNORECASE).search(narration, cursor)
    return m.start() if m else -1


@dataclass
class Cue:
    start: float
    end: float
    text: str


def caption_cues(timeline: Timeline, max_chars: int = 44) -> list[Cue]:
    """Captions of at most two short lines' worth, timed within each spoken chunk."""
    cues: list[Cue] = []
    for chunk in timeline.chunks:
        words = chunk.text.split()
        # Split into as few captions as fit, of even length (no one-word leftovers).
        parts = max(1, -(-len(chunk.text) // max_chars))
        target = len(chunk.text) / parts
        lines: list[list[str]] = [[]]
        for word in words:
            if lines[-1] and len(lines) < parts and len(" ".join(lines[-1] + [word])) > target + 4:
                lines.append([])
            lines[-1].append(word)
        total = sum(len(" ".join(line)) for line in lines) or 1
        t = chunk.t0
        for line in lines:
            share = len(" ".join(line)) / total
            end = t + share * (chunk.t1 - chunk.t0)
            cues.append(Cue(t, end, " ".join(line)))
            t = end
    return cues


def _stamp(seconds: float, sep: str = ",") -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def srt(cues: list[Cue]) -> str:
    return "\n".join(f"{i}\n{_stamp(c.start)} --> {_stamp(c.end)}\n{c.text}\n" for i, c in enumerate(cues, 1))


def ass(cues: list[Cue], width: int, height: int, font: str = "DejaVu Sans") -> str:
    """Burned-in captions: white on a soft dark box, low in the frame, readable on phones."""
    size = round(height * 0.052)

    def t(seconds: float) -> str:
        cs = int(round(seconds * 100))
        h, cs = divmod(cs, 360_000)
        m, cs = divmod(cs, 6_000)
        s, cs = divmod(cs, 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}", f"PlayResY: {height}", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Caption,{font},{size},&H00F2F2F2,&H00F2F2F2,&H00101010,&H90000000,1,0,0,0,100,100,0,0,"
        f"1,{max(2, size // 14)},0,2,80,80,{round(height * 0.07)},1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for c in cues:
        text = c.text.replace("{", "(").replace("}", ")").replace("\n", " ")
        lines.append(f"Dialogue: 0,{t(c.start)},{t(c.end)},Caption,,0,0,0,,{text}")
    return "\n".join(lines) + "\n"


def split_lines(narration: str, chunks: list[Chunk], lines: list[Any], handoff_ms: int = 90) -> list[Chunk]:
    """Cut chunks where a quoted line starts or ends, so each character reads their own words.
    `lines` have start, end (the quote's offsets, marks included) and speaker."""
    if not lines:
        return chunks
    out: list[Chunk] = []
    for chunk in chunks:
        cuts = sorted({x for ln in lines for x in (ln.start, ln.end) if chunk.start < x < chunk.end})
        bounds = [chunk.start, *cuts, chunk.end]
        pieces: list[Chunk] = []
        for a, b in zip(bounds, bounds[1:]):
            while a < b and narration[a].isspace():
                a += 1
            while b > a and narration[b - 1].isspace():
                b -= 1
            text = narration[a:b]
            if not re.search(r"\w", text):  # a lone mark or comma joins the piece before it
                if pieces:
                    pieces[-1].text, pieces[-1].end = narration[pieces[-1].start:b], b
                continue
            who = next((ln.speaker for ln in lines if ln.start <= a and b <= ln.end), "narrator")
            pieces.append(Chunk(text, a, b, handoff_ms, speaker=who))
        if not pieces:
            out.append(chunk)
            continue
        pieces[-1].gap_ms = chunk.gap_ms
        # Speech in the same voice on both sides of a cut reads as one chunk again.
        merged = [pieces[0]]
        for piece in pieces[1:]:
            if piece.speaker == merged[-1].speaker:
                merged[-1].text = narration[merged[-1].start:piece.end]
                merged[-1].end, merged[-1].gap_ms = piece.end, piece.gap_ms
            else:
                merged.append(piece)
        out.extend(merged)
    return out


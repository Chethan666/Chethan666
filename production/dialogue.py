"""Characters speak their own lines.

The narration is a told story with sparse, clearly attributed dialogue. Every quoted line
is found, an open model says who speaks it (a cast member, a stranger, or the narrator
quoting themself) and what kind of voice they have, and the voice step reads that line in
the character's own voice while the narrator reads the rest. If the model fails, the
words around the quote decide ("she whispered", "Delano said").
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from story_factory.llm import LLMError

log = logging.getLogger(__name__)

QUOTE = re.compile(r"[“\"]([^”\"\n]{1,400})[”\"]")
KINDS = ("man", "woman", "old_man", "old_woman", "boy", "girl", "narrator")
SAY = (r"(?:said|says|asked|asks|whispered|whispers|shouted|shouts|yelled|screamed|called|cried|muttered|"
       r"murmured|replied|answered|told|hissed|breathed|added|begged|sang|laughed)")

SYSTEM = ("You attribute dialogue in a horror story so each line can be voiced by the right actor. "
          "Answer with JSON only.")
SCHEMA = {
    "type": "object", "required": ["lines", "speakers"],
    "properties": {
        "lines": {"type": "array", "items": {"type": "object", "required": ["index", "speaker"],
                                             "properties": {"index": {"type": "integer"},
                                                            "speaker": {"type": "string"}}}},
        "speakers": {"type": "array", "items": {"type": "object", "required": ["name", "voice"],
                                                "properties": {"name": {"type": "string"},
                                                               "voice": {"type": "string", "enum": list(KINDS)}}}},
    },
}


STRANGER = re.compile(rf"\b(?:a|an|the)\s+(old\s+)?(woman|man|girl|boy)(?:'s voice)?\s+{SAY}\b[^\"“]{{0,20}}$",
                      re.IGNORECASE)
STRANGER_KINDS = {"the woman": "woman", "the man": "man", "the old woman": "old_woman", "the old man": "old_man",
                  "the girl": "girl", "the boy": "boy"}


@dataclass
class Line:
    start: int  # offsets of the quote, marks included
    end: int
    text: str  # what is spoken, without the marks
    speaker: str = "narrator"


def find(narration: str) -> list[Line]:
    return [Line(m.start(), m.end(), m.group(1).strip()) for m in QUOTE.finditer(narration) if m.group(1).strip()]


def _context(narration: str, line: Line, width: int = 220) -> str:
    before = narration[max(0, line.start - width):line.start]
    after = narration[line.end:line.end + width]
    return f"...{before}[[{narration[line.start:line.end]}]]{after}..."


def attribute(router: Any, narration: str, cast: list[dict[str, Any]],
              batch: int = 25) -> tuple[list[Line], dict[str, str]]:
    """Who speaks each quoted line, and each speaker's kind of voice. Returns (lines, voice kinds)."""
    lines = find(narration)
    names = [str(c.get("name", "")) for c in cast if c.get("name")]
    kinds: dict[str, str] = {"narrator": "narrator"}
    if not lines:
        return lines, kinds
    answered: dict[int, str] = {}
    if router is not None:
        for start in range(0, len(lines), batch):
            group = lines[start:start + batch]
            listing = "\n".join(f"{i}. {_context(narration, ln)}" for i, ln in enumerate(group, start + 1))
            prompt = (f"Cast: {json.dumps([{'name': c.get('name'), 'look': c.get('look', '')} for c in cast])}\n"
                      "The story is told by a narrator, often in the first person. For each numbered quote "
                      "(the text in [[ ]]), say who speaks it: a cast name exactly as written, \"narrator\" when "
                      "the narrator or the first-person teller says it, or a short label for a minor speaker "
                      "(\"the operator\"). Then give every speaker a voice: man, woman, old_man, old_woman, "
                      "boy, girl, or narrator.\n\n" + listing)
            try:
                result = router.one("planner").generate_json("dialogue_speakers", SYSTEM, prompt, SCHEMA)
            except (LLMError, ValueError, TypeError) as exc:
                log.warning("Dialogue attribution fell back to the text around the quotes: %s", exc)
                continue
            for item in result.get("lines", []):
                if isinstance(item, dict) and plausible(str(item.get("speaker", ""))):
                    answered[int(item.get("index", 0))] = str(item["speaker"]).strip()
            for item in result.get("speakers", []):
                if isinstance(item, dict) and item.get("voice") in KINDS and plausible(str(item.get("name", ""))):
                    kinds[_canonical(str(item.get("name", "")), names)] = item["voice"]
    for i, line in enumerate(lines, 1):
        line.speaker = _canonical(answered.get(i) or guess(narration, line, names), names)
    for line in lines:
        kinds.setdefault(line.speaker, STRANGER_KINDS.get(line.speaker, "man"))
    return lines, kinds


def plausible(name: str) -> bool:
    """A speaker name, not a model's reasoning pasted into the field (seen in a live run)."""
    name = name.strip()
    return 0 < len(name) <= 40 and "\n" not in name and len(name.split()) <= 5 and "?" not in name


def _canonical(name: str, names: list[str]) -> str:
    low = name.strip().lower()
    if low in ("", "narrator", "i", "me", "the narrator", "storyteller", "the storyteller"):
        return "narrator"
    for n in names:
        if low == n.lower() or low == n.lower().split()[0]:
            return n
    return name.strip()


def guess(narration: str, line: Line, names: list[str]) -> str:
    """From the words around the quote: "...," Delano said / she whispered / I said."""
    after = narration[line.end:line.end + 60]
    before = narration[max(0, line.start - 80):line.start]
    for n in names:
        first = re.escape(n.split()[0])
        if re.match(rf"\s*,?\s*(?:{first}\b[^.]{{0,20}}{SAY}|{SAY}\s+{first}\b)", after, re.IGNORECASE):
            return n
    if re.match(rf"\s*,?\s*I\s+{SAY}", after):
        return "narrator"
    stranger = STRANGER.search(before[-60:])  # 'A woman's voice whispered, "..."'
    if stranger:
        return f"the {(stranger.group(1) or '').lower()}{stranger.group(2).lower()}"
    mentioned = [n for n in names if re.search(rf"\b{re.escape(n.split()[0])}\b", before)]
    return mentioned[-1] if mentioned else "narrator"

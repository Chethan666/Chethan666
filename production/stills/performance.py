"""The performance director: how each sentence of the narration is delivered.

A real storyteller doesn't read a horror story in one tone. They speak plainly most of the
time, drop their voice for the line that matters, hesitate before the reveal and hurry when
things go wrong. The planner model reads the narration sentence by sentence and gives each
one at most one ElevenLabs v3 audio tag from a fixed list (or none). The words are never
changed. Rules keep it believable: most sentences plain, whispers rare and never back to
back. Without a model, the scenes' delivery cues are used as before.
"""

from __future__ import annotations

import logging
from typing import Any

from story_factory.llm import LLMError

from .. import timing

log = logging.getLogger(__name__)

# v3 tags that suit a horror narrator. One word each, so a fallback voice can strip them.
ALLOWED = ("[quietly]", "[whispers]", "[softly]", "[nervously]", "[hesitantly]", "[fearfully]",
           "[urgently]", "[ominously]", "[sighs]", "[exhales]", "[gasps]", "[curious]")
SYSTEM = ("You are the performance director for Hollow Lines, a horror storytelling channel. You direct "
          "the narrator's voice sentence by sentence so the story sounds told by a real person, not read "
          "by a machine. Answer with JSON only.")
SCHEMA = {"type": "object", "required": ["marks"], "properties": {"marks": {"type": "array", "items": {
    "type": "object", "required": ["n", "tag"],
    "properties": {"n": {"type": "integer"}, "tag": {"type": "string"}}}}}}


def sentences(narration: str, pauses: list[dict[str, Any]], lines: list[Any]) -> list[timing.Chunk]:
    """The narrator's sentences, split exactly as voice.narrate splits them."""
    chunks = timing.split_lines(narration, timing.split_chunks(narration, pauses), lines or [])
    return [c for c in chunks if c.speaker == "narrator"]


def prompt(title: str, chunks: list[timing.Chunk], scene_of: list[dict[str, Any]]) -> str:
    rows = []
    for i, (c, sc) in enumerate(zip(chunks, scene_of), 1):
        rows.append(f"{i}. [{sc.get('mood', '')}/{sc.get('delivery', 'steady')}"
                    f"{', KEY MOMENT' if sc.get('key_moment') else ''}] {c.text}")
    return (f"Story: {title}\n\n" + "\n".join(rows) +
            "\n\nIn brackets: the scene's mood and delivery. For every numbered sentence choose its delivery "
            f"tag: one of {', '.join(ALLOWED)}, or \"\" to say it plainly.\n"
            "How a gripping horror narrator sounds:\n"
            "- Plain, natural speech carries most of the story: leave at least half the sentences \"\".\n"
            "- Change the colour from sentence to sentence; never the same tag more than twice in a row.\n"
            "- [whispers] only for one or two short lines that land a chill (never twice in a row).\n"
            "- Build: calm and curious early, [nervously]/[hesitantly] as things turn wrong, "
            "[fearfully]/[urgently] at the scare, [quietly] or [ominously] for the final line.\n"
            "- [sighs], [exhales], [gasps] at most once each, where a person really would.\n"
            "Return every sentence once: {\"marks\": [{\"n\": 1, \"tag\": \"\"}, ...]}")


def direct(router: Any, story: dict[str, Any], narration: str, pauses: list[dict[str, Any]], lines: list[Any],
           scenes: list[dict[str, Any]], spans: list[tuple[int, int]]) -> tuple[dict[int, str], dict[str, Any]]:
    """Chunk start -> tag ("" = plain) for every narrator sentence, and a short report."""
    chunks = sentences(narration, pauses, lines)
    if router is None or not chunks:
        return {}, {"directed": False}
    scene_of = []
    for c in chunks:
        sc = next((s for (a, b), s in zip(spans, scenes) if a <= c.start < b), {})
        scene_of.append(sc)
    try:
        result = router.one("planner").generate_json(
            "performance", SYSTEM, prompt(str(story.get("title", "")), chunks, scene_of), SCHEMA)
    except (LLMError, ValueError, TypeError) as exc:
        log.warning("Performance by scene cues: %s", exc)
        return {}, {"directed": False, "error": str(exc)[:200]}
    raw: dict[int, str] = {}
    unknown: list[str] = []
    for m in result.get("marks", []) if isinstance(result, dict) else []:
        try:
            n, tag = int(m.get("n")), normal(m.get("tag"))
        except (AttributeError, TypeError, ValueError):
            continue
        if 1 <= n <= len(chunks):
            raw[n] = tag if tag in ALLOWED else ""
            if tag and tag not in ALLOWED:
                unknown.append(tag)
    if len(raw) < len(chunks) / 2:  # too little answered to trust: keep the scene cues
        return {}, {"directed": False, "answered": len(raw)}
    tags = rules([raw.get(i, "") for i in range(1, len(chunks) + 1)])
    marks = {c.start: tag for c, tag in zip(chunks, tags)}
    used = [t for t in tags if t]
    return marks, {"directed": True, "sentences": len(chunks), "plain": len(tags) - len(used),
                   "tags": {t: used.count(t) for t in sorted(set(used))}, "unknown_tags": unknown[:10]}


def normal(tag: Any) -> str:
    """A tag as the model may write it ("whispers", "Whispers", "[whispers]", "none") -> "[whispers]" or ""."""
    t = str(tag or "").strip().lower().strip("[]()\"' .").strip()
    return "" if t in ("", "none", "plain", "neutral", "normal", "-") else f"[{t}]"


def rules(tags: list[str]) -> list[str]:
    """What the model is told, enforced: no whisper twice in a row, no tag three times running,
    each breath sound once, whispers on at most a fifth of the sentences."""
    out: list[str] = []
    whispers, once = 0, set()
    limit = max(1, len(tags) // 5)
    for tag in tags:
        if tag == "[whispers]" and ((out and out[-1] == tag) or whispers >= limit):
            tag = "[quietly]"
        if tag in ("[sighs]", "[exhales]", "[gasps]"):
            tag = "" if tag in once else tag
            once.add(tag)
        if tag and len(out) >= 2 and out[-1] == out[-2] == tag:
            tag = ""
        whispers += tag == "[whispers]"
        out.append(tag)
    return out

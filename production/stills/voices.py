"""Casting and voices: the host frames the story, the teller tells it, characters speak.

- The Storyteller (host) reads the intro and the outro, in the third person, in one fixed
  channel voice.
- A first-person story is told by its teller in a voice of their own (a man, a woman, an
  older person: decided from the story), so it sounds like a real person's account; a
  third-person story is told by the host throughout.
- Every quoted line is read by its speaker in a distinct voice (never the host's or the
  teller's).
ElevenLabs v3 (on fal) performs every part when FAL_KEY is set; Kokoro reads any line it
cannot, and everything when there is no key.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from story_factory.llm import LLMError

from .. import dialogue, timing, voice
from ..run import _json
from ..speech import ElevenVoice
from .board import Board

log = logging.getLogger(__name__)

FIRST = re.compile(r"\b(I|me|my|mine|myself|I'm|I'd|I've|I'll)\b")
TELLER_KINDS = ("man", "woman", "old_man", "old_woman")
TELLER_SCHEMA = {"type": "object", "required": ["kind", "reason"],
                 "properties": {"kind": {"type": "string", "enum": list(TELLER_KINDS)},
                                "reason": {"type": "string"}}}


def point_of_view(body: str) -> str:
    """'first' when the teller says I (outside the quotes) often enough, else 'third'."""
    told = dialogue.QUOTE.sub(" ", body)
    words = max(len(told.split()), 1)
    return "first" if len(FIRST.findall(told)) / words >= 0.015 else "third"


def teller_kind(router: Any, story: dict[str, Any], body: str, default: str = "man") -> tuple[str, str]:
    """Who the first-person teller is, so the voice fits them."""
    if router is None:
        return default, "default"
    hero = (_json(story.get("bible"), {}) or {}).get("hero") or {}
    prompt = (f"Hero in the story bible: {hero}\n\nOpening of the story, told in the first person:\n"
              f"{body[:2500]}\n\nWho is telling this story (the 'I')? Pick the voice that fits them: man, "
              "woman, old_man or old_woman.")
    try:
        result = router.one("planner").generate_json(
            "teller_voice", "You cast voice actors for horror audio stories. JSON only.", prompt, TELLER_SCHEMA)
    except (LLMError, ValueError, TypeError) as exc:
        log.warning("Teller voice by default: %s", exc)
        return default, "default"
    kind = result.get("kind") if isinstance(result, dict) else None
    return (kind, str(result.get("reason", ""))[:200]) if kind in TELLER_KINDS else (default, "default")


def frame_spans(story: dict[str, Any], narration: str) -> list[tuple[int, int]]:
    """Where the host's intro and outro sit in the narration."""
    spans = []
    intro, outro = str(story.get("intro") or "").strip(), str(story.get("outro") or "").strip()
    if intro and narration.startswith(intro) and len(intro) < len(narration):
        spans.append((0, len(intro)))
    if outro and narration.endswith(outro) and len(outro) < len(narration):
        spans.append((len(narration) - len(outro), len(narration)))
    return spans


class CastingAgent:
    """Decides who reads what and with which voice."""

    name = "casting"
    critical = True

    def run(self, board: Board) -> dict[str, Any]:
        story, cfg = board.story, board.cfg
        narration = str(story.get("narration") or "").strip()
        scenes = _json(story.get("scenes"), [])
        if not narration or not scenes:
            raise ValueError(f"story #{story.get('id')} has no narration or scene plan")
        cast = [c for c in _json(story.get("cast"), []) if isinstance(c, dict)]
        body = str(story.get("body") or narration)
        pov = cfg.get("point_of_view") or point_of_view(body)
        kind, why = (teller_kind(board.router, story, body, cfg.get("teller_default", "man"))
                     if pov == "first" else ("host", "third-person story: the host tells it"))
        lines, kinds = dialogue.attribute(board.router, narration, cast)
        paid = board.studio is not None and "voice" in board.studio.models
        engine = "elevenlabs" if paid else "kokoro"
        host_name = cfg.get("host_voice", {}).get(engine)
        teller_name = cfg.get("teller_voices", {}).get(engine, {}).get(kind, host_name) if pov == "first" else host_name
        taken = {host_name, teller_name}
        pools = {k: [v for v in names if v not in taken]
                 for k, names in (cfg.get("cast_voices", {}).get(engine) or {}).items()}
        character_names = voice.pick_voices(kinds, pools, host_name)
        board.data.update(narration=narration, scenes=scenes, cast=cast, lines=lines, pov=pov,
                          frames=frame_spans(story, narration),
                          kinds=kinds,
                          casting={"engine": engine, "host": host_name, "teller": teller_name,
                                   "teller_kind": kind, "characters": character_names})
        return {"point_of_view": pov, "teller": kind, "why": why, "engine": engine, "host_voice": host_name,
                "teller_voice": teller_name, "quoted_lines": len(lines), "character_voices": character_names}


class VoiceAgent:
    """Reads the whole narration with the cast's voices; every chunk is timed."""

    name = "voice"
    critical = True

    def _speaker(self, board: Board, eleven: ElevenVoice | None, name: str, kokoro_name: str) -> Any:
        kokoro = board.kokoro
        free = kokoro.speaker(kokoro_name, getattr(kokoro, "base_speed", 1.0)) if hasattr(kokoro, "speaker") else kokoro
        return eleven.speaker(name, free) if eleven is not None else free

    def run(self, board: Board) -> dict[str, Any]:
        cfg, d = board.cfg, board.data
        casting = d["casting"]
        eleven = None
        if casting["engine"] == "elevenlabs":
            eleven = ElevenVoice(board.studio, casting["host"], board.kokoro, board.work / "voice",
                                 int(cfg.get("parallel", 4)))
        kokoro_cfg = cfg.get("cast_voices", {}).get("kokoro") or voice.KOKORO_CAST
        host_k = cfg.get("host_voice", {}).get("kokoro", "am_michael")
        teller_k = cfg.get("teller_voices", {}).get("kokoro", {}).get(casting["teller_kind"], host_k)
        host = self._speaker(board, eleven, casting["host"], host_k)
        teller = host if d["pov"] != "first" else self._speaker(board, eleven, casting["teller"], teller_k)
        # Characters: the paid name when paid, else a Kokoro voice of the same kind.
        k_names = voice.pick_voices(d["kinds"],
                                    {k: [v for v in names if v not in (host_k, teller_k)]
                                     for k, names in kokoro_cfg.items()}, host_k)
        cast_voices = {who: self._speaker(board, eleven, name, k_names.get(who, "am_fenrir"))
                       for who, name in casting["characters"].items()}
        lines = list(d["lines"])
        if teller is not host and d["frames"]:  # only a first-person story changes voice
            cast_voices["host"] = host
            lines += [dialogue.Line(a, b, d["narration"][a:b], "host") for a, b in d["frames"]]
        spans = timing.scene_spans(d["narration"], d["scenes"])
        audio, timeline = voice.narrate(teller, d["narration"], _json(board.story.get("pauses"), []),
                                        _json(board.story.get("pronunciations"), []), d["scenes"], spans, lines,
                                        cast_voices)
        d.update(audio=audio, timeline=timeline, rate=teller.sample_rate, spans=spans)
        failed = eleven.shared["failed"] if eleven is not None else 0
        return {"seconds_of_speech": round(timeline.duration, 1), "chunks": len(timeline.chunks),
                "speakers": sorted({c.speaker for c in timeline.chunks}), "lines_read_by_kokoro": failed}


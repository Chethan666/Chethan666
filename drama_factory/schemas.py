"""JSON shapes for every drama writers' room task (strict subset, every field required)."""

from __future__ import annotations

from typing import Any

from story_factory.schemas import SAFETY_FLAGS


def obj(**props: Any) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def arr(items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": items}


S, I, B = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}


def enum(*values: str) -> dict[str, Any]:
    return {"type": "string", "enum": list(values)}


SHOTS = ("establishing", "wide", "medium", "close_up", "extreme_close_up", "over_shoulder", "insert", "two_shot")
EMOTIONS = ("calm", "tense", "afraid", "angry", "sad", "cold", "pleading", "whisper", "shocked", "warm", "mocking")
PITCH_CRITERIA = ("hook", "bingeability", "originality", "emotional_core", "clarity")
EPISODE_CRITERIA = ("hook", "conflict", "character", "dialogue", "cliffhanger", "continuity")

PAIRWISE = obj(reason=S, winner=enum("A", "B"))

PITCHES = obj(pitches=arr(obj(title=S, logline=S, hook=S, central_secret=S, why_bingeable=S)))
PITCH_SCORE = obj(scores=obj(**{c: I for c in PITCH_CRITERIA}), verdict=S)

CHARACTER = obj(name=S, role=S, age=I, gender=enum("female", "male"), look=S, wardrobe=S, voice=S,
                want=S, secret=S)
BIBLE = obj(title=S, logline=S, genre=S, tone=S, world=S, hook=S, central_secret=S,
            cast=arr(CHARACTER), locations=arr(obj(name=S, look=S)), arc=S,
            episodes=arr(obj(number=I, title=S, hook=S, beats=arr(S), cliffhanger=S)))
PLAN_REVIEW = obj(score=I, problems=arr(S), approved=B)

SHOT = obj(shot=enum(*SHOTS), location=S, characters=arr(S), action=S, speaker=S, line=S,
           emotion=enum(*EMOTIONS), inner_voice=B, motion=B)
EPISODE = obj(title=S, hook_line=S, cliffhanger=S, shots=arr(SHOT))

# Editors' notes point at shot numbers, so a revision can rewrite only those shots.
NOTE = obj(shot=I, problem=S, fix=S)
REVIEW = obj(scores=obj(**{c: I for c in EPISODE_CRITERIA}), flags=obj(**{f: B for f in SAFETY_FLAGS}),
             notes=arr(NOTE))
SHOT_REWRITE = obj(shots=arr(obj(number=I, shot=SHOT)))
ALTERNATIVES = obj(options=arr(obj(shots=arr(SHOT))))
VIEWER = obj(shots=arr(obj(shot=I, interest=I, would_swipe=B, note=S)), unlock_next=I, overall=S)
FINAL_CHECK = obj(flags=obj(**{f: B for f in SAFETY_FLAGS}), issues=arr(S))
PACKAGING = obj(title=S, caption=S, hashtags=arr(S), pinned_comment=S)

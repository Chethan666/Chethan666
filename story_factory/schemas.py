"""JSON schemas for every model task.

Written in the strict subset that both Claude structured outputs and OpenAI
strict JSON schemas accept: every property required, no extra properties, no
length or range keywords (ranges are checked in code instead).
"""

from __future__ import annotations

from typing import Any


def obj(**props: Any) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def arr(items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": items}


S: dict[str, Any] = {"type": "string"}
I: dict[str, Any] = {"type": "integer"}
B: dict[str, Any] = {"type": "boolean"}


def enum(*values: str) -> dict[str, Any]:
    return {"type": "string", "enum": list(values)}


PITCH_CRITERIA = ("click_appeal", "hook", "originality", "emotional_core", "series_fit", "talkability")
REVIEW_CRITERIA = ("hook", "tension", "character", "emotional_payoff", "originality", "prose_for_the_ear",
                   "drawability", "follows_recipe")
SAFETY_FLAGS = ("gore", "harm_to_children", "sexual_content", "real_people", "claims_true",
                "copyrighted_character", "self_harm_method", "stereotype")

PITCHES = obj(pitches=arr(obj(
    title=S, logline=S, synopsis=S, hook_line=S, thumbnail_concept=S, why_it_works=S)))

PITCH_SCORE = obj(scores=obj(**{c: I for c in PITCH_CRITERIA}), verdict=S)

PAIRWISE = obj(reason=S, winner=enum("A", "B"))

DUPLICATE = obj(reason=S, same_story=B)

DOSSIER = obj(summary=S, core_elements=arr(S), regional_variants=arr(S), cultural_notes=arr(S),
              avoid=arr(S), sources=arr(S))

# The story bible: the decisions a storyteller makes before plotting, which every
# later stage writes and is checked against.
BIBLE = obj(
    hero=obj(name=S, who=S, want=S, wound=S, flaw=S, false_belief=S, human_detail=S, voice=S),
    threat=obj(what_it_is=S, what_it_wants=S, rules=arr(S), limits=arr(S), tell=S),
    theme=S, theme_question=S, hero_choice=S, dramatic_irony=S, breath_moment=S,
    refrain=S, refrain_turn=S, sound_motif=S, red_accent=S, open_detail=S, aftertaste=S,
    clues=arr(obj(plant=S, payoff=S)))

BEAT_SHEET = obj(
    central_question=S, emotional_core=S,
    # Planned retention: new questions raised along the way, and the big moments.
    rehooks=arr(obj(at_seconds=I, question=S)),
    midpoint_reversal=S, peak_moment=S,
    beats=arr(obj(name=S, purpose=S, summary=S, target_seconds=I, tension=I,
                  turn=S, open_loop=S, plants=arr(S), payoffs=arr(S))))

OUTLINE_REVIEW = obj(problems=arr(S), approved=B)

ENTITY = obj(kind=enum("character", "place", "object", "rule"), name=S, description=S)

SCRIPT = obj(intro=S, body=S, outro=S, comment_question=S, hook_line=S, thumbnail_idea=S,
             description_credit=S, new_entities=arr(ENTITY), referenced_entities=arr(S))

# Long drafts are written part by part, then framed by the Storyteller.
SECTION = obj(text=S)

FRAME = obj(intro=S, outro=S, comment_question=S, thumbnail_idea=S, description_credit=S,
            new_entities=arr(ENTITY), referenced_entities=arr(S))

# Editors' notes name the passage, the problem and the fix.
NOTE = obj(quote=S, problem=S, fix=S)

REVIEW = obj(scores=obj(**{c: I for c in REVIEW_CRITERIA}),
             flags=obj(**{f: B for f in SAFETY_FLAGS}), notes=arr(NOTE))

# Does the script deliver each promise of the story bible? Quotes are verified in code.
BIBLE_CHECK = obj(items=arr(obj(id=S, present=B, quote=S)), threat_rule_breaks=arr(S))

AUDIENCE = obj(chunks=arr(obj(chunk=I, interest=I, would_leave=B, note=S)), overall=S)

CHUNK_REWRITE = obj(chunks=arr(obj(chunk=I, text=S)))

POLISH = obj(intro=S, body=S, outro=S, pronunciations=arr(obj(word=S, say_as=S)),
             # Silences for the narrator: after this exact phrase, pause this long.
             pauses=arr(obj(after=S, ms=I)))

FINAL_CHECK = obj(flags=obj(**{f: B for f in SAFETY_FLAGS}), accuracy_issues=arr(S),
                  cultural_issues=arr(S))

# How the narrator should deliver a scene; the voice step turns these into direction.
DELIVERIES = ("steady", "slow", "tense", "hushed", "urgent", "tender")

SCENES = obj(
    # Fixed looks so every image draws each character the same way.
    cast=arr(obj(name=S, look=S)),
    scenes=arr(obj(text=S, location=S, characters=arr(S), visual=S, mood=S,
                   delivery=enum(*DELIVERIES), sfx=arr(S), key_moment=B)))

HOOKS = obj(openings=arr(S))

PACKAGING = obj(titles=arr(S), thumbnail_brief=S, thumbnail_text=S, description=S,
                tags=arr(S), pinned_comment=S)

LEARNINGS = obj(insights=arr(S))

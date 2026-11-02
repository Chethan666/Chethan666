"""Prompt templates. Every prompt asks for a single JSON object."""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Sequence

CHANNEL_BIBLE = """\
CHANNEL: Hollow Lines, an animated horror storytelling channel for a global audience of
teens and adults (not made for kids).

VISUAL STYLE (you write narration, but every moment must be drawable): dark hand-drawn
stickman animation; white chalk-like stick figures on black ink paper; one blood-red accent
(eyes, a lantern flame, a warning light). Prefer few characters, clear silhouettes, strong
light and darkness, single objects over crowds. Fear comes from what is barely shown.

HOST: The Storyteller, a tall stick figure with a red-flamed lantern. Calm, warm, slightly
ominous. He opens and closes long episodes in the first person and speaks directly to the
viewer. The whole video is narrated by one voice, so write prose for the ear: short clear
sentences, concrete sounds and images, no lists of dialogue tags, no stage directions.

HARD RULES:
- Fiction or folklore only. Never claim the story is true or happened to real people.
- No real living or recently deceased people, no real crimes, no real companies as villains.
- No gore, torture or detailed injury. Violence is implied, brief, off-screen.
- No sexual content. No harm to children; children may appear only safely.
- No suicide or self-harm methods.
- No copyrighted characters, creatures or plots (famous creepypastas, film, TV or game monsters).
- No hateful stereotypes. Treat every culture's folklore with respect and accuracy.
"""

AUDIENCE_NEEDS = """\
EVERY STORY MUST DELIVER THESE TO THE VIEWER:
1. Curiosity: open a concrete, unanswered question in the first 10 seconds; answer it at the end.
2. Safe thrill: fear rises in steps; each step is worse than the last.
3. "That could be me": an ordinary setting and an ordinary person the viewer recognizes.
4. Attachment: show why we care about the protagonist before anything goes wrong.
5. Fair clues: plant 2-3 small details early that the ending pays off, so attentive viewers can solve it.
6. Earned ending: the twist or emotion must follow from what came before, never from nowhere.
7. An emotional core: love, guilt, loneliness or loyalty underneath the fear.
8. Something to talk about: a lingering question the audience will argue about in the comments.
"""

LONG_STRUCTURE = """\
STRUCTURE (long episode):
- intro: The Storyteller, 25-60 words. Sets the mood and hints at the question. Do not summarize.
- body, in this order:
  a) hook: the very first sentence plants something wrong or an unanswered question;
  b) normal life: who the protagonist is and why we care;
  c) the first wrong detail;
  d) three escalating beats of dread, each worse than the last;
  e) false relief: a moment where it seems over;
  f) climax;
  g) ending that matches the requested twist and ending type.
- outro: The Storyteller, 25-60 words, ending with exactly one question to the viewer.
"""

SHORT_STRUCTURE = """\
STRUCTURE (60-second Short):
- intro: empty string (no host introduction; every second counts).
- body: first sentence is the hook; one fast escalation; a sting in the final line.
- outro: one short line from The Storyteller that ends with a question to the viewer.
"""


def _recipe_block(recipe: Any, formats: dict[str, str], endings: dict[str, str]) -> str:
    return (
        f"- setting: {recipe.setting}\n"
        f"- core fear: {recipe.core_fear}\n"
        f"- protagonist: {recipe.protagonist}\n"
        f"- format: {formats[recipe.format]}\n"
        f"- twist: {recipe.twist}\n"
        f"- ending: {endings[recipe.ending]}\n"
    )


def _world_block(entities: Sequence[sqlite3.Row]) -> str:
    lines = [f"- {e['name']} ({e['kind']}): {e['description']}" for e in entities]
    return (
        "SHARED WORLD (canon; never contradict it):\n" + "\n".join(lines) + "\n"
        "You may reference at most two existing entries as subtle details for loyal viewers, "
        "and introduce at most two new named characters, places or objects.\n"
    )


def premises(*, series: Any, recipe: Any, formats: dict[str, str], endings: dict[str, str],
             count: int, legend: sqlite3.Row | None, sparks: Sequence[sqlite3.Row],
             world: Sequence[sqlite3.Row], avoid: Sequence[sqlite3.Row],
             trends: Sequence[sqlite3.Row]) -> str:
    parts = [CHANNEL_BIBLE, AUDIENCE_NEEDS,
             f"SERIES: {series.title}. {series.description}\n",
             "RECIPE (use every ingredient):\n" + _recipe_block(recipe, formats, endings)]
    if legend is not None:
        parts.append(
            f"LEGEND TO RETELL: {legend['title']} ({legend['region']}).\n"
            f"Research notes: {legend['summary']}\n"
            "Keep the legend's traditional core accurate and respectful. The framing story about "
            "the protagonist must be original. Use your own words; never copy the notes.\n"
        )
    if sparks:
        parts.append(
            "MOTIF SPARKS (borrow at most one abstract element, such as a creature's nature or a "
            "superstition; never a plot or a name):\n"
            + "\n".join(f"- {s['title']} ({s['region']}): {(s['summary'] or '')[:300]}" for s in sparks)
            + "\n"
        )
    if world:
        parts.append(_world_block(world))
    if trends:
        parts.append(
            "AUDIENCE DEMAND (popular horror video titles; use only to understand what viewers "
            "want; never copy their titles or plots):\n"
            + "\n".join(f"- {t['title']}" for t in trends) + "\n"
        )
    if avoid:
        parts.append(
            "ALREADY MADE (your premises must be clearly different from all of these):\n"
            + "\n".join(f"- {s['title']}: {s['logline'] or ''}" for s in avoid) + "\n"
        )
    parts.append(
        f"TASK: Pitch {count} different premises for one episode built from the recipe. Make each "
        "distinct in hook, characters and central image. Titles: at most 60 characters, intriguing, "
        "honest about the story.\n"
        'Return JSON: {"premises": [{"title": "...", "logline": "one sentence", '
        '"synopsis": "80-120 words, including the ending"}]}'
    )
    return "\n".join(parts)


def write_script(*, series: Any, kind: str, recipe: Any, formats: dict[str, str],
                 endings: dict[str, str], premise: dict[str, Any], min_words: int, max_words: int,
                 legend: sqlite3.Row | None, world: Sequence[sqlite3.Row]) -> str:
    parts = [CHANNEL_BIBLE, AUDIENCE_NEEDS,
             LONG_STRUCTURE if kind == "long" else SHORT_STRUCTURE,
             f"SERIES: {series.title}. {series.description}\n",
             "RECIPE:\n" + _recipe_block(recipe, formats, endings),
             f"APPROVED PREMISE:\nTitle: {premise['title']}\nLogline: {premise['logline']}\n"
             f"Synopsis: {premise['synopsis']}\n"]
    if legend is not None:
        parts.append(
            f"LEGEND: {legend['title']} ({legend['region']}). Notes: {legend['summary']}\n"
            "Retell its traditional core accurately, in your own words.\n"
        )
    if world:
        parts.append(_world_block(world))
    parts.append(
        f"TASK: Write the full narration. intro + body + outro together must be {min_words}-"
        f"{max_words} words.\n"
        'Return JSON: {"intro": "...", "body": "...", "outro": "...", '
        '"comment_question": "the question that ends the outro", '
        '"hook_line": "the first sentence of the body", '
        '"thumbnail_idea": "one simple, striking drawable image, no text", '
        '"description_credit": "for legends: one sentence crediting the tradition and region; '
        'otherwise: This is an original work of fiction.", '
        '"new_entities": [{"kind": "character|place|object|rule", "name": "...", "description": "..."}], '
        '"referenced_entities": ["names of existing shared-world entries used"]}'
    )
    return "\n".join(parts)


def review(*, kind: str, recipe: Any, formats: dict[str, str], endings: dict[str, str],
           script: dict[str, Any]) -> str:
    return "\n".join([
        CHANNEL_BIBLE, AUDIENCE_NEEDS,
        LONG_STRUCTURE if kind == "long" else SHORT_STRUCTURE,
        "RECIPE:\n" + _recipe_block(recipe, formats, endings),
        "SCRIPT TO REVIEW:\n" + json.dumps(
            {k: script.get(k, "") for k in ("intro", "body", "outro")}, ensure_ascii=False),
        "TASK: You are a strict horror story editor. Score 1-10 (7 means publishable). Be honest; "
        "a generic or predictable story cannot score above 6 on originality. Flag any hard-rule "
        "violation.\n"
        'Return JSON: {"scores": {"hook": n, "tension": n, "emotional_payoff": n, "originality": n, '
        '"clarity_for_listening": n, "drawability": n, "follows_recipe": n}, '
        '"flags": {"gore": bool, "harm_to_children": bool, "sexual_content": bool, '
        '"real_people": bool, "claims_true": bool, "copyrighted_character": bool, '
        '"self_harm_method": bool, "stereotype": bool}, '
        '"fix_instructions": ["specific, actionable changes"]}',
    ])


def revise(*, kind: str, script: dict[str, Any], feedback: list[str], min_words: int,
           max_words: int) -> str:
    return "\n".join([
        CHANNEL_BIBLE, AUDIENCE_NEEDS,
        LONG_STRUCTURE if kind == "long" else SHORT_STRUCTURE,
        "CURRENT SCRIPT:\n" + json.dumps(script, ensure_ascii=False),
        "EDITOR FEEDBACK (fix every point):\n" + "\n".join(f"- {f}" for f in feedback),
        f"TASK: Rewrite the script to fix the feedback, keeping the premise. intro + body + outro "
        f"must total {min_words}-{max_words} words. Return the same JSON fields as CURRENT SCRIPT.",
    ])


def judge_duplicate(premise: dict[str, Any], existing: sqlite3.Row) -> str:
    return "\n".join([
        "Decide whether two horror story premises are essentially the same story: the same "
        "central situation, threat and twist, even if names, places or wording differ. Sharing "
        "only a genre, a setting type or a common trope does NOT make them the same.",
        f"STORY A: {existing['title']}. {existing['logline'] or ''} {existing['synopsis'] or ''}",
        f"STORY B: {premise['title']}. {premise['logline']} {premise['synopsis']}",
        'Return JSON: {"same_story": true|false, "reason": "one sentence"}',
    ])


def scene_plan(*, kind: str, narration: str, scene_range: tuple[int, int],
               key_range: tuple[int, int]) -> str:
    return "\n".join([
        CHANNEL_BIBLE,
        "NARRATION:\n" + narration,
        f"TASK: Split the narration into {scene_range[0]}-{scene_range[1]} visual scenes for the "
        "animator. Each scene's `text` must be copied exactly from the narration, the scenes must "
        "be in order, and together they must cover the whole narration with nothing skipped. "
        "Describe each picture in words only (no image prompts with camera jargon). Keep scenes "
        "drawable in the channel style: few figures, clear silhouettes, darkness and one red accent "
        f"at most. Mark {key_range[0]}-{key_range[1]} scenes as key_moment: the scares, reveals "
        "and emotional peaks that need a brand-new image.\n"
        'Return JSON: {"scenes": [{"text": "exact narration span", "location": "...", '
        '"characters": ["..."], "visual": "what is drawn and what moves", '
        '"mood": "...", "sfx": ["..."], "key_moment": true|false}]}',
    ])

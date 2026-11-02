"""System prompts and task prompts for every writers' room role.

Each task's output shape is enforced separately by `schemas`, so prompts focus
on judgment and craft rather than formatting.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Sequence

CHANNEL = """\
Hollow Lines is an animated horror storytelling channel for a global audience of teens and
adults (not made for kids). Every video is narrated by one calm voice and drawn as dark,
hand-drawn stickman animation: white chalk-like figures on black ink paper with a single
blood-red accent. The host, The Storyteller, is a tall stick figure with a red-flamed lantern
who opens and closes long episodes, speaking directly to the viewer.

Rules that are never broken:
- Fiction or folklore only; never claim a story is true or happened to real people.
- No real living or recently deceased people, real crimes, or real companies as villains.
- No gore, torture or lingering injury; violence is implied, brief and off-screen.
- No sexual content. No harm to children; children may appear only safely.
- No suicide or self-harm methods.
- No copyrighted characters, creatures or plots (famous creepypastas, film, TV or game monsters).
- Treat every culture's folklore with accuracy and respect; no stereotypes.
"""

AUDIENCE_NEEDS = """\
What a great Hollow Lines story gives its viewer:
1. Curiosity: a concrete, unanswered question within the first ten seconds, answered at the end.
2. A safe thrill: fear that climbs in steps, each worse than the last.
3. "That could be me": an ordinary person in a place the viewer knows.
4. Attachment: we care about the protagonist before anything goes wrong.
5. Fair clues: two or three early details that the ending pays off.
6. An earned ending: the twist or emotion follows from what came before.
7. An emotional core: love, guilt, loneliness or loyalty beneath the fear.
8. Something to argue about in the comments afterwards.
"""

CRAFT = """\
Writing for the ear:
- This is narration heard once, at speaking pace, over simple drawings. Write sentences a
  listener can follow without rereading. Vary rhythm: longer sentences to settle in, short
  ones as dread tightens.
- Make it concrete: specific sounds, smells, objects and numbers rather than abstract fear.
  One precise detail beats three adjectives.
- Keep dialogue sparse and always clear about who is speaking.
- Every image must be drawable: few figures, clear silhouettes, darkness, one red accent.
- Never explain the twist twice. Trust the listener.
- Avoid worn horror phrasing: "a chill ran down", "blood ran cold", "little did they know",
  "suddenly", "out of nowhere", "it was all a dream", "heart pounding in my chest",
  "something was off", "eerie silence".
"""

SYSTEM_WRITER = (
    "You are the head writer of Hollow Lines, a world-class horror storyteller whose work is "
    "known for originality, dread that builds, and endings people still think about days later.\n\n"
    + CHANNEL + "\n" + AUDIENCE_NEEDS + "\n" + CRAFT
)

SYSTEM_EDITOR = (
    "You are a demanding story editor for Hollow Lines who has read thousands of horror stories "
    "and greenlights only the exceptional. You are specific, honest and hard to impress: a "
    "competent but familiar story scores 6, not 8. You judge only what is on the page.\n\n"
    + CHANNEL + "\n" + AUDIENCE_NEEDS
)

SYSTEM_RESEARCHER = (
    "You are a folklore researcher preparing accurate notes for a horror storyteller. Prefer "
    "scholarly and cultural sources over entertainment sites, separate the traditional legend "
    "from modern pop-culture versions, and note anything that would be disrespectful to get wrong."
)


def system_audience(persona: str) -> str:
    return (
        f"You are {persona}. You are watching a horror story video on YouTube. React honestly as "
        "this person would, not as a critic. You have many other videos you could watch instead."
    )


def _recipe(recipe: Any, formats: dict[str, str], endings: dict[str, str]) -> str:
    return (f"- setting: {recipe.setting}\n- core fear: {recipe.core_fear}\n"
            f"- protagonist: {recipe.protagonist}\n- format: {formats[recipe.format]}\n"
            f"- twist: {recipe.twist}\n- ending: {endings[recipe.ending]}\n")


def _pitch(p: dict[str, Any]) -> str:
    return (f"Title: {p['title']}\nLogline: {p['logline']}\nSynopsis: {p['synopsis']}\n"
            f"Opening line: {p.get('hook_line', '')}\nThumbnail: {p.get('thumbnail_concept', '')}")


def _world(entities: Sequence[sqlite3.Row]) -> str:
    if not entities:
        return ""
    lines = "\n".join(f"- {e['name']} ({e['kind']}): {e['description']}" for e in entities)
    return ("\nShared world canon (never contradict it). Loyal viewers love small connections: you "
            "may weave in at most two existing entries as subtle details, and introduce at most two "
            f"new named characters, places or objects.\n{lines}\n")


def _legend(legend: sqlite3.Row | None, dossier: dict | None) -> str:
    if legend is None:
        return ""
    notes = json.dumps(dossier, ensure_ascii=False) if dossier else (legend["summary"] or "")
    return (f"\nLegend to retell: {legend['title']} ({legend['region']}).\nResearch notes: {notes}\n"
            "Keep the legend's traditional core accurate and respectful; the framing story about the "
            "protagonist is original. Use your own words, never the notes' wording.\n")


def _script(script: dict[str, Any]) -> str:
    return json.dumps({k: script.get(k, "") for k in ("intro", "body", "outro")}, ensure_ascii=False)


# --- Research ----------------------------------------------------------------

def dossier_research(legend: sqlite3.Row) -> str:
    return (f"Research the traditional legend \"{legend['title']}\" from {legend['region']}. "
            f"Starting notes: {legend['summary']}\n\nFind: the core traditional story and what the "
            "entity is and does; how versions differ by region or era; cultural meaning and context; "
            "details that would be inaccurate or disrespectful to get wrong; and which well-known "
            "versions are modern copyrighted works rather than tradition. Cite your sources.")


def dossier_structure(legend: sqlite3.Row, research: str) -> str:
    return (f"Turn these research notes on \"{legend['title']}\" into a storyteller's dossier. Keep "
            f"only what the notes support.\n\n{research}")


# --- Pitching ----------------------------------------------------------------

def pitches(*, series: Any, recipe: Any, formats: dict[str, str], endings: dict[str, str], count: int,
            legend: sqlite3.Row | None, dossier: dict | None, sparks: Sequence[sqlite3.Row],
            world: Sequence[sqlite3.Row], avoid: Sequence[sqlite3.Row], trends: Sequence[sqlite3.Row],
            learnings: Sequence[str]) -> str:
    parts = [f"Series: {series.title}. {series.description}",
             "Recipe for this episode (use every ingredient):\n" + _recipe(recipe, formats, endings),
             _legend(legend, dossier), _world(world)]
    if sparks:
        parts.append("Folklore motifs you may borrow one abstract element from (a creature's nature, "
                     "a superstition), never a plot or a name:\n"
                     + "\n".join(f"- {s['title']} ({s['region']}): {(s['summary'] or '')[:300]}" for s in sparks))
    if learnings:
        parts.append("What our audience has responded to so far (from real analytics):\n"
                     + "\n".join(f"- {i}" for i in learnings))
    if trends:
        parts.append("Popular horror video titles right now, to understand appetite only; never copy "
                     "their titles or plots:\n" + "\n".join(f"- {t['title']}" for t in trends))
    if avoid:
        parts.append("Episodes we already have; every pitch must be clearly different from all of "
                     "these:\n" + "\n".join(f"- {s['title']}: {s['logline'] or ''}" for s in avoid))
    parts.append(
        f"Pitch {count} genuinely different episodes built from the recipe: different central images, "
        "different hooks, different emotional cores. Before writing each, ask whether a viewer "
        "scrolling past would stop for its title and thumbnail. Titles: at most 60 characters, "
        "intriguing, honest about the story. Synopsis: 80-120 words including the ending. "
        "Thumbnail concept: one simple drawable image with no text.")
    return "\n\n".join(p for p in parts if p)


def score_pitch(series: Any, pitch: dict[str, Any]) -> str:
    return (f"Series: {series.title}. {series.description}\n\nPitch:\n{_pitch(pitch)}\n\n"
            "Score 1-10 on: click_appeal (would a viewer stop scrolling for this title and thumbnail), "
            "hook, originality (have you seen this before?), emotional_core, series_fit and "
            "talkability (will viewers debate it in the comments). Then give a one-line verdict.")


def compare_pitches(series: Any, a: dict[str, Any], b: dict[str, Any]) -> str:
    return (f"Series: {series.title}. We can only make one of these episodes.\n\nPITCH A:\n{_pitch(a)}"
            f"\n\nPITCH B:\n{_pitch(b)}\n\nWhich would more viewers click, watch to the very end and "
            "talk about afterwards? Reason briefly, then choose.")


def judge_duplicate(premise: dict[str, Any], existing: sqlite3.Row) -> str:
    return ("Decide whether two horror premises are essentially the same story: the same central "
            "situation, threat and twist, even if names, places or wording differ. Sharing only a "
            "genre, a setting type or a common trope does NOT make them the same.\n\n"
            f"STORY A: {existing['title']}. {existing['logline'] or ''} {existing['synopsis'] or ''}\n\n"
            f"STORY B: {premise['title']}. {premise['logline']} {premise['synopsis']}")


# --- Outline and drafts ------------------------------------------------------

def beat_sheet(*, series: Any, kind: str, recipe: Any, formats: dict[str, str], endings: dict[str, str],
               pitch: dict[str, Any], legend: sqlite3.Row | None, dossier: dict | None,
               world: Sequence[sqlite3.Row], target_seconds: int) -> str:
    shape = ("a Storyteller intro, a hook in the first sentence, ordinary life and why we care, the "
             "first wrong detail, three escalations each worse than the last, a false relief, the "
             "climax, the ending, and a Storyteller outro ending in one question to the viewer"
             if kind == "long" else
             "a hook in the first sentence, one fast escalation, a sting in the final line, and a "
             "one-line Storyteller question")
    return "\n\n".join(p for p in [
        f"Series: {series.title}. {series.description}",
        "Recipe:\n" + _recipe(recipe, formats, endings),
        "Approved pitch:\n" + _pitch(pitch), _legend(legend, dossier), _world(world),
        f"Outline this episode beat by beat for about {target_seconds} seconds of narration: {shape}. "
        "For each beat give its purpose, what happens, its target length in seconds, its tension "
        "from 1 to 10, the clues it plants and the earlier clues it pays off. Tension should "
        "climb overall, dipping only at the false relief. State the central question and the "
        "emotional core."] if p)


def review_outline(kind: str, pitch: dict[str, Any], outline: dict[str, Any]) -> str:
    return (f"Pitch:\n{_pitch(pitch)}\n\nOutline:\n{json.dumps(outline, ensure_ascii=False)}\n\n"
            "Approve only if: the hook raises a concrete question at once; tension climbs; every "
            "payoff was planted earlier and every plant pays off; the ending is earned and surprising; "
            "the emotional core is felt, not stated; and no beat is filler. List each specific "
            "problem otherwise.")


def revise_outline(outline: dict[str, Any], problems: list[str]) -> str:
    return (f"Outline:\n{json.dumps(outline, ensure_ascii=False)}\n\nEditor's problems:\n"
            + "\n".join(f"- {p}" for p in problems)
            + "\n\nRevise the outline to solve every problem while keeping what works.")


def write_draft(*, series: Any, kind: str, recipe: Any, formats: dict[str, str], endings: dict[str, str],
                pitch: dict[str, Any], outline: dict[str, Any], min_words: int, max_words: int,
                legend: sqlite3.Row | None, dossier: dict | None, world: Sequence[sqlite3.Row]) -> str:
    structure = ("intro: The Storyteller, 25-60 words, sets the mood without summarizing. body: the "
                 "story. outro: The Storyteller, 25-60 words, ending with exactly one question to "
                 "the viewer." if kind == "long" else
                 "intro: empty string. body: the story; the first sentence is the hook. outro: one "
                 "short Storyteller line ending with a question to the viewer.")
    return "\n\n".join(p for p in [
        f"Series: {series.title}. {series.description}",
        "Recipe:\n" + _recipe(recipe, formats, endings),
        "Approved pitch:\n" + _pitch(pitch),
        "Approved outline (follow its beats, plants and payoffs):\n" + json.dumps(outline, ensure_ascii=False),
        _legend(legend, dossier), _world(world),
        f"Write the full narration. {structure} intro + body + outro together: {min_words}-{max_words} "
        "words. comment_question repeats the outro's final question exactly. hook_line is the body's "
        "first sentence. description_credit: for a legend, one sentence crediting the tradition and "
        "region; otherwise \"This is an original work of fiction.\" List new shared-world entities "
        "you introduced and existing ones you referenced."] if p)


def compare_drafts(kind: str, pitch: dict[str, Any], a: dict[str, Any], b: dict[str, Any]) -> str:
    return (f"Two drafts of the same {'episode' if kind == 'long' else 'Short'} ({pitch['title']}).\n\n"
            f"DRAFT A:\n{_script(a)}\n\nDRAFT B:\n{_script(b)}\n\nRead each as a listener hearing it "
            "once. Which keeps you more gripped from first line to last, lands its ending harder and "
            "sounds better aloud? Ignore length unless it hurts the story. Reason briefly, then choose.")


def review(kind: str, recipe: Any, formats: dict[str, str], endings: dict[str, str],
           script: dict[str, Any], dossier: dict | None) -> str:
    accuracy = (f"\nLegend dossier for accuracy:\n{json.dumps(dossier, ensure_ascii=False)}\n"
                if dossier else "")
    return (f"Recipe:\n{_recipe(recipe, formats, endings)}{accuracy}\nScript:\n{_script(script)}\n\n"
            "Score 1-10 (8 means a standout we would be proud to publish): hook, tension, "
            "emotional_payoff, originality, prose_for_the_ear, drawability and follows_recipe. Flag "
            "every rule violation. In notes, give specific, line-level changes that would raise the "
            "lowest scores; quote the passage each note is about.")


def revise(kind: str, script: dict[str, Any], notes: list[str], min_words: int, max_words: int) -> str:
    return (f"Current script:\n{json.dumps(script, ensure_ascii=False)}\n\nEditors' notes:\n"
            + "\n".join(f"- {n}" for n in notes)
            + f"\n\nRevise the script to address every note while keeping what already works. "
            f"intro + body + outro: {min_words}-{max_words} words. Return every field.")


# --- Audience test and polish -----------------------------------------------

def audience(kind: str, chunks: list[str]) -> str:
    numbered = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))
    return (f"The narration below is split into numbered parts of about 30 seconds each. Read them in "
            f"order as if hearing the video.\n\n{numbered}\n\nFor every part give your interest from 1 "
            "to 10, whether you would have left the video at that point, and a short honest note on "
            "why. Once you would have left, keep rating as if you had stayed. Then sum up your "
            "experience in one line.")


def rewrite_chunks(script: dict[str, Any], chunks: list[str], weak: dict[int, list[str]]) -> str:
    numbered = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))
    issues = "\n".join(f"[{n}] " + " | ".join(notes) for n, notes in sorted(weak.items()))
    return (f"Full script for context:\n{_script(script)}\n\nThe story body split into numbered parts:\n"
            f"{numbered}\n\nTest viewers lost interest in these parts:\n{issues}\n\nRewrite only those "
            "parts so they grip, keeping their story events, the clues they carry and how they connect "
            "to the parts around them. Keep each part about the same length. Return only the rewritten "
            "parts with their numbers.")


def polish(kind: str, script: dict[str, Any], min_words: int, max_words: int) -> str:
    return (f"Script:\n{_script(script)}\n\nDo a final read-aloud line edit for a single narrator voice: "
            "smooth any sentence that is hard to say or easy to mishear, tune rhythm and pauses with "
            "punctuation, and remove anything that only works on the page. Do not change events, "
            f"clues, the ending or the final question. Keep {min_words}-{max_words} words. List names "
            "or foreign words a voice engine might mispronounce, with a simple respelling.")


def final_check(script: dict[str, Any], dossier: dict | None) -> str:
    accuracy = (f"Check the legend against this dossier:\n{json.dumps(dossier, ensure_ascii=False)}\n\n"
                if dossier else "")
    return (f"{accuracy}Script:\n{_script(script)}\n\nFinal compliance check before publishing. Flag "
            "any rule violation, list factual errors about the legend (if any) and anything "
            "culturally insensitive. Empty lists mean it is clear.")


def scene_plan(narration: str, scene_range: tuple[int, int], key_range: tuple[int, int]) -> str:
    return (f"NARRATION:\n{narration}\nTASK: Split the narration into {scene_range[0]}-{scene_range[1]} "
            "visual scenes for the animator. Each scene's `text` is copied exactly from the narration; "
            "scenes are in order and together cover the whole narration with nothing skipped. "
            "Describe each picture in words (what is drawn and what moves), keeping it drawable: few "
            "figures, clear silhouettes, darkness and at most one red accent. List sound effects. "
            f"Mark {key_range[0]}-{key_range[1]} scenes as key_moment: the scares, reveals and "
            "emotional peaks that deserve a brand-new image.")


def packaging(series: Any, script: dict[str, Any], pitch: dict[str, Any], credit: str) -> str:
    return (f"Series: {series.title}.\nPitch:\n{_pitch(pitch)}\n\nFinal script:\n{_script(script)}\n\n"
            "Package this video for YouTube. titles: five options, best first, at most 60 characters, "
            "curiosity without lying. thumbnail_brief: one simple, striking drawable image in our "
            "style. thumbnail_text: two to four words, or empty if the image works alone. description: "
            f"two short paragraphs that tease without spoiling, ending with: \"{credit}\". tags: ten to "
            "fifteen. pinned_comment: a question that invites theories or personal stories.")


def learn(top: Sequence[sqlite3.Row], bottom: Sequence[sqlite3.Row]) -> str:
    def rows(items: Sequence[sqlite3.Row]) -> str:
        return "\n".join(
            f"- {r['title']} | {r['series']} | setting: {r['setting']} | twist: {r['twist']} | "
            f"ending: {r['ending']} | views: {r['views']} | avg viewed: {r['avg_view_percentage']}% | "
            f"CTR: {r['ctr']}%" for r in items)
    return (f"Our best-performing videos:\n{rows(top)}\n\nOur worst-performing videos:\n{rows(bottom)}"
            "\n\nWhat separates them? Write up to ten concrete, actionable insights for our writers "
            "about hooks, settings, twists, endings and titles. Only claim what the data supports.")

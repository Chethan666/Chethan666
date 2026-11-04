"""The writers' room: how one story is made.

    research legend (web) -> pitch contest -> plan contest (each writer's story bible
    and outline; Shorts plan once) -> competing drafts (long ones written part by part;
    Shorts write several takes) -> editors' loop (free checks, story-bible check, scored
    review, legend recognition; the best version is always kept, and a showrunner picks
    the fix: new plan, fresh draft, section rewrite, revision or hook lab) -> hook lab
    -> ending lab -> surprise test -> simulated audience -> read-aloud polish -> final
    checks -> narration notes -> scene plan with cast -> packaging with title contest
    -> approved (or rejected / parked with its best draft and a diagnosis)

Different companies' models fill different roles, and work is always judged by
models from a different family than its author.
"""

from __future__ import annotations

import json
import logging
import random
import re
import sqlite3
import statistics
from dataclasses import dataclass, field
from typing import Any

from . import catalog, db, dedup, learning, prompts, schemas, tournament
from .llm import BudgetExceeded, LLMError, ModelSpec
from .router import FailoverClient as FailoverJudge, Router

log = logging.getLogger(__name__)

CLAIMS_TRUE = re.compile(
    r"\b(true story|really happened|based on (a )?true|actual events|this happened to me)\b", re.IGNORECASE)
CLICHES = re.compile(
    r"\b(chills? ran down|blood ran cold|little did (he|she|they|i|we) know|out of nowhere|"
    r"it was all a dream|heart pounding in (my|his|her|their) chest|eerie silence|something was off)\b",
    re.IGNORECASE)
# Phrasing that marks text as machine-written; each one is easy to replace.
AI_ISMS = re.compile(
    r"\b(a sense of (dread|unease|foreboding|doom)|the air (was|grew|hung|felt) (thick|heavy)|"
    r"a testament to|tapestry|palpable|sen(t|ding) (a )?shivers? down|shivers? down (my|his|her|their) spine|"
    r"couldn'?t shake the feeling|deafening silence|in that moment|time seemed to (stop|slow|freeze)|"
    r"every fib(er|re) of|hung in the air|for what felt like (an eternity|hours|forever)|"
    r"(a|the) weight of (it all|the world)|as if on cue|a chill (crept|settled)|"
    r"breath (he|she|they|i) didn'?t know)", re.IGNORECASE)
# Strong profanity costs ad revenue; mild words are left to the editors.
PROFANITY = re.compile(r"\b(fuck\w*|motherfuck\w*|shit\w*|cunt\w*|bullshit)\b", re.IGNORECASE)
# Naming a feeling instead of causing it.
NAMED_FEELINGS = re.compile(
    r"\b(felt|feel|feels|feeling) (so |very |truly |completely )?(scared|terrified|afraid|frightened|"
    r"uneasy|nervous|anxious|horrified|fear|dread|panic(ked)?)\b", re.IGNORECASE)
# The Storyteller's ritual: every long episode's intro opens with it.
RITUAL_OPENING = "The lantern is lit."
# The ritual as a whole sentence anywhere (Shorts must never use it).
RITUAL_SENTENCE = re.compile(r"(?:^|(?<=[.!?…\"”]))\s*the lantern is lit[.!,]?(?=\s|$)", re.IGNORECASE)
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
SCRIPT_FIELDS = ("intro", "body", "outro", "comment_question", "hook_line", "thumbnail_idea",
                 "description_credit", "new_entities", "referenced_entities")
FORMATS, ENDINGS = catalog.FORMATS, catalog.ENDINGS


@dataclass
class Context:
    conn: sqlite3.Connection
    cfg: dict
    router: Router
    rng: random.Random


@dataclass
class Outcome:
    series: str
    status: str  # approved | rejected | skipped
    story_id: int | None = None
    detail: str = ""


@dataclass
class Story:
    """Everything known about the story being developed."""

    series: catalog.Series
    recipe: catalog.Recipe
    pitch: dict[str, Any]
    legend: sqlite3.Row | None
    dossier: dict | None
    sparks: list[sqlite3.Row]
    world: list[sqlite3.Row]
    authors: dict[str, str] = field(default_factory=dict)
    bible: dict[str, Any] | None = None
    log: list[dict[str, Any]] = field(default_factory=list)

    def note(self, stage: str, **info: Any) -> None:
        self.log.append({"stage": stage, **info})
        log.info("%s: %s", stage, info)


# --- Pure checks -------------------------------------------------------------

def narration_of(script: dict[str, Any]) -> str:
    return "\n\n".join(p.strip() for p in (script.get("intro"), script.get("body"), script.get("outro"))
                       if isinstance(p, str) and p.strip())


def word_range(cfg: dict, kind: str) -> tuple[int, int]:
    return cfg["lengths"][f"{kind}_min_words"], cfg["lengths"][f"{kind}_max_words"]


def scene_ranges(kind: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """(number of scenes, number of key moments) for a story kind."""
    return ((18, 45), (3, 6)) if kind == "long" else ((4, 10), (1, 2))


def tidy(script: dict[str, Any], kind: str | None = None) -> dict[str, Any]:
    """Recompute the fields derived from the text, so a model's copying slip can't sink a story.

    comment_question is the outro's final question (appended to the outro if the model
    only returned it separately); hook_line is the body's first sentence. A long
    episode's intro always opens with the Storyteller's ritual line.
    """
    if kind == "short":  # a Short's first seconds belong to the hook, never the ritual
        for key in ("intro", "body"):
            script[key] = RITUAL_SENTENCE.sub("", str(script.get(key) or "")).strip()
    intro = str(script.get("intro") or "").strip()
    if kind == "long" and intro and dedup.normalize(RITUAL_OPENING) not in dedup.normalize(intro):
        script["intro"] = f"{RITUAL_OPENING} {intro}"  # the cold open should precede it; better late than missing
    outro = str(script.get("outro", "")).strip()
    question = str(script.get("comment_question", "")).strip()
    if outro and not outro.endswith("?") and question.endswith("?") and question not in outro:
        # The model wrote the closing question separately; the Storyteller says it last.
        outro = script["outro"] = f"{outro} {question}"
    if outro.endswith("?"):
        script["comment_question"] = re.split(r"(?<=[.!?…])\s+", outro)[-1]
    body = str(script.get("body", "")).strip()
    if body:
        script["hook_line"] = re.split(r"(?<=[.!?…])\s+", body)[0]
    return script


def syllables(word: str) -> int:
    """A rough syllable count (vowel groups, minus a silent final e)."""
    word = word.lower()
    groups = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and not word.endswith(("le", "ee")) and groups > 1:
        groups -= 1
    return max(groups, 1)


def machine_phrasing(text: str) -> list[str]:
    """Phrases that mark prose as machine-written. Each is easy to replace, so they block."""
    return [f"replace the overused phrase '{m.group(0)}' with something specific to this story"
            for m in AI_ISMS.finditer(text)]


def style_issues(text: str, cfg: dict, kind: str) -> list[str]:
    """Free checks for prose that sounds machine-written or is hard to follow by ear.

    Only machine phrasing blocks a story; the rest are judgment calls (a repeated
    opening can be deliberate rhetoric), so they go to the line edit and the editors
    as advice.
    """
    rules = cfg["style"]
    issues = machine_phrasing(text)
    feelings = [m.group(0) for m in NAMED_FEELINGS.finditer(text)]
    if len(feelings) > (rules["max_named_feelings"] if kind == "long" else 1):
        issues.append(f"show fear through the body and actions instead of naming it ({', '.join(feelings[:4])})")
    sentences = [s for s in SENTENCE_END.split(text.replace("\n", " ").strip()) if s.split()]
    if sentences:
        lengths = [len(s.split()) for s in sentences]
        average = sum(lengths) / len(lengths)
        if average > rules["max_avg_sentence_words"]:
            issues.append(f"sentences average {average:.0f} words; shorten them for listeners "
                          f"(under {rules['max_avg_sentence_words']})")
        long_ones = [s for s, n in zip(sentences, lengths) if n > rules["long_sentence_words"]]
        if len(long_ones) > (rules["max_long_sentences"] if kind == "long" else 0):
            issues.append(f"split long sentences, such as: '{' '.join(long_ones[0].split()[:12])}...'")
        firsts = [dedup.normalize(s).split()[0] if dedup.normalize(s) else "" for s in sentences]
        run, limit = 1, rules["max_same_openings_in_a_row"]
        for i in range(1, len(firsts)):
            run = run + 1 if firsts[i] and firsts[i] == firsts[i - 1] else 1
            if run > limit:
                issues.append(f"{run} sentences in a row start with '{firsts[i]}'; vary the openings")
                break
    words = text.split()
    dashes = len(re.findall(r"—|–|\s-\s|--", text))
    if words and dashes > max(1, len(words) // rules["words_per_dash"]):
        issues.append(f"use fewer dashes ({dashes}); the narrator reads them as awkward breaks")
    plain = [w for w in dedup.normalize(text).split() if w.isalpha()]
    hard = [w for w in plain if syllables(w) >= 4]
    if plain and len(hard) / len(plain) > rules["max_hard_word_share"]:
        issues.append("use plainer words for a global audience, for example instead of: "
                      + ", ".join(dict.fromkeys(hard[:5])))
    return issues


def structure_issues(script: dict[str, Any], kind: str, cfg: dict) -> list[str]:
    """Length, frame and rule checks that need no model call (derived fields are tidied first)."""
    tidy(script, kind)
    issues: list[str] = []
    lo, hi = word_range(cfg, kind)
    words = len(narration_of(script).split())
    if not lo <= words <= hi:
        issues.append(f"intro + body + outro is {words} words; it must be {lo}-{hi}")
    if not str(script.get("body", "")).strip():
        issues.append("body is empty")
    if kind == "long" and not str(script.get("intro", "")).strip():
        issues.append("long episodes need a Storyteller intro")
    outro = str(script.get("outro", "")).strip()
    question = str(script.get("comment_question", "")).strip()
    if not outro.endswith("?"):
        issues.append("the outro must end with one question to the viewer")
    if not question.endswith("?") or dedup.normalize(question) not in dedup.normalize(outro):
        issues.append("comment_question must be the question that ends the outro, copied exactly")
    text = narration_of(script)
    if CLAIMS_TRUE.search(text):
        issues.append("never claim the story is true or based on real events")
    for match in CLICHES.finditer(text):
        issues.append(f"replace the cliche '{match.group(0)}' with something specific")
    for match in PROFANITY.finditer(text):
        issues.append(f"remove the strong profanity '{match.group(0)}'; the channel stays advertiser-friendly")
    return issues


def script_issues(script: dict[str, Any], kind: str, cfg: dict) -> list[str]:
    """Every blocking check that needs no model call."""
    issues = structure_issues(script, kind, cfg)
    return issues + machine_phrasing(narration_of(script))


def _note_text(note: Any) -> str:
    if not isinstance(note, dict):
        return str(note).strip()
    quote, problem, fix = (str(note.get(k, "")).strip() for k in ("quote", "problem", "fix"))
    text = f'At "{quote}": {problem}' if quote else problem
    return f"{text} Fix: {fix}" if fix else text


def aggregate_reviews(reviews: list[dict[str, Any]]) -> dict[str, Any]:
    """Median score per criterion; a flag counts if any editor raised it."""
    scores = {c: statistics.median(int(r["scores"].get(c, 0)) for r in reviews)
              for c in schemas.REVIEW_CRITERIA}
    flags = sorted({f for r in reviews for f, raised in r.get("flags", {}).items() if raised is True})
    notes = list(dict.fromkeys(t for r in reviews for t in map(_note_text, r.get("notes", [])) if t))
    # Quoted notes are kept whole too, so a revision can go to the passage they point at.
    quoted = [n for r in reviews for n in r.get("notes", []) if isinstance(n, dict) and str(n.get("quote", "")).strip()]
    return {"scores": scores, "flags": flags, "notes": notes, "quoted_notes": quoted}


def review_issues(agg: dict[str, Any], min_score: float) -> list[str]:
    issues = [f"raise {c.replace('_', ' ')} (editors' median {s}/10)"
              for c, s in agg["scores"].items() if s < min_score]
    issues += [f"remove rule violation: {f.replace('_', ' ')}" for f in agg["flags"]]
    if issues:
        issues += agg["notes"]
    return issues


def scene_issues(narration: str, scenes: Any, kind: str, cast: Any = None) -> list[str]:
    """Checks a scene plan; when `cast` is given, also checks names, looks and delivery cues."""
    (lo, hi), (key_lo, key_hi) = scene_ranges(kind)
    if not isinstance(scenes, list) or not scenes:
        return ["no scenes returned"]
    issues: list[str] = []
    if cast is not None:
        names = {dedup.normalize(str(m.get("name", ""))) for m in cast if isinstance(m, dict)}
        if any(not str(m.get("look", "")).strip() for m in cast if isinstance(m, dict)):
            issues.append("every cast member needs a fixed look")
        for i, scene in enumerate(scenes):
            if not isinstance(scene, dict):
                continue
            unknown = [c for c in scene.get("characters", []) if dedup.normalize(str(c)) not in names]
            if unknown:
                issues.append(f"scene {i + 1} uses characters missing from the cast: {', '.join(map(str, unknown))}")
            if scene.get("delivery") not in schemas.DELIVERIES:
                issues.append(f"scene {i + 1} needs a delivery cue from {', '.join(schemas.DELIVERIES)}")
    if not lo <= len(scenes) <= hi:
        issues.append(f"{len(scenes)} scenes; need {lo}-{hi}")
    keys = sum(1 for s in scenes if isinstance(s, dict) and s.get("key_moment") is True)
    if not key_lo <= keys <= key_hi:
        issues.append(f"{keys} key moments; need {key_lo}-{key_hi}")
    full = dedup.normalize(narration)
    cursor, covered = 0, 0
    for i, scene in enumerate(scenes):
        if not isinstance(scene, dict) or not all(str(scene.get(k, "")).strip() for k in ("text", "visual", "location")):
            issues.append(f"scene {i + 1} is missing text, visual or location")
            continue
        span = dedup.normalize(scene["text"])
        found = full.find(span, cursor)
        if not span or found < 0:
            issues.append(f"scene {i + 1} text is not an exact, in-order span of the narration")
            continue
        cursor = found + len(span)
        covered += len(span.split())
    total = len(full.split())
    if total and covered / total < 0.95:
        issues.append(f"scenes cover {covered}/{total} narration words; they must cover all of it")
    return issues


def bible_issues(bible: dict[str, Any], kind: str) -> list[str]:
    """Missing decisions in a story bible."""
    long = kind == "long"
    hero, threat = bible.get("hero") or {}, bible.get("threat") or {}
    blank = lambda value: not str(value or "").strip()  # noqa: E731
    issues = [f"hero.{k} is empty" for k in (("name", "want", "wound", "flaw", "human_detail") if long
                                            else ("name", "want", "human_detail")) if blank(hero.get(k))]
    issues += [f"threat.{k} is empty" for k in ("what_it_is", "what_it_wants") if blank(threat.get(k))]
    rules = [r for r in threat.get("rules") or [] if not blank(r)]
    if len(rules) < (2 if long else 1):
        issues.append(f"give the threat at least {2 if long else 1} rules it always obeys")
    if long and not [x for x in threat.get("limits") or [] if not blank(x)]:
        issues.append("give the threat at least one limit (something it cannot do)")
    required = (("theme_question", "hero_choice", "dramatic_irony", "breath_moment", "refrain",
                 "refrain_turn", "sound_motif", "red_accent", "open_detail", "aftertaste") if long
                else ("red_accent", "aftertaste"))
    issues += [f"{k} is empty" for k in required if blank(bible.get(k))]
    if len(str(bible.get("refrain") or "").split()) > 12:
        issues.append("the refrain must be at most 12 words")
    if dedup.normalize(RITUAL_OPENING) in dedup.normalize(str(bible.get("refrain") or "")):
        issues.append("the refrain must be this story's own line, not the Storyteller's ritual")
    clues = [c for c in bible.get("clues") or [] if isinstance(c, dict)
             and not blank(c.get("plant")) and not blank(c.get("payoff"))]
    if len(clues) < (2 if long else 1):
        issues.append(f"plant at least {2 if long else 1} clues, each with its payoff")
    insider = [d for d in bible.get("insider_details") or [] if not blank(d)]
    if len(insider) < (4 if long else 2):
        issues.append(f"give at least {4 if long else 2} insider_details only someone who lived it would know")
    return issues


def bible_promises(bible: dict[str, Any], kind: str, dossier: dict | None = None) -> list[tuple[str, str]]:
    """What the finished script must visibly deliver, as (id, requirement) pairs.

    For a retold legend, its traditional core elements are promises too, so accuracy
    is fixed during editing rather than discovered at the final check.
    """
    hero, threat = bible.get("hero") or {}, bible.get("threat") or {}
    items = [("human_detail", f"the hero's human detail appears: {hero.get('human_detail', '')}"),
             ("want", f"we learn what the hero wants: {hero.get('want', '')}")]
    if kind == "long":
        items += [
            ("wound", f"the hero's wound is revealed: {hero.get('wound', '')}"),
            ("choice", f"the hero makes this choice: {bible.get('hero_choice', '')}"),
            ("irony", f"the viewer knows or suspects this before the hero: {bible.get('dramatic_irony', '')}"),
            ("breath", f"a moment of warmth or quiet humor: {bible.get('breath_moment', '')}"),
            ("tell", f"the threat's tell appears: {threat.get('tell', '')}"),
            ("refrain", f"the refrain appears early: {bible.get('refrain', '')}"),
            ("refrain_return", f"the refrain returns later with a darker meaning: {bible.get('refrain_turn', '')}"),
            ("open_detail", f"the deliberate ambiguity is on the page: {bible.get('open_detail', '')}"),
            ("aftertaste", f"the ending or outro ties the fear to: {bible.get('aftertaste', '')}"),
        ]
        for i, detail in enumerate((bible.get("insider_details") or [])[:2], 1):
            items.append((f"insider{i}", f"this insider detail makes the world feel lived-in: {detail}"))
    for i, clue in enumerate(bible.get("clues") or [], 1):
        if isinstance(clue, dict):
            items += [(f"clue{i}_plant", f"clue planted early: {clue.get('plant', '')}"),
                      (f"clue{i}_payoff", f"clue paid off later: {clue.get('payoff', '')}")]
    for i, element in enumerate((dossier or {}).get("core_elements") or [], 1):
        if i <= 3 and str(element).strip():
            items.append((f"legend{i}", f"the legend's traditional core is told accurately: {element}"))
    return [(i, r) for i, r in items if r.split(":", 1)[1].strip()]


def quote_found(quote: str, narration: str) -> bool:
    """Whether a judge's quote really is on the page (allowing small copying slips)."""
    words = dedup.normalize(quote).split()
    if len(words) < 3:
        return False
    full = dedup.normalize(narration)
    if " ".join(words) in full:
        return True
    mine = dedup.shingles(words, 3)
    return len(mine & dedup.shingles(full.split(), 3)) >= 0.6 * len(mine)


def outline_issues(outline: dict[str, Any], kind: str) -> list[str]:
    """Structure every outline needs: retention hooks, turns, a midpoint and a peak."""
    beats = [b for b in outline.get("beats") or [] if isinstance(b, dict)]
    issues: list[str] = []
    if len(beats) < (7 if kind == "long" else 3):
        issues.append(f"{len(beats)} beats is too few; break the story into at least {7 if kind == 'long' else 3}")
    no_turn = [str(b.get("name", i + 1)) for i, b in enumerate(beats) if not str(b.get("turn", "")).strip()]
    if no_turn:
        issues.append("every beat needs a turn (what changes); missing in: " + ", ".join(no_turn[:5]))
    if not str(outline.get("peak_moment", "")).strip():
        issues.append("name the peak moment")
    if kind == "long":
        rehooks = [r for r in outline.get("rehooks") or [] if isinstance(r, dict) and str(r.get("question", "")).strip()]
        if len(rehooks) < 3:
            issues.append("plan at least three rehooks: by about 30 seconds, about 2 minutes and the midpoint")
        elif not any(int(r.get("at_seconds") or 999) <= 45 for r in rehooks):
            issues.append("the first rehook must land within about 30 seconds")
        if not str(outline.get("midpoint_reversal", "")).strip():
            issues.append("add a midpoint reversal that changes what the hero is fighting")
        no_loop = [str(b.get("name", i + 1)) for i, b in enumerate(beats[:-1]) if not str(b.get("open_loop", "")).strip()]
        if len(no_loop) > len(beats) // 4:
            issues.append("most beats must leave an open question; missing in: " + ", ".join(no_loop[:5]))
    ladder = [e for e in outline.get("escalations") or [] if isinstance(e, dict)
              and all(str(e.get(k, "")).strip() for k in ("closer", "more_personal", "safety_lost"))]
    need = 3 if kind == "long" else 1
    if len(ladder) < need:
        issues.append(f"plan {need} escalation{'s' if need > 1 else ''}, each saying how the threat comes closer, "
                      "gets more personal and which safety the hero loses")
    return issues


def split_beats(beats: list[dict[str, Any]], parts: int) -> list[list[dict[str, Any]]]:
    """Contiguous groups of beats with roughly equal running time."""
    parts = max(1, min(parts, len(beats)))
    seconds = [max(int(b.get("target_seconds") or 0), 1) for b in beats]
    total, groups, current, done = sum(seconds), [], [], 0
    for i, (beat, secs) in enumerate(zip(beats, seconds)):
        current.append(beat)
        done += secs
        remaining_beats, remaining_groups = len(beats) - i - 1, parts - len(groups) - 1
        if remaining_groups and (done >= total * (len(groups) + 1) / parts or remaining_beats == remaining_groups):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def split_sections(body: str, parts: int) -> list[str]:
    """The body as `parts` contiguous sections of similar length, cut at paragraphs when it has
    enough of them, otherwise at sentences. Joining them with blank lines restores the body."""
    pieces = [p.strip() for p in body.split("\n\n") if p.strip()]
    if len(pieces) < parts:
        pieces = [x for x in SENTENCE_END.split(body.strip()) if x.strip()]
    parts = max(1, min(parts, len(pieces)))
    total = sum(len(p.split()) for p in pieces)
    sections: list[list[str]] = [[]]
    done = 0
    for i, piece in enumerate(pieces):
        sections[-1].append(piece)
        done += len(piece.split())
        left_pieces, left_sections = len(pieces) - i - 1, parts - len(sections)
        if left_sections and (done >= total * len(sections) / parts or left_pieces == left_sections):
            sections.append([])
    joiner = "\n\n" if "\n\n" in body else " "
    return [joiner.join(s) for s in sections if s]


def locate_notes(sections: list[str], quoted_notes: list[dict[str, Any]]) -> dict[int, list[str]]:
    """Which section (numbered from 1) each quoted editor's note points at."""
    found: dict[int, list[str]] = {}
    for note in quoted_notes:
        quote = str(note.get("quote", ""))
        for n, text in enumerate(sections, 1):
            if quote_found(quote, text):
                found.setdefault(n, []).append(_note_text(note))
                break
    return {n: list(dict.fromkeys(items)) for n, items in found.items()}


def _merge_scene(into: dict[str, Any], other: dict[str, Any]) -> None:
    into["key_moment"] = into.get("key_moment") is True or other.get("key_moment") is True
    for key in ("characters", "sfx"):
        into[key] = list(dict.fromkeys([*(into.get(key) or []), *(other.get(key) or [])]))


def snap_scenes(narration: str, scenes: Any, kind: str) -> Any:
    """Fix a scene plan's bookkeeping instead of rejecting a finished story.

    Scenes are snapped to whole sentences of the narration so together they cover
    all of it (a skipped passage joins the scene before it); a scene whose text
    can't be found joins its neighbor; too many scenes are merged and too many key
    moments are thinned, evenly. Visual choices are left as the planner made them.
    """
    if not isinstance(scenes, list):
        return scenes
    scenes = [dict(s) for s in scenes if isinstance(s, dict) and str(s.get("text", "")).strip()]
    sentences = [x for x in SENTENCE_END.split(narration.strip()) if x.strip()]
    if not scenes or not sentences:
        return scenes
    starts, offset = [], 0
    for sentence in sentences:
        starts.append(offset)
        offset += len(dedup.normalize(sentence)) + 1
    full = " ".join(dedup.normalize(x) for x in sentences)

    kept: list[tuple[int, dict[str, Any]]] = []
    for scene in scenes:
        words = dedup.normalize(scene["text"]).split()
        at = full.find(" ".join(words[:6]), starts[kept[-1][0]] if kept else 0) if words else -1
        index = max(i for i, start in enumerate(starts) if start <= at) if at >= 0 else -1
        if kept and index <= kept[-1][0]:
            _merge_scene(kept[-1][1], scene)  # unfound, or starts in the same sentence
        else:
            kept.append((0 if not kept else index, scene))
    bounds = [i for i, _ in kept] + [len(sentences)]
    fixed = []
    for n, (_, scene) in enumerate(kept):
        scene["text"] = " ".join(sentences[bounds[n]:bounds[n + 1]])
        fixed.append(scene)

    (_, hi), (_, key_hi) = scene_ranges(kind)
    while len(fixed) > hi:  # merge the shortest neighboring pair
        n = min(range(1, len(fixed)), key=lambda i: len(fixed[i - 1]["text"]) + len(fixed[i]["text"]))
        fixed[n - 1]["text"] = f"{fixed[n - 1]['text']} {fixed[n]['text']}"
        _merge_scene(fixed[n - 1], fixed.pop(n))
    keys = [i for i, scene in enumerate(fixed) if scene.get("key_moment") is True]
    if len(keys) > key_hi:
        chosen = {keys[round(i * (len(keys) - 1) / (key_hi - 1))] for i in range(key_hi)} if key_hi > 1 else {keys[-1]}
        for i in keys:
            fixed[i]["key_moment"] = i in chosen
    return fixed


def chunk_text(text: str, target_words: int) -> list[str]:
    """Group whole sentences into chunks of roughly `target_words` words."""
    sentences = [s for s in re.split(r"(?<=[.!?…])\s+", text.strip()) if s]
    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        current.append(sentence)
        if sum(len(s.split()) for s in current) >= target_words:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


def _valid_pitch(p: Any) -> bool:
    return isinstance(p, dict) and all(isinstance(p.get(k), str) and p[k].strip()
                                       for k in ("title", "logline", "synopsis"))


# --- Model-backed stages -----------------------------------------------------

def ensure_dossier(ctx: Context, legend: sqlite3.Row) -> dict | None:
    """Web-researched notes on a legend, cached on the inspiration row."""
    if legend["dossier"]:
        return json.loads(legend["dossier"])
    researcher = ctx.router.one("researcher")
    if getattr(researcher, "supports_web_search", False):
        notes = researcher.research("dossier_research", prompts.SYSTEM_RESEARCHER, prompts.dossier_research(legend))
    else:
        notes = legend["summary"] or ""
    dossier = researcher.generate_json("dossier", prompts.SYSTEM_RESEARCHER,
                                       prompts.dossier_structure(legend, notes), schemas.DOSSIER)
    db.set_dossier(ctx.conn, legend["id"], dossier)
    return dossier


def choose_legend(ctx: Context, kind: str) -> tuple[sqlite3.Row, dict | None] | None:
    """An unused legend that judges think can carry a faithful horror story.

    A harmless or comic being (a rain spirit, a household helper) would have to be
    given invented traits to frighten, which a faithful retelling can't do; such
    legends are marked and never offered again.
    """
    for _ in range(ctx.cfg["quality"]["legend_fit_tries"]):
        legends = db.unused_legends(ctx.conn)
        if not legends:
            return None
        legend = ctx.rng.choice(legends)
        dossier = ensure_dossier(ctx, legend)
        try:
            verdict = ctx.router.one("judges").generate_json(
                "legend_fit", prompts.SYSTEM_EDITOR, prompts.legend_fit(legend, dossier, kind), schemas.LEGEND_FIT)
        except LLMError as exc:
            log.warning("Legend fit check skipped: %s", exc)
            return legend, dossier
        if verdict.get("fit") is not False:
            return legend, dossier
        log.info("Legend '%s' is unfit for horror: %s", legend["title"], verdict.get("reason", ""))
        db.mark_unsuitable(ctx.conn, legend["id"], str(verdict.get("reason", "")))
    return None


def _same_story(ctx: Context):
    def ask(premise: dict[str, Any], row: sqlite3.Row) -> tuple[bool, str]:
        judge = ctx.router.one("judges")
        answer = judge.generate_json("judge_duplicate", prompts.SYSTEM_EDITOR,
                                     prompts.judge_duplicate(premise, row), schemas.DUPLICATE)
        return answer.get("same_story") is True, str(answer.get("reason", ""))
    return ask


def gather_pitches(ctx: Context, series: catalog.Series, legend: sqlite3.Row | None, dossier: dict | None,
                   sparks: list[sqlite3.Row], world: list[sqlite3.Row]) -> list[dict[str, Any]]:
    """Every pitcher pitches against its own fresh recipe; repeats are dropped."""
    rules = ctx.cfg["dedup"]
    used = db.used_signatures(ctx.conn)
    recent = db.recent_recipes(ctx.conn)
    weights = learning.ingredient_weights(ctx.conn)
    jobs = []
    for pitcher in ctx.router.pool("pitchers"):
        recipe = catalog.pick_recipe(ctx.rng, series, recent, used, rules, weights)
        used.add(recipe.signature)
        jobs.append((pitcher, recipe))
    avoid = db.all_stories(ctx.conn)[: rules["recent_titles_in_prompt"]]
    trends = db.top_trends(ctx.conn, ctx.cfg["research"]["trends_in_prompt"])
    learnings = db.latest_learnings(ctx.conn)

    def pitch(job: tuple[Any, catalog.Recipe]) -> list[dict[str, Any]]:
        pitcher, recipe = job
        try:
            result = pitcher.generate_json("pitches", prompts.SYSTEM_WRITER, prompts.pitches(
                series=series, recipe=recipe, formats=FORMATS, endings=ENDINGS,
                count=ctx.cfg["quality"]["pitches_per_pitcher"], legend=legend, dossier=dossier,
                sparks=sparks, world=world, avoid=avoid, trends=trends, learnings=learnings), schemas.PITCHES)
        except LLMError as exc:
            log.warning("Pitcher %s failed: %s", pitcher.spec.label, exc)
            return []
        return [dict(p, _recipe=recipe, _author=pitcher.spec.label, _family=pitcher.spec.family)
                for p in result.get("pitches", []) if _valid_pitch(p)]

    candidates = [p for batch in ctx.router.parallel(pitch, jobs) for p in batch]
    accepted: list[dict[str, Any]] = []
    for p in candidates:
        verdict = dedup.check_premise(ctx.conn, ctx.cfg, p, p["_recipe"].signature,
                                      ctx.router.embed, _same_story(ctx))
        if not verdict.ok:
            log.info("Pitch '%s' rejected: %s", p["title"], verdict.reason)
            continue
        twin = next((a for a in accepted if dedup.title_similarity(a["title"], p["title"]) >= rules["title_similarity"]
                     or dedup.cosine(a["_embedding"], verdict.embedding) >= rules["semantic_hard"]), None)
        if twin:
            log.info("Pitch '%s' duplicates '%s' from this round", p["title"], twin["title"])
            continue
        accepted.append(dict(p, _embedding=verdict.embedding))
    return accepted


def score_pitches(ctx: Context, series: catalog.Series, pitches: list[dict[str, Any]],
                  legend: sqlite3.Row | None = None, dossier: dict | None = None) -> None:
    limit = ctx.cfg["quality"]["judges_per_comparison"]
    jobs = [(p, judge) for p in pitches for judge in ctx.router.judges([p["_family"]], limit)]

    def score(job: tuple[dict, Any]) -> tuple[int, float]:
        p, judge = job
        result = judge.generate_json("score_pitch", prompts.SYSTEM_EDITOR,
                                     prompts.score_pitch(series, p, legend, dossier), schemas.PITCH_SCORE)
        values = [int(result["scores"].get(c, 0)) for c in schemas.PITCH_CRITERIA]
        return id(p), sum(values) / len(values)

    totals: dict[int, list[float]] = {}
    for key, value in ctx.router.panel(score, jobs, "score pitches"):
        totals.setdefault(key, []).append(value)
    for p in pitches:
        p["_score"] = round(statistics.mean(totals.get(id(p), [0.0])), 2)


def pitch_contest(ctx: Context, series: catalog.Series, pitches: list[dict[str, Any]]) -> list[tournament.Ranked]:
    finalists = sorted(pitches, key=lambda p: p["_score"], reverse=True)[: ctx.cfg["quality"]["tournament_size"]]

    def compare(judge: Any, a: dict, b: dict) -> str:
        return judge.generate_json("compare_pitches", prompts.SYSTEM_EDITOR,
                                   prompts.compare_pitches(series, a, b), schemas.PAIRWISE)["winner"]

    limit = ctx.cfg["quality"]["judges_per_comparison"]
    return tournament.rank(finalists, lambda a, b: ctx.router.judges([a["_family"], b["_family"]], limit),
                           compare, ctx.router.parallel, prior=lambda p: p["_score"])


def _plan_score(review: dict[str, Any]) -> float:
    try:
        return float(min(max(int(review.get("score") or 0), 0), 10))
    except (TypeError, ValueError):
        return 0.0


def refine_plan(ctx: Context, story: Story, item: dict[str, Any], *, stage: str,
                check: Any, review_prompt: Any, revise: Any, author_family: str,
                judges: int | None, review: bool = True) -> dict[str, Any]:
    """Review and revise a story bible or outline, keeping the best-scoring version.

    Free structural checks come first; editors then score the plan and name the few
    problems that matter most. A version is approved when the median score clears
    `plan_min_score`, and every revision is reviewed again, so the plan that goes
    forward is never an unreviewed rewrite.
    """
    q = ctx.cfg["quality"]
    best, best_key = item, (-1, -1.0)
    current = item
    for round_ in range(q["plan_revisions"] + 1):
        problems = check(current)
        score = 0.0
        if not problems and review:
            reviews = ctx.router.panel(
                lambda judge: judge.generate_json(f"review_{stage}", prompts.SYSTEM_EDITOR,
                                                  review_prompt(current), schemas.OUTLINE_REVIEW),
                ctx.router.judges([author_family], judges))
            score = statistics.median(_plan_score(r) for r in reviews)
            approved = score >= q["plan_min_score"]
            # Each judge's most important problems first, so a long list doesn't drown them.
            problems = [] if approved else list(dict.fromkeys(
                str(p) for rank in range(5) for r in reviews
                for p in (r.get("problems") or [])[rank:rank + 1] if str(p).strip()))[:6]
        if review:
            story.note(f"{stage}_review", round=round_ + 1, score=score, problems=len(problems))
        else:  # free checks only (Shorts, and each plan before the plan contest)
            story.note("plan_check", plan=stage, round=round_ + 1, problems=len(problems))
        key = (0 if check(current) else 1, score)
        if key > best_key:
            best, best_key = current, key
        if not problems:
            return current
        if round_ == q["plan_revisions"]:
            break
        current = revise(current, problems)
    if best is not current:
        story.note(f"{stage}_kept_best", score=best_key[1])
    return best


def make_bible(ctx: Context, story: Story, writer: Any = None, review: bool = True,
               recent_heroes: list[str] | None = None) -> dict[str, Any]:
    """The storyteller's decisions before plotting: hero, threat, theme, devices, clues."""
    kind = story.series.kind
    writer = writer or ctx.router.one("writers")
    if recent_heroes is None:
        recent_heroes = db.recent_heroes(ctx.conn, ctx.cfg["quality"]["recent_heroes"])
    bible = writer.generate_json("story_bible", prompts.SYSTEM_WRITER, prompts.story_bible(
        series=story.series, kind=kind, recipe=story.recipe, formats=FORMATS, endings=ENDINGS,
        pitch=story.pitch, legend=story.legend, dossier=story.dossier, world=story.world,
        recent_heroes=recent_heroes), schemas.BIBLE)

    def revise(current: dict[str, Any], problems: list[str]) -> dict[str, Any]:
        revised = writer.generate_json("revise_bible", prompts.SYSTEM_WRITER,
                                       prompts.revise_bible(current, problems), schemas.BIBLE)
        # A revision that lost decisions the current version had is not progress.
        return revised if len(bible_issues(revised, kind)) <= len(bible_issues(current, kind)) else current

    bible = refine_plan(ctx, story, bible, stage="bible", check=lambda b: bible_issues(b, kind),
                        review_prompt=lambda b: prompts.review_bible(kind, story.pitch, b, story.dossier),
                        revise=revise, author_family=writer.spec.family,
                        judges=ctx.cfg["quality"]["quick_judges"], review=review)
    story.authors["bible"] = writer.spec.label
    return bible


def make_outline(ctx: Context, story: Story, writer: Any = None, bible: dict | None = None,
                 review: bool = True) -> dict[str, Any]:
    kind = story.series.kind
    writer = writer or ctx.router.one("writers")
    bible = bible if bible is not None else story.bible
    target = 60 if kind == "short" else round(ctx.cfg["lengths"]["long_min_words"] / 2.3)
    outline = writer.generate_json("beat_sheet", prompts.SYSTEM_WRITER, prompts.beat_sheet(
        series=story.series, kind=kind, recipe=story.recipe, formats=FORMATS, endings=ENDINGS,
        pitch=story.pitch, legend=story.legend, dossier=story.dossier, world=story.world,
        target_seconds=target, bible=bible), schemas.BEAT_SHEET)

    def revise(current: dict[str, Any], problems: list[str]) -> dict[str, Any]:
        revised = writer.generate_json("revise_outline", prompts.SYSTEM_WRITER,
                                       prompts.revise_outline(current, problems), schemas.BEAT_SHEET)
        return revised if revised.get("beats") else current

    outline = refine_plan(ctx, story, outline, stage="outline", check=lambda o: outline_issues(o, kind),
                          review_prompt=lambda o: prompts.review_outline(kind, story.pitch, o, bible),
                          revise=revise, author_family=writer.spec.family,
                          judges=ctx.cfg["quality"]["judges_per_comparison"], review=review)
    if not outline.get("beats"):
        raise LLMError("outline has no beats")
    story.authors["outline"] = writer.spec.label
    return outline


def make_plan(ctx: Context, story: Story) -> tuple[dict[str, Any], dict[str, Any]]:
    """Story bible and outline. Returns (bible, outline).

    Long episodes: every writer plans the story, judges from other companies pick the
    better plan head to head, and only the winner's outline gets the full review.
    Shorts plan with one writer and free checks only: at 60 seconds, editors' lists of
    plan problems are mostly noise, and the Shorts draft contest does the selecting.
    """
    kind, q = story.series.kind, ctx.cfg["quality"]
    long = kind == "long"
    writers = ctx.router.pool("writers") if long and q["plan_contest"] else [ctx.router.one("writers")]
    heroes = db.recent_heroes(ctx.conn, q["recent_heroes"])

    def plan(writer: Any) -> dict[str, Any] | None:
        try:
            bible = make_bible(ctx, story, writer, review=long, recent_heroes=heroes)
            outline = make_outline(ctx, story, writer, bible, review=False)
        except LLMError as exc:
            if len(writers) == 1:
                raise
            log.warning("Planner %s failed: %s", writer.spec.label, exc)
            return None
        return {"bible": bible, "outline": outline, "_author": writer.spec.label,
                "_family": writer.spec.family, "_writer": writer}

    plans = [p for p in ctx.router.parallel(plan, writers) if p]
    if not plans:
        raise LLMError("every planner failed")
    winner = plans[0]
    if len(plans) > 1:
        def compare(judge: Any, a: dict, b: dict) -> str:
            return judge.generate_json("compare_plans", prompts.SYSTEM_EDITOR,
                                       prompts.compare_plans(kind, story.pitch, a, b), schemas.PAIRWISE)["winner"]

        ranked = tournament.rank(
            plans, lambda a, b: ctx.router.judges([a["_family"], b["_family"]], q["judges_per_comparison"]),
            compare, ctx.router.parallel,
            prior=lambda p: -len(bible_issues(p["bible"], kind)) - len(outline_issues(p["outline"], kind)))
        winner = ranked[0].item
        story.note("plan_contest", plans=len(plans), winner=winner["_author"], points=[r.points for r in ranked])
    story.authors["bible"] = story.authors["outline"] = winner["_author"]
    outline = winner["outline"]
    if long:
        story.bible = winner["bible"]
        outline = refine_plan(
            ctx, story, outline, stage="outline", check=lambda o: outline_issues(o, kind),
            review_prompt=lambda o: prompts.review_outline(kind, story.pitch, o, winner["bible"]),
            revise=lambda o, problems: _revised_outline(winner["_writer"], o, problems),
            author_family=winner["_family"], judges=q["judges_per_comparison"])
    return winner["bible"], outline


def _revised_outline(writer: Any, outline: dict[str, Any], problems: list[str]) -> dict[str, Any]:
    revised = writer.generate_json("revise_outline", prompts.SYSTEM_WRITER,
                                   prompts.revise_outline(outline, problems), schemas.BEAT_SHEET)
    return revised if revised.get("beats") else outline


def write_in_parts(ctx: Context, story: Story, writer: Any, outline: dict[str, Any],
                   avoid: tuple[str, ...] | list[str] = ()) -> dict[str, Any]:
    """Write a long body part by part (each part sees everything before it), then frame it.

    Long single outputs are where models lose the thread or loop; parts keep each
    call short and focused.
    """
    lo, hi = word_range(ctx.cfg, "long")
    groups = split_beats(outline["beats"], ctx.cfg["quality"]["draft_sections"])
    seconds = [sum(max(int(b.get("target_seconds") or 0), 1) for b in g) for g in groups]
    body_words = (lo + hi) // 2 - 80  # intro and outro take about 80 words
    parts: list[str] = []
    for i, group in enumerate(groups):
        part = writer.generate_json("write_section", prompts.SYSTEM_WRITER, prompts.write_section(
            series=story.series, recipe=story.recipe, formats=FORMATS, endings=ENDINGS, pitch=story.pitch,
            outline=outline, bible=story.bible, beats=group, part=i + 1, parts=len(groups),
            so_far="\n\n".join(parts), words=round(body_words * seconds[i] / sum(seconds)),
            legend=story.legend, dossier=story.dossier, world=story.world, avoid=avoid), schemas.SECTION)
        text = str(part.get("text", "")).strip()
        if not text:
            raise LLMError(f"{writer.spec.label} wrote an empty part {i + 1}")
        parts.append(text)
    body = "\n\n".join(parts)
    frame = writer.generate_json("write_frame", prompts.SYSTEM_WRITER, prompts.write_frame(
        series=story.series, kind="long", pitch=story.pitch, bible=story.bible, body=body,
        legend=story.legend, world=story.world), schemas.FRAME)
    return tidy(dict(frame, body=body), "long")


def _quick_score(ctx: Context, story: Story, draft: dict[str, Any]) -> float:
    """Mean editor score from the quick judges (other companies), minus free-check problems."""
    penalty = len(script_issues(draft, story.series.kind, ctx.cfg))
    try:
        reviews = ctx.router.panel(
        lambda judge: judge.generate_json("review", prompts.SYSTEM_EDITOR, prompts.review(
            story.series.kind, story.recipe, FORMATS, ENDINGS, draft, story.dossier, story.bible), schemas.REVIEW),
        ctx.router.judges([draft["_family"]], ctx.cfg["quality"]["quick_judges"]), "score drafts")
    except LLMError as exc:
        log.warning("Quick score skipped: %s", exc)
        return -float(penalty)
    scores = [int(r["scores"].get(c, 0)) for r in reviews for c in schemas.REVIEW_CRITERIA]
    return (statistics.mean(scores) if scores else 0.0) - penalty


def draft_contest(ctx: Context, story: Story, outline: dict[str, Any],
                  avoid: tuple[str, ...] | list[str] = ()) -> tuple[dict[str, Any], str]:
    """Competing drafts; judges from other companies pick the best. Returns (script, author family).

    Long episodes: one draft per writer, written part by part. Shorts compete on
    breadth: a 60-second draft is cheap, so several takes per writer are written,
    quick-scored, and the finalists go head to head.
    """
    kind, q = story.series.kind, ctx.cfg["quality"]
    lo, hi = word_range(ctx.cfg, kind)
    in_parts = kind == "long" and q["draft_sections"] > 1 and len(outline["beats"]) > 1
    writers = ctx.router.pool("writers")
    if kind == "short":
        jobs = [(writers[i % len(writers)], prompts.SHORT_ANGLES[i % len(prompts.SHORT_ANGLES)])
                for i in range(max(q["short_drafts"], len(writers)))]
    else:
        jobs = [(w, "") for w in writers]

    def write(job: tuple[Any, str]) -> dict[str, Any] | None:
        # A model that loops to its output limit usually manages on a second try, and a
        # contest with one draft is no contest; long drafts get that second try.
        for attempt in range(2 if kind == "long" else 1):
            draft = write_once(job)
            if draft is not None:
                return draft
            log.info("Retrying writer %s", job[0].spec.label)
        return None

    def write_once(job: tuple[Any, str]) -> dict[str, Any] | None:
        writer, angle = job
        try:
            if in_parts:
                script = write_in_parts(ctx, story, writer, outline, avoid)
            else:
                script = tidy(writer.generate_json("write_draft", prompts.SYSTEM_WRITER, prompts.write_draft(
                    series=story.series, kind=kind, recipe=story.recipe, formats=FORMATS, endings=ENDINGS,
                    pitch=story.pitch, outline=outline, min_words=lo, max_words=hi, legend=story.legend,
                    dossier=story.dossier, world=story.world, bible=story.bible, angle=angle, avoid=avoid),
                    schemas.SCRIPT), kind)
        except LLMError as exc:
            log.warning("Writer %s failed: %s", writer.spec.label, exc)
            return None
        return dict(script, _author=writer.spec.label, _family=writer.spec.family)

    drafts = [d for d in ctx.router.parallel(write, jobs) if d]
    if not drafts:
        raise LLMError("every writer failed")
    finalists = drafts
    if len(drafts) > q["short_finalists"] and kind == "short":
        scored = list(zip(ctx.router.parallel(lambda d: _quick_score(ctx, story, d), drafts), drafts))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        finalists = [d for _, d in scored[: q["short_finalists"]]]
        story.note("short_drafts", written=len(drafts), quick_scores=[round(x, 2) for x, _ in scored])

    def compare(judge: Any, a: dict, b: dict) -> str:
        return judge.generate_json("compare_drafts", prompts.SYSTEM_EDITOR,
                                   prompts.compare_drafts(kind, story.pitch, a, b), schemas.PAIRWISE)["winner"]

    limit = q["judges_per_comparison"]
    ranked = tournament.rank(finalists, lambda a, b: ctx.router.judges([a["_family"], b["_family"]], limit),
                             compare, ctx.router.parallel, prior=lambda d: -len(script_issues(d, kind, ctx.cfg)))
    best = ranked[0].item
    story.note("draft_contest", drafts=len(drafts), finalists=len(finalists), winner=best["_author"],
               points=[r.points for r in ranked], fresh=bool(avoid))
    story.authors["draft"] = best["_author"]
    return {k: best.get(k) for k in SCRIPT_FIELDS}, best["_family"]


def fit_length(ctx: Context, story: Story, script: dict[str, Any]) -> dict[str, Any]:
    """Bring the script inside the word range with a dedicated edit (models count words badly).

    Keeps whichever version came closest to the range.
    """
    lo, hi = word_range(ctx.cfg, story.series.kind)
    reviser = ctx.router.one("reviser")

    def distance(candidate: dict[str, Any]) -> int:
        words = len(narration_of(candidate).split())
        return max(lo - words, 0, words - hi)

    best = candidate = script
    for _ in range(ctx.cfg["quality"]["length_fix_attempts"]):
        if distance(candidate) == 0:
            break
        words = len(narration_of(candidate).split())
        candidate = tidy(reviser.generate_json("fit_length", prompts.SYSTEM_WRITER,
                                               prompts.fit_length(candidate, words, lo, hi), schemas.SCRIPT))
        if distance(candidate) < distance(best):
            best = candidate
    if best is not script:
        story.note("fit_length", words=len(narration_of(best).split()), in_range=distance(best) == 0)
    return best


def repair(ctx: Context, story: Story, script: dict[str, Any], notes: list[str]) -> dict[str, Any]:
    """Fix specific problems without breaking the script's structure.

    Revise against the notes, refit the length, then make one targeted repair of
    anything structural (length, closing question) the revision broke.
    """
    kind = story.series.kind
    lo, hi = word_range(ctx.cfg, kind)
    reviser = ctx.router.one("reviser")
    keep = ("Keep the structure: the outro must still end with exactly one question to the viewer, "
            "and change only what these notes require.")
    fixed = tidy(reviser.generate_json("revise", prompts.SYSTEM_WRITER, prompts.revise(
        kind, script, [*notes, keep], lo, hi, story.bible), schemas.SCRIPT), kind)
    fixed = fit_length(ctx, story, fixed)
    structural = structure_issues(fixed, kind, ctx.cfg)
    if structural:
        fixed = tidy(reviser.generate_json("revise", prompts.SYSTEM_WRITER, prompts.revise(
            kind, fixed, [*structural, keep], lo, hi, story.bible), schemas.SCRIPT), kind)
        fixed = fit_length(ctx, story, fixed)
    return fixed


def clean_style(ctx: Context, story: Story, script: dict[str, Any]) -> dict[str, Any]:
    """Line-edit away machine-sounding or hard-to-follow prose that the free checks found.

    Keeps the version with the fewest style problems that didn't break the structure.
    """
    kind = story.series.kind
    problems = style_issues(narration_of(script), ctx.cfg, kind)
    if not problems:
        return script
    reviser = ctx.router.one("reviser")
    best, best_count = script, len(problems)
    structure = len(structure_issues(script, kind, ctx.cfg))
    candidate = script
    for _ in range(ctx.cfg["quality"]["style_fix_attempts"]):
        candidate = tidy(reviser.generate_json("line_edit", prompts.SYSTEM_WRITER,
                                               prompts.line_edit(candidate, problems), schemas.SCRIPT), kind)
        problems = style_issues(narration_of(candidate), ctx.cfg, kind)
        if len(problems) < best_count and len(structure_issues(candidate, kind, ctx.cfg)) <= structure:
            best, best_count = candidate, len(problems)
        if not problems:
            break
    story.note("line_edit", style_problems_left=best_count)
    return best


@dataclass
class BibleVerdict:
    breaks: list[str] = field(default_factory=list)  # confirmed by two independent judges
    missing: list[str] = field(default_factory=list)
    low_coverage: bool = False
    coverage: float = 1.0


def check_bible(ctx: Context, story: Story, script: dict[str, Any], author_family: str) -> BibleVerdict:
    """Does the script deliver the story bible?

    A promise only counts when the judge's quote is really in the script. A claimed
    break of the threat's rules (or, for a legend, a trait the research doesn't
    support) counts only when a second, independent judge confirms it: one model's
    claim is too noisy to veto a story.
    """
    if not story.bible:
        return BibleVerdict()
    promises = bible_promises(story.bible, story.series.kind, story.dossier)
    if not promises:
        return BibleVerdict()
    judges = ctx.router.judges([author_family])
    try:
        result = FailoverJudge(judges).generate_json(
            "bible_check", prompts.SYSTEM_EDITOR,
            prompts.bible_check(story.bible, promises, script, story.dossier), schemas.BIBLE_CHECK)
    except LLMError as exc:
        log.warning("Story-bible check skipped: %s", exc)
        return BibleVerdict()
    narration = narration_of(script)
    shown = {str(r.get("id")) for r in result.get("items", [])
             if isinstance(r, dict) and r.get("present") is True and quote_found(str(r.get("quote", "")), narration)}
    missing = [f"deliver the story bible: {requirement}" for item_id, requirement in promises if item_id not in shown]
    claimed = [str(b).strip() for b in result.get("threat_rule_breaks", []) if _real_issue(b)]
    breaks = confirm_breaks(ctx, story, script, claimed, judges[1:]) if claimed else []
    coverage = 1 - len(missing) / len(promises)
    story.note("bible_check", coverage=round(coverage, 2), missing=len(missing),
               rule_breaks_claimed=len(claimed), rule_breaks_confirmed=len(breaks))
    floor = ctx.cfg["quality"]["bible_min_coverage" if story.series.kind == "long" else "bible_min_coverage_short"]
    return BibleVerdict([f"the threat breaks its own rules: {b}" for b in breaks], missing, coverage < floor,
                        round(coverage, 2))


def confirm_breaks(ctx: Context, story: Story, script: dict[str, Any], claimed: list[str],
                   others: list[Any]) -> list[str]:
    """Keep only the claimed rule breaks a second judge agrees are real."""
    if not others:
        return claimed
    try:
        result = FailoverJudge(others).generate_json(
            "confirm_breaks", prompts.SYSTEM_EDITOR,
            prompts.confirm_breaks(story.bible, script, claimed, story.dossier), schemas.CONFIRM)
    except LLMError as exc:
        log.warning("Could not confirm rule breaks, keeping them: %s", exc)
        return claimed
    real = {int(r.get("index", 0)) for r in result.get("items", []) if isinstance(r, dict) and r.get("real") is True}
    return [c for n, c in enumerate(claimed, 1) if n in real]


# --- The editors' loop and the rewrite ladder ----------------------------------

# Low scores here are usually the plan's fault, not the prose's.
STRUCTURAL_CRITERIA = frozenset({"tension", "character", "emotional_payoff"})


@dataclass
class Version:
    """One judged draft in the editors' loop."""

    script: dict[str, Any]
    author_family: str
    structural: list[str] = field(default_factory=list)  # free-check problems; the panel skips these drafts
    agg: dict[str, Any] = field(default_factory=dict)
    panel: list[str] = field(default_factory=list)  # scores below the bar and safety flags, with notes
    bible: BibleVerdict = field(default_factory=BibleVerdict)
    unrecognized: list[str] = field(default_factory=list)  # a retold legend's being not recognized
    advice: list[str] = field(default_factory=list)

    @property
    def scores(self) -> dict[str, Any]:
        return self.agg.get("scores") or {}

    def low(self, min_score: float) -> set[str]:
        return {c for c, v in self.scores.items() if v < min_score}

    def blocking(self, accept_gaps: bool = False) -> list[str]:
        if self.structural:
            return list(self.structural)
        gaps = self.bible.missing if self.bible.low_coverage and not accept_gaps else []
        return list(dict.fromkeys(self.panel + self.bible.breaks + self.unrecognized + gaps))

    def passes_but_gaps(self) -> bool:
        """Approved by the editors, with only unquoted story-bible promises left."""
        return bool(self.scores) and not self.blocking(accept_gaps=True)

    def key(self) -> tuple:
        """Ordering of versions: free checks passed, editors satisfied, then the weakest and mean score."""
        if self.structural or not self.scores:
            return (0, 0, 0.0, 0.0, 0.0)
        values = [float(v) for v in self.scores.values()]
        return (1, int(self.passes_but_gaps()), min(values), statistics.mean(values), self.bible.coverage)

    def notes(self) -> list[str]:
        """What a revision should fix: blocking issues first, then notes, missing promises, style advice."""
        return list(dict.fromkeys(self.blocking() + list(self.agg.get("notes", []))
                                  + self.bible.missing + self.advice))


def recognition_issues(ctx: Context, story: Story, script: dict[str, Any], author_family: str) -> list[str]:
    """For a retold legend: would most judges recognize its being? One judge can't veto,
    but a majority saying "this is a different creature" blocks."""
    if not (story.legend is not None and ctx.cfg["quality"]["recognition_check"]):
        return []
    try:
        verdicts = ctx.router.panel(
            lambda judge: judge.generate_json("recognition", prompts.SYSTEM_EDITOR, prompts.recognition(
                story.legend, story.dossier, script), schemas.RECOGNITION),
            ctx.router.judges([author_family], ctx.cfg["quality"]["judges_per_comparison"]), "legend recognition")
    except LLMError as exc:
        log.warning("Legend recognition check skipped: %s", exc)
        return []
    against = [v for v in verdicts if v.get("recognizable") is False]
    story.note("recognition", judges=len(verdicts), not_recognized=len(against))
    if len(against) * 2 <= len(verdicts):
        return []
    traits = list(dict.fromkeys(str(c).strip() for v in against for c in v.get("contradicted") or [] if _real_issue(c)))
    detail = "; ".join(traits[:4]) or "; ".join(str(v.get("reason", "")) for v in against)[:400]
    return [f"make the being recognizably {story.legend['title']} as the research describes it "
            f"(most judges did not recognize it): {detail}"]


def judge_version(ctx: Context, story: Story, script: dict[str, Any], author_family: str,
                  story_id: int) -> Version:
    """Free checks, then the story-bible check, the editors' panel and (legends) recognition."""
    kind, cfg = story.series.kind, ctx.cfg
    script = fit_length(ctx, story, script)
    script = fit_length(ctx, story, clean_style(ctx, story, script))
    structural = script_issues(script, kind, cfg)
    copy = dedup.check_script(ctx.conn, cfg, story_id, narration_of(script), _sources(story))
    if not copy.ok:
        structural.append(f"rewrite in original words: {copy.reason}")
    advice = [i for i in style_issues(narration_of(script), cfg, kind) if i not in structural]
    version = Version(script, author_family, structural, advice=advice)
    if structural:
        return version
    version.bible = check_bible(ctx, story, script, author_family)
    reviews = ctx.router.panel(
        lambda judge: judge.generate_json("review", prompts.SYSTEM_EDITOR, prompts.review(
            kind, story.recipe, FORMATS, ENDINGS, script, story.dossier, story.bible), schemas.REVIEW),
        ctx.router.judges([author_family], cfg["quality"]["judges_per_comparison"]))
    version.agg = aggregate_reviews(reviews)
    version.panel = review_issues(version.agg, cfg["quality"]["min_score"])
    version.unrecognized = recognition_issues(ctx, story, script, author_family)
    return version


def diagnose(version: Version, kind: str, q: dict[str, Any], fresh_left: int, stuck: set[str],
             last_action: str, stalled: bool) -> tuple[str, str]:
    """The showrunner's call: fix the deepest broken level first. Returns (action, why).

    Actions, deepest first: kill (the premise), replan (new outline, then fresh drafts),
    redraft (fresh drafts from the same plan), sections (rewrite only the passages the
    notes quote), lines (revise the whole script against the notes), hook (the hook
    lab), structure (repair what the free checks found).
    """
    if version.structural:
        return "structure", "free checks failed: " + "; ".join(version.structural[:2])
    scores = version.scores
    floor = min(float(v) for v in scores.values())
    originality = float(scores.get("originality", 10))
    if originality <= q["kill_at_or_below_originality"]:
        return "kill", f"the premise itself is familiar (originality {originality:g}); rewriting can't fix that"
    if version.unrecognized:
        if fresh_left:
            return "replan", "the legend's being isn't recognizable: re-plan around the real legend"
        return "lines", "the legend's being isn't recognizable"
    # A low promise count alone isn't enough: on a short script the check's quotes often
    # miss what is there. It takes weak scores too before the draft is thrown away.
    far_off = floor <= q["redraft_at_or_below_score"] or (
        version.bible.coverage < q["redraft_below_coverage"] and floor < q["min_score"] - 1)
    if fresh_left and far_off:
        return "redraft", (f"the draft is far from its plan (lowest score {floor:g}, promises delivered "
                           f"{version.bible.coverage:.0%}): write it again rather than patch it")
    if stuck and fresh_left:
        return "replan", f"{', '.join(sorted(stuck))} stayed below the bar after a revision: the plan needs fixing"
    if stalled and last_action in ("lines", "sections", "hook"):
        if fresh_left and floor <= q["min_score"] - 2:
            return "redraft", "patching made the draft worse"
        other = "sections" if kind == "long" and last_action != "sections" else "lines"
        return other, f"the last revision ({last_action}) made it worse; trying a different kind of fix"
    if version.low(q["min_score"]) == {"hook"} and last_action != "hook":
        return "hook", "only the hook is below the bar"
    if kind == "long":
        return "sections", "revise the passages the notes point at"
    return "lines", "revise against the notes"


def rewrite_sections(ctx: Context, story: Story, version: Version) -> dict[str, Any]:
    """Rewrite only the sections the editors' quoted notes point at; the rest stays frozen,
    so a fix in one place can't undo what already works elsewhere."""
    kind, script = story.series.kind, version.script
    sections = split_sections(script["body"], 2 * ctx.cfg["quality"]["draft_sections"])
    located = locate_notes(sections, version.agg.get("quoted_notes", []))
    if not located or len(sections) < 2:
        return repair(ctx, story, script, version.notes())
    reviser = ctx.router.one("reviser")
    result = reviser.generate_json("rewrite_sections", prompts.SYSTEM_WRITER, prompts.rewrite_sections(
        script, sections, located, story.bible), schemas.CHUNK_REWRITE)
    new = list(sections)
    for row in result.get("chunks", []):
        try:
            n = int(row.get("chunk", 0))
        except (TypeError, ValueError):
            continue
        if n in located and str(row.get("text", "")).strip():
            new[n - 1] = str(row["text"]).strip()
    story.note("section_rewrite", sections=len(sections), rewritten=sorted(located))
    if new == sections:
        return repair(ctx, story, script, version.notes())
    joiner = "\n\n" if "\n\n" in script["body"] else " "
    candidate = fit_length(ctx, story, tidy(dict(script, body=joiner.join(new)), kind))
    structural = structure_issues(candidate, kind, ctx.cfg)
    return repair(ctx, story, candidate, structural) if structural else candidate


def replan(ctx: Context, story: Story, outline: dict[str, Any], version: Version) -> dict[str, Any]:
    """Go back to the plan: revise the outline (and, for an unrecognized legend, the story
    bible) against what the editors found, and review it again."""
    kind = story.series.kind
    writer = ctx.router.one("writers")
    problems = [i for i in version.blocking() if not i.startswith('At "')][:6] + version.agg.get("notes", [])[:4]
    if version.unrecognized and story.bible:
        revised = writer.generate_json("revise_bible", prompts.SYSTEM_WRITER,
                                       prompts.revise_bible(story.bible, version.unrecognized), schemas.BIBLE)
        if len(bible_issues(revised, kind)) <= len(bible_issues(story.bible, kind)):
            story.bible = revised
    bible = story.bible
    new = refine_plan(ctx, story, _revised_outline(writer, outline, problems), stage="outline",
                      check=lambda o: outline_issues(o, kind),
                      review_prompt=lambda o: prompts.review_outline(kind, story.pitch, o, bible),
                      revise=lambda o, p: _revised_outline(writer, o, p), author_family=writer.spec.family,
                      judges=ctx.cfg["quality"]["judges_per_comparison"], review=kind == "long")
    story.note("replan", problems=len(problems))
    return new if new.get("beats") else outline


def apply_action(ctx: Context, story: Story, action: str, version: Version,
                 outline: dict[str, Any]) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Carry out the showrunner's call on the best version. Returns (script, author family, outline)."""
    reviser = ctx.router.one("reviser")
    revised_by = reviser.spec.family
    if action == "structure":
        script = repair(ctx, story, version.script, version.structural)
    elif action == "hook":
        script = hook_lab(ctx, story, version.script)
        if script is version.script:  # no better opening: revise against the notes instead
            script = repair(ctx, story, version.script, version.notes())
        else:
            revised_by = version.author_family  # the body is the author's own
    elif action == "sections":
        script = rewrite_sections(ctx, story, version)
    elif action == "lines":
        script = repair(ctx, story, version.script, version.notes())
    else:  # redraft / replan: fresh drafts that know what went wrong last time
        if action == "replan":
            outline = replan(ctx, story, outline, version)
        script, family = draft_contest(ctx, story, outline, avoid=version.blocking()[:8])
        return script, family, outline
    story.authors["reviser"] = reviser.spec.label
    return script, revised_by, outline


@dataclass
class Edit:
    script: dict[str, Any] | None  # None: rejected
    review: dict[str, Any]
    reason: str
    outline: dict[str, Any]
    best: Version | None = None
    diagnosis: str = ""


def editors_loop(ctx: Context, story: Story, script: dict[str, Any], author_family: str,
                 story_id: int, outline: dict[str, Any] | None = None) -> Edit:
    """Review, diagnose and fix until the editors approve, always keeping the best version.

    Each round judges the current draft. If it is worse than the best so far, the best
    is restored, so a revision can never lose ground. The showrunner (`diagnose`) then
    picks the fix for the deepest problem: a fresh plan, a fresh draft, a rewrite of
    the sections the notes point at, a whole-script revision, or the hook lab.
    """
    q, kind = ctx.cfg["quality"], story.series.kind
    outline = outline or {"beats": []}
    rounds, fresh_left = q["max_revisions"] + 1, q["max_fresh_starts"]
    best: Version | None = None
    previous: Version | None = None
    last_action, why, gap_fix_tried, round_ = "", "", False, 0
    while True:
        round_ += 1
        version = judge_version(ctx, story, script, author_family, story_id)
        improved = best is None or version.key() > best.key()
        if improved:
            best = version
        story.note("editors", round=round_, after=last_action or "first draft", issues=len(version.blocking()),
                   scores=version.scores or None, coverage=version.bible.coverage if version.scores else None,
                   kept="this version" if improved else "the earlier best",
                   **({"free_check_failed": version.structural[:2]} if version.structural else {}))
        if not version.blocking():
            return Edit(version.script, version.agg, "", outline, version)
        if best.passes_but_gaps() and (gap_fix_tried or round_ >= rounds):
            story.note("bible_gaps_accepted", missing=best.bible.missing[:4])
            return Edit(best.script, best.agg, "", outline, best)
        if round_ >= rounds:
            break
        stuck = (version.low(q["min_score"]) & previous.low(q["min_score"]) & STRUCTURAL_CRITERIA
                 if previous and version.scores else set())
        target = best
        if version.structural and version is not best and last_action != "structure":
            # A new draft (often a fresh start) that only fails a free check, such as its
            # length, is repaired before it is judged against the best, not thrown away.
            target, action, why = version, "structure", "repair the new draft's free-check problems first"
        else:
            action, why = diagnose(best, kind, q, fresh_left, stuck, last_action, stalled=not improved)
        story.note("showrunner", action=action, why=why)
        if action == "kill":
            break
        gap_fix_tried = gap_fix_tried or best.passes_but_gaps()
        if version.scores:
            previous = version
        script, author_family, outline = apply_action(ctx, story, action, target, outline)
        if action in ("redraft", "replan"):
            fresh_left -= 1
            rounds += 1  # a fresh start gets its own review
        last_action = action
    reason = "; ".join(best.blocking()[:6])
    if last_action == "kill" or why.startswith("the premise"):
        reason = f"premise: {why}"
    return Edit(None, best.agg, reason, outline, best, "kill" if reason.startswith("premise") else last_action)


def audience_test(ctx: Context, story: Story, script: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Simulated viewers rate each ~30-second part; weak parts are rewritten if that's judged better."""
    q, kind = ctx.cfg["quality"], story.series.kind
    chunks = chunk_text(script["body"], q["audience_chunk_words"])
    if len(chunks) < 2:
        return script, {"skipped": "too short to chunk"}
    # Viewers hear the Storyteller's intro first; it is rated as part 1 but only body parts are rewritten.
    intro = str(script.get("intro") or "").strip()
    shown = ([intro] if intro else []) + chunks
    first_body = 2 if intro else 1
    readers = ctx.router.pool("audience")
    jobs = [(persona, readers[i % len(readers)]) for i, persona in enumerate(q["audience_personas"])]

    def watch(job: tuple[str, Any]) -> dict[str, Any] | None:
        persona, reader = job
        try:
            return reader.generate_json("audience", prompts.system_audience(persona),
                                        prompts.audience(kind, shown), schemas.AUDIENCE)
        except LLMError as exc:
            log.warning("Audience reader failed: %s", exc)
            return None

    reports = [r for r in ctx.router.parallel(watch, jobs) if r]
    if not reports:
        return script, {"skipped": "no audience reports"}
    interest: dict[int, list[int]] = {}
    leaves: dict[int, int] = {}
    notes: dict[int, list[str]] = {}
    for report in reports:
        left = False
        for row in report.get("chunks", []):
            n = int(row.get("chunk", 0)) - (first_body - 1)  # body part number; 0 is the intro
            if not (0 if intro else 1) <= n <= len(chunks):
                continue
            interest.setdefault(n, []).append(int(row.get("interest", 0)))
            if row.get("note"):
                notes.setdefault(n, []).append(str(row["note"]))
            if row.get("would_leave") and not left:  # count each viewer's exit once
                leaves[n] = leaves.get(n, 0) + 1
                left = True
    means = {n: round(statistics.mean(v), 2) for n, v in interest.items()}
    intro_interest = means.pop(0, None)
    weak = sorted((n for n in means if means[n] < q["audience_min_interest"] or leaves.get(n, 0) >= 2),
                  key=lambda n: (means[n], -leaves.get(n, 0)))[: q["max_chunks_to_rewrite"]]
    report = {"chunk_interest": means, "exits": leaves, "weak_chunks": weak,
              "viewers": [r.get("overall", "") for r in reports]}
    if intro_interest is not None:
        report["intro_interest"] = intro_interest
    story.note("audience", viewers=len(reports), weak=weak)
    if not weak:
        return script, report

    reviser = ctx.router.one("reviser")
    rewritten = reviser.generate_json("rewrite_chunks", prompts.SYSTEM_WRITER, prompts.rewrite_chunks(
        script, chunks, {n: notes.get(n, ["lost interest"]) for n in weak}), schemas.CHUNK_REWRITE)
    new_chunks = list(chunks)
    for row in rewritten.get("chunks", []):
        n = int(row.get("chunk", 0))
        if n in weak and str(row.get("text", "")).strip():
            new_chunks[n - 1] = str(row["text"]).strip()
    candidate = dict(script, body="\n\n".join(new_chunks))
    if candidate["body"] == script["body"]:
        report["rewrite"] = "discarded: no change"
        return script, report
    candidate = fit_length(ctx, story, tidy(candidate, kind))  # rewritten parts often run long
    problems = script_issues(candidate, kind, ctx.cfg)
    if problems:
        report["rewrite"] = "discarded: failed checks (" + "; ".join(problems[:2]) + ")"
        return script, report

    def compare(judge: Any, a: dict, b: dict) -> str:
        return judge.generate_json("compare_drafts", prompts.SYSTEM_EDITOR,
                                   prompts.compare_drafts(kind, story.pitch, a, b), schemas.PAIRWISE)["winner"]

    quick = ctx.cfg["quality"]["quick_judges"]
    ranked = tournament.rank([script, candidate], lambda a, b: ctx.router.judges([reviser.spec.family], quick),
                             compare, ctx.router.parallel)
    improved = ranked[0].item is candidate or ranked[0].points == ranked[1].points
    report["rewrite"] = "kept" if improved else "discarded: judges preferred the original"
    return (candidate if improved else script), report


def valid_pauses(pauses: Any, narration: str, limit: int) -> list[dict[str, Any]]:
    """Pauses whose phrase is really in the narration, in reading order, with sane lengths."""
    full = dedup.normalize(narration)
    found: list[tuple[int, dict[str, Any]]] = []
    for pause in pauses if isinstance(pauses, list) else []:
        if not isinstance(pause, dict):
            continue
        phrase = str(pause.get("after", "")).strip()
        at = full.find(dedup.normalize(phrase)) if len(phrase.split()) >= 2 else -1
        if at >= 0 and all(at != f[0] for f in found):
            found.append((at, {"after": phrase, "ms": min(max(int(pause.get("ms") or 800), 300), 3000)}))
    return [p for _, p in sorted(found, key=lambda f: f[0])][:limit]


def polish(ctx: Context, story: Story, script: dict[str, Any]) -> dict[str, Any]:
    """Read-aloud line edit; kept only if it is a light touch that passes the checks."""
    kind = story.series.kind
    lo, hi = word_range(ctx.cfg, kind)
    polisher = ctx.router.one("polisher")
    result = polisher.generate_json("polish", prompts.SYSTEM_WRITER, prompts.polish(kind, script, lo, hi),
                                    schemas.POLISH)
    candidate = tidy(dict(script, intro=result.get("intro", ""), body=result.get("body", ""),
                          outro=result.get("outro", "")), kind)
    if str(candidate.get("body", "")).strip():
        candidate = fit_length(ctx, story, candidate)  # a word or two over shouldn't cost the polish
    similarity = dedup.lexical_similarity(narration_of(script), narration_of(candidate), size=1)
    problems = script_issues(candidate, kind, ctx.cfg)
    if problems or similarity < 0.6:
        story.note("polish", kept=False, similarity=round(similarity, 2), problems=problems[:3])
        return script
    story.note("polish", kept=True, similarity=round(similarity, 2))
    story.authors["polisher"] = polisher.spec.label
    return script if not candidate.get("body") else candidate


def annotate(ctx: Context, story: Story, script: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """Narration notes for the voice engine, made on the final script: (pronunciations, pauses)."""
    kind = story.series.kind
    limit = ctx.cfg["quality"]["max_pauses_long" if kind == "long" else "max_pauses_short"]
    try:
        result = ctx.router.one("polisher").generate_json("annotate", prompts.SYSTEM_WRITER,
                                                          prompts.annotate(script, limit), schemas.ANNOTATE)
    except LLMError as exc:
        log.warning("Narration notes skipped: %s", exc)
        return [], []
    pauses = valid_pauses(result.get("pauses"), narration_of(script), limit)
    words = narration_of(script).lower()
    pronunciations = [p for p in result.get("pronunciations", []) if isinstance(p, dict)
                      and str(p.get("word", "")).strip() and str(p.get("word")).lower() in words]
    story.note("narration_notes", pauses=len(pauses), pronunciations=len(pronunciations))
    return pronunciations, pauses


def final_check(ctx: Context, story: Story, script: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Returns (must-fix issues, minor suggestions).

    Safety flags block on any one judge. Accuracy and cultural issues block only when
    most judges raise a must-fix issue of that kind: single judges contradict each
    other on fine detail, and one of them shouldn't veto what the rest find sound.
    """
    results = ctx.router.panel(
        lambda judge: judge.generate_json("final_check", prompts.SYSTEM_EDITOR,
                                          prompts.final_check(script, story.dossier), schemas.FINAL_CHECK),
        ctx.router.judges([ctx.router.one("polisher").spec.family]))
    blocking = sorted({f"rule violation: {f.replace('_', ' ')}" for r in results
                       for f, raised in r.get("flags", {}).items() if raised is True})
    minor: list[str] = []
    labels = (["accuracy"] if story.dossier else []) + ["cultural"]  # originals have no legend to get wrong
    for label in labels:
        severe_by_judge: list[list[str]] = []
        for r in results:
            severe: list[str] = []
            for item in r.get(f"{label}_issues", []):
                # A plain string (from a model that ignored the schema) counts as must-fix.
                text = str(item.get("issue", "") if isinstance(item, dict) else item).strip()
                if not _real_issue(text):
                    continue
                if not isinstance(item, dict) or item.get("severity") != "minor":
                    severe.append(f"{label}: {text}")
                else:
                    minor.append(f"{label}: {text}")
            severe_by_judge.append(severe)
        raised = [s for s in severe_by_judge if s]
        if len(raised) * 2 > len(results):  # a majority of judges
            blocking += [i for s in raised for i in s]
        else:
            minor += [i for s in raised for i in s]
    return list(dict.fromkeys(blocking)), list(dict.fromkeys(minor))


NOT_AN_ISSUE = re.compile(
    r"^\W*(none|no (issues?|problems?|errors?|concerns?)|n/?a|not applicable|nothing|clear|"
    r"(the )?story is (an )?original|no (specific )?legend)", re.IGNORECASE)


def _real_issue(text: Any) -> bool:
    """Judges sometimes write "None" or "story is original fiction" into an issue list."""
    text = str(text).strip()
    return bool(text) and not NOT_AN_ISSUE.match(text)


QUALIFIER = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]")


def snap_cast(scenes: Any, cast: Any, drop_unknown: bool = False) -> Any:
    """Point every scene's characters at cast names: "Delano (Memory)" or "Del" is the cast's
    "Delano Marchetti". With `drop_unknown`, names that match nobody are dropped (the scene's
    visual description still says what to draw)."""
    if not isinstance(scenes, list) or not isinstance(cast, list):
        return scenes
    names = [str(m.get("name", "")) for m in cast if isinstance(m, dict) and str(m.get("name", "")).strip()]
    exact = {dedup.normalize(n): n for n in names}
    firsts: dict[str, list[str]] = {}
    for n in names:
        firsts.setdefault(dedup.normalize(n).split()[0], []).append(n)

    def match(name: str) -> str | None:
        if dedup.normalize(name) in exact:
            return exact[dedup.normalize(name)]
        base = dedup.normalize(QUALIFIER.sub("", name))
        if base in exact:
            return exact[base]
        first = base.split()[0] if base else ""
        if first in ("the", "a", "an", "old", "young", "mr", "mrs", "ms", "dr"):
            return None  # "The Woman" is not "The Storyteller"
        candidates = firsts.get(first, [])
        if len(candidates) == 1:
            return candidates[0]
        nicknames = [n for n in names if first and len(first) >= 3
                     and dedup.normalize(n).split()[0].startswith(first)]
        return nicknames[0] if len(nicknames) == 1 else None

    for scene in scenes:
        if not isinstance(scene, dict) or not isinstance(scene.get("characters"), list):
            continue
        snapped = [match(str(c)) or (None if drop_unknown else str(c)) for c in scene["characters"]]
        scene["characters"] = list(dict.fromkeys(c for c in snapped if c))
    return scenes


def plan_scenes(ctx: Context, kind: str, narration: str,
                bible: dict | None = None) -> tuple[list | None, list[dict], list[str]]:
    """Returns (scenes, cast, problems). Recurring characters keep their canon look."""
    scene_range, key_range = scene_ranges(kind)
    planner = ctx.router.one("planner")
    canon = dict(db.known_looks(ctx.conn))
    base = prompts.scene_plan(narration, scene_range, key_range, list(canon.items()), bible)
    issues: list[str] = []
    for attempt in range(2):
        prompt = base if not issues else base + "\nFIX THESE PROBLEMS:\n" + "\n".join(issues)
        plan = planner.generate_json("scene_plan", prompts.SYSTEM_EDITOR, prompt, schemas.SCENES)
        cast = plan.get("cast") or []
        # Bookkeeping is repaired in code; on the last try, uncast extras are dropped.
        scenes = snap_cast(snap_scenes(narration, plan.get("scenes"), kind), cast, drop_unknown=attempt == 1)
        issues = scene_issues(narration, scenes, kind, cast)
        if not issues:
            by_norm = {dedup.normalize(name): look for name, look in canon.items()}
            for member in cast:  # enforce the canon look rather than trusting a rewrite
                member["look"] = by_norm.get(dedup.normalize(member["name"]), member["look"])
            return scenes, cast, []
    return None, [], issues


def hook_lab(ctx: Context, story: Story, script: dict[str, Any]) -> dict[str, Any]:
    """Try alternative openings; keep one only if judges prefer it to the original."""
    sentences = [x for x in re.split(r"(?<=[.!?…])\s+", script["body"].strip()) if x]
    if len(sentences) < 3:
        return script
    cut = 1
    while cut < min(3, len(sentences) - 1) and len(" ".join(sentences[:cut]).split()) < 25:
        cut += 1
    opening, rest = " ".join(sentences[:cut]), sentences[cut:]
    continuation = " ".join(rest[:2])
    reviser = ctx.router.one("reviser")
    result = reviser.generate_json("alternative_openings", prompts.SYSTEM_WRITER, prompts.alternative_openings(
        script, opening, ctx.cfg["quality"]["hook_alternatives"]), schemas.HOOKS)
    limit = 2 * len(opening.split()) + 10
    candidates = [opening] + list(dict.fromkeys(
        o.strip() for o in result.get("openings", []) if o.strip() and o.strip() != opening and len(o.split()) <= limit))
    if len(candidates) == 1:
        return script

    def compare(judge: Any, a: str, b: str) -> str:
        return judge.generate_json("compare_openings", prompts.SYSTEM_EDITOR,
                                   prompts.compare_openings(a, b, continuation), schemas.PAIRWISE)["winner"]

    quick = ctx.cfg["quality"]["quick_judges"]
    ranked = tournament.rank(candidates, lambda a, b: ctx.router.judges([reviser.spec.family], quick), compare,
                             ctx.router.parallel, prior=lambda o: 1.0 if o is opening else 0.0)
    best = ranked[0].item
    story.note("hook_lab", candidates=len(candidates), replaced=best is not opening)
    if best is opening:
        return script
    candidate = dict(script, body=" ".join([best, *rest]),
                     hook_line=re.split(r"(?<=[.!?…])\s+", best.strip())[0])
    return script if script_issues(candidate, story.series.kind, ctx.cfg) else candidate


def ending_lab(ctx: Context, story: Story, script: dict[str, Any]) -> dict[str, Any]:
    """Try alternative closing passages; keep one only if judges prefer it to the original.

    Viewers remember a story's worst moment and its last line most (the peak-end rule),
    and the ending decides whether they comment, share and come back.
    """
    q = ctx.cfg["quality"]
    sentences = [x for x in SENTENCE_END.split(script["body"].strip()) if x]
    if q["ending_alternatives"] < 1 or len(sentences) < 4:
        return script
    cut = 1
    while cut < min(4, len(sentences) - 2) and len(" ".join(sentences[-cut:]).split()) < 35:
        cut += 1
    ending, kept = " ".join(sentences[-cut:]), sentences[:-cut]
    before = " ".join(" ".join(kept).split()[-150:])
    reviser = ctx.router.one("reviser")
    try:
        result = reviser.generate_json("alternative_endings", prompts.SYSTEM_WRITER, prompts.alternative_endings(
            script, ending, q["ending_alternatives"]), schemas.ALT_ENDINGS)
    except LLMError as exc:
        log.warning("Ending lab skipped: %s", exc)
        return script
    # Writers often rewrite the whole closing scene, not only the last lines: a fuller
    # ending is fine (the length check still applies), a new story isn't.
    limit = max(3 * len(ending.split()) + 20, 180)
    offered = [e.strip() for e in result.get("endings", []) if isinstance(e, str) and e.strip()]
    candidates = [ending] + list(dict.fromkeys(e for e in offered if e != ending and len(e.split()) <= limit))
    if len(candidates) == 1:
        story.note("ending_lab", candidates=1, offered=len(offered), replaced=False,
                   offered_words=[len(e.split()) for e in offered], limit=limit)
        return script

    def compare(judge: Any, a: str, b: str) -> str:
        return judge.generate_json("compare_endings", prompts.SYSTEM_EDITOR,
                                   prompts.compare_endings(a, b, before), schemas.PAIRWISE)["winner"]

    ranked = tournament.rank(candidates, lambda a, b: ctx.router.judges([reviser.spec.family], q["quick_judges"]),
                             compare, ctx.router.parallel, prior=lambda e: 1.0 if e is ending else 0.0)
    best = ranked[0].item
    story.note("ending_lab", candidates=len(candidates), replaced=best is not ending)
    if best is ending:
        return script
    body = script["body"].rstrip()
    at = body.rfind(sentences[-cut])  # keep the paragraphs before the ending as they are
    candidate = dict(script, body=(body[:at] + best) if at >= 0 else " ".join([*kept, best]))
    return script if script_issues(candidate, story.series.kind, ctx.cfg) else candidate


def _opening_part(script: dict[str, Any], share: float) -> str:
    """What a viewer has heard by `share` of the story: the intro and the start of the body."""
    sentences = [x for x in SENTENCE_END.split(script["body"].strip()) if x]
    total = sum(len(x.split()) for x in sentences)
    heard, words = [], 0
    for sentence in sentences:
        if words >= total * share:
            break
        heard.append(sentence)
        words += len(sentence.split())
    intro = str(script.get("intro") or "").strip()
    return "\n\n".join(p for p in (intro, " ".join(heard)) if p)


def predictability(ctx: Context, story: Story, script: dict[str, Any], author_family: str) -> tuple[int, int, list[str]]:
    """Readers guess the ending from the first part; a judge grades the guesses.

    Returns (readers, correct guesses, guesses). A twist most readers see coming
    doesn't surprise; one nobody could have seen isn't fair (that's the clue check's job).
    """
    q = ctx.cfg["quality"]
    readers = (ctx.router.pool("audience") + ctx.router.judges([author_family]))[: q["surprise_readers"]]
    opening = _opening_part(script, q["surprise_cut"])
    prompt = prompts.predict_ending(story.series.kind, opening)

    def guess(reader: Any) -> str:
        try:
            return str(reader.generate_json("predict_ending", prompts.SYSTEM_EDITOR, prompt,
                                            schemas.PREDICTION).get("guess", "")).strip()
        except LLMError as exc:
            log.warning("A surprise-test reader failed: %s", exc)
            return ""

    guesses = [g for g in ctx.router.parallel(guess, readers) if g]
    if not guesses:
        return 0, 0, []
    try:
        grades = FailoverJudge(ctx.router.judges([author_family])).generate_json(
            "grade_guesses", prompts.SYSTEM_EDITOR, prompts.grade_guesses(script, guesses), schemas.GUESS_GRADES)
    except LLMError as exc:
        log.warning("Surprise test skipped: %s", exc)
        return 0, 0, []
    right = {int(g.get("index", 0)) for g in grades.get("items", [])
             if isinstance(g, dict) and g.get("guessed") is True}
    correct = [g for n, g in enumerate(guesses, 1) if n in right]
    return len(guesses), len(correct), correct


def surprise_lab(ctx: Context, story: Story, script: dict[str, Any], author_family: str) -> tuple[dict[str, Any], dict]:
    """If most readers guess the ending early, add misdirection, and keep the revision only if
    fewer readers see it coming and judges don't prefer the original."""
    if not ctx.cfg["quality"]["surprise_test"]:
        return script, {}
    readers, guessed, correct = predictability(ctx, story, script, author_family)
    report = {"readers": readers, "guessed": guessed}
    story.note("surprise_test", readers=readers, guessed=guessed)
    if not readers or guessed * 2 <= readers:
        return script, report
    candidate = repair(ctx, story, script, prompts.misdirection_notes(correct))
    if script_issues(candidate, story.series.kind, ctx.cfg):
        report["revision"] = "discarded: failed checks"
        return script, report
    reviser_family = ctx.router.one("reviser").spec.family
    readers2, guessed2, _ = predictability(ctx, story, candidate, reviser_family)
    report.update(readers_after=readers2, guessed_after=guessed2)
    if not readers2 or guessed2 * readers >= guessed * readers2:  # not less predictable
        report["revision"] = "discarded: no less predictable"
        return script, report

    def compare(judge: Any, a: dict, b: dict) -> str:
        return judge.generate_json("compare_drafts", prompts.SYSTEM_EDITOR,
                                   prompts.compare_drafts(story.series.kind, story.pitch, a, b), schemas.PAIRWISE)["winner"]

    ranked = tournament.rank([script, candidate], lambda a, b: ctx.router.judges([reviser_family],
                                                                                  ctx.cfg["quality"]["quick_judges"]),
                             compare, ctx.router.parallel)
    kept = ranked[0].item is candidate or ranked[0].points == ranked[1].points
    report["revision"] = "kept" if kept else "discarded: judges preferred the original"
    story.note("surprise_lab", **{k: v for k, v in report.items() if k not in ("readers", "guessed")})
    return (candidate if kept else script), report


def packaging_issues(result: dict[str, Any], cfg: dict) -> list[str]:
    q = cfg["quality"]
    issues = []
    brief = len(str(result.get("thumbnail_brief", "")).split())
    if brief > q["thumbnail_brief_max_words"]:
        issues.append(f"thumbnail_brief is {brief} words; describe one simple image in at most 35 words, "
                      "one or two figures and one red element")
    if len(str(result.get("thumbnail_text", "")).split()) > q["thumbnail_text_max_words"] - 1:
        issues.append("thumbnail_text must be at most three words")
    if not str(result.get("end_screen_line", "")).strip():
        issues.append("write the end_screen_line")
    return issues


def make_teaser(ctx: Context, story: Story, script: dict[str, Any]) -> dict[str, Any] | None:
    """A 60-second Short cut from a long episode, ending on a cliffhanger that sends viewers to it.

    Optional: a teaser that fails its checks is simply left out.
    """
    lo, hi = word_range(ctx.cfg, "short")
    reviser = ctx.router.one("reviser")
    for _ in range(2):
        try:
            result = reviser.generate_json("teaser_short", prompts.SYSTEM_WRITER,
                                           prompts.teaser_short(script, lo, hi), schemas.TEASER)
        except LLMError as exc:
            log.warning("Teaser Short skipped: %s", exc)
            return None
        teaser = tidy({"intro": "", "body": str(result.get("body", "")).strip(),
                       "outro": str(result.get("outro", "")).strip(), "comment_question": ""}, "short")
        if not structure_issues(teaser, "short", ctx.cfg) and not machine_phrasing(narration_of(teaser)):
            narration = narration_of(teaser)
            scenes, cast, _ = plan_scenes(ctx, "short", narration, story.bible)
            story.note("teaser_short", words=len(narration.split()), scenes=bool(scenes))
            return {"narration": narration, "body": teaser["body"], "outro": teaser["outro"],
                    "comment_question": teaser["comment_question"], "scenes": scenes, "cast": cast}
    story.note("teaser_short", made=False)
    return None


def package(ctx: Context, story: Story, script: dict[str, Any]) -> dict[str, Any]:
    writer = ctx.router.one("reviser")
    credit = script.get("description_credit") or "This is an original work of fiction."
    base = prompts.packaging(story.series, script, story.pitch, credit, story.bible)
    result = writer.generate_json("packaging", prompts.SYSTEM_WRITER, base, schemas.PACKAGING)
    problems = packaging_issues(result, ctx.cfg)
    if problems:
        retry = writer.generate_json("packaging", prompts.SYSTEM_WRITER,
                                     base + "\n\nFIX THESE PROBLEMS:\n" + "\n".join(problems), schemas.PACKAGING)
        if len(packaging_issues(retry, ctx.cfg)) < len(problems):
            result = retry
    q = ctx.cfg["quality"]
    result["thumbnail_text"] = " ".join(str(result.get("thumbnail_text", "")).split()[: q["thumbnail_text_max_words"]])
    titles = list(dict.fromkeys(t.strip() for t in result.get("titles", []) if t.strip() and len(t.strip()) <= 70))
    if len(titles) > 1:
        def compare(judge: Any, a: str, b: str) -> str:
            return judge.generate_json("compare_titles", prompts.SYSTEM_EDITOR, prompts.compare_titles(
                a, b, result.get("thumbnail_brief", ""), story.pitch["synopsis"]), schemas.PAIRWISE)["winner"]

        top = ctx.cfg["quality"]["title_finalists"]
        quick = ctx.cfg["quality"]["quick_judges"]
        ranked = tournament.rank(titles[:top], lambda a, b: ctx.router.judges([writer.spec.family], quick),
                                 compare, ctx.router.parallel)
        titles = [r.item for r in ranked] + titles[top:]
        story.note("title_contest", winner=titles[0])
    result["titles"] = titles or [story.pitch["title"]]
    return result


# --- Orchestration -----------------------------------------------------------

def _sources(story: Story) -> list[str]:
    texts = [row["summary"] or "" for row in ([story.legend] if story.legend else []) + story.sparks]
    if story.dossier:
        texts.append(json.dumps(story.dossier, ensure_ascii=False))
    return texts


def _reject(ctx: Context, story_id: int, series: str, reason: str, review: Any = None,
            story: Story | None = None) -> Outcome:
    db.update_story(ctx.conn, story_id, status="rejected", reject_reason=reason[:1000], review=review,
                    production_log=story.log if story else None)
    log.info("Story #%d rejected: %s", story_id, reason)
    return Outcome(series, "rejected", story_id, reason)


def _insert(ctx: Context, series: catalog.Series, pitch: dict[str, Any], legend: sqlite3.Row | None,
            sparks: list[sqlite3.Row], status: str) -> int:
    recipe = pitch["_recipe"]
    return db.insert_story(
        ctx.conn, kind=series.kind, series=series.name, status=status,
        title=pitch["title"].strip(), title_norm=dedup.normalize(pitch["title"]),
        logline=pitch["logline"].strip(), synopsis=pitch["synopsis"].strip(),
        setting=recipe.setting, core_fear=recipe.core_fear, protagonist=recipe.protagonist,
        format=recipe.format, twist=recipe.twist, ending=recipe.ending, recipe_sig=recipe.signature,
        legend_key=f"{legend['source']}:{legend['source_id']}" if legend else None,
        inspiration_ids=[row["id"] for row in ([legend] if legend else []) + sparks],
        synopsis_embedding=json.dumps(pitch["_embedding"]),
        pitch={k: v for k, v in pitch.items() if not k.startswith("_")},
        pitch_score=pitch.get("_score"), author_models={"pitch": pitch["_author"]},
    )


def _from_shelf(ctx: Context, row: sqlite3.Row) -> Story:
    series = catalog.SERIES[row["series"]]
    ids = json.loads(row["inspiration_ids"] or "[]")
    inspirations = [ctx.conn.execute("SELECT * FROM inspirations WHERE id = ?", (i,)).fetchone() for i in ids]
    legend = next((r for r in inspirations if r and row["legend_key"] == f"{r['source']}:{r['source_id']}"), None)
    recipe = catalog.Recipe(row["setting"], row["core_fear"], row["protagonist"], row["format"],
                            row["twist"], row["ending"])
    pitch = json.loads(row["pitch"]) if row["pitch"] else {
        "title": row["title"], "logline": row["logline"], "synopsis": row["synopsis"]}
    return Story(series=series, recipe=recipe, pitch=pitch, legend=legend,
                 dossier=ensure_dossier(ctx, legend) if legend else None,
                 sparks=[r for r in inspirations if r and r is not legend],
                 world=db.world_entities(ctx.conn) if series.uses_world else [],
                 authors=json.loads(row["author_models"] or "{}"))


def generate_story(ctx: Context, series_name: str) -> Outcome:
    series = catalog.SERIES[series_name]
    shelved = db.best_shelved(ctx.conn, series_name, ctx.cfg["quality"]["retry_parked"])
    if shelved is not None:
        story = _from_shelf(ctx, shelved)
        story.note("pitch", source="shelf", title=story.pitch["title"])
        db.update_story(ctx.conn, shelved["id"], status="idea")
        return _guarded_develop(ctx, shelved["id"], story)

    for attempt in range(1, ctx.cfg["quality"]["max_attempts_per_story"] + 1):
        legend = dossier = None
        if series.uses_legend:
            chosen = choose_legend(ctx, series.kind)
            if chosen is None:
                return Outcome(series_name, "skipped", detail="no suitable unused legend; run `research` first")
            legend, dossier = chosen
        sparks = db.folklore_sparks(ctx.conn, 2) if series.uses_world else []
        world = db.world_entities(ctx.conn) if series.uses_world else []

        pitches = gather_pitches(ctx, series, legend, dossier, sparks, world)
        if not pitches:
            log.info("Attempt %d for %s produced no original pitch", attempt, series_name)
            continue
        score_pitches(ctx, series, pitches, legend, dossier)
        ranked = pitch_contest(ctx, series, pitches)
        winner = ranked[0].item
        story = Story(series=series, recipe=winner["_recipe"], pitch=winner, legend=legend, dossier=dossier,
                      sparks=sparks, world=world, authors={"pitch": winner["_author"]})
        story.note("pitch", candidates=len(pitches), winner=winner["title"], score=winner["_score"],
                   points=[r.points for r in ranked])
        story_id = _insert(ctx, series, winner, legend, sparks, status="idea")
        if not series.uses_legend:  # a legend is retold once, so its runners-up can't wait
            taken = {winner["_recipe"].signature}
            for runner_up in ranked[1:]:
                pitch = runner_up.item
                # Pitches from one pitcher share a recipe, and a recipe is only ever used once.
                if pitch["_score"] >= ctx.cfg["quality"]["shelve_min_score"] and pitch["_recipe"].signature not in taken:
                    _insert(ctx, series, pitch, None, sparks, status="shelved")
                    taken.add(pitch["_recipe"].signature)
        story.pitch = {k: v for k, v in winner.items() if not k.startswith("_")}
        return _guarded_develop(ctx, story_id, story)
    return Outcome(series_name, "skipped", detail="no original pitch found this run")


def _guarded_develop(ctx: Context, story_id: int, story: Story) -> Outcome:
    ctx.router.recorder.story_id = story_id  # attribute every call's cost to this story
    try:
        return develop(ctx, story_id, story)
    except (LLMError, BudgetExceeded) as exc:
        # Don't leave a half-made story stuck in the queue; its recipe is freed.
        _reject(ctx, story_id, story.series.name, f"interrupted: {exc}", story=story)
        raise
    finally:
        ctx.router.recorder.story_id = None


def develop(ctx: Context, story_id: int, story: Story) -> Outcome:
    series, kind = story.series, story.series.kind
    story.bible, outline = make_plan(ctx, story)
    db.update_story(ctx.conn, story_id, bible=story.bible, beat_sheet=outline)

    script, author_family = draft_contest(ctx, story, outline)
    edit = editors_loop(ctx, story, script, author_family, story_id, outline)
    if edit.outline is not outline or story.bible is not None:
        db.update_story(ctx.conn, story_id, bible=story.bible, beat_sheet=edit.outline)
    if edit.script is None:
        return _fail_editing(ctx, story_id, story, edit)
    script = edit.script
    review = edit.review
    author_family = edit.best.author_family if edit.best else author_family

    script = hook_lab(ctx, story, script)
    script = ending_lab(ctx, story, script)
    script, surprise = surprise_lab(ctx, story, script, author_family)
    script, audience_report = audience_test(ctx, story, script)
    if surprise:
        audience_report["surprise_test"] = surprise
    script = polish(ctx, story, script)

    problems, minor = final_check(ctx, story, script)
    for _ in range(ctx.cfg["quality"]["final_check_repairs"]):
        if not problems:
            break
        story.note("final_check", problems=problems[:4], minor=len(minor))
        fixed = repair(ctx, story, script, problems + minor)
        problems = script_issues(fixed, kind, ctx.cfg)
        if not problems:
            script = fixed
            problems, minor = final_check(ctx, story, script)
    if problems:
        return _reject(ctx, story_id, series.name, "final check: " + "; ".join(problems[:4]), review, story)
    story.note("final_check", clear=True, minor_left=minor[:4])

    narration = narration_of(script)
    copy = dedup.check_script(ctx.conn, ctx.cfg, story_id, narration, _sources(story))
    if not copy.ok:
        return _reject(ctx, story_id, series.name, copy.reason, review, story)
    pronunciations, pauses = annotate(ctx, story, script)

    db.update_story(
        ctx.conn, story_id, status="scripted",
        intro=str(script.get("intro", "")).strip(), body=str(script["body"]).strip(),
        outro=str(script["outro"]).strip(), narration=narration, word_count=len(narration.split()),
        comment_question=str(script["comment_question"]).strip(),
        hook_line=str(script.get("hook_line", "")).strip(),
        thumbnail_idea=str(script.get("thumbnail_idea", "")).strip(),
        description_credit=str(script.get("description_credit", "")).strip(),
        review=review, audience_report=audience_report, pronunciations=pronunciations, pauses=pauses)

    scenes, cast, issues = plan_scenes(ctx, kind, narration, story.bible)
    if scenes is None:
        return _reject(ctx, story_id, series.name, "scene plan invalid: " + "; ".join(issues[:4]), review, story)
    packaging = package(ctx, story, script)
    if kind == "long" and ctx.cfg["quality"]["teaser_shorts"]:
        packaging["teaser_short"] = make_teaser(ctx, story, script)

    if series.uses_world:
        db.link_entities(ctx.conn, story_id, script.get("new_entities") or [],
                         [str(n) for n in script.get("referenced_entities") or []])
        db.fix_looks(ctx.conn, cast)
    db.mark_inspirations_used(ctx.conn, [row["id"] for row in ([story.legend] if story.legend else []) + story.sparks])
    db.update_story(ctx.conn, story_id, status="approved", scenes=scenes, cast=cast, packaging=packaging,
                    author_models=story.authors, production_log=story.log)
    log.info("Story #%d approved: %s", story_id, story.pitch["title"])
    return Outcome(series.name, "approved", story_id, story.pitch["title"])


def _fail_editing(ctx: Context, story_id: int, story: Story, edit: Edit) -> Outcome:
    """A story the editors never approved. Its best draft and the diagnosis are kept.

    If the premise wasn't the problem and it pitched well, the story is parked for a
    stronger model (see `retry_parked`) instead of being thrown away. A legend is never
    parked: its being stays free for a fresh retelling.
    """
    best = edit.best
    best_draft = {k: best.script.get(k) for k in SCRIPT_FIELDS} if best else None
    row = db.get_story(ctx.conn, story_id)
    pitch_score = (row["pitch_score"] if row is not None else None) or 0
    park = (story.legend is None and edit.diagnosis != "kill"
            and pitch_score >= ctx.cfg["quality"]["shelve_min_score"])
    status = "parked" if park else "rejected"
    db.update_story(ctx.conn, story_id, status=status, reject_reason=edit.reason[:1000], review=edit.review,
                    production_log=story.log, best_draft=best_draft,
                    diagnosis=edit.diagnosis or "revisions ran out")
    log.info("Story #%d %s: %s", story_id, status, edit.reason)
    return Outcome(story.series.name, "rejected", story_id, edit.reason)


def recheck_legends(ctx: Context) -> list[dict[str, Any]]:
    """Run the legend recognition test on approved retellings that were never given it.

    One approved before the test existed may not really depict its legend; if most
    judges don't recognize the being, the story goes back to `rejected` (its script is
    kept in the row) and the legend is free to be retold.
    """
    results = []
    rows = ctx.conn.execute("SELECT * FROM stories WHERE status = 'approved' AND legend_key IS NOT NULL").fetchall()
    for row in rows:
        review = json.loads(row["review"]) if row["review"] else {}
        if "recognition" in review:
            continue
        story = _from_shelf(ctx, row)
        if story.legend is None:
            continue
        script = {"intro": row["intro"] or "", "body": row["body"] or "", "outro": row["outro"] or ""}
        authors = json.loads(row["author_models"] or "{}")
        family = ModelSpec.parse(authors["draft"]).family if authors.get("draft") else ""
        issues = recognition_issues(ctx, story, script, family)
        review["recognition"] = {"recognized": not issues, "issues": issues}
        if issues:
            db.update_story(ctx.conn, row["id"], status="rejected", review=review,
                            reject_reason=("recheck: " + issues[0])[:1000])
        else:
            db.update_story(ctx.conn, row["id"], review=review)
        results.append({"id": row["id"], "title": row["title"], "recognized": not issues,
                        "issues": issues, "log": story.log})
    return results


def fill_backlog(ctx: Context, max_new: int = 10) -> list[Outcome]:
    """Generate stories for the series furthest below its backlog target."""
    outcomes: list[Outcome] = []
    blocked: set[str] = set()
    for _ in range(max_new * 3):
        if sum(1 for o in outcomes if o.status == "approved") >= max_new:
            break
        counts = db.backlog_counts(ctx.conn)
        deficits = {name: target - counts.get(name, 0)
                    for name, target in ctx.cfg["backlog"].items()
                    if name in catalog.SERIES and name not in blocked}
        name, gap = max(deficits.items(), key=lambda kv: kv[1], default=(None, 0))
        if name is None or gap <= 0:
            break
        try:
            outcome = generate_story(ctx, name)
        except BudgetExceeded as exc:
            log.warning("Stopping: %s", exc)
            outcomes.append(Outcome(name, "skipped", detail=str(exc)))
            break
        except LLMError as exc:
            log.warning("Model error while generating %s: %s", name, exc)
            outcome = Outcome(name, "skipped", detail=str(exc))
        outcomes.append(outcome)
        if outcome.status == "skipped":
            blocked.add(name)
    return outcomes

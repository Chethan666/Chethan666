"""The writers' room: how one story is made.

    research legend (web) -> pitch contest -> story bible -> beat outline
    -> competing drafts (long ones written part by part) -> editors' loop (free style
    checks, story-bible check, scored review) -> hook lab -> simulated audience
    -> read-aloud polish with pauses -> final checks -> scene plan with cast
    -> packaging with title contest -> approved

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
from .llm import BudgetExceeded, LLMError
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
RITUAL_AT_START = re.compile(r"^\s*the lantern is lit[.!,]?\s*", re.IGNORECASE)
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
            script[key] = RITUAL_AT_START.sub("", str(script.get(key) or "")).strip()
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
    return {"scores": scores, "flags": flags, "notes": notes}


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


def make_bible(ctx: Context, story: Story) -> dict[str, Any]:
    """The storyteller's decisions before plotting: hero, threat, theme, devices, clues."""
    kind = story.series.kind
    writer = ctx.router.one("writers")
    bible = writer.generate_json("story_bible", prompts.SYSTEM_WRITER, prompts.story_bible(
        series=story.series, kind=kind, recipe=story.recipe, formats=FORMATS, endings=ENDINGS,
        pitch=story.pitch, legend=story.legend, dossier=story.dossier, world=story.world,
        recent_heroes=db.recent_heroes(ctx.conn, ctx.cfg["quality"]["recent_heroes"])), schemas.BIBLE)
    problems = bible_issues(bible, kind)
    if not problems:
        reviews = ctx.router.panel(
            lambda judge: judge.generate_json("review_bible", prompts.SYSTEM_EDITOR,
                                              prompts.review_bible(kind, story.pitch, bible, story.dossier),
                                              schemas.OUTLINE_REVIEW),
            ctx.router.judges([writer.spec.family], ctx.cfg["quality"]["quick_judges"]))
        problems = list(dict.fromkeys(p for r in reviews if not r.get("approved") for p in r.get("problems", [])))
    story.note("story_bible", problems=len(problems))
    if problems:
        revised = writer.generate_json("revise_bible", prompts.SYSTEM_WRITER,
                                       prompts.revise_bible(bible, problems), schemas.BIBLE)
        # Keep the revision unless it lost decisions the first version had.
        if len(bible_issues(revised, kind)) <= len(bible_issues(bible, kind)):
            bible = revised
    story.authors["bible"] = writer.spec.label
    return bible


def make_outline(ctx: Context, story: Story) -> dict[str, Any]:
    kind = story.series.kind
    writer = ctx.router.one("writers")
    target = 60 if kind == "short" else round(ctx.cfg["lengths"]["long_min_words"] / 2.3)
    outline = writer.generate_json("beat_sheet", prompts.SYSTEM_WRITER, prompts.beat_sheet(
        series=story.series, kind=kind, recipe=story.recipe, formats=FORMATS, endings=ENDINGS,
        pitch=story.pitch, legend=story.legend, dossier=story.dossier, world=story.world,
        target_seconds=target, bible=story.bible), schemas.BEAT_SHEET)
    for attempt in range(2):
        problems = outline_issues(outline, kind)
        if not problems:  # structure first; editors only read outlines that have it
            reviews = ctx.router.panel(
                lambda judge: judge.generate_json("review_outline", prompts.SYSTEM_EDITOR,
                                                  prompts.review_outline(kind, story.pitch, outline, story.bible),
                                                  schemas.OUTLINE_REVIEW),
                ctx.router.judges([writer.spec.family], ctx.cfg["quality"]["judges_per_comparison"]))
            problems = list(dict.fromkeys(p for r in reviews if not r.get("approved") for p in r.get("problems", [])))
        story.note("outline_review", round=attempt + 1, problems=len(problems))
        if not problems:
            break
        outline = writer.generate_json("revise_outline", prompts.SYSTEM_WRITER,
                                       prompts.revise_outline(outline, problems), schemas.BEAT_SHEET)
    if not outline.get("beats"):
        raise LLMError("outline has no beats")
    story.authors["outline"] = writer.spec.label
    return outline


def write_in_parts(ctx: Context, story: Story, writer: Any, outline: dict[str, Any]) -> dict[str, Any]:
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
            legend=story.legend, dossier=story.dossier, world=story.world), schemas.SECTION)
        text = str(part.get("text", "")).strip()
        if not text:
            raise LLMError(f"{writer.spec.label} wrote an empty part {i + 1}")
        parts.append(text)
    body = "\n\n".join(parts)
    frame = writer.generate_json("write_frame", prompts.SYSTEM_WRITER, prompts.write_frame(
        series=story.series, kind="long", pitch=story.pitch, bible=story.bible, body=body,
        legend=story.legend, world=story.world), schemas.FRAME)
    return tidy(dict(frame, body=body), "long")


def draft_contest(ctx: Context, story: Story, outline: dict[str, Any]) -> tuple[dict[str, Any], str]:
    kind = story.series.kind
    lo, hi = word_range(ctx.cfg, kind)

    in_parts = kind == "long" and ctx.cfg["quality"]["draft_sections"] > 1 and len(outline["beats"]) > 1

    def write(writer: Any) -> dict[str, Any] | None:
        try:
            if in_parts:
                script = write_in_parts(ctx, story, writer, outline)
            else:
                script = writer.generate_json("write_draft", prompts.SYSTEM_WRITER, prompts.write_draft(
                    series=story.series, kind=kind, recipe=story.recipe, formats=FORMATS, endings=ENDINGS,
                    pitch=story.pitch, outline=outline, min_words=lo, max_words=hi, legend=story.legend,
                    dossier=story.dossier, world=story.world, bible=story.bible), schemas.SCRIPT)
        except LLMError as exc:
            log.warning("Writer %s failed: %s", writer.spec.label, exc)
            return None
        return dict(script, _author=writer.spec.label, _family=writer.spec.family)

    drafts = [d for d in ctx.router.parallel(write, ctx.router.pool("writers")) if d]
    if not drafts:
        raise LLMError("every writer failed")

    def compare(judge: Any, a: dict, b: dict) -> str:
        return judge.generate_json("compare_drafts", prompts.SYSTEM_EDITOR,
                                   prompts.compare_drafts(kind, story.pitch, a, b), schemas.PAIRWISE)["winner"]

    limit = ctx.cfg["quality"]["judges_per_comparison"]
    ranked = tournament.rank(drafts, lambda a, b: ctx.router.judges([a["_family"], b["_family"]], limit), compare,
                             ctx.router.parallel, prior=lambda d: -len(script_issues(d, kind, ctx.cfg)))
    best = ranked[0].item
    story.note("draft_contest", drafts=len(drafts), winner=best["_author"],
               points=[r.points for r in ranked])
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


def check_bible(ctx: Context, story: Story, script: dict[str, Any],
                author_family: str) -> tuple[list[str], list[str]]:
    """Does the script deliver the story bible? Returns (blocking issues, advisory notes).

    A promise only counts when the judge's quote is really in the script. Missing
    promises block when coverage falls below the bar; broken threat rules always do.
    """
    if not story.bible:
        return [], []
    promises = bible_promises(story.bible, story.series.kind, story.dossier)
    if not promises:
        return [], []
    judge = FailoverJudge(ctx.router.judges([author_family]))
    try:
        result = judge.generate_json("bible_check", prompts.SYSTEM_EDITOR,
                                     prompts.bible_check(story.bible, promises, script, story.dossier),
                                     schemas.BIBLE_CHECK)
    except LLMError as exc:
        log.warning("Story-bible check skipped: %s", exc)
        return [], []
    narration = narration_of(script)
    shown = {str(r.get("id")) for r in result.get("items", [])
             if isinstance(r, dict) and r.get("present") is True and quote_found(str(r.get("quote", "")), narration)}
    missing = [f"deliver the story bible: {requirement}" for item_id, requirement in promises if item_id not in shown]
    breaks = [f"the threat breaks its own rules: {b}" for b in result.get("threat_rule_breaks", []) if _real_issue(b)]
    coverage = 1 - len(missing) / len(promises)
    story.note("bible_check", coverage=round(coverage, 2), missing=len(missing), rule_breaks=len(breaks))
    if coverage < ctx.cfg["quality"]["bible_min_coverage"]:
        return breaks + missing, []
    return breaks, missing


def editors_loop(ctx: Context, story: Story, script: dict[str, Any], author_family: str,
                 story_id: int) -> tuple[dict[str, Any] | None, dict[str, Any], str]:
    """Review and revise until every editor median clears the bar. Returns (script|None, review, reason)."""
    kind, cfg = story.series.kind, ctx.cfg
    lo, hi = word_range(cfg, kind)
    reviser = ctx.router.one("reviser")
    sources = _sources(story)
    agg: dict[str, Any] = {}
    for round_ in range(cfg["quality"]["max_revisions"] + 1):
        script = fit_length(ctx, story, script)
        script = fit_length(ctx, story, clean_style(ctx, story, script))
        issues = script_issues(script, kind, cfg)
        copy = dedup.check_script(ctx.conn, cfg, story_id, narration_of(script), sources)
        if not copy.ok:
            issues.append(f"rewrite in original words: {copy.reason}")
        advice = [i for i in style_issues(narration_of(script), cfg, kind) if i not in issues]
        if not issues:
            issues, missing = check_bible(ctx, story, script, author_family)
            advice += missing
        if not issues:
            reviews = ctx.router.panel(
                lambda judge: judge.generate_json("review", prompts.SYSTEM_EDITOR, prompts.review(
                    kind, story.recipe, FORMATS, ENDINGS, script, story.dossier, story.bible), schemas.REVIEW),
                ctx.router.judges([author_family], cfg["quality"]["judges_per_comparison"]))
            agg = aggregate_reviews(reviews)
            issues = review_issues(agg, cfg["quality"]["min_score"])
            if issues:
                issues += advice  # style advice and unmet story-bible promises help the revision
        story.note("editors", round=round_ + 1, issues=len(issues), scores=agg.get("scores"))
        if not issues:
            return script, agg, ""
        if round_ == cfg["quality"]["max_revisions"]:
            return None, agg, "; ".join(issues[:6])
        script = repair(ctx, story, script, issues)
        author_family = reviser.spec.family
        story.authors["reviser"] = reviser.spec.label
    raise AssertionError("unreachable")


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
    if script_issues(candidate, kind, ctx.cfg) or candidate["body"] == script["body"]:
        report["rewrite"] = "discarded: failed checks or no change"
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


def polish(ctx: Context, story: Story, script: dict[str, Any]) -> tuple[dict[str, Any], list[dict], list[dict]]:
    """Read-aloud line edit. Returns (script, pronunciations, pauses)."""
    kind = story.series.kind
    lo, hi = word_range(ctx.cfg, kind)
    limit = ctx.cfg["quality"]["max_pauses_long" if kind == "long" else "max_pauses_short"]
    polisher = ctx.router.one("polisher")
    result = polisher.generate_json("polish", prompts.SYSTEM_WRITER, prompts.polish(kind, script, lo, hi, limit),
                                    schemas.POLISH)
    candidate = dict(script, intro=result.get("intro", ""), body=result.get("body", ""),
                     outro=result.get("outro", ""))
    unchanged_meaning = dedup.lexical_similarity(narration_of(script), narration_of(candidate), size=1) >= 0.6
    if script_issues(candidate, kind, ctx.cfg) or not unchanged_meaning:
        story.note("polish", kept=False)
        return script, [], []
    pauses = valid_pauses(result.get("pauses"), narration_of(candidate), limit)
    story.note("polish", kept=True, pauses=len(pauses))
    story.authors["polisher"] = polisher.spec.label
    return candidate, [p for p in result.get("pronunciations", []) if p.get("word")], pauses


def final_check(ctx: Context, story: Story, script: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Returns (must-fix issues, minor suggestions). Rule violations always must be fixed."""
    results = ctx.router.panel(
        lambda judge: judge.generate_json("final_check", prompts.SYSTEM_EDITOR,
                                          prompts.final_check(script, story.dossier), schemas.FINAL_CHECK),
        ctx.router.judges([ctx.router.one("polisher").spec.family]))
    blocking = sorted({f"rule violation: {f.replace('_', ' ')}" for r in results
                       for f, raised in r.get("flags", {}).items() if raised is True})
    minor: list[str] = []
    for r in results:
        groups = [("cultural", r.get("cultural_issues", []))]
        if story.dossier:  # original stories have no legend to be inaccurate about
            groups.insert(0, ("accuracy", r.get("accuracy_issues", [])))
        for label, items in groups:
            for item in items:
                # A plain string (from a model that ignored the schema) counts as must-fix.
                text = item.get("issue", "") if isinstance(item, dict) else item
                if not _real_issue(text):
                    continue
                severe = not isinstance(item, dict) or item.get("severity") != "minor"
                (blocking if severe else minor).append(f"{label}: {str(text).strip()}")
    return list(dict.fromkeys(blocking)), list(dict.fromkeys(minor))


NOT_AN_ISSUE = re.compile(
    r"^\W*(none|no (issues?|problems?|errors?|concerns?)|n/?a|not applicable|nothing|clear|"
    r"(the )?story is (an )?original|no (specific )?legend)", re.IGNORECASE)


def _real_issue(text: Any) -> bool:
    """Judges sometimes write "None" or "story is original fiction" into an issue list."""
    text = str(text).strip()
    return bool(text) and not NOT_AN_ISSUE.match(text)


def plan_scenes(ctx: Context, kind: str, narration: str,
                bible: dict | None = None) -> tuple[list | None, list[dict], list[str]]:
    """Returns (scenes, cast, problems). Recurring characters keep their canon look."""
    scene_range, key_range = scene_ranges(kind)
    planner = ctx.router.one("planner")
    canon = dict(db.known_looks(ctx.conn))
    base = prompts.scene_plan(narration, scene_range, key_range, list(canon.items()), bible)
    issues: list[str] = []
    for _ in range(2):
        prompt = base if not issues else base + "\nFIX THESE PROBLEMS:\n" + "\n".join(issues)
        plan = planner.generate_json("scene_plan", prompts.SYSTEM_EDITOR, prompt, schemas.SCENES)
        scenes, cast = snap_scenes(narration, plan.get("scenes"), kind), plan.get("cast") or []
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
    shelved = db.best_shelved(ctx.conn, series_name)
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
    story.bible = make_bible(ctx, story)
    db.update_story(ctx.conn, story_id, bible=story.bible)
    outline = make_outline(ctx, story)
    db.update_story(ctx.conn, story_id, beat_sheet=outline)

    script, author_family = draft_contest(ctx, story, outline)
    script, review, reason = editors_loop(ctx, story, script, author_family, story_id)
    if script is None:
        return _reject(ctx, story_id, series.name, reason, review, story)

    script = hook_lab(ctx, story, script)
    script, audience_report = audience_test(ctx, story, script)
    script, pronunciations, pauses = polish(ctx, story, script)

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

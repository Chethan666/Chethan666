"""Story generation: recipe -> premises -> duplicate gate -> script -> review -> scenes."""

from __future__ import annotations

import json
import logging
import random
import re
import sqlite3
from dataclasses import dataclass
from typing import Any

from . import catalog, db, dedup, prompts
from .llm import LLM, BudgetExceeded, LLMError

log = logging.getLogger(__name__)

CLAIMS_TRUE = re.compile(
    r"\b(true story|really happened|based on (a )?true|actual events|this happened to me)\b", re.IGNORECASE
)
SCRIPT_FIELDS = ("intro", "body", "outro", "comment_question", "hook_line", "thumbnail_idea",
                 "description_credit", "new_entities", "referenced_entities")


@dataclass
class Context:
    conn: sqlite3.Connection
    cfg: dict
    llm: LLM
    rng: random.Random


@dataclass
class Outcome:
    series: str
    status: str  # approved | rejected | skipped
    story_id: int | None = None
    detail: str = ""


def narration_of(script: dict[str, Any]) -> str:
    return "\n\n".join(p.strip() for p in (script.get("intro"), script.get("body"), script.get("outro"))
                       if isinstance(p, str) and p.strip())


def word_range(cfg: dict, kind: str) -> tuple[int, int]:
    lengths = cfg["lengths"]
    return lengths[f"{kind}_min_words"], lengths[f"{kind}_max_words"]


def scene_ranges(kind: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """(number of scenes, number of key moments) for a story kind."""
    return ((18, 45), (3, 6)) if kind == "long" else ((4, 10), (1, 2))


def _valid_premise(p: Any) -> bool:
    return isinstance(p, dict) and all(isinstance(p.get(k), str) and p[k].strip()
                                       for k in ("title", "logline", "synopsis"))


def script_issues(script: dict[str, Any], kind: str, cfg: dict) -> list[str]:
    """Checks that need no model call."""
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
    if CLAIMS_TRUE.search(narration_of(script)):
        issues.append("never claim the story is true or based on real events")
    return issues


def review_issues(review: dict[str, Any], min_score: int) -> list[str]:
    issues: list[str] = []
    for name, value in (review.get("scores") or {}).items():
        if not isinstance(value, (int, float)) or value < min_score:
            issues.append(f"improve {name.replace('_', ' ')} (scored {value}/10)")
    if not review.get("scores"):
        issues.append("review returned no scores")
    for flag, raised in (review.get("flags") or {}).items():
        if raised is True:
            issues.append(f"remove hard-rule violation: {flag.replace('_', ' ')}")
    if issues:
        issues.extend(str(f) for f in review.get("fix_instructions") or [])
    return issues


def scene_issues(narration: str, scenes: Any, kind: str) -> list[str]:
    (lo, hi), (key_lo, key_hi) = scene_ranges(kind)
    if not isinstance(scenes, list) or not scenes:
        return ["no scenes returned"]
    issues: list[str] = []
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


def _reject(ctx: Context, story_id: int, series: str, reason: str, review: Any = None) -> Outcome:
    db.update_story(ctx.conn, story_id, status="rejected", reject_reason=reason[:1000], review=review)
    log.info("Story #%d rejected: %s", story_id, reason)
    return Outcome(series, "rejected", story_id, reason)


def generate_story(ctx: Context, series_name: str) -> Outcome:
    series = catalog.SERIES[series_name]
    quality, rules = ctx.cfg["quality"], ctx.cfg["dedup"]
    for attempt in range(1, quality["max_attempts_per_story"] + 1):
        recipe = catalog.pick_recipe(ctx.rng, series, db.recent_recipes(ctx.conn),
                                     db.used_signatures(ctx.conn), rules)
        legend = None
        if series.uses_legend:
            legends = db.unused_legends(ctx.conn)
            if not legends:
                return Outcome(series_name, "skipped", detail="no unused legends; run `research` first")
            legend = ctx.rng.choice(legends)
        sparks = db.folklore_sparks(ctx.conn, 2) if series.uses_world else []
        world = db.world_entities(ctx.conn) if series.uses_world else []
        avoid = db.all_stories(ctx.conn)[: rules["recent_titles_in_prompt"]]
        trends = db.top_trends(ctx.conn, ctx.cfg["research"]["trends_in_prompt"])

        pitched = ctx.llm.generate_json("premises", prompts.premises(
            series=series, recipe=recipe, formats=catalog.FORMATS, endings=catalog.ENDINGS,
            count=quality["candidates_per_call"], legend=legend, sparks=sparks, world=world,
            avoid=avoid, trends=trends))
        for premise in filter(_valid_premise, pitched.get("premises") or []):
            verdict = dedup.check_premise(ctx.conn, ctx.llm, ctx.cfg, premise, recipe.signature)
            if not verdict.ok:
                log.info("Premise '%s' rejected: %s", premise["title"], verdict.reason)
                continue
            story_id = db.insert_story(
                ctx.conn, kind=series.kind, series=series.name, status="idea",
                title=premise["title"].strip(), title_norm=dedup.normalize(premise["title"]),
                logline=premise["logline"].strip(), synopsis=premise["synopsis"].strip(),
                setting=recipe.setting, core_fear=recipe.core_fear, protagonist=recipe.protagonist,
                format=recipe.format, twist=recipe.twist, ending=recipe.ending,
                recipe_sig=recipe.signature,
                legend_key=f"{legend['source']}:{legend['source_id']}" if legend else None,
                inspiration_ids=[row["id"] for row in ([legend] if legend else []) + list(sparks)],
                synopsis_embedding=json.dumps(verdict.embedding),
            )
            try:
                return _develop(ctx, story_id, series, recipe, premise, legend, sparks, world)
            except (LLMError, BudgetExceeded) as exc:
                # Don't leave a half-made story stuck in the queue; its recipe is freed.
                _reject(ctx, story_id, series_name, f"interrupted: {exc}")
                raise
        log.info("Attempt %d for %s produced no original premise", attempt, series_name)
    return Outcome(series_name, "skipped", detail="no original premise found this run")


def _develop(ctx: Context, story_id: int, series: catalog.Series, recipe: catalog.Recipe,
             premise: dict[str, Any], legend: sqlite3.Row | None, sparks: list[sqlite3.Row],
             world: list[sqlite3.Row]) -> Outcome:
    kind, cfg = series.kind, ctx.cfg
    lo, hi = word_range(cfg, kind)
    sources = [row["summary"] or "" for row in ([legend] if legend else []) + list(sparks)]

    script = ctx.llm.generate_json("write_script", prompts.write_script(
        series=series, kind=kind, recipe=recipe, formats=catalog.FORMATS, endings=catalog.ENDINGS,
        premise=premise, min_words=lo, max_words=hi, legend=legend, world=world))
    review: dict[str, Any] | None = None
    max_revisions = cfg["quality"]["max_revisions"]
    for round_ in range(max_revisions + 1):
        issues = script_issues(script, kind, cfg)
        copy_check = dedup.check_script(ctx.conn, cfg, story_id, narration_of(script), sources)
        if not copy_check.ok:
            issues.append(f"rewrite in original words: {copy_check.reason}")
        if not issues:
            review = ctx.llm.generate_json("review", prompts.review(
                kind=kind, recipe=recipe, formats=catalog.FORMATS, endings=catalog.ENDINGS,
                script=script), temperature=0.2)
            issues = review_issues(review, cfg["quality"]["min_score"])
        if not issues:
            break
        if round_ == max_revisions:
            return _reject(ctx, story_id, series.name, "; ".join(issues[:6]), review)
        script = ctx.llm.generate_json("revise", prompts.revise(
            kind=kind, script={k: script.get(k) for k in SCRIPT_FIELDS}, feedback=issues,
            min_words=lo, max_words=hi))

    narration = narration_of(script)
    db.update_story(
        ctx.conn, story_id, status="scripted",
        intro=str(script.get("intro", "")).strip(), body=str(script["body"]).strip(),
        outro=str(script["outro"]).strip(), narration=narration, word_count=len(narration.split()),
        comment_question=str(script["comment_question"]).strip(),
        hook_line=str(script.get("hook_line", "")).strip(),
        thumbnail_idea=str(script.get("thumbnail_idea", "")).strip(),
        description_credit=str(script.get("description_credit", "")).strip(), review=review)

    scene_range, key_range = scene_ranges(kind)
    scene_prompt = prompts.scene_plan(kind=kind, narration=narration, scene_range=scene_range,
                                      key_range=key_range)
    issues: list[str] = []
    for _ in range(2):
        prompt = scene_prompt if not issues else scene_prompt + "\nFIX THESE PROBLEMS:\n" + "\n".join(issues)
        scenes = ctx.llm.generate_json("scene_plan", prompt, temperature=0.4).get("scenes")
        issues = scene_issues(narration, scenes, kind)
        if not issues:
            break
    else:
        return _reject(ctx, story_id, series.name, "scene plan invalid: " + "; ".join(issues[:4]), review)

    if series.uses_world:
        db.link_entities(ctx.conn, story_id, script.get("new_entities") or [],
                         [str(n) for n in script.get("referenced_entities") or []])
    db.mark_inspirations_used(ctx.conn, [row["id"] for row in ([legend] if legend else []) + list(sparks)])
    db.update_story(ctx.conn, story_id, status="approved", scenes=scenes)
    log.info("Story #%d approved: %s", story_id, premise["title"])
    return Outcome(series.name, "approved", story_id, premise["title"])


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

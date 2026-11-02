"""The writers' room: how one story is made.

    research legend (web) -> pitch contest -> beat outline -> competing drafts
    -> editors' loop -> simulated audience -> read-aloud polish -> final checks
    -> scene plan -> packaging -> approved

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
from .router import Router

log = logging.getLogger(__name__)

CLAIMS_TRUE = re.compile(
    r"\b(true story|really happened|based on (a )?true|actual events|this happened to me)\b", re.IGNORECASE)
CLICHES = re.compile(
    r"\b(chills? ran down|blood ran cold|little did (he|she|they|i|we) know|out of nowhere|"
    r"it was all a dream|heart pounding in (my|his|her|their) chest|eerie silence|something was off)\b",
    re.IGNORECASE)
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
    text = narration_of(script)
    if CLAIMS_TRUE.search(text):
        issues.append("never claim the story is true or based on real events")
    for match in CLICHES.finditer(text):
        issues.append(f"replace the cliche '{match.group(0)}' with something specific")
    return issues


def aggregate_reviews(reviews: list[dict[str, Any]]) -> dict[str, Any]:
    """Median score per criterion; a flag counts if any editor raised it."""
    scores = {c: statistics.median(int(r["scores"].get(c, 0)) for r in reviews)
              for c in schemas.REVIEW_CRITERIA}
    flags = sorted({f for r in reviews for f, raised in r.get("flags", {}).items() if raised is True})
    notes = list(dict.fromkeys(n for r in reviews for n in r.get("notes", []) if str(n).strip()))
    return {"scores": scores, "flags": flags, "notes": notes}


def review_issues(agg: dict[str, Any], min_score: float) -> list[str]:
    issues = [f"raise {c.replace('_', ' ')} (editors' median {s}/10)"
              for c, s in agg["scores"].items() if s < min_score]
    issues += [f"remove rule violation: {f.replace('_', ' ')}" for f in agg["flags"]]
    if issues:
        issues += agg["notes"]
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
    if hasattr(researcher, "research"):
        notes = researcher.research("dossier_research", prompts.SYSTEM_RESEARCHER, prompts.dossier_research(legend))
    else:
        notes = legend["summary"] or ""
    dossier = researcher.generate_json("dossier", prompts.SYSTEM_RESEARCHER,
                                       prompts.dossier_structure(legend, notes), schemas.DOSSIER)
    db.set_dossier(ctx.conn, legend["id"], dossier)
    return dossier


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


def score_pitches(ctx: Context, series: catalog.Series, pitches: list[dict[str, Any]]) -> None:
    jobs = [(p, judge) for p in pitches for judge in ctx.router.judges([p["_family"]])]

    def score(job: tuple[dict, Any]) -> tuple[int, float]:
        p, judge = job
        result = judge.generate_json("score_pitch", prompts.SYSTEM_EDITOR, prompts.score_pitch(series, p),
                                     schemas.PITCH_SCORE)
        values = [int(result["scores"].get(c, 0)) for c in schemas.PITCH_CRITERIA]
        return id(p), sum(values) / len(values)

    totals: dict[int, list[float]] = {}
    for key, value in ctx.router.parallel(score, jobs):
        totals.setdefault(key, []).append(value)
    for p in pitches:
        p["_score"] = round(statistics.mean(totals.get(id(p), [0.0])), 2)


def pitch_contest(ctx: Context, series: catalog.Series, pitches: list[dict[str, Any]]) -> list[tournament.Ranked]:
    finalists = sorted(pitches, key=lambda p: p["_score"], reverse=True)[: ctx.cfg["quality"]["tournament_size"]]

    def compare(judge: Any, a: dict, b: dict) -> str:
        return judge.generate_json("compare_pitches", prompts.SYSTEM_EDITOR,
                                   prompts.compare_pitches(series, a, b), schemas.PAIRWISE)["winner"]

    return tournament.rank(finalists, lambda a, b: ctx.router.judges([a["_family"], b["_family"]]),
                           compare, ctx.router.parallel, prior=lambda p: p["_score"])


def make_outline(ctx: Context, story: Story) -> dict[str, Any]:
    kind = story.series.kind
    writer = ctx.router.one("writers")
    target = 60 if kind == "short" else round(ctx.cfg["lengths"]["long_min_words"] / 2.3)
    outline = writer.generate_json("beat_sheet", prompts.SYSTEM_WRITER, prompts.beat_sheet(
        series=story.series, kind=kind, recipe=story.recipe, formats=FORMATS, endings=ENDINGS,
        pitch=story.pitch, legend=story.legend, dossier=story.dossier, world=story.world,
        target_seconds=target), schemas.BEAT_SHEET)
    for attempt in range(2):
        reviews = ctx.router.parallel(
            lambda judge: judge.generate_json("review_outline", prompts.SYSTEM_EDITOR,
                                              prompts.review_outline(kind, story.pitch, outline),
                                              schemas.OUTLINE_REVIEW),
            ctx.router.judges([writer.spec.family]))
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


def draft_contest(ctx: Context, story: Story, outline: dict[str, Any]) -> tuple[dict[str, Any], str]:
    kind = story.series.kind
    lo, hi = word_range(ctx.cfg, kind)

    def write(writer: Any) -> dict[str, Any] | None:
        try:
            script = writer.generate_json("write_draft", prompts.SYSTEM_WRITER, prompts.write_draft(
                series=story.series, kind=kind, recipe=story.recipe, formats=FORMATS, endings=ENDINGS,
                pitch=story.pitch, outline=outline, min_words=lo, max_words=hi, legend=story.legend,
                dossier=story.dossier, world=story.world), schemas.SCRIPT)
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

    ranked = tournament.rank(drafts, lambda a, b: ctx.router.judges([a["_family"], b["_family"]]), compare,
                             ctx.router.parallel, prior=lambda d: -len(script_issues(d, kind, ctx.cfg)))
    best = ranked[0].item
    story.note("draft_contest", drafts=len(drafts), winner=best["_author"],
               points=[r.points for r in ranked])
    story.authors["draft"] = best["_author"]
    return {k: best.get(k) for k in SCRIPT_FIELDS}, best["_family"]


def editors_loop(ctx: Context, story: Story, script: dict[str, Any], author_family: str,
                 story_id: int) -> tuple[dict[str, Any] | None, dict[str, Any], str]:
    """Review and revise until every editor median clears the bar. Returns (script|None, review, reason)."""
    kind, cfg = story.series.kind, ctx.cfg
    lo, hi = word_range(cfg, kind)
    reviser = ctx.router.one("reviser")
    sources = _sources(story)
    agg: dict[str, Any] = {}
    for round_ in range(cfg["quality"]["max_revisions"] + 1):
        issues = script_issues(script, kind, cfg)
        copy = dedup.check_script(ctx.conn, cfg, story_id, narration_of(script), sources)
        if not copy.ok:
            issues.append(f"rewrite in original words: {copy.reason}")
        if not issues:
            reviews = ctx.router.parallel(
                lambda judge: judge.generate_json("review", prompts.SYSTEM_EDITOR, prompts.review(
                    kind, story.recipe, FORMATS, ENDINGS, script, story.dossier), schemas.REVIEW),
                ctx.router.judges([author_family]))
            agg = aggregate_reviews(reviews)
            issues = review_issues(agg, cfg["quality"]["min_score"])
        story.note("editors", round=round_ + 1, issues=len(issues), scores=agg.get("scores"))
        if not issues:
            return script, agg, ""
        if round_ == cfg["quality"]["max_revisions"]:
            return None, agg, "; ".join(issues[:6])
        script = reviser.generate_json("revise", prompts.SYSTEM_WRITER,
                                       prompts.revise(kind, script, issues, lo, hi), schemas.SCRIPT)
        author_family = reviser.spec.family
        story.authors["reviser"] = reviser.spec.label
    raise AssertionError("unreachable")


def audience_test(ctx: Context, story: Story, script: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Simulated viewers rate each ~30-second part; weak parts are rewritten if that's judged better."""
    q, kind = ctx.cfg["quality"], story.series.kind
    chunks = chunk_text(script["body"], q["audience_chunk_words"])
    if len(chunks) < 2:
        return script, {"skipped": "too short to chunk"}
    readers = ctx.router.pool("audience")
    jobs = [(persona, readers[i % len(readers)]) for i, persona in enumerate(q["audience_personas"])]

    def watch(job: tuple[str, Any]) -> dict[str, Any] | None:
        persona, reader = job
        try:
            return reader.generate_json("audience", prompts.system_audience(persona),
                                        prompts.audience(kind, chunks), schemas.AUDIENCE)
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
            n = int(row.get("chunk", 0))
            if not 1 <= n <= len(chunks):
                continue
            interest.setdefault(n, []).append(int(row.get("interest", 0)))
            if row.get("note"):
                notes.setdefault(n, []).append(str(row["note"]))
            if row.get("would_leave") and not left:  # count each viewer's exit once
                leaves[n] = leaves.get(n, 0) + 1
                left = True
    means = {n: round(statistics.mean(v), 2) for n, v in interest.items()}
    weak = sorted((n for n in means if means[n] < q["audience_min_interest"] or leaves.get(n, 0) >= 2),
                  key=lambda n: (means[n], -leaves.get(n, 0)))[: q["max_chunks_to_rewrite"]]
    report = {"chunk_interest": means, "exits": leaves, "weak_chunks": weak,
              "viewers": [r.get("overall", "") for r in reports]}
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

    ranked = tournament.rank([script, candidate], lambda a, b: ctx.router.judges([reviser.spec.family]),
                             compare, ctx.router.parallel)
    improved = ranked[0].item is candidate or ranked[0].points == ranked[1].points
    report["rewrite"] = "kept" if improved else "discarded: judges preferred the original"
    return (candidate if improved else script), report


def polish(ctx: Context, story: Story, script: dict[str, Any]) -> tuple[dict[str, Any], list[dict]]:
    kind = story.series.kind
    lo, hi = word_range(ctx.cfg, kind)
    polisher = ctx.router.one("polisher")
    result = polisher.generate_json("polish", prompts.SYSTEM_WRITER, prompts.polish(kind, script, lo, hi),
                                    schemas.POLISH)
    candidate = dict(script, intro=result.get("intro", ""), body=result.get("body", ""),
                     outro=result.get("outro", ""))
    unchanged_meaning = dedup.lexical_similarity(narration_of(script), narration_of(candidate), size=1) >= 0.6
    if script_issues(candidate, kind, ctx.cfg) or not unchanged_meaning:
        story.note("polish", kept=False)
        return script, []
    story.note("polish", kept=True)
    story.authors["polisher"] = polisher.spec.label
    return candidate, [p for p in result.get("pronunciations", []) if p.get("word")]


def final_check(ctx: Context, story: Story, script: dict[str, Any]) -> list[str]:
    results = ctx.router.parallel(
        lambda judge: judge.generate_json("final_check", prompts.SYSTEM_EDITOR,
                                          prompts.final_check(script, story.dossier), schemas.FINAL_CHECK),
        ctx.router.judges([ctx.router.one("polisher").spec.family]))
    issues = sorted({f"rule violation: {f.replace('_', ' ')}" for r in results
                     for f, raised in r.get("flags", {}).items() if raised is True})
    for r in results:
        issues += [f"accuracy: {i}" for i in r.get("accuracy_issues", []) if str(i).strip()]
        issues += [f"cultural: {i}" for i in r.get("cultural_issues", []) if str(i).strip()]
    return list(dict.fromkeys(issues))


def plan_scenes(ctx: Context, kind: str, narration: str) -> tuple[list | None, list[str]]:
    scene_range, key_range = scene_ranges(kind)
    planner = ctx.router.one("planner")
    base = prompts.scene_plan(narration, scene_range, key_range)
    issues: list[str] = []
    for _ in range(2):
        prompt = base if not issues else base + "\nFIX THESE PROBLEMS:\n" + "\n".join(issues)
        scenes = planner.generate_json("scene_plan", prompts.SYSTEM_EDITOR, prompt, schemas.SCENES).get("scenes")
        issues = scene_issues(narration, scenes, kind)
        if not issues:
            return scenes, []
    return None, issues


def package(ctx: Context, story: Story, script: dict[str, Any]) -> dict[str, Any]:
    writer = ctx.router.one("reviser")
    credit = script.get("description_credit") or "This is an original work of fiction."
    result = writer.generate_json("packaging", prompts.SYSTEM_WRITER,
                                  prompts.packaging(story.series, script, story.pitch, credit), schemas.PACKAGING)
    titles = [t.strip() for t in result.get("titles", []) if t.strip() and len(t.strip()) <= 70]
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
            legends = db.unused_legends(ctx.conn)
            if not legends:
                return Outcome(series_name, "skipped", detail="no unused legends; run `research` first")
            legend = ctx.rng.choice(legends)
            dossier = ensure_dossier(ctx, legend)
        sparks = db.folklore_sparks(ctx.conn, 2) if series.uses_world else []
        world = db.world_entities(ctx.conn) if series.uses_world else []

        pitches = gather_pitches(ctx, series, legend, dossier, sparks, world)
        if not pitches:
            log.info("Attempt %d for %s produced no original pitch", attempt, series_name)
            continue
        score_pitches(ctx, series, pitches)
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
    try:
        return develop(ctx, story_id, story)
    except (LLMError, BudgetExceeded) as exc:
        # Don't leave a half-made story stuck in the queue; its recipe is freed.
        _reject(ctx, story_id, story.series.name, f"interrupted: {exc}", story=story)
        raise


def develop(ctx: Context, story_id: int, story: Story) -> Outcome:
    series, kind = story.series, story.series.kind
    outline = make_outline(ctx, story)
    db.update_story(ctx.conn, story_id, beat_sheet=outline)

    script, author_family = draft_contest(ctx, story, outline)
    script, review, reason = editors_loop(ctx, story, script, author_family, story_id)
    if script is None:
        return _reject(ctx, story_id, series.name, reason, review, story)

    script, audience_report = audience_test(ctx, story, script)
    script, pronunciations = polish(ctx, story, script)

    problems = final_check(ctx, story, script)
    if problems:
        reviser = ctx.router.one("reviser")
        lo, hi = word_range(ctx.cfg, kind)
        fixed = reviser.generate_json("revise", prompts.SYSTEM_WRITER,
                                      prompts.revise(kind, script, problems, lo, hi), schemas.SCRIPT)
        problems = script_issues(fixed, kind, ctx.cfg) or final_check(ctx, story, fixed)
        if problems:
            return _reject(ctx, story_id, series.name, "final check: " + "; ".join(problems[:4]), review, story)
        script = fixed
    story.note("final_check", clear=True)

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
        review=review, audience_report=audience_report, pronunciations=pronunciations)

    scenes, issues = plan_scenes(ctx, kind, narration)
    if scenes is None:
        return _reject(ctx, story_id, series.name, "scene plan invalid: " + "; ".join(issues[:4]), review, story)
    packaging = package(ctx, story, script)

    if series.uses_world:
        db.link_entities(ctx.conn, story_id, script.get("new_entities") or [],
                         [str(n) for n in script.get("referenced_entities") or []])
    db.mark_inspirations_used(ctx.conn, [row["id"] for row in ([story.legend] if story.legend else []) + story.sparks])
    db.update_story(ctx.conn, story_id, status="approved", scenes=scenes, packaging=packaging,
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

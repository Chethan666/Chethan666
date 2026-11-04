"""Pipeline 3: the drama writers' room, cloned from pipeline 1 and rebuilt for series.

    SERIES   pitch contest (every pitcher on its own trope recipe; repeats dropped; judges
             from other companies score; finalists head to head; strong runners-up shelved)
             -> plan contest (every writer writes a series bible; judges pick one) -> the
             winning bible is scored and revised, keeping the best version
    EPISODE  draft contest (every writer drafts; judges pick) -> editors' loop with the
             showrunner (free checks, scored review with notes on shot numbers; the best
             version is always kept; the fix is chosen by what is wrong: rewrite the noted
             shots, revise, hook or cliffhanger lab, a fresh draft, or a new outline)
             -> hook lab -> cliffhanger lab -> simulated viewers (where would they swipe,
             would they unlock the next one?) -> final check -> packaging -> approved

Models, judges, the database and the cost log are pipeline 1's; the stages are this
pipeline's own.
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, field
from typing import Any

from story_factory import db, dedup, tournament
from story_factory.llm import LLMError
from story_factory.pipeline import Context

from . import catalog, prompts, schemas

log = logging.getLogger(__name__)

STRUCTURAL = frozenset({"conflict", "character", "cliffhanger"})
OUTLINE_ENTRY = schemas.obj(number=schemas.I, title=schemas.S, hook=schemas.S, beats=schemas.arr(schemas.S),
                            cliffhanger=schemas.S)


@dataclass
class Log:
    entries: list[dict[str, Any]] = field(default_factory=list)

    def note(self, stage: str, **info: Any) -> None:
        self.entries.append({"stage": stage, **info})
        log.info("%s: %s", stage, info)


def settings(ctx: Context) -> dict[str, Any]:
    return ctx.cfg["drama"]


def _judges(ctx: Context, families: list[str], limit_key: str = "judges_per_comparison") -> list[Any]:
    return ctx.router.judges(families, ctx.cfg["quality"][limit_key])


def _compare(task: str, build: Any) -> Any:
    def compare(judge: Any, a: Any, b: Any) -> str:
        return judge.generate_json(task, prompts.SYSTEM_EDITOR, build(a, b), schemas.PAIRWISE)["winner"]
    return compare


# --- Series ------------------------------------------------------------------

def bible_issues(bible: dict[str, Any], episodes: int) -> list[str]:
    issues = []
    cast = [c for c in bible.get("cast") or [] if isinstance(c, dict)]
    if not 2 <= len(cast) <= 7:
        issues.append(f"{len(cast)} characters; give 3-6")
    if any(len(str(c.get("look", "")).split()) < 8 for c in cast):
        issues.append("every character needs a precise look of at least a sentence")
    if len(bible.get("locations") or []) < 2:
        issues.append("give 3-6 recurring locations")
    if len(bible.get("episodes") or []) < episodes:
        issues.append(f"outline all {episodes} episodes")
    if any(not str(e.get("cliffhanger", "")).strip() for e in bible.get("episodes") or []):
        issues.append("every episode needs a cliffhanger")
    return issues


def pitch_contest(ctx: Context, journal: Log) -> list[dict[str, Any]]:
    """Ranked pitches, best first (empty if none survived the repeat checks)."""
    d = settings(ctx)
    rows = ctx.conn.execute("SELECT title, pitch FROM series").fetchall()
    used = {json.loads(r["pitch"]).get("_recipe", "") for r in rows if r["pitch"]}
    taken = [r["title"] for r in rows]
    pitchers = ctx.router.pool("pitchers")
    jobs = list(zip(pitchers, catalog.recipes(ctx.rng, len(pitchers), used)))

    def pitch(job: tuple[Any, catalog.Recipe]) -> list[dict[str, Any]]:
        pitcher, recipe = job
        try:
            result = pitcher.generate_json("drama_pitches", prompts.SYSTEM_WRITER, prompts.pitches(
                d["genre"], recipe.describe(), d["pitches_per_pitcher"], taken[:40]), schemas.PITCHES)
        except LLMError as exc:
            log.warning("Pitcher %s failed: %s", pitcher.spec.label, exc)
            return []
        return [dict(p, _recipe=recipe.signature, _author=pitcher.spec.label, _family=pitcher.spec.family)
                for p in result.get("pitches", []) if str(p.get("title", "")).strip()]

    candidates: list[dict[str, Any]] = []
    for p in (p for batch in ctx.router.parallel(pitch, jobs) for p in batch):
        twin = next((t for t in taken + [c["title"] for c in candidates]
                     if dedup.title_similarity(t, p["title"]) >= ctx.cfg["dedup"]["title_similarity"]), None)
        if twin:
            log.info("Pitch '%s' repeats '%s'", p["title"], twin)
            continue
        candidates.append(p)
    if not candidates:
        return []

    def score(job: tuple[dict, Any]) -> tuple[int, float]:
        p, judge = job
        result = judge.generate_json("drama_score_pitch", prompts.SYSTEM_EDITOR, prompts.score_pitch(
            d["genre"], {k: v for k, v in p.items() if not k.startswith("_")}), schemas.PITCH_SCORE)
        return id(p), statistics.mean(int(result["scores"].get(c, 0)) for c in schemas.PITCH_CRITERIA)

    totals: dict[int, list[float]] = {}
    for key, value in ctx.router.panel(score, [(p, j) for p in candidates for j in _judges(ctx, [p["_family"]])],
                                       "score pitches"):
        totals.setdefault(key, []).append(value)
    for p in candidates:
        p["_score"] = round(statistics.mean(totals.get(id(p), [0.0])), 2)
    finalists = sorted(candidates, key=lambda p: p["_score"], reverse=True)[: d["tournament_size"]]
    clean = lambda p: {k: v for k, v in p.items() if not k.startswith("_")}  # noqa: E731
    ranked = tournament.rank(finalists, lambda a, b: _judges(ctx, [a["_family"], b["_family"]]),
                             _compare("drama_compare_pitches", lambda a, b: prompts.compare_pitches(clean(a), clean(b))),
                             ctx.router.parallel, prior=lambda p: p["_score"])
    journal.note("pitch_contest", candidates=len(candidates), winner=ranked[0].item["title"],
                 points=[r.points for r in ranked])
    return [r.item for r in ranked]


def refine_bible(ctx: Context, bible: dict[str, Any], writer: Any, journal: Log) -> dict[str, Any]:
    """Scored review and revision of a bible, keeping the best-scoring version."""
    d = settings(ctx)
    best, best_key, current = bible, (-1, -1.0), bible
    for round_ in range(d["plan_revisions"] + 1):
        problems = bible_issues(current, d["episodes_per_series"])
        score = 0.0
        if not problems:
            reviews = ctx.router.panel(lambda j: j.generate_json("drama_review_bible", prompts.SYSTEM_EDITOR,
                                                                 prompts.review_bible(current), schemas.PLAN_REVIEW),
                                       _judges(ctx, [writer.spec.family], "quick_judges"))
            score = statistics.median(min(max(int(r.get("score") or 0), 0), 10) for r in reviews)
            problems = [] if score >= d["plan_min_score"] else list(dict.fromkeys(
                p for r in reviews for p in (r.get("problems") or [])[:3]))[:6]
        journal.note("bible_review", round=round_ + 1, score=score, problems=len(problems))
        key = (0 if bible_issues(current, d["episodes_per_series"]) else 1, score)
        if key > best_key:
            best, best_key = current, key
        if not problems or round_ == d["plan_revisions"]:
            break
        current = writer.generate_json("drama_revise_bible", prompts.SYSTEM_WRITER,
                                       prompts.revise_bible(current, problems), schemas.BIBLE)
    return best


def develop_series(ctx: Context, pitch: dict[str, Any], journal: Log) -> int:
    d = settings(ctx)
    writers = ctx.router.pool("writers")
    clean = {k: v for k, v in pitch.items() if not k.startswith("_")}

    def plan(writer: Any) -> dict[str, Any] | None:
        try:
            b = writer.generate_json("drama_bible", prompts.SYSTEM_WRITER,
                                     prompts.bible(d["genre"], clean, d["episodes_per_series"]), schemas.BIBLE)
        except LLMError as exc:
            log.warning("Planner %s failed: %s", writer.spec.label, exc)
            return None
        return dict(b, _writer=writer, _family=writer.spec.family, _author=writer.spec.label)

    plans = [p for p in ctx.router.parallel(plan, writers) if p]
    if not plans:
        raise LLMError("every planner failed")
    strip = lambda b: {k: v for k, v in b.items() if not k.startswith("_")}  # noqa: E731
    winner = plans[0]
    if len(plans) > 1:
        ranked = tournament.rank(plans, lambda a, b: _judges(ctx, [a["_family"], b["_family"]]),
                                 _compare("drama_compare_bibles", lambda a, b: prompts.compare_bibles(strip(a), strip(b))),
                                 ctx.router.parallel,
                                 prior=lambda b: -len(bible_issues(b, d["episodes_per_series"])))
        winner = ranked[0].item
        journal.note("plan_contest", plans=len(plans), winner=winner["_author"], points=[r.points for r in ranked])
    bible = refine_bible(ctx, strip(winner), winner["_writer"], journal)
    if len(bible.get("cast") or []) < 2 or not bible.get("episodes"):
        raise LLMError("the series bible has no cast or episodes")
    now = db.now()
    cur = ctx.conn.execute(
        "INSERT INTO series (title, genre, status, bible, pitch, production_log, created_at, updated_at) "
        "VALUES (?, ?, 'active', ?, ?, ?, ?, ?)",
        (bible["title"].strip(), d["genre"], json.dumps(bible, ensure_ascii=False),
         json.dumps({**clean, "_recipe": pitch.get("_recipe", ""), "_score": pitch.get("_score")}),
         json.dumps(journal.entries), now, now))
    return int(cur.lastrowid)


def create_series(ctx: Context) -> int:
    """A new series: a shelved runner-up first, otherwise a fresh pitch contest."""
    d = settings(ctx)
    journal = Log()
    shelved = ctx.conn.execute("SELECT * FROM series WHERE status = 'shelved' ORDER BY id LIMIT 1").fetchone()
    if shelved is not None:
        pitch = json.loads(shelved["pitch"])
        ctx.conn.execute("DELETE FROM series WHERE id = ?", (shelved["id"],))
        journal.note("pitch", source="shelf", title=pitch.get("title"))
    else:
        ranked = pitch_contest(ctx, journal)
        if not ranked:
            raise LLMError("no original pitch survived")
        pitch = ranked[0]
        now = db.now()
        for runner_up in ranked[1:]:
            if runner_up["_score"] >= d["shelve_min_score"]:
                ctx.conn.execute(
                    "INSERT INTO series (title, genre, status, bible, pitch, created_at, updated_at) "
                    "VALUES (?, ?, 'shelved', '{}', ?, ?, ?)",
                    (runner_up["title"], d["genre"], json.dumps({k: v for k, v in runner_up.items()
                                                                 if k not in ("_family", "_author")}), now, now))
    series_id = develop_series(ctx, pitch, journal)
    log.info("Series #%d created: %s", series_id, pitch.get("title"))
    return series_id


# --- Episodes ----------------------------------------------------------------

def episode_issues(ep: dict[str, Any], bible: dict[str, Any]) -> list[str]:
    """Free checks: enough shots, known characters and places, mostly dialogue, short lines."""
    shots = ep.get("shots") or []
    issues = []
    if not 10 <= len(shots) <= 32:
        issues.append(f"{len(shots)} shots; write 14-26")
    names = {c["name"].lower() for c in bible.get("cast", [])}
    places = {loc["name"].lower() for loc in bible.get("locations", [])}
    strangers = {c for s in shots for c in s.get("characters", []) if c.lower() not in names}
    if strangers:
        issues.append("characters not in the cast: " + ", ".join(sorted(strangers)))
    if any(str(s.get("speaker", "")).lower() not in names for s in shots if str(s.get("line", "")).strip()):
        issues.append("every speaker must be a cast member")
    unknown = {s.get("location", "") for s in shots if str(s.get("location", "")).lower() not in places}
    if unknown:
        issues.append("use only the bible's locations; unknown: " + ", ".join(sorted(unknown)))
    if sum(1 for s in shots if str(s.get("line", "")).strip()) < len(shots) * 0.5:
        issues.append("this is a dialogue drama: at least half the shots carry a spoken line")
    if sum(1 for s in shots if s.get("motion")) > 4:
        issues.append("mark at most four shots as motion")
    if any(len(str(s.get("line", "")).split()) > 24 for s in shots):
        issues.append("keep every line under 20 words")
    if shots and not str(shots[0].get("line", "") or shots[0].get("action", "")).strip():
        issues.append("the first shot must be the hook")
    return issues


@dataclass
class Version:
    episode: dict[str, Any]
    family: str
    issues: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    notes: list[dict[str, Any]] = field(default_factory=list)

    def low(self, bar: float) -> set[str]:
        return {c for c, v in self.scores.items() if v < bar}

    def passes(self, bar: float) -> bool:
        return not self.issues and bool(self.scores) and not self.low(bar) and not self.flags

    def key(self, bar: float) -> tuple:
        if self.issues or not self.scores:
            return (0, 0, 0.0, 0.0)
        values = list(self.scores.values())
        return (1, int(self.passes(bar)), min(values), statistics.mean(values))

    def note_lines(self) -> list[str]:
        return [f"shot {n.get('shot')}: {n.get('problem')} Fix: {n.get('fix')}" for n in self.notes]


def judge(ctx: Context, bible: dict[str, Any], ep: dict[str, Any], family: str) -> Version:
    v = Version(ep, family, episode_issues(ep, bible))
    if v.issues:
        return v
    reviews = ctx.router.panel(lambda j: j.generate_json("drama_review", prompts.SYSTEM_EDITOR,
                                                         prompts.review_episode(bible, ep), schemas.REVIEW),
                               _judges(ctx, [family]))
    v.scores = {c: statistics.median(int(r["scores"].get(c, 0)) for r in reviews) for c in schemas.EPISODE_CRITERIA}
    v.flags = sorted({f for r in reviews for f, raised in r.get("flags", {}).items() if raised is True})
    v.notes = [n for r in reviews for n in r.get("notes", []) if isinstance(n, dict)][:10]
    return v


def diagnose(v: Version, bar: float, fresh_left: int, stuck: set[str], last: str, stalled: bool) -> tuple[str, str]:
    """The showrunner's call for an episode: fix the deepest problem first."""
    if v.issues:
        return "repair", "free checks failed: " + "; ".join(v.issues[:2])
    floor = min(v.scores.values())
    if fresh_left and floor <= 5:
        return "redraft", f"the draft is far off (lowest score {floor:g}): write it again"
    if stuck and fresh_left:
        return "replan", f"{', '.join(sorted(stuck))} stayed low after a revision: rethink the episode's outline"
    low = v.low(bar)
    if low == {"hook"} and last != "hook":
        return "hook", "only the hook is weak"
    if low == {"cliffhanger"} and last != "cliffhanger":
        return "cliffhanger", "only the cliffhanger is weak"
    if stalled and last == "shots":
        return "revise", "rewriting the noted shots made it worse; revise the whole episode"
    if v.notes and last != "shots":
        return "shots", "rewrite the shots the editors point at"
    return "revise", "revise against the notes"


def draft_contest(ctx: Context, bible: dict[str, Any], number: int, previous: list[dict[str, Any]],
                  journal: Log, avoid: list[str] | None = None) -> tuple[dict[str, Any], str]:
    d = settings(ctx)
    writers = ctx.router.pool("writers")

    def write(writer: Any) -> dict[str, Any] | None:
        for _ in range(2):  # a model that loops to its output limit usually manages a second time
            try:
                ep = writer.generate_json("drama_episode", prompts.SYSTEM_WRITER, prompts.episode(
                    bible, number, previous, d["episode_seconds"], avoid or []), schemas.EPISODE)
                return dict(ep, _family=writer.spec.family, _author=writer.spec.label)
            except LLMError as exc:
                log.warning("Writer %s failed: %s", writer.spec.label, exc)
        return None

    drafts = [x for x in ctx.router.parallel(write, writers) if x]
    if not drafts:
        raise LLMError("every writer failed")
    strip = lambda e: {k: v for k, v in e.items() if not k.startswith("_")}  # noqa: E731
    ranked = tournament.rank(drafts, lambda a, b: _judges(ctx, [a["_family"], b["_family"]]),
                             _compare("drama_compare_episodes", lambda a, b: prompts.compare_episodes(strip(a), strip(b))),
                             ctx.router.parallel, prior=lambda e: -len(episode_issues(e, bible)))
    best = ranked[0].item
    journal.note("draft_contest", drafts=len(drafts), winner=best["_author"], points=[r.points for r in ranked],
                 fresh=bool(avoid))
    return strip(best), best["_family"]


def rewrite_shots(ctx: Context, bible: dict[str, Any], v: Version, journal: Log) -> dict[str, Any]:
    shots = v.episode.get("shots", [])
    notes: dict[int, list[str]] = {}
    for n in v.notes:
        try:
            k = int(n.get("shot", 0))
        except (TypeError, ValueError):
            continue
        if 1 <= k <= len(shots):
            notes.setdefault(k, []).append(f"{n.get('problem')} Fix: {n.get('fix')}")
    if not notes:
        return revise(ctx, bible, v)
    result = ctx.router.one("reviser").generate_json("drama_rewrite_shots", prompts.SYSTEM_WRITER,
                                                     prompts.rewrite_shots(bible, v.episode, notes), schemas.SHOT_REWRITE)
    new = list(shots)
    for row in result.get("shots", []):
        try:
            k = int(row.get("number", 0))
        except (TypeError, ValueError):
            continue
        if k in notes and isinstance(row.get("shot"), dict):
            new[k - 1] = row["shot"]
    journal.note("shot_rewrite", rewritten=sorted(notes))
    return dict(v.episode, shots=new)


def revise(ctx: Context, bible: dict[str, Any], v: Version) -> dict[str, Any]:
    notes = v.issues + [f"raise {c} (editors' median {s:g})" for c, s in v.scores.items()
                        if s < settings(ctx)["min_score"]] + v.flags + v.note_lines()
    return ctx.router.one("reviser").generate_json("drama_revise", prompts.SYSTEM_WRITER,
                                                   prompts.revise_episode(bible, v.episode, notes), schemas.EPISODE)


def lab(ctx: Context, part: str, ep: dict[str, Any], journal: Log) -> dict[str, Any]:
    """Hook or cliffhanger lab: alternative first or last two shots against the original."""
    shots = ep.get("shots") or []
    if len(shots) < 6:
        return ep
    reviser = ctx.router.one("reviser")
    try:
        result = reviser.generate_json(f"drama_{part}_options", prompts.SYSTEM_WRITER,
                                       prompts.alternatives(ep, part, settings(ctx)["alternatives"]), schemas.ALTERNATIVES)
    except LLMError as exc:
        log.warning("%s lab skipped: %s", part, exc)
        return ep
    original = shots[:2] if part == "hook" else shots[-2:]
    options = [original] + [o["shots"] for o in result.get("options", [])
                            if isinstance(o, dict) and isinstance(o.get("shots"), list) and len(o["shots"]) == 2]
    if len(options) == 1:
        return ep
    context = "The episode continues: " + json.dumps(shots[2:4] if part == "hook" else shots[-4:-2], ensure_ascii=False)
    ranked = tournament.rank(options, lambda a, b: _judges(ctx, [reviser.spec.family], "quick_judges"),
                             _compare(f"drama_compare_{part}", lambda a, b: prompts.compare_parts(part, a, b, context)),
                             ctx.router.parallel, prior=lambda o: 1.0 if o is original else 0.0)
    best = ranked[0].item
    journal.note(f"{part}_lab", options=len(options), replaced=best is not original)
    if best is original:
        return ep
    return dict(ep, shots=(best + shots[2:]) if part == "hook" else (shots[:-2] + best))


def audience(ctx: Context, ep: dict[str, Any], journal: Log) -> dict[str, Any]:
    d = settings(ctx)
    readers = ctx.router.pool("audience")
    jobs = [(p, readers[i % len(readers)]) for i, p in enumerate(d["viewers"])]

    def watch(job: tuple[str, Any]) -> dict[str, Any] | None:
        persona, reader = job
        try:
            return reader.generate_json("drama_viewer", prompts.system_viewer(persona), prompts.viewer(ep), schemas.VIEWER)
        except LLMError as exc:
            log.warning("A viewer failed: %s", exc)
            return None

    reports = [r for r in ctx.router.parallel(watch, jobs) if r]
    if not reports:
        return {"skipped": "no viewers answered"}
    swipes: dict[int, int] = {}
    for r in reports:
        first = next((int(s.get("shot", 0)) for s in r.get("shots", []) if s.get("would_swipe")), None)
        if first:
            swipes[first] = swipes.get(first, 0) + 1
    report = {"unlock_next": round(statistics.mean(int(r.get("unlock_next", 0)) for r in reports), 2),
              "swipes": swipes, "viewers": [r.get("overall", "") for r in reports]}
    journal.note("audience", unlock_next=report["unlock_next"], swipes=swipes)
    return report


def write_episode(ctx: Context, series_id: int, number: int | None = None) -> tuple[int, str]:
    """Write the next episode through the whole room. Returns (episode id, status)."""
    d = settings(ctx)
    row = ctx.conn.execute("SELECT * FROM series WHERE id = ?", (series_id,)).fetchone()
    if row is None:
        raise ValueError(f"no series #{series_id}")
    bible = json.loads(row["bible"])
    done = [dict(r) for r in ctx.conn.execute(
        "SELECT number, title, script FROM episodes WHERE series_id = ? AND status != 'rejected' ORDER BY number",
        (series_id,))]
    number = number or (max((x["number"] for x in done), default=0) + 1)
    previous = [{"number": x["number"], "title": x["title"],
                 "cliffhanger": json.loads(x["script"]).get("cliffhanger", "")} for x in done if x["number"] < number]
    journal = Log()
    bar = float(d["min_score"])

    episode, family = draft_contest(ctx, bible, number, previous, journal)
    rounds, fresh_left = d["max_revisions"] + 1, d["max_fresh_starts"]
    best: Version | None = None
    previous_version: Version | None = None
    last, round_ = "", 0
    while True:
        round_ += 1
        v = judge(ctx, bible, episode, family)
        improved = best is None or v.key(bar) > best.key(bar)
        if improved:
            best = v
        journal.note("editors", round=round_, after=last or "first draft", scores=v.scores or None,
                     issues=v.issues[:3], kept="this version" if improved else "the earlier best")
        if v.passes(bar) or round_ >= rounds:
            break
        stuck = (v.low(bar) & previous_version.low(bar) & STRUCTURAL) if previous_version and v.scores else set()
        target = v if (v.issues and v is not best and last != "repair") else best
        action, why = diagnose(target, bar, fresh_left, stuck, last, stalled=not improved)
        journal.note("showrunner", action=action, why=why)
        if v.scores:
            previous_version = v
        if action in ("redraft", "replan"):
            fresh_left -= 1
            rounds += 1
            avoid = (best.issues + best.note_lines())[:8]
            if action == "replan":
                outline = ctx.router.one("writers").generate_json(
                    "drama_replan", prompts.SYSTEM_WRITER,
                    f"Series bible:\n{json.dumps(bible, ensure_ascii=False)}\n\nRethink episode {number}'s outline so it "
                    f"fixes: {'; '.join(avoid)}. Keep it consistent with the story so far.", OUTLINE_ENTRY)
                bible = dict(bible, episodes=[outline if int(e.get("number", 0)) == number else e
                                              for e in bible.get("episodes", [])])
            episode, family = draft_contest(ctx, bible, number, previous, journal, avoid)
        elif action == "hook":
            episode, family = lab(ctx, "hook", target.episode, journal), target.family
        elif action == "cliffhanger":
            episode, family = lab(ctx, "cliffhanger", target.episode, journal), target.family
        elif action == "shots":
            episode, family = rewrite_shots(ctx, bible, target, journal), ctx.router.one("reviser").spec.family
        else:
            episode, family = revise(ctx, bible, target), ctx.router.one("reviser").spec.family
        last = action

    now = db.now()
    ctx.conn.execute("DELETE FROM episodes WHERE series_id = ? AND number = ? AND status = 'rejected'",
                     (series_id, number))
    status, reason, extra = "approved", "", {}
    ep = best.episode
    if not best.passes(bar):
        status = "rejected"
        reason = "; ".join(best.issues or [f"raise {c} ({s:g})" for c, s in best.scores.items() if s < bar] + best.flags)
    else:
        ep = lab(ctx, "hook", ep, journal)
        ep = lab(ctx, "cliffhanger", ep, journal)
        extra["audience"] = audience(ctx, ep, journal)
        flags: list[str] = []
        for attempt in range(2):
            checks = ctx.router.panel(lambda j: j.generate_json("drama_final_check", prompts.SYSTEM_EDITOR,
                                                                prompts.final_check(ep), schemas.FINAL_CHECK),
                                      _judges(ctx, [best.family], "quick_judges"))
            flags = sorted({f for c in checks for f, raised in c.get("flags", {}).items() if raised is True})
            journal.note("final_check", flags=flags)
            if not flags or attempt == 1:
                break
            fixed = revise(ctx, bible, Version(ep, best.family, flags=flags))
            if not episode_issues(fixed, bible):
                ep = fixed
        if flags:
            status, reason = "rejected", "final check: " + ", ".join(flags)
        else:
            try:
                extra["packaging"] = ctx.router.one("reviser").generate_json(
                    "drama_packaging", prompts.SYSTEM_WRITER, prompts.packaging(bible, ep, number), schemas.PACKAGING)
            except LLMError as exc:
                log.warning("Packaging skipped: %s", exc)
    review = {"scores": best.scores, "notes": best.note_lines()[:6], "reason": reason}
    cur = ctx.conn.execute(
        "INSERT INTO episodes (series_id, number, title, status, script, review, audience_report, packaging, "
        "production_log, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (series_id, number, ep.get("title", f"Episode {number}"), status, json.dumps(ep, ensure_ascii=False),
         json.dumps(review), json.dumps(extra.get("audience")), json.dumps(extra.get("packaging")),
         json.dumps(journal.entries), now, now))
    log.info("Episode %d of series #%d %s %s", number, series_id, status, reason)
    return int(cur.lastrowid), status

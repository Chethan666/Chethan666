"""Blind benchmark: do two model setups write better stories than each other?

Both setups write the same fixed pitches through the writers' room (plan contest,
competing drafts, style line edit, polish). Judges drawn from both setups
then compare the two scripts without knowing who wrote which, in both orders.
Use it before switching models, so an upgrade is proven rather than assumed.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Callable

from . import catalog, db, pipeline, prompts, schemas, tournament
from .llm import LLMError, ModelSpec
from .router import Router

log = logging.getLogger(__name__)

# Original pitches kept fixed so every benchmark run is comparable.
PITCHES: list[dict[str, Any]] = [
    {
        "title": "The Night Clerk's Ledger",
        "logline": "A new night clerk finds tonight's guest ledger already filled in, in her own handwriting.",
        "synopsis": (
            "On her first night at a roadside motel, Ines opens the guest ledger and finds every arrival "
            "for tonight already written down, in her handwriting, with times. The guests arrive exactly "
            "as listed. The last line, at 4:10, reads only: she stays. As the night repeats details she "
            "half remembers, Ines realizes she has worked this shift many times and the ledger is her "
            "own warning. She breaks the loop by writing a new last line, and walks out at dawn, leaving "
            "the pen for the next clerk."),
        "hook_line": "The ledger already had tonight's guests in it, and the handwriting was mine.",
        "thumbnail_concept": "A hand holding a pen over a ledger, one red line of ink",
        "why_it_works": "A concrete impossible object, a loop and a choice.",
        "recipe": {"setting": "a roadside motel with one working sign", "core_fear": "a night that keeps repeating",
                   "protagonist": "a newly hired hotel clerk", "format": "logbook",
                   "twist": "the loop has run many times before", "ending": "bittersweet"},
    },
    {
        "title": "Seat 14 Was Taken",
        "logline": "On an overnight train, a translator meets a woman who knows her whole life and wants her ticket.",
        "synopsis": (
            "Noor, a translator travelling alone on a sleeper train, finds a stranger in her berth who "
            "speaks her mother's dialect and knows details of Noor's childhood. The stranger is calm and "
            "certain: the ticket is hers. Through the night small things change, until the conductor "
            "greets the stranger by Noor's name. In the end we learn the stranger is the sister Noor "
            "was mistaken for at birth, come to claim the life she lost, and Noor steps off at a dark "
            "station alone."),
        "hook_line": "The woman in my berth said my name like she had been using it for years.",
        "thumbnail_concept": "Two identical silhouettes in a train corridor, one red ticket",
        "why_it_works": "Identity fear with a human wound underneath.",
        "recipe": {"setting": "a sleeper car on an overnight train", "core_fear": "being replaced by a copy",
                   "protagonist": "a translator travelling alone", "format": "classic",
                   "twist": "a case of mistaken identity", "ending": "twist"},
    },
    {
        "title": "Keep the Lamp for Me",
        "logline": "A lighthouse keeper keeps lighting the lamp for a son lost at sea, until someone answers.",
        "synopsis": (
            "Old Tomas has kept the lighthouse every night since his son's boat went down. One storm, a "
            "light flashes back from the sea in the family's old signal. Each night it comes closer, and "
            "the town says he is losing his mind. On the last night wet footprints climb the tower stairs "
            "and stop beside his chair. He realizes his son never left: he has been keeping the lamp lit "
            "for his father all along, and Tomas finally lets the light go out."),
        "hook_line": "Every night for nine years, Tomas lit the lamp, and every night the sea stayed dark.",
        "thumbnail_concept": "A lighthouse beam over black water, one red light answering",
        "why_it_works": "Grief, a signal and a heartbreaking reversal.",
        "recipe": {"setting": "a lighthouse on a remote coast", "core_fear": "losing someone you love",
                   "protagonist": "a lighthouse keeper", "format": "heartbreak_ghost",
                   "twist": "the one they were mourning never left", "ending": "heartbreak"},
    },
]

Factory = Callable[[dict, Any], Callable[[ModelSpec], Any]]


def _router(cfg: dict, conn: Any, factory: Factory | None) -> Router:
    return Router(cfg, conn, factory(cfg, conn) if factory else None)


def write(cfg: dict, pitch: dict[str, Any], factory: Factory | None = None) -> tuple[dict[str, Any] | None, float]:
    """One setup writes one pitch. Returns (script or None, estimated cost in USD)."""
    conn = db.connect(":memory:")
    ctx = pipeline.Context(conn=conn, cfg=cfg, router=_router(cfg, conn, factory), rng=random.Random(0))
    series = catalog.SERIES["hollow_world"]
    story = pipeline.Story(series=series, recipe=catalog.Recipe(**pitch["recipe"]),
                           pitch={k: v for k, v in pitch.items() if k != "recipe"}, legend=None, dossier=None,
                           sparks=[], world=db.world_entities(conn))
    try:
        story.bible, outline = pipeline.make_plan(ctx, story)
        script, _ = pipeline.draft_contest(ctx, story, outline)
        script = pipeline.fit_length(ctx, story, pipeline.clean_style(ctx, story, pipeline.fit_length(ctx, story, script)))
        script = pipeline.polish(ctx, story, script)
    except LLMError as exc:
        log.warning("Benchmark writer failed on %s: %s", pitch["title"], exc)
        script = None
    cost = conn.execute("SELECT COALESCE(SUM(usd), 0) FROM costs").fetchone()[0]
    return script, float(cost)


def run(cfg_a: dict, cfg_b: dict, count: int = 2, factory: Factory | None = None) -> dict[str, Any]:
    """Setup A and setup B each write `count` fixed pitches; judges from both compare them blind."""
    judge_labels = list(dict.fromkeys(cfg_a["roles"]["judges"] + cfg_b["roles"]["judges"]))
    judge_cfg = dict(cfg_a, roles=dict(cfg_a["roles"], judges=judge_labels))
    conn = db.connect(":memory:")
    judges = _router(judge_cfg, conn, factory).pool("judges")
    results = []
    for pitch in PITCHES[:count]:
        a, cost_a = write(cfg_a, pitch, factory)
        b, cost_b = write(cfg_b, pitch, factory)
        row = {"title": pitch["title"], "cost_a": round(cost_a, 3), "cost_b": round(cost_b, 3),
               "words_a": len(pipeline.narration_of(a).split()) if a else 0,
               "words_b": len(pipeline.narration_of(b).split()) if b else 0}
        if a is None or b is None:
            row.update(a_points=float(a is not None), b_points=float(b is not None))
        else:
            def compare(judge: Any, first: dict, second: dict) -> str:
                return judge.generate_json("compare_drafts", prompts.SYSTEM_EDITOR, prompts.compare_drafts(
                    "long", pitch, first, second), schemas.PAIRWISE)["winner"]

            ranked = tournament.rank([a, b], lambda *_: judges, compare, lambda fn, jobs: [fn(j) for j in jobs])
            points = {id(r.item): r.points for r in ranked}
            row.update(a_points=points[id(a)], b_points=points[id(b)])
        row["winner"] = "A" if row["a_points"] > row["b_points"] else "B" if row["b_points"] > row["a_points"] else "tie"
        results.append(row)
    return {"results": results, "judges": judge_labels,
            "a_wins": sum(r["winner"] == "A" for r in results),
            "b_wins": sum(r["winner"] == "B" for r in results),
            "ties": sum(r["winner"] == "tie" for r in results)}


def markdown(summary: dict[str, Any], name_a: str, name_b: str) -> str:
    lines = [f"## Benchmark: A = {name_a}, B = {name_b}", "",
             f"Judges (blind, both orders): {', '.join(summary['judges'])}", "",
             "| Pitch | A points | B points | Winner | A words | B words | A cost | B cost |",
             "|---|---|---|---|---|---|---|---|"]
    for r in summary["results"]:
        lines.append(f"| {r['title']} | {r['a_points']} | {r['b_points']} | {r['winner']} | {r['words_a']} | "
                     f"{r['words_b']} | ${r['cost_a']:.2f} | ${r['cost_b']:.2f} |")
    lines += ["", f"**A won {summary['a_wins']}, B won {summary['b_wins']}, ties {summary['ties']}.**"]
    return "\n".join(lines)

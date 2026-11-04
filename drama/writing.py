"""Writing a drama series: the series bible, then each episode's shot-by-shot script.

Micro-drama craft, spelled out for the models: a hook in the first three seconds, one
clear conflict per episode, a reversal, and a cliffhanger in the last line. Everything is
said or shown: characters speak their lines, a character's inner voice covers what we
can't see, and there is no narrator. Each script is judged by models from other companies
and revised once if it falls short.
"""

from __future__ import annotations

import json
import logging
import statistics
from typing import Any

from story_factory import db
from story_factory.llm import LLMError

log = logging.getLogger(__name__)


def obj(**props: Any) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def arr(items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": items}


S, I, B = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}

SHOTS = ("establishing", "wide", "medium", "close_up", "extreme_close_up", "over_shoulder", "insert", "two_shot")
EMOTIONS = ("calm", "tense", "afraid", "angry", "sad", "cold", "pleading", "whisper", "shocked", "warm", "mocking")

CHARACTER = obj(name=S, role=S, age=I, gender={"type": "string", "enum": ["female", "male"]},
                look=S, wardrobe=S, voice=S, want=S, secret=S)
BIBLE = obj(title=S, logline=S, genre=S, tone=S, world=S, hook=S, central_secret=S,
            cast=arr(CHARACTER), locations=arr(obj(name=S, look=S)), arc=S,
            episodes=arr(obj(number=I, title=S, hook=S, beats=arr(S), cliffhanger=S)))
SHOT = obj(shot={"type": "string", "enum": list(SHOTS)}, location=S, characters=arr(S), action=S,
           speaker=S, line=S, emotion={"type": "string", "enum": list(EMOTIONS)}, inner_voice=B, motion=B)
EPISODE = obj(title=S, hook_line=S, cliffhanger=S, shots=arr(SHOT))
REVIEW = obj(scores=obj(hook=I, conflict=I, dialogue=I, cliffhanger=I, continuity=I), problems=arr(S))

SYSTEM_WRITER = (
    "You are the head writer of a hit vertical short-drama series, the kind people binge on their "
    "phones one 90-second episode after another. You know exactly what keeps a thumb from "
    "scrolling: a hook in the first three seconds, characters who want something badly and hide "
    "something worse, a reversal in every episode, and a cliffhanger in the last line that makes "
    "the next episode impossible to skip. Dialogue is short, sharp and spoken: people interrupt, "
    "lie, threaten, confess. Everything is said or shown; there is no narrator. A character's "
    "inner voice may carry one or two thoughts we can't see. Original work only: never copy an "
    "existing show, film or book. No sexual content, no gore, no harm to children, no real people.")

SYSTEM_EDITOR = (
    "You are a demanding development executive for a vertical short-drama studio. You have "
    "watched thousands of episodes and you know what makes people pay to unlock the next one. "
    "Score honestly: a competent episode is 6, an 8 needs a reason you can name.")


def bible_prompt(genre: str, episodes: int, avoid: list[str]) -> str:
    seen = ("\nSeries we already made (be clearly different):\n" + "\n".join(f"- {t}" for t in avoid)) if avoid else ""
    return (f"Create a new original {genre} short-drama series of {episodes} episodes, each 60-120 "
            "seconds, vertical, for a global English-speaking audience.\n"
            "title; logline (one sentence that makes people click); genre; tone; world (where and when); "
            "hook (the first image of episode 1); central_secret (what the audience slowly learns).\n"
            "cast: 3-6 characters. For each: name (invented, not a famous person's), role, age, gender, "
            "look (a precise visual description an illustrator could repeat in every shot: face, hair, "
            "build, skin tone, distinctive features), wardrobe (their signature clothes), voice (how they "
            "sound), want, secret.\n"
            "locations: 3-6 recurring places, each with a precise look.\n"
            f"arc: how the series turns at the start, the middle and the end.\n"
            f"episodes: all {episodes} episodes, each with title, hook (first three seconds), 3-5 beats "
            f"and the cliffhanger. Every episode ends on a new question.{seen}")


def episode_prompt(bible: dict[str, Any], number: int, previous: list[dict[str, Any]], seconds: int) -> str:
    outline = next((e for e in bible.get("episodes", []) if int(e.get("number", 0)) == number), {})
    recap = "\n".join(f"Episode {p['number']}: {p['title']} - ended on: {p['cliffhanger']}" for p in previous[-3:])
    cast = ", ".join(c["name"] for c in bible.get("cast", []))
    places = ", ".join(loc["name"] for loc in bible.get("locations", []))
    return (f"SERIES BIBLE:\n{json.dumps(bible, ensure_ascii=False)}\n\nStory so far:\n{recap or '(first episode)'}\n\n"
            f"Write episode {number}: {json.dumps(outline, ensure_ascii=False)}\n\n"
            f"About {seconds} seconds of screen time: 14-26 shots. Each shot is one camera setup: "
            f"shot (establishing, wide, medium, close_up, extreme_close_up, over_shoulder, insert, two_shot); "
            f"location (one of: {places}); characters in frame (from: {cast}); action (what we see, in "
            "one vivid sentence a cinematographer could shoot); speaker and line (the one line spoken in "
            "this shot, at most 20 words; empty speaker and line for a silent shot); emotion of the line; "
            "inner_voice true if the line is the speaker's unspoken thought; motion true for at most four "
            "shots where movement is the point (a slap, a door bursting open, a reveal). The first shot is "
            "the hook. Cut between speakers like TV: close-ups and over-the-shoulder shots in arguments, "
            "inserts on the object that matters. The last shot is the cliffhanger. hook_line and "
            "cliffhanger repeat the first and last spoken lines.")


def review_prompt(bible: dict[str, Any], episode: dict[str, Any]) -> str:
    return (f"Series: {bible.get('title')} - {bible.get('logline')}\n\nEPISODE:\n{json.dumps(episode, ensure_ascii=False)}\n\n"
            "Score 1-10: hook (are the first three seconds impossible to scroll past?), conflict (does "
            "something change, is there a reversal?), dialogue (short, sharp, in character, no "
            "exposition dumps?), cliffhanger (must the viewer see the next episode?), continuity (fits "
            "the bible and the story so far, characters and places consistent?). List the specific "
            "problems that would most raise the lowest scores.")


def episode_issues(episode: dict[str, Any], bible: dict[str, Any]) -> list[str]:
    """Free checks: enough shots, known characters and places, the hook and the cliffhanger spoken."""
    shots = episode.get("shots") or []
    issues = []
    if not 10 <= len(shots) <= 32:
        issues.append(f"{len(shots)} shots; write 14-26")
    names = {c["name"].lower() for c in bible.get("cast", [])}
    places = {loc["name"].lower() for loc in bible.get("locations", [])}
    strangers = {c for s in shots for c in s.get("characters", []) if c.lower() not in names}
    if strangers:
        issues.append("characters not in the cast: " + ", ".join(sorted(strangers)))
    speakers = {s.get("speaker", "") for s in shots if s.get("line")} - {""}
    if any(sp.lower() not in names for sp in speakers):
        issues.append("every speaker must be a cast member")
    unknown_places = {s.get("location", "") for s in shots if s.get("location", "").lower() not in places}
    if unknown_places:
        issues.append("use only the bible's locations; unknown: " + ", ".join(sorted(unknown_places)))
    spoken = [s for s in shots if str(s.get("line", "")).strip()]
    if len(spoken) < len(shots) * 0.5:
        issues.append("this is a dialogue drama: at least half the shots carry a spoken line")
    if sum(1 for s in shots if s.get("motion")) > 4:
        issues.append("mark at most four shots as motion")
    if any(len(str(s.get("line", "")).split()) > 24 for s in shots):
        issues.append("keep every line under 20 words")
    return issues


def _judged(ctx: Any, task: str, prompt: str, schema: dict, author_family: str) -> list[dict[str, Any]]:
    return ctx.router.panel(lambda judge: judge.generate_json(task, SYSTEM_EDITOR, prompt, schema),
                            ctx.router.judges([author_family], ctx.cfg["quality"]["quick_judges"]))


def create_series(ctx: Any, genre: str, episodes: int) -> int:
    """Write and store a new series bible. Returns the series id."""
    writer = ctx.router.one("writers")
    avoid = [r["title"] for r in ctx.conn.execute("SELECT title FROM series ORDER BY id DESC LIMIT 20")]
    bible = writer.generate_json("drama_bible", SYSTEM_WRITER, bible_prompt(genre, episodes, avoid), BIBLE)
    if len(bible.get("cast") or []) < 2 or len(bible.get("episodes") or []) < min(episodes, 3):
        raise LLMError("the series bible came back without a cast or episode outlines")
    now = db.now()
    cur = ctx.conn.execute(
        "INSERT INTO series (title, genre, status, bible, created_at, updated_at) VALUES (?, ?, 'active', ?, ?, ?)",
        (bible["title"].strip(), genre, json.dumps(bible, ensure_ascii=False), now, now))
    log.info("Series #%d created: %s (%d episodes, %d characters)", cur.lastrowid, bible["title"],
             len(bible["episodes"]), len(bible["cast"]))
    return int(cur.lastrowid)


def write_episode(ctx: Any, series_id: int, number: int | None = None) -> tuple[int, str]:
    """Write, judge and store the next episode. Returns (episode id, status)."""
    row = ctx.conn.execute("SELECT * FROM series WHERE id = ?", (series_id,)).fetchone()
    if row is None:
        raise ValueError(f"no series #{series_id}")
    bible = json.loads(row["bible"])
    done = [dict(r) for r in ctx.conn.execute(
        "SELECT number, title, script FROM episodes WHERE series_id = ? AND status != 'rejected' ORDER BY number",
        (series_id,))]
    number = number or (max((d["number"] for d in done), default=0) + 1)
    previous = [{"number": d["number"], "title": d["title"], "cliffhanger": json.loads(d["script"]).get("cliffhanger", "")}
                for d in done if d["number"] < number]
    q = ctx.cfg.get("drama", {})
    seconds = int(q.get("episode_seconds", 90))
    writer = ctx.router.one("writers")
    episode = writer.generate_json("drama_episode", SYSTEM_WRITER, episode_prompt(bible, number, previous, seconds),
                                   EPISODE)
    best, best_score, review = episode, -1.0, {}
    for attempt in range(2):
        problems = episode_issues(episode, bible)
        score = 0.0
        if not problems:
            reviews = _judged(ctx, "drama_review", review_prompt(bible, episode), REVIEW, writer.spec.family)
            scores = {k: statistics.median(int(r["scores"].get(k, 0)) for r in reviews)
                      for k in REVIEW["properties"]["scores"]["properties"]}
            score = min(scores.values())
            review = {"scores": scores, "problems": [p for r in reviews for p in r.get("problems", [])][:8]}
            problems = review["problems"] if score < int(q.get("min_score", 7)) else []
        if score > best_score:
            best, best_score = episode, score
        log.info("Episode %d round %d: %s", number, attempt + 1, review.get("scores") or problems[:3])
        if not problems or attempt == 1:
            break
        episode = writer.generate_json("drama_revise", SYSTEM_WRITER, episode_prompt(bible, number, previous, seconds)
                                       + "\n\nA previous draft had these problems; fix them all:\n"
                                       + "\n".join(f"- {p}" for p in problems), EPISODE)
    status = "approved" if best_score >= int(q.get("min_score", 7)) else "rejected"
    now = db.now()
    ctx.conn.execute("DELETE FROM episodes WHERE series_id = ? AND number = ? AND status = 'rejected'", (series_id, number))
    cur = ctx.conn.execute(
        "INSERT INTO episodes (series_id, number, title, status, script, review, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (series_id, number, best.get("title", f"Episode {number}"), status, json.dumps(best, ensure_ascii=False),
         json.dumps(review), now, now))
    log.info("Episode %d of series #%d %s (lowest score %s)", number, series_id, status, best_score)
    return int(cur.lastrowid), status

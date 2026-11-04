"""Prompts for the drama writers' room (pipeline 3)."""

from __future__ import annotations

import json
from typing import Any, Sequence

RULES = ("Original work only: never copy an existing show, film or book. No sexual content, no gore "
         "or lingering injury, no harm to children, no real people, no hate. Violence is implied and brief.")

CRAFT = """\
What makes a vertical short drama impossible to stop watching:
1. The first three seconds: a shocking line or image that poses a question (a slap, a ring on the
   wrong hand, "You married a dead man").
2. One clear conflict per episode and at least one reversal: someone gains or loses power.
3. Characters who want something badly and hide something worse; every secret is a bomb with a timer.
4. Dialogue that is short, spoken and sharp: people interrupt, lie, threaten, confess. No exposition
   dumps; show it in a look or an object.
5. TV grammar: establish the place, then cut between speakers in close-ups and over-the-shoulder
   shots; inserts on the object that matters; a wide shot when someone enters or leaves.
6. The last line is a cliffhanger that makes the next episode impossible to skip: a new question,
   never an answer.
7. Everything is said or shown. There is no narrator; a character's inner voice may carry one or two
   thoughts we can't see.
"""

SYSTEM_WRITER = ("You are the head writer of a hit vertical short-drama series, the kind people binge "
                 "one 90-second episode after another on their phones.\n\n" + CRAFT + "\n" + RULES)
SYSTEM_EDITOR = ("You are a demanding development executive at a vertical short-drama studio. You have "
                 "watched thousands of episodes and know exactly what makes people unlock the next one. "
                 "Score honestly: a competent episode is 6, an 8 needs a reason you can name.\n\n" + CRAFT)


def system_viewer(persona: str) -> str:
    return (f"You are {persona}, scrolling short dramas on your phone. You swipe away the moment you're "
            "bored and you pay to unlock the next episode only when you must know what happens.")


def pitches(genre: str, recipe: str, count: int, avoid: Sequence[str], learnings: Sequence[str] = ()) -> str:
    seen = ("\nSeries we already made or considered (be clearly different):\n" + "\n".join(f"- {t}" for t in avoid)
            if avoid else "")
    return (f"Pitch {count} original {genre} short-drama series built on this recipe: {recipe}.\n"
            "For each: title (short, intriguing), logline (one sentence that makes people tap), hook (the "
            "first three seconds of episode 1), central_secret (what the audience slowly learns), and "
            f"why_bingeable (why viewers will unlock episode after episode).{seen}")


def score_pitch(genre: str, pitch: dict[str, Any]) -> str:
    return (f"A {genre} short-drama pitch:\n{json.dumps(pitch, ensure_ascii=False)}\n\nScore 1-10: hook (would a "
            "stranger stop scrolling?), bingeability (does it promise many reversals and cliffhangers?), "
            "originality, emotional_core (a real human want or wound, not only shocks), clarity (understood "
            "in one sentence). Then a one-line verdict.")


def compare_pitches(a: dict[str, Any], b: dict[str, Any]) -> str:
    return (f"PITCH A:\n{json.dumps(a, ensure_ascii=False)}\n\nPITCH B:\n{json.dumps(b, ensure_ascii=False)}\n\n"
            "Which series would more viewers binge to the last episode? Reason briefly, then choose.")


def bible(genre: str, pitch: dict[str, Any], episodes: int) -> str:
    return (f"Approved pitch:\n{json.dumps(pitch, ensure_ascii=False)}\n\nWrite the series bible for {episodes} "
            f"episodes of 60-120 seconds, {genre}, vertical, for a global English-speaking audience.\n"
            "title, logline, genre, tone, world (where and when), hook (the first image of episode 1), "
            "central_secret.\ncast: 3-6 characters. For each: name (invented, not a famous person's), role, "
            "age, gender, look (a precise visual description an illustrator could repeat in every shot: face, "
            "hair, build, skin tone, distinctive features), wardrobe (signature clothes), voice (how they "
            "sound), want, secret.\nlocations: 3-6 recurring places, each with a precise look.\narc: how the "
            f"series turns at the start, the middle and the end.\nepisodes: all {episodes}, each with title, "
            "hook (its first three seconds), 3-5 beats and a cliffhanger; power shifts back and forth, and "
            "every cliffhanger is a new question.")


def compare_bibles(a: dict[str, Any], b: dict[str, Any]) -> str:
    return (f"BIBLE A:\n{json.dumps(a, ensure_ascii=False)}\n\nBIBLE B:\n{json.dumps(b, ensure_ascii=False)}\n\n"
            "Which series plan will make the more bingeable show: sharper characters and secrets, stronger "
            "reversals, better cliffhangers? Reason briefly, then choose.")


def review_bible(b: dict[str, Any]) -> str:
    return (f"Series bible:\n{json.dumps(b, ensure_ascii=False)}\n\nReview it. Approve only if every character "
            "wants something and hides something that drives the plot; the secrets detonate at the right "
            "moments; power shifts in every episode; every cliffhanger is a new question; the arc pays off; "
            "looks and places are precise enough to draw the same way every time. Score the plan 1-10 (7 "
            "solid, 8 exceptional) and list at most five problems, most important first.")


def revise_bible(b: dict[str, Any], problems: list[str]) -> str:
    return (f"Series bible:\n{json.dumps(b, ensure_ascii=False)}\n\nProblems:\n" + "\n".join(f"- {p}" for p in problems)
            + "\n\nRevise the bible to fix every problem while keeping what works. Return the whole bible.")


def episode(b: dict[str, Any], number: int, previous: list[dict[str, Any]], seconds: int,
            avoid: Sequence[str] = ()) -> str:
    outline = next((e for e in b.get("episodes", []) if int(e.get("number", 0)) == number), {})
    recap = "\n".join(f"Episode {p['number']}: {p['title']} - ended on: {p['cliffhanger']}" for p in previous[-3:])
    cast = ", ".join(c["name"] for c in b.get("cast", []))
    places = ", ".join(loc["name"] for loc in b.get("locations", []))
    redo = ("\nAn earlier draft was turned down. Write a fresh one that avoids its problems:\n"
            + "\n".join(f"- {n}" for n in avoid)) if avoid else ""
    return (f"SERIES BIBLE:\n{json.dumps(b, ensure_ascii=False)}\n\nStory so far:\n{recap or '(first episode)'}\n\n"
            f"Write episode {number}: {json.dumps(outline, ensure_ascii=False)}{redo}\n\n"
            f"About {seconds} seconds of screen time: 24-32 shots of 2-5 seconds each, numbered by their order, "
            "most of them carrying a line, the way a TV scene cuts between speakers and reactions. Every shot is "
            "filmed as a real moving clip, so each action must show visible movement (a turn, a step, a hand "
            "reaching, a door, breath, a flicker of light) - never a frozen pose. Each shot is one camera setup: "
            "shot (establishing, wide, medium, close_up, extreme_close_up, over_shoulder, insert, two_shot); "
            f"location (one of: {places}); characters in frame (from: {cast}); action (what we see move, one vivid "
            "sentence a cinematographer could shoot); speaker and line (the one line spoken in this shot, 4-20 "
            "words; empty for a silent shot); emotion; inner_voice true if the line is an unspoken thought; "
            "motion true for the shots where big movement is the point (they are filmed first). The first shot "
            "is the hook; the last is the cliffhanger. hook_line and cliffhanger repeat the first and last "
            "spoken lines.")


def compare_episodes(a: dict[str, Any], b: dict[str, Any]) -> str:
    return (f"EPISODE A:\n{json.dumps(a, ensure_ascii=False)}\n\nEPISODE B:\n{json.dumps(b, ensure_ascii=False)}\n\n"
            "Which episode keeps a viewer from swiping and makes them unlock the next one? Reason briefly, "
            "then choose.")


def review_episode(b: dict[str, Any], ep: dict[str, Any]) -> str:
    numbered = [dict(s, number=i + 1) for i, s in enumerate(ep.get("shots", []))]
    return (f"Series: {b.get('title')} - {b.get('logline')}\nCast: {json.dumps(b.get('cast', []), ensure_ascii=False)}\n\n"
            f"EPISODE (shots numbered from 1):\n{json.dumps(dict(ep, shots=numbered), ensure_ascii=False)}\n\n"
            "Score 1-10: hook (first three seconds), conflict (something changes; a reversal), character (wants "
            "and secrets drive the choices), dialogue (short, sharp, in character), cliffhanger (the next episode "
            "is a must), continuity (fits the bible). Flag rule violations. Then up to six notes on the shots "
            "that would most raise the lowest scores: shot number, problem, fix (ideally the rewritten line).")


def rewrite_shots(b: dict[str, Any], ep: dict[str, Any], notes: dict[int, list[str]]) -> str:
    numbered = [dict(s, number=i + 1) for i, s in enumerate(ep.get("shots", []))]
    asks = "\n".join(f"[{n}] " + " | ".join(v) for n, v in sorted(notes.items()))
    return (f"Cast: {json.dumps(b.get('cast', []), ensure_ascii=False)}\n\nEPISODE:\n"
            f"{json.dumps(dict(ep, shots=numbered), ensure_ascii=False)}\n\nEditors' notes:\n{asks}\n\nRewrite only "
            "these shots to fix their notes; every other shot stays as it is, so each rewritten shot must still "
            "connect to the shots around it. Return each rewritten shot with its number.")


def revise_episode(b: dict[str, Any], ep: dict[str, Any], notes: list[str]) -> str:
    return (f"SERIES BIBLE:\n{json.dumps(b, ensure_ascii=False)}\n\nEPISODE:\n{json.dumps(ep, ensure_ascii=False)}\n\n"
            "Notes:\n" + "\n".join(f"- {n}" for n in notes) + "\n\nRevise the episode to fix every note while "
            "keeping what works. Return the whole episode.")


def alternatives(ep: dict[str, Any], part: str, count: int) -> str:
    which = "opening (the first two shots)" if part == "hook" else "ending (the last two shots)"
    shots = ep["shots"][:2] if part == "hook" else ep["shots"][-2:]
    goal = ("must stop a scrolling thumb within three seconds" if part == "hook"
            else "must end on a new question that makes the next episode impossible to skip")
    return (f"EPISODE:\n{json.dumps(ep, ensure_ascii=False)}\n\nThe {which} currently is:\n"
            f"{json.dumps(shots, ensure_ascii=False)}\n\nWrite {count} alternative versions of exactly these two "
            f"shots. Each {goal}, keeps the same characters and places, and flows with the rest of the "
            "episode. Make them genuinely different approaches.")


def compare_parts(part: str, a: list[dict[str, Any]], b: list[dict[str, Any]], context: str) -> str:
    goal = "stops a viewer scrolling past" if part == "hook" else "makes a viewer unlock the next episode"
    return (f"{context}\n\nOPTION A:\n{json.dumps(a, ensure_ascii=False)}\n\nOPTION B:\n{json.dumps(b, ensure_ascii=False)}"
            f"\n\nWhich {goal}? Reason briefly, then choose.")


def viewer(ep: dict[str, Any]) -> str:
    numbered = [{"shot": i + 1, "see": s.get("action", ""), "hear": (f"{s.get('speaker')}: {s.get('line')}"
                                                                     if s.get("line") else "")}
                for i, s in enumerate(ep.get("shots", []))]
    return (f"Watch this episode shot by shot:\n{json.dumps(numbered, ensure_ascii=False)}\n\nFor every shot: "
            "your interest 1-10, whether you would swipe away there, and a short honest note. Keep rating "
            "after a swipe. Then unlock_next 1-10 (how badly you need the next episode) and one line overall.")


def final_check(ep: dict[str, Any]) -> str:
    return (f"EPISODE:\n{json.dumps(ep, ensure_ascii=False)}\n\nFinal compliance check before publishing: {RULES} "
            "Flag any violation and list anything that would offend or stereotype a group. Empty means clear.")


def packaging(b: dict[str, Any], ep: dict[str, Any], number: int) -> str:
    return (f"Series: {b.get('title')} - {b.get('logline')}\nEpisode {number}:\n{json.dumps(ep, ensure_ascii=False)}\n\n"
            "Package this episode for TikTok, YouTube Shorts and Reels: title (at most 60 characters, "
            "curiosity without spoilers, starting with the series title and the episode number), caption "
            "(one or two short sentences ending on a question), hashtags (8-12), pinned_comment (a question "
            "that starts arguments about what happens next).")

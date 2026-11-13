"""The team's verifiers before and after the work.

- ScriptCheckAgent (first): the script is in a state to be produced: every scene and pause
  is found in the narration, quotes are closed, and the narration's length suits the plan.
  Code only, free; problems are reported, and the ones that would break timing stop the video.
- ArtReviewAgent (after the illustrator, before the edit): the pictures as one sequence. Each
  picture was checked alone by the critic; the vision model now sees them all together with
  the narration and flags what only shows side by side (a person who changes, a picture that
  breaks the story). The worst few are redrawn with its note. About $0.002 plus any redraw.
- thumbnail_ok (used by the packager): the thumbnail's words are spelled right and readable.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from production.media import MediaError, data_uri

from .. import package
from ..paid import OverBudget, _small
from ..run import _json
from .board import Board

log = logging.getLogger(__name__)

WORDS_PER_SECOND = 2.5  # the narrator's natural speed before voice_pace (about 150 words a minute)


class ScriptCheckAgent:
    name = "script_check"
    critical = True  # only for a script that cannot be timed at all; everything else is a warning

    def run(self, board: Board) -> dict[str, Any]:
        story, cfg = board.story, board.cfg
        narration = str(story.get("narration") or "").strip()
        scenes = [s for s in _json(story.get("scenes"), []) if isinstance(s, dict)]
        if not narration or not scenes:
            raise ValueError(f"story #{story.get('id')} has no narration or scene plan")
        issues: list[str] = []
        cursor = 0
        for i, sc in enumerate(scenes, 1):
            text = str(sc.get("text", "")).strip()
            at = narration.find(text, cursor) if text else -1
            if at < 0:
                issues.append(f"scene {i}'s text is not in the narration in order (it is placed by word count)")
            else:
                cursor = at + len(text)
            if not str(sc.get("visual", "")).strip():
                issues.append(f"scene {i} has no visual")
        for p in _json(story.get("pauses"), []):
            after = str((p or {}).get("after", "")).strip() if isinstance(p, dict) else ""
            if after and after not in narration:
                issues.append(f"pause after {after[:40]!r}: phrase not in the narration")
        if (narration.count('"') + narration.count("“") + narration.count("”")) % 2:
            issues.append("a quotation mark is not closed (a character's line may be read by the narrator)")
        words = len(narration.split())
        pace = float(cfg.get("voice_pace", 1.0)) or 1.0
        seconds = words / (WORDS_PER_SECOND * pace)
        target = float(cfg.get("target_seconds", 0) or 0)
        if target and not 0.8 * target <= seconds <= 1.2 * target:
            issues.append(f"the narration reads in about {seconds:.0f} s; the plan is {target:.0f} s "
                          f"(about {int(target * WORDS_PER_SECOND * pace)} words)")
        board.data["script_issues"] = issues
        return {"words": words, "estimated_seconds": round(seconds, 1), "target_seconds": target or None,
                "scenes": len(scenes), "issues": issues}


def ask_vision(studio: Any, prompt: str, image: Path, task: str, system: str, width: int = 1536) -> dict[str, Any] | None:
    """One look by the vision model; its JSON verdict, or None when there is no model, budget or answer."""
    if studio is None or "vision" not in studio.models or not image.exists():
        return None
    model = studio.models["vision"]
    try:
        usd = studio.budget.reserve(model)
    except OverBudget:
        return None
    try:
        result = studio.fal.run(model, dict(studio.args.get("vision", {}), prompt=prompt, system_prompt=system,
                                            image_urls=[data_uri(_small(image, width))]))
    except MediaError as exc:
        studio.budget.settle(usd, task, model, ok=False)
        log.warning("%s: vision model unavailable: %s", task, str(exc)[:200])
        return None
    studio.budget.settle(usd, task, model, ok=True)
    text = str(result.get("output") or result.get("text") or result)
    match = re.search(r"\{.*\}", text, re.S)
    try:
        verdict = json.loads(match.group(0)) if match else None
    except ValueError:
        return None
    return verdict if isinstance(verdict, dict) else None


SYSTEM = ("You are the art director of Hollow Lines, a horror storytelling channel, looking at all the pictures "
          "of one video side by side before it is edited. Be specific and brief. Answer with JSON only.")


class ArtReviewAgent:
    """The pictures as a sequence: each was already checked alone; this checks them together
    (the same person looks the same, the pictures tell the story in order, the scare lands) and
    has the worst few redrawn with a note on what to correct. The critic checks every redraw."""

    name = "art_review"
    critical = False

    def run(self, board: Board) -> dict[str, Any]:
        d, cfg = board.data, board.cfg
        beats = d.get("beats", [])
        fixes_allowed = int(cfg.get("art_review_fixes", 2))
        sheet = board.work / "review_sheet.jpg"
        package.contact_sheet(beats, d["pictures"], {}, sheet, columns=4, thumb=(480, 270))
        lines = "\n".join(f"{k}. {str(b.text)[:140]}" for k, b in enumerate(beats, 1))
        cast = "; ".join(f"{c.get('name')}: {str(c.get('design', ''))[:160]}" for c in d.get("cast", [])
                         if isinstance(c, dict) and c.get("design"))
        prompt = (f"Story: {board.story.get('title', '')}\nPeople: {cast or 'none named'}\n"
                  "The sheet shows the video's pictures in order, numbered, left to right, top to bottom. Each is "
                  f"on screen while its line is narrated:\n{lines}\n\n"
                  "Judge the pictures TOGETHER: does each named person look the same in every picture (face, hair, "
                  "age, clothes); do the pictures follow the story in order; does a picture show something its line "
                  "does not say, or contradict an earlier one; is anything broken (warped faces or hands, text, "
                  "panels); is the scare picture strong; do they all look like one dark horror film? List only "
                  "clear problems worth a redraw, worst first. Reply exactly as JSON: "
                  '{"score": 1-10, "fix": [{"picture": N, "problem": "what is wrong and what it should show"}], '
                  '"strongest": N}')
        verdict = ask_vision(board.studio, prompt, sheet, "art_review", SYSTEM)
        if verdict is None:
            return {"reviewed": False}
        try:
            score: int | None = max(1, min(10, int(verdict.get("score"))))
        except (TypeError, ValueError):
            score = None  # no readable score: only its list of fixes counts
        wanted = []
        for f in verdict.get("fix", []) if isinstance(verdict.get("fix"), list) else []:
            try:
                k, problem = int(f.get("picture")), str(f.get("problem", "")).strip()
            except (AttributeError, TypeError, ValueError):
                continue
            if 1 <= k <= len(beats) and problem and k not in [w[0] for w in wanted]:
                wanted.append((k, problem))
        redrawn = []
        repaint = d.get("repaint")
        for k, problem in wanted[:fixes_allowed] if repaint is not None else []:
            beat = beats[k - 1]
            try:
                path = repaint(beat, problem)
            except Exception as exc:  # a failed redraw keeps the first picture
                log.warning("Redraw of picture %d failed: %s", k, str(exc)[:200])
                path = None
            if path is not None:
                d["pictures"][beat.index] = path
                redrawn.append({"picture": k, "problem": problem[:200]})
        return {"reviewed": True, "score": score, "flagged": [{"picture": k, "problem": p[:200]} for k, p in wanted],
                "redrawn": redrawn, "strongest_picture": verdict.get("strongest")}


THUMB_SYSTEM = "You check YouTube thumbnails for a horror channel. Answer with JSON only."


def thumbnail_ok(studio: Any, image: Path, words: str) -> tuple[bool, str]:
    """Do the thumbnail's words read exactly as planned, large and clear, with nothing else written?"""
    want = f'the words "{words.upper()}" spelled exactly, large and readable at phone size' if words else "no text"
    verdict = ask_vision(studio, f"This thumbnail should have {want}, nothing else written, a dark horror image "
                                 "with one clear focal point and no warped faces or hands. Reply exactly as JSON: "
                                 '{"ok": true or false, "reason": "short reason"}', image, "thumbnail_check",
                         THUMB_SYSTEM, width=1024)
    if verdict is None or "ok" not in verdict:
        return True, "not checked"
    return bool(verdict.get("ok")), str(verdict.get("reason", ""))[:200]

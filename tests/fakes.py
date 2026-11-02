"""Deterministic stand-ins for the model providers, so the pipeline is tested offline."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import deque
from typing import Any, Callable

from story_factory import config, db, dedup, pipeline
from story_factory.llm import LLMError, ModelSpec
from story_factory.router import Router


def test_config() -> dict:
    cfg = copy.deepcopy(config.DEFAULTS)
    cfg["database"] = ":memory:"
    cfg["roles"] = {
        "pitchers": ["anthropic:writer-a", "openai:writer-b", "google:writer-c"],
        "judges": ["anthropic:judge-a", "openai:judge-b", "google:judge-c", "xai:judge-d"],
        "writers": ["anthropic:writer-a", "openai:writer-b"],
        "reviser": ["anthropic:writer-a"],
        "audience": ["openai:viewer-b", "google:viewer-c"],
        "polisher": ["anthropic:writer-a"],
        "researcher": ["anthropic:judge-a"],
        "planner": ["anthropic:judge-a"],
        "embedder": ["google:embed"],
    }
    cfg["routing"]["parallel_calls"] = 4
    cfg["lengths"] = {"long_min_words": 150, "long_max_words": 320,
                      "short_min_words": 40, "short_max_words": 110}
    cfg["quality"].update(pitches_per_pitcher=2, max_revisions=2, audience_chunk_words=40,
                          audience_personas=["viewer one", "viewer two"])
    return cfg


def unique_words(seed: str, count: int) -> list[str]:
    digest = hashlib.sha256(seed.encode()).hexdigest()
    return [f"{seed[:2]}{digest[i % 60:i % 60 + 4]}{i}" for i in range(count)]


def premise(n: int, family: str = "x") -> dict[str, str]:
    tag = hashlib.md5(f"{n}{family}".encode()).hexdigest()[:10]
    return {"title": f"Tale {tag} by {family}", "logline": " ".join(unique_words(f"l{n}{family}", 12)) + ".",
            "synopsis": " ".join(unique_words(f"s{n}{family}", 90)) + ".",
            "hook_line": "The bus had one passenger too many.", "thumbnail_concept": "A red lantern",
            "why_it_works": "It opens a question."}


def _between(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


class Brain:
    """Shared state and behavior behind every fake client."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []  # (task, model label)
        self.violations: list[str] = []
        self.counter = 0
        self.pitch_queue: deque[list[dict]] = deque()
        self.review_scores = 9
        self.pitch_scores: Callable[[str], int] = lambda title: 8
        self.judge_same = False
        self.audience_interest: Callable[[int], int] = lambda chunk: 9
        self.final_issues: deque[list[str]] = deque()
        self.overrides: dict[str, Callable[[str, ModelSpec], dict]] = {}
        self.failing: set[str] = set()  # model labels that raise
        self.prefer_opening: str | None = None  # text the hook-lab judges prefer
        self.prefer_title: str | None = None
        self.cast_override: Callable[[list[str]], list[dict]] | None = None

    def tasks(self, name: str) -> list[str]:
        return [label for task, label in self.calls if task == name]

    def call(self, spec: ModelSpec, task: str, system: str, prompt: str) -> dict[str, Any]:
        self.calls.append((task, spec.label))
        if spec.label in self.failing:
            raise LLMError(f"{spec.label} is down")
        if task in self.overrides:
            return self.overrides[task](prompt, spec)
        return getattr(self, f"_{task}")(prompt, spec)

    def make_script(self, long: bool, marker: str = "") -> dict[str, Any]:
        self.counter += 1
        n = self.counter
        sentences = 30 if long else 8
        body = " ".join(" ".join(unique_words(f"b{n}x{i}", 6)).capitalize() + "." for i in range(sentences))
        if marker:
            body = body.replace(".", f" {marker}.", 1)
        question = "Would you have opened the door?"
        return {"intro": "I found this one in the hollow beside a quiet road." if long else "",
                "body": body, "outro": f"Some doors are better left closed. {question}",
                "comment_question": question, "hook_line": body.split(".")[0] + ".",
                "thumbnail_idea": "A red lantern in an empty doorway",
                "description_credit": "This is an original work of fiction.",
                "new_entities": [{"kind": "place", "name": f"Exit {n}", "description": "A road sign that moves."}],
                "referenced_entities": ["The red lantern"]}

    # --- Task handlers ------------------------------------------------------

    def _pitches(self, prompt: str, spec: ModelSpec) -> dict:
        if self.pitch_queue:
            return {"pitches": self.pitch_queue.popleft()}
        self.counter += 1
        return {"pitches": [premise(self.counter * 10 + i, spec.family) for i in range(2)]}

    def _score_pitch(self, prompt: str, spec: ModelSpec) -> dict:
        title = _between(prompt, "Title: ", "\n")
        if f"by {spec.family}" in title:
            self.violations.append(f"{spec.label} scored its own family's pitch")
        value = self.pitch_scores(title)
        return {"scores": {c: value for c in ("click_appeal", "hook", "originality", "emotional_core",
                                             "series_fit", "talkability")}, "verdict": "fine"}

    def _compare_pitches(self, prompt: str, spec: ModelSpec) -> dict:
        a = _between(prompt, "PITCH A:\nTitle: ", "\n")
        b = _between(prompt, "PITCH B:\nTitle: ", "\n")
        for title in (a, b):
            if f"by {spec.family}" in title:
                self.violations.append(f"{spec.label} judged its own family's pitch")
        return {"reason": "higher score", "winner": "A" if self.pitch_scores(a) >= self.pitch_scores(b) else "B"}

    def _compare_drafts(self, prompt: str, spec: ModelSpec) -> dict:
        a = _between(prompt, "DRAFT A:\n", "\n\nDRAFT B:")
        winner = "A" if "PREFERRED" in a else "B" if "PREFERRED" in prompt else "A"
        return {"reason": "grips harder", "winner": winner}

    def _judge_duplicate(self, prompt: str, spec: ModelSpec) -> dict:
        return {"reason": "fake judge", "same_story": self.judge_same}

    def _dossier(self, prompt: str, spec: ModelSpec) -> dict:
        return {"summary": "A spirit of the roads.", "core_elements": ["walks at night"],
                "regional_variants": [], "cultural_notes": [], "avoid": [], "sources": ["https://example.org"]}

    def _beat_sheet(self, prompt: str, spec: ModelSpec) -> dict:
        return {"central_question": "Who is the extra passenger?", "emotional_core": "loneliness",
                "beats": [{"name": "hook", "purpose": "question", "summary": "x", "target_seconds": 10,
                           "tension": 3, "plants": ["ticket"], "payoffs": []}]}

    _revise_outline = _beat_sheet

    def _review_outline(self, prompt: str, spec: ModelSpec) -> dict:
        return {"problems": [], "approved": True}

    def _write_draft(self, prompt: str, spec: ModelSpec) -> dict:
        return self.make_script("150-320" in prompt, marker="PREFERRED" if spec.model == "writer-b" else "")

    def _revise(self, prompt: str, spec: ModelSpec) -> dict:
        return self.make_script("150-320" in prompt)

    def _review(self, prompt: str, spec: ModelSpec) -> dict:
        names = ("hook", "tension", "emotional_payoff", "originality", "prose_for_the_ear", "drawability",
                 "follows_recipe")
        return {"scores": {k: self.review_scores for k in names}, "flags": {"gore": False},
                "notes": ["raise the tension in the middle"]}

    def _audience(self, prompt: str, spec: ModelSpec) -> dict:
        count = len(re.findall(r"^\[(\d+)\] ", prompt, re.MULTILINE))
        return {"chunks": [{"chunk": n, "interest": self.audience_interest(n), "would_leave": False,
                            "note": "drags"} for n in range(1, count + 1)], "overall": "ok"}

    def _rewrite_chunks(self, prompt: str, spec: ModelSpec) -> dict:
        weak = re.findall(r"^\[(\d+)\] ", prompt.split("Test viewers lost interest")[1], re.MULTILINE)
        return {"chunks": [{"chunk": int(n), "text": " ".join(unique_words(f"r{n}{self.counter}", 40)) + "."}
                           for n in weak]}

    def _polish(self, prompt: str, spec: ModelSpec) -> dict:
        script = json.loads(_between(prompt, "Script:\n", "\n\nDo a final"))
        return dict(script, pronunciations=[{"word": "Llorona", "say_as": "yoh-ROH-nah"}])

    def _final_check(self, prompt: str, spec: ModelSpec) -> dict:
        issues = self.final_issues.popleft() if self.final_issues else []
        return {"flags": {"gore": False}, "accuracy_issues": issues, "cultural_issues": []}

    def _scene_plan(self, prompt: str, spec: ModelSpec) -> dict:
        narration = _between(prompt, "NARRATION:\n", "\nTASK: Split")
        parts = [p for p in re.split(r"(?<=[.?!])\s+", narration.strip()) if p]
        keys = 4 if len(parts) > 15 else 1
        cast = [{"name": "Protagonist", "look": "Tall, long grey coat, flat cap, a red scarf"},
                {"name": "The Storyteller", "look": "Invented look that canon must override"}]
        if self.cast_override:
            cast = self.cast_override([c["name"] for c in cast])
        return {"cast": cast, "scenes": [
            {"text": t, "location": "a dark road", "characters": ["Protagonist"] if i % 2 else ["The Storyteller"],
             "visual": "A stick figure under one streetlight", "mood": "uneasy", "delivery": "slow",
             "sfx": ["wind"], "key_moment": i < keys} for i, t in enumerate(parts)]}

    def _alternative_openings(self, prompt: str, spec: ModelSpec) -> dict:
        return {"openings": ["A brand new opening that asks who knocked at midnight. It never stopped.",
                             "The knocking started at 3:07 every night. Nobody else heard it."]}

    def _compare_openings(self, prompt: str, spec: ModelSpec) -> dict:
        a = _between(prompt, "OPENING A:\n", "\n\nOPENING B:")
        b = _between(prompt, "OPENING B:\n", "\n\nA viewer")
        winner = "A" if self.prefer_opening and self.prefer_opening in a else \
            "B" if self.prefer_opening and self.prefer_opening in b else "A"
        return {"reason": "hook", "winner": winner}

    def _compare_titles(self, prompt: str, spec: ModelSpec) -> dict:
        a = _between(prompt, "TITLE A: ", "\n")
        b = _between(prompt, "TITLE B: ", "\n")
        return {"reason": "click", "winner": "B" if self.prefer_title == b else "A"}

    def _packaging(self, prompt: str, spec: ModelSpec) -> dict:
        return {"titles": ["The Extra Passenger", "Night Bus 9", "Seat Ten", "Do Not Board"],
                "thumbnail_brief": "A red lantern", "thumbnail_text": "WHO?",
                "description": "A story. This is an original work of fiction.",
                "tags": ["horror"], "pinned_comment": "Would you get off the bus?"}

    def _learn(self, prompt: str, spec: ModelSpec) -> dict:
        return {"insights": ["Night-shift settings keep viewers longest."]}


class FakeClient:
    supports_web_search = True

    def __init__(self, spec: ModelSpec, brain: Brain):
        self.spec = spec
        self.brain = brain

    def generate_json(self, task: str, system: str, prompt: str, schema: dict) -> dict:
        return self.brain.call(self.spec, task, system, prompt)

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * 256
        for word in dedup.normalize(text).split():
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def research(self, task: str, system: str, prompt: str) -> str:
        self.brain.calls.append((task, self.spec.label))
        return "Researched notes."


def make_context(seed: int = 7, cfg: dict | None = None) -> tuple[pipeline.Context, Brain]:
    import random

    cfg = cfg or test_config()
    brain = Brain()
    conn = db.connect(":memory:")
    router = Router(cfg, conn, factory=lambda spec: FakeClient(spec, brain))
    return pipeline.Context(conn=conn, cfg=cfg, router=router, rng=random.Random(seed)), brain

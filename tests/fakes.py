"""A deterministic stand-in for Gemini so the pipeline can be tested offline."""

from __future__ import annotations

import copy
import hashlib
import math
import re
from collections import deque
from typing import Any, Callable

from story_factory import config, dedup


def test_config() -> dict:
    cfg = copy.deepcopy(config.DEFAULTS)
    cfg["database"] = ":memory:"
    cfg["lengths"] = {"long_min_words": 150, "long_max_words": 320,
                      "short_min_words": 40, "short_max_words": 110}
    return cfg


def unique_words(seed: str, count: int) -> list[str]:
    digest = hashlib.sha256(seed.encode()).hexdigest()
    return [f"{seed[:2]}{digest[i % 60:i % 60 + 4]}{i}" for i in range(count)]


def premise(n: int) -> dict[str, str]:
    tag = hashlib.md5(str(n).encode()).hexdigest()[:10]
    return {"title": f"Tale {tag}", "logline": " ".join(unique_words(f"l{n}", 12)) + ".",
            "synopsis": " ".join(unique_words(f"s{n}", 90)) + "."}


class FakeLLM:
    """Routes each task to a handler; tests can queue premises or override handlers."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.premise_queue: deque[list[dict]] = deque()
        self.counter = 0
        self.review_scores = 8
        self.judge_same = False
        self.overrides: dict[str, Callable[[str], dict]] = {}

    # --- LLM protocol -------------------------------------------------------

    def generate_json(self, task: str, prompt: str, *, temperature: float = 0.9) -> dict[str, Any]:
        self.calls.append(task)
        if task in self.overrides:
            return self.overrides[task](prompt)
        return getattr(self, f"_{task}")(prompt)

    def embed(self, text: str) -> list[float]:
        self.calls.append("embed")
        vector = [0.0] * 256
        for word in dedup.normalize(text).split():
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    # --- Handlers -----------------------------------------------------------

    def _premises(self, prompt: str) -> dict:
        if self.premise_queue:
            return {"premises": self.premise_queue.popleft()}
        self.counter += 1
        return {"premises": [premise(self.counter * 10 + i) for i in range(3)]}

    def make_script(self, long: bool) -> dict:
        self.counter += 1
        n = self.counter
        sentences = 30 if long else 8
        body = " ".join(" ".join(unique_words(f"b{n}x{i}", 6)).capitalize() + "."
                        for i in range(sentences))
        question = "Would you have opened the door?"
        return {
            "intro": "I found this one in the hollow beside a quiet road." if long else "",
            "body": body,
            "outro": f"Some doors are better left closed. {question}",
            "comment_question": question,
            "hook_line": body.split(".")[0] + ".",
            "thumbnail_idea": "A red lantern in an empty doorway",
            "description_credit": "This is an original work of fiction.",
            "new_entities": [{"kind": "place", "name": f"Exit {n}", "description": "A road sign that moves."}],
            "referenced_entities": ["The red lantern"],
        }

    def _write_script(self, prompt: str) -> dict:
        return self.make_script("STRUCTURE (long episode)" in prompt)

    _revise = _write_script

    def _review(self, prompt: str) -> dict:
        names = ("hook", "tension", "emotional_payoff", "originality",
                 "clarity_for_listening", "drawability", "follows_recipe")
        return {"scores": {k: self.review_scores for k in names},
                "flags": {"gore": False, "claims_true": False},
                "fix_instructions": ["raise the tension in the middle"]}

    def _judge_duplicate(self, prompt: str) -> dict:
        return {"same_story": self.judge_same, "reason": "fake judge"}

    def _scene_plan(self, prompt: str) -> dict:
        narration = prompt.split("NARRATION:\n", 1)[1].split("\nTASK: Split", 1)[0]
        parts = [p for p in re.split(r"(?<=[.?!])\s+", narration.strip()) if p]
        long = len(parts) > 15
        keys = 4 if long else 1
        return {"scenes": [
            {"text": text, "location": "a dark road", "characters": ["Protagonist"],
             "visual": "A stick figure under one streetlight", "mood": "uneasy",
             "sfx": ["wind"], "key_moment": i < keys}
            for i, text in enumerate(parts)
        ]}

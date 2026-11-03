"""Deterministic stand-ins for the model providers, so the pipeline is tested offline."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import deque
from typing import Any, Callable

from story_factory import config, db, dedup, pipeline, schemas
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
        self.bible_present: Callable[[str], bool] = lambda item_id: True
        self.threat_breaks: list[str] = []
        self.unfit_legends: set[str] = set()
        self.pauses: list[dict] = [{"after": "Some doors are better left closed", "ms": 1200}]

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

    def _legend_fit(self, prompt: str, spec: ModelSpec) -> dict:
        title = _between(prompt, "Legend: ", " (")
        return {"reason": "harmless", "fit": title not in self.unfit_legends}

    def _judge_duplicate(self, prompt: str, spec: ModelSpec) -> dict:
        return {"reason": "fake judge", "same_story": self.judge_same}

    def _dossier(self, prompt: str, spec: ModelSpec) -> dict:
        return {"summary": "A spirit of the roads.", "core_elements": ["walks at night"],
                "regional_variants": [], "cultural_notes": [], "avoid": [], "sources": ["https://example.org"]}

    def _beat_sheet(self, prompt: str, spec: ModelSpec) -> dict:
        names = ["hook", "ordinary life", "first wrong detail", "escalation", "midpoint", "breath",
                 "choice", "climax"]
        return {"central_question": "Who is the extra passenger?", "emotional_core": "loneliness",
                "rehooks": [{"at_seconds": 25, "question": "Who rang?"}, {"at_seconds": 120, "question": "Why her?"},
                            {"at_seconds": 260, "question": "What is it?"}],
                "midpoint_reversal": "The passenger is her brother", "peak_moment": "The mirror",
                "beats": [{"name": n, "purpose": "question", "summary": "x", "target_seconds": 60,
                           "tension": i + 2, "turn": "it gets worse", "open_loop": "" if i == 7 else "why?",
                           "plants": ["ticket"], "payoffs": []} for i, n in enumerate(names)]}

    def _story_bible(self, prompt: str, spec: ModelSpec) -> dict:
        return {"hero": {"name": "Marisol", "who": "a bus driver, 52", "want": "to finish her last shift",
                         "wound": "her brother vanished on this route", "flaw": "stubborn",
                         "false_belief": "the road took him", "human_detail": "she hums three notes",
                         "voice": "dry, kind"},
                "threat": {"what_it_is": "a passenger who never boards", "what_it_wants": "a seat",
                           "rules": ["rings the bell at 12:06", "cannot cross a lit door"],
                           "limits": ["cannot speak"], "tell": "the bell"},
                "theme": "grief", "theme_question": "can you let someone go?",
                "hero_choice": "she stops the bus", "dramatic_irony": "the viewer sees the handprint",
                "breath_moment": "a joke with the dispatcher", "refrain": "Next stop, nowhere.",
                "refrain_turn": "it means her stop", "sound_motif": "the stop bell",
                "red_accent": "the stop button", "open_detail": "whose handprint?",
                "aftertaste": "the stop bell on your bus home",
                "clues": [{"plant": "the timetable", "payoff": "the date"},
                          {"plant": "the cold seat", "payoff": "her brother's seat"}]}

    _revise_bible = _story_bible

    def _review_bible(self, prompt: str, spec: ModelSpec) -> dict:
        return {"problems": [], "approved": True}

    def _bible_check(self, prompt: str, spec: ModelSpec) -> dict:
        script = json.loads(_between(prompt, "SCRIPT:\n", "\n\nTASK:"))
        quote = " ".join(script["body"].split()[:5])
        ids = re.findall(r"^- \[(\w+)\]", prompt, re.MULTILINE)
        return {"items": [{"id": i, "present": self.bible_present(i), "quote": quote} for i in ids],
                "threat_rule_breaks": list(self.threat_breaks)}

    def _write_section(self, prompt: str, spec: ModelSpec) -> dict:
        self.counter += 1
        words = int(re.search(r"About (\d+) words", prompt).group(1))
        text = " ".join(" ".join(unique_words(f"p{self.counter}x{i}", 6)).capitalize() + "."
                        for i in range(max(words // 6, 1)))
        if spec.model == "writer-b" and "part 1 of" in prompt:
            text = text.replace(".", " PREFERRED.", 1)
        return {"text": text}

    def _write_frame(self, prompt: str, spec: ModelSpec) -> dict:
        question = "Would you have opened the door?"
        return {"intro": "The phone rings at 3:07 and the voice is his own. The lantern is lit. Listen closely.",
                "outro": f"Some doors are better left closed. {question}", "comment_question": question,
                "thumbnail_idea": "A red stop button", "description_credit": "This is an original work of fiction.",
                "new_entities": [{"kind": "place", "name": f"Exit {self.counter}", "description": "A moving sign."}],
                "referenced_entities": ["The red lantern"]}

    def _line_edit(self, prompt: str, spec: ModelSpec) -> dict:
        return json.loads(_between(prompt, "Script:\n", "\n\nA line editor"))

    _revise_outline = _beat_sheet

    def _review_outline(self, prompt: str, spec: ModelSpec) -> dict:
        return {"problems": [], "approved": True}

    def _write_draft(self, prompt: str, spec: ModelSpec) -> dict:
        return self.make_script("150-320" in prompt, marker="PREFERRED" if spec.model == "writer-b" else "")

    def _revise(self, prompt: str, spec: ModelSpec) -> dict:
        return self.make_script("150-320" in prompt)

    def _review(self, prompt: str, spec: ModelSpec) -> dict:
        return {"scores": {k: self.review_scores for k in schemas.REVIEW_CRITERIA}, "flags": {"gore": False},
                "notes": [{"quote": "the door", "problem": "the middle sags", "fix": "raise the tension"}]}

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
        return dict(script, pronunciations=[{"word": "Llorona", "say_as": "yoh-ROH-nah"}], pauses=self.pauses)

    def _final_check(self, prompt: str, spec: ModelSpec) -> dict:
        issues = self.final_issues.popleft() if self.final_issues else []
        # Cultural issues apply to every story (accuracy only to retold legends).
        return {"flags": {"gore": False}, "accuracy_issues": [], "cultural_issues": issues}

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
                "tags": ["horror"], "pinned_comment": "Would you get off the bus?",
                "end_screen_line": "There is another story waiting down the line."}

    def _teaser_short(self, prompt: str, spec: ModelSpec) -> dict:
        self.counter += 1
        body = " ".join(" ".join(unique_words(f"t{self.counter}x{i}", 6)).capitalize() + "." for i in range(10))
        return {"body": body, "outro": "The full story is on the channel. Would you have answered?"}

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

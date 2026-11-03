"""Shared model-call plumbing: errors, JSON parsing, usage logging, spending cap."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any, Protocol

from . import db


class LLMError(RuntimeError):
    pass


def temperature_for(cfg: dict, task: str) -> float | None:
    """The configured sampling temperature for a task, or None to use the model's default."""
    value = (cfg.get("temperature") or {}).get(task)
    return float(value) if value is not None else None


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    """`provider:model` with an optional `@effort`, e.g. `anthropic:claude-fable-5-1@high`.

    Amazon Bedrock specs name the model's maker in the model ID, e.g.
    `aws:anthropic.claude-opus-5-5`; the maker is the model's family.
    """

    provider: str
    model: str
    effort: str | None = None

    @classmethod
    def parse(cls, text: str) -> "ModelSpec":
        provider, _, rest = text.partition(":")
        model, _, effort = rest.partition("@")
        if provider not in ("anthropic", "openai", "google", "xai", "aws", "local") or not model:
            raise ValueError(f"Bad model spec {text!r}; use provider:model[@effort]")
        return cls(provider, model, effort or None)

    @property
    def family(self) -> str:
        if self.provider == "aws":
            return self.model.split(".", 1)[0]
        return self.provider

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


class Client(Protocol):
    spec: ModelSpec

    def generate_json(self, task: str, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]: ...


def parse_json(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating a Markdown code fence around it."""
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Model returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise LLMError("Model returned JSON that is not an object")
    return value


class UsageRecorder:
    """Logs every call's tokens (and an estimated price) to the shared costs table.

    Calls run on worker threads, so writes are serialized with a lock.
    """

    def __init__(self, conn: sqlite3.Connection, cfg: dict, pipeline: str = "story_factory"):
        self.conn = conn
        self.prices = cfg.get("prices", {})
        self.cap = cfg["budget"].get("monthly_cap_usd")
        self.pipeline = pipeline
        # Set by the pipeline while it develops a story, so costs can be reported per story.
        self.story_id: int | None = None
        self.lock = threading.Lock()

    def check(self) -> None:
        if not self.cap:
            return
        with self.lock:
            spent = db.month_spend(self.conn)
        if spent >= self.cap:
            raise BudgetExceeded(f"Monthly spend ${spent:.2f} reached the ${self.cap:.2f} cap")

    def record(self, task: str, spec: ModelSpec, input_tokens: int, output_tokens: int) -> None:
        price_in, price_out = self.prices.get(spec.model, (0.0, 0.0))
        usd = (input_tokens * price_in + output_tokens * price_out) / 1_000_000
        with self.lock:
            db.record_cost(self.conn, pipeline=self.pipeline, task=task, model=spec.label,
                           input_units=input_tokens, output_units=output_tokens, usd=usd,
                           story_id=self.story_id)

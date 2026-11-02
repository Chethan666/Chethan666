"""Gemini API client for text (JSON output) and embeddings.

Text and embedding calls are safe to retry: a failed request returns nothing
to bill twice for. Every call is logged to the shared `costs` table and checked
against the monthly cap first.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

from . import db

log = logging.getLogger(__name__)

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class LLMError(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


class LLM(Protocol):
    def generate_json(self, task: str, prompt: str, *, temperature: float = 0.9) -> dict[str, Any]: ...
    def embed(self, text: str) -> list[float]: ...


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


class BudgetGuard:
    def __init__(self, conn: sqlite3.Connection, cap_usd: float):
        self.conn = conn
        self.cap_usd = cap_usd

    def check(self, upcoming_usd: float = 0.0) -> None:
        spent = db.month_spend(self.conn)
        if spent + upcoming_usd >= self.cap_usd:
            raise BudgetExceeded(f"Monthly spend ${spent:.2f} reached the ${self.cap_usd:.2f} cap")


class GeminiClient:
    def __init__(self, cfg: dict, conn: sqlite3.Connection, api_key: str | None = None,
                 pipeline: str = "story_factory"):
        self.cfg = cfg["gemini"]
        self.conn = conn
        self.pipeline = pipeline
        self.guard = BudgetGuard(conn, cfg["budget"]["monthly_cap_usd"])
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise LLMError("Set GEMINI_API_KEY in the environment")
        self._last_call = 0.0

    # --- HTTP ---------------------------------------------------------------

    def _request(self, method: str, path: str, body: dict | None = None) -> dict[str, Any]:
        wait = self.cfg["min_interval_s"] - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        data = json.dumps(body).encode() if body is not None else None
        for attempt in range(self.cfg["max_retries"] + 1):
            request = urllib.request.Request(
                BASE_URL + path,
                data=data,
                method=method,
                headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            )
            try:
                with urllib.request.urlopen(request, timeout=self.cfg["timeout_s"]) as response:
                    self._last_call = time.monotonic()
                    return json.loads(response.read())
            except urllib.error.HTTPError as exc:
                self._last_call = time.monotonic()
                retryable = exc.code == 429 or exc.code >= 500
                if not retryable or attempt == self.cfg["max_retries"]:
                    detail = exc.read().decode(errors="replace")[:500]
                    raise LLMError(f"Gemini HTTP {exc.code}: {detail}") from exc
                delay = float(exc.headers.get("Retry-After") or 2 ** (attempt + 2))
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == self.cfg["max_retries"]:
                    raise LLMError(f"Gemini request failed: {exc}") from exc
                delay = 2 ** (attempt + 2)
            log.warning("Gemini call failed, retrying in %.0fs", delay)
            time.sleep(delay)
        raise AssertionError("unreachable")

    def _record(self, task: str, model: str, usage: dict) -> None:
        input_tokens = int(usage.get("promptTokenCount", 0))
        # Thinking tokens are billed as output.
        output_tokens = int(usage.get("candidatesTokenCount", 0)) + int(usage.get("thoughtsTokenCount", 0))
        usd = (input_tokens * self.cfg["input_usd_per_million"]
               + output_tokens * self.cfg["output_usd_per_million"]) / 1_000_000
        db.record_cost(self.conn, pipeline=self.pipeline, task=task, model=model,
                       input_units=input_tokens, output_units=output_tokens, usd=usd)

    # --- API ----------------------------------------------------------------

    def generate_json(self, task: str, prompt: str, *, temperature: float = 0.9) -> dict[str, Any]:
        self.guard.check()
        model = self.cfg["text_model"]
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"},
        }
        data = self._request("POST", f"/models/{model}:generateContent", body)
        self._record(task, model, data.get("usageMetadata", {}))
        candidates = data.get("candidates") or []
        if not candidates or not candidates[0].get("content", {}).get("parts"):
            reason = candidates[0].get("finishReason") if candidates else data.get("promptFeedback")
            raise LLMError(f"Gemini returned no text ({reason})")
        text = "".join(part.get("text", "") for part in candidates[0]["content"]["parts"])
        return parse_json(text)

    def embed(self, text: str) -> list[float]:
        self.guard.check()
        model = self.cfg["embedding_model"]
        body = {
            "content": {"parts": [{"text": text}]},
            "taskType": "SEMANTIC_SIMILARITY",
            "outputDimensionality": self.cfg["embedding_dimensions"],
        }
        data = self._request("POST", f"/models/{model}:embedContent", body)
        self._record("embed", model, {"promptTokenCount": len(text.split())})
        return [float(v) for v in data["embedding"]["values"]]

    def list_models(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/models?pageSize=1000")
        return [
            {"name": m["name"].removeprefix("models/"), "methods": m.get("supportedGenerationMethods", [])}
            for m in data.get("models", [])
        ]

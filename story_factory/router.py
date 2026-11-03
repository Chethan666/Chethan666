"""Role-based model routing.

Each role in `[roles]` lists one or more `provider:model[@effort]` specs. Judges
are drawn from families other than the author's whenever possible, because
models measurably favor text written by their own family.
"""

from __future__ import annotations

import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, TypeVar

from . import providers
from .llm import Client, LLMError, ModelSpec, UsageRecorder

T = TypeVar("T")
R = TypeVar("R")

log = logging.getLogger(__name__)

ROLES = ("pitchers", "judges", "writers", "reviser", "audience", "polisher", "researcher",
         "planner", "embedder")


class Router:
    def __init__(self, cfg: dict, conn: sqlite3.Connection,
                 factory: Callable[[ModelSpec], Any] | None = None):
        self.cfg = cfg
        self.recorder = UsageRecorder(conn, cfg)
        self._factory = factory or (lambda spec: providers.build(spec, cfg, self.recorder))
        self._clients: dict[str, Any] = {}
        missing = [r for r in ROLES if not cfg["roles"].get(r)]
        if missing:
            raise ValueError(f"Configure models for roles: {', '.join(missing)}")
        self.workers = cfg["routing"]["parallel_calls"]

    def _client(self, text: str) -> Any:
        if text not in self._clients:
            self._clients[text] = self._factory(ModelSpec.parse(text))
        return self._clients[text]

    def pool(self, role: str) -> list[Client]:
        return [self._client(t) for t in self.cfg["roles"][role]]

    def one(self, role: str) -> Client:
        """The role's first model, failing over to the next ones listed if a call fails."""
        pool = self.pool(role)
        return pool[0] if len(pool) == 1 else FailoverClient(pool)

    def judges(self, authors: Iterable[str] = ()) -> list[Client]:
        """Judges from families that wrote none of the items being judged.

        Falls back to the whole panel when every family is an author, so a
        verdict is always possible.
        """
        authors = set(authors)
        panel = self.pool("judges")
        independent = [j for j in panel if j.spec.family not in authors]
        return independent or panel

    def embed(self, text: str) -> list[float]:
        return self.one("embedder").embed(text)

    def parallel(self, fn: Callable[[T], R], items: list[T]) -> list[R]:
        """Run independent model calls concurrently, keeping input order."""
        if len(items) <= 1 or self.workers <= 1:
            return [fn(i) for i in items]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(fn, items))

    def panel(self, fn: Callable[[T], R], items: list[T], what: str = "judge panel") -> list[R]:
        """Like `parallel`, but one failing model doesn't sink the panel.

        Failed calls are logged and left out; raises only if every call fails.
        """
        def safe(item: T) -> R | None:
            try:
                return fn(item)
            except LLMError as exc:
                log.warning("%s: one panel member failed: %s", what, exc)
                return None

        results = [r for r in self.parallel(safe, items) if r is not None]
        if items and not results:
            raise LLMError(f"{what}: every panel member failed")
        return results


class FailoverClient:
    """Acts as the first model; on a failed call, tries the next ones in order.

    Some models occasionally loop until their output limit or time out; a single
    such failure shouldn't stop a story.
    """

    def __init__(self, clients: list[Any]):
        self.clients = clients
        self.spec = clients[0].spec
        self.supports_web_search = getattr(clients[0], "supports_web_search", False)

    def generate_json(self, task: str, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        last: LLMError | None = None
        for client in self.clients:
            try:
                return client.generate_json(task, system, prompt, schema)
            except LLMError as exc:
                log.warning("%s failed on %s, trying the next model: %s", client.spec.label, task, exc)
                last = exc
        raise last  # every model failed

    def __getattr__(self, name: str) -> Any:  # research, embed, list_models: the primary model
        return getattr(self.clients[0], name)

"""Role-based model routing.

Each role in `[roles]` lists one or more `provider:model[@effort]` specs. Judges
are drawn from families other than the author's whenever possible, because
models measurably favor text written by their own family.
"""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, TypeVar

from . import providers
from .llm import Client, ModelSpec, UsageRecorder

T = TypeVar("T")
R = TypeVar("R")

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
        return self.pool(role)[0]

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

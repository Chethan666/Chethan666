"""Pairwise ranking with position-bias control.

Each judge sees every pair twice, once in each order. A judge's vote counts as
a win only when it picks the same item both times; otherwise it's a tie.
Items are ranked by wins (ties worth half), then by their prior score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Sequence

from .llm import LLMError

log = logging.getLogger(__name__)

# compare(judge, first, second) -> "A" if `first` wins, "B" if `second` wins.
Compare = Callable[[Any, Any, Any], str]


@dataclass
class Ranked:
    item: Any
    points: float
    prior: float


def rank(items: Sequence[Any], judges_for: Callable[[Any, Any], list[Any]], compare: Compare,
         run: Callable[[Callable[[Any], Any], list[Any]], list[Any]],
         prior: Callable[[Any], float] = lambda _: 0.0) -> list[Ranked]:
    """Round-robin `items`; `run` executes the comparison jobs (possibly in parallel)."""
    if len(items) <= 1:
        return [Ranked(i, 0.0, prior(i)) for i in items]
    jobs = []
    for a, b in combinations(range(len(items)), 2):
        for judge in judges_for(items[a], items[b]):
            jobs.append((a, b, judge))

    def judge_pair(job: tuple[int, int, Any]) -> tuple[int, int, float]:
        a, b, judge = job
        try:
            forward = compare(judge, items[a], items[b])
            backward = compare(judge, items[b], items[a])
        except LLMError as exc:  # a judge that can't answer abstains: a tie
            log.warning("Comparison skipped: %s", exc)
            return a, b, 0.5
        if forward == "A" and backward == "B":
            return a, b, 1.0
        if forward == "B" and backward == "A":
            return a, b, 0.0
        return a, b, 0.5

    points = [0.0] * len(items)
    for a, b, score in run(judge_pair, jobs):
        points[a] += score
        points[b] += 1.0 - score
    ranked = [Ranked(items[i], points[i], prior(items[i])) for i in range(len(items))]
    ranked.sort(key=lambda r: (r.points, r.prior), reverse=True)
    return ranked

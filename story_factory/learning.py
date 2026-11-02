"""Feedback from real YouTube performance into future stories.

Two signals:
- Ingredient weights: settings, fears, twists, formats and endings whose videos
  keep viewers watching longer are drawn more often (shrunk toward the average
  so a single lucky video can't dominate).
- Written lessons: a judge model compares the best and worst performers and
  writes insights that every future pitch prompt sees.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict

from . import db, prompts, schemas
from .router import Router

DIMENSIONS = ("setting", "core_fear", "twist", "format", "ending")
SHRINK = 3.0  # pretend each ingredient also has this many average-performing videos
MIN_WEIGHT, MAX_WEIGHT = 0.5, 2.0


def ingredient_weights(conn: sqlite3.Connection) -> dict[str, dict[str, float]]:
    rows = [r for r in db.latest_performance(conn) if r["avg_view_percentage"] is not None]
    if len(rows) < 5:
        return {}
    overall = sum(r["avg_view_percentage"] for r in rows) / len(rows)
    weights: dict[str, dict[str, float]] = {}
    for dim in DIMENSIONS:
        groups: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            if r[dim]:
                groups[r[dim]].append(r["avg_view_percentage"])
        weights[dim] = {}
        for value, scores in groups.items():
            shrunk = (sum(scores) + SHRINK * overall) / (len(scores) + SHRINK)
            ratio = (shrunk / overall) ** 2 if overall else 1.0
            weights[dim][value] = min(MAX_WEIGHT, max(MIN_WEIGHT, ratio))
    return weights


def learn(conn: sqlite3.Connection, router: Router, sample: int = 6) -> list[str]:
    rows = [r for r in db.latest_performance(conn) if r["avg_view_percentage"] is not None]
    if len(rows) < 2 * sample:
        return []
    ranked = sorted(rows, key=lambda r: (r["avg_view_percentage"], r["views"] or 0), reverse=True)
    judge = router.one("judges")
    result = judge.generate_json("learn", prompts.SYSTEM_EDITOR,
                                 prompts.learn(ranked[:sample], ranked[-sample:]), schemas.LEARNINGS)
    insights = [str(i) for i in result.get("insights", []) if str(i).strip()][:10]
    if insights:
        db.save_learnings(conn, insights)
    return insights

"""Repeat and copy detection.

A new premise must pass, in order:
1. title check: not a near-copy of an existing title;
2. recipe check: setting + fear + twist never used before (also a DB constraint);
3. lexical check: synopsis doesn't share too many word runs with an old one;
4. semantic check: meaning-level similarity via embeddings, with an LLM judge
   deciding the borderline cases.
Finished scripts are then checked again for near-copies of old scripts and for
any copied passage from the research sources.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable, Iterable

from . import db

STOPWORDS = frozenset(
    "a an and are as at be but by for from has have he her his i in into is it its of on or "
    "she that the their them they this to was were will with you your".split()
)


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", text.lower()).split())


def content_words(text: str) -> list[str]:
    return [w for w in normalize(text).split() if w not in STOPWORDS]


def shingles(words: list[str], size: int) -> set[tuple[str, ...]]:
    if len(words) < size:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + size]) for i in range(len(words) - size + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def lexical_similarity(a: str, b: str, size: int = 2) -> float:
    return jaccard(shingles(content_words(a), size), shingles(content_words(b), size))


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def copied_passages(text: str, sources: Iterable[str], ngram: int) -> list[str]:
    """Runs of `ngram` words that `text` shares verbatim with any source."""
    words = normalize(text).split()
    mine = shingles(words, ngram)
    hits: list[str] = []
    for source in sources:
        for run in mine & shingles(normalize(source).split(), ngram):
            hits.append(" ".join(run))
    return hits


def premise_text(premise: dict[str, Any]) -> str:
    return f"{premise['title']}. {premise['logline']} {premise['synopsis']}"


@dataclass
class Verdict:
    ok: bool
    reason: str = ""
    embedding: list[float] = field(default_factory=list)


def check_premise(conn: sqlite3.Connection, cfg: dict, premise: dict[str, Any], recipe_sig: str,
                  embed: Callable[[str], list[float]],
                  same_story: Callable[[dict[str, Any], sqlite3.Row], tuple[bool, str]]) -> Verdict:
    """`embed` turns text into a vector; `same_story` asks an AI judge about a borderline pair."""
    rules = cfg["dedup"]
    # Rejected ideas block repeats too, so the factory never pitches them again.
    existing = db.all_stories(conn)

    for row in existing:
        score = title_similarity(premise["title"], row["title"])
        if score >= rules["title_similarity"]:
            return Verdict(False, f"title too close to story #{row['id']} '{row['title']}' ({score:.2f})")

    # A rejected story frees its recipe so a different premise can use it.
    if any(row["recipe_sig"] == recipe_sig for row in existing if row["status"] != "rejected"):
        return Verdict(False, "recipe (setting + fear + twist) already used")

    for row in existing:
        if not row["synopsis"]:
            continue
        score = lexical_similarity(premise["synopsis"], row["synopsis"])
        if score >= rules["synopsis_lexical_similarity"]:
            return Verdict(False, f"synopsis wording too close to story #{row['id']} ({score:.2f})")

    embedding = embed(premise_text(premise))
    scored = []
    for row in existing:
        if row["synopsis_embedding"]:
            scored.append((cosine(embedding, json.loads(row["synopsis_embedding"])), row))
    scored.sort(key=lambda pair: pair[0], reverse=True)

    if scored and scored[0][0] >= rules["semantic_hard"]:
        score, row = scored[0]
        return Verdict(False, f"same meaning as story #{row['id']} '{row['title']}' ({score:.2f})")

    borderline = [(s, r) for s, r in scored if s >= rules["semantic_soft"]][: rules["judge_top_k"]]
    for score, row in borderline:
        same, reason = same_story(premise, row)
        if same:
            return Verdict(False, f"judge: same story as #{row['id']} ({score:.2f}): {reason}")

    return Verdict(True, embedding=embedding)


def check_script(conn: sqlite3.Connection, cfg: dict, story_id: int, narration: str,
                 source_texts: list[str]) -> Verdict:
    rules = cfg["dedup"]
    copied = copied_passages(narration, source_texts, rules["copy_ngram"])
    if copied:
        return Verdict(False, f"copies research text verbatim: '{copied[0]}'")
    for row in db.live_stories(conn):
        if row["id"] == story_id or not row["narration"]:
            continue
        score = lexical_similarity(narration, row["narration"], size=5)
        if score >= rules["script_lexical_similarity"]:
            return Verdict(False, f"script too close to story #{row['id']} ({score:.2f})")
    return Verdict(True)

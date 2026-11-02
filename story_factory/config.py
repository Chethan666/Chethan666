"""Configuration: built-in defaults, optionally overridden by a TOML file."""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "database": "data/stories.db",
    "budget": {
        # Shared by both pipelines; paid calls stop once the month reaches this.
        "monthly_cap_usd": 18.0,
    },
    "gemini": {
        # Check current model IDs with `python -m story_factory models`.
        "text_model": "gemini-2.5-flash",
        "embedding_model": "gemini-embedding-001",
        "embedding_dimensions": 768,
        # Free tier costs nothing; set real prices if you move to paid text calls.
        "input_usd_per_million": 0.0,
        "output_usd_per_million": 0.0,
        "timeout_s": 180,
        "max_retries": 4,
        # Spacing between calls keeps the free tier's per-minute limit happy.
        "min_interval_s": 4.0,
    },
    "backlog": {
        # Approved stories to keep ready for the production pipeline.
        "hollow_world": 4,
        "legends_retold": 2,
        "sixty_seconds": 8,
    },
    "lengths": {
        # About 140 spoken words per minute: 1,200 words is roughly 8.5 minutes.
        "long_min_words": 1200,
        "long_max_words": 1500,
        "short_min_words": 110,
        "short_max_words": 170,
    },
    "quality": {
        "min_score": 7,
        "max_revisions": 2,
        "candidates_per_call": 3,
        "max_attempts_per_story": 3,
    },
    "dedup": {
        "title_similarity": 0.85,
        "synopsis_lexical_similarity": 0.30,
        "script_lexical_similarity": 0.15,
        # Above `semantic_hard` a premise is a repeat; between soft and hard an
        # LLM judge compares the two stories directly.
        "semantic_hard": 0.92,
        "semantic_soft": 0.82,
        "judge_top_k": 3,
        "setting_cooldown": 8,
        "twist_cooldown": 4,
        "fear_cooldown": 3,
        # Any shared run of this many words with a source text counts as copying.
        "copy_ngram": 8,
        "recent_titles_in_prompt": 40,
    },
    "research": {
        "wikipedia_page_limit": 40,
        "wikipedia_categories": [
            {"category": "Category:Ghosts", "region": "Global"},
            {"category": "Category:Urban legends", "region": "Global"},
            {"category": "Category:Ghost ships", "region": "Global"},
            {"category": "Category:Yōkai", "region": "Japan"},
            {"category": "Category:Philippine legendary creatures", "region": "Philippines"},
            {"category": "Category:Mexican folklore", "region": "Mexico"},
            {"category": "Category:Scottish folklore", "region": "Scotland"},
            {"category": "Category:Slavic legendary creatures", "region": "Eastern Europe"},
            {"category": "Category:Indian legendary creatures", "region": "India"},
            {"category": "Category:Indonesian folklore", "region": "Indonesia"},
            {"category": "Category:African legendary creatures", "region": "Africa"},
            {"category": "Category:Brazilian legendary creatures", "region": "Brazil"},
        ],
        "youtube_queries": [
            "animated horror story",
            "scary rules story",
            "horror story animated night shift",
            "sad ghost story animated",
            "folklore horror story",
        ],
        "youtube_max_results": 15,
        "youtube_published_within_days": 120,
        "trends_in_prompt": 15,
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Return defaults merged with `path` (or ./config.toml when it exists)."""
    if path is None and Path("config.toml").is_file():
        path = "config.toml"
    if path is None:
        return copy.deepcopy(DEFAULTS)
    with open(path, "rb") as handle:
        return _merge(DEFAULTS, tomllib.load(handle))

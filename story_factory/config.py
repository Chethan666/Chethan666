"""Configuration: built-in defaults, optionally overridden by a TOML file."""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "database": "data/stories.db",
    "budget": {
        # Optional monthly cap in USD across both pipelines; 0 disables it.
        "monthly_cap_usd": 0,
    },
    # Writers' room: which models fill each role. Format: provider:model[@effort].
    # "Balanced" defaults: the strongest models write the story; cheaper models score,
    # simulate viewers and plan scenes. See config.example.toml for a premium preset.
    # Run `python -m story_factory models` to confirm IDs your keys can use.
    "roles": {
        # Pitch competing ideas; several companies give genuinely different ideas.
        "pitchers": ["anthropic:claude-sonnet-5-5@high", "google:gemini-3-pro-preview",
                     "openai:gpt-5-mini@high"],
        # Score and rank pitches and drafts; never judge their own family's work.
        "judges": ["anthropic:claude-sonnet-5-5@high", "google:gemini-3-pro-preview",
                   "openai:gpt-5-mini@high"],
        # Each entry writes one competing full draft.
        "writers": ["anthropic:claude-opus-5-5@high", "google:gemini-3-pro-preview"],
        "reviser": ["anthropic:claude-opus-5-5@high"],
        # Simulated viewers; personas are spread across these models.
        "audience": ["google:gemini-3-flash-preview", "openai:gpt-5-mini@low"],
        "polisher": ["anthropic:claude-opus-5-5@high"],
        # Needs web search: Claude's server-side web search tool.
        "researcher": ["anthropic:claude-sonnet-5-5@high"],
        "planner": ["anthropic:claude-sonnet-5-5@medium"],
        "embedder": ["google:gemini-embedding-001"],
    },
    "routing": {"parallel_calls": 6},
    "providers": {
        "anthropic": {"max_tokens": 64000, "timeout_s": 900},
        "openai": {"max_tokens": 64000, "timeout_s": 900, "embedding_dimensions": None},
        "google": {"timeout_s": 600, "embedding_dimensions": 768},
    },
    # USD per million input/output tokens, for the cost log only. Claude prices are
    # Anthropic's list prices; the others are from third-party trackers, so check them.
    "prices": {
        "claude-fable-5-1": [10.0, 50.0],
        "claude-opus-5-5": [4.0, 20.0],
        "claude-sonnet-5-5": [2.0, 10.0],
        "claude-haiku-5-5": [0.10, 0.50],
        "gpt-5.5": [5.0, 30.0],
        "gpt-5-mini": [0.25, 2.0],
        "gemini-3-pro-preview": [2.0, 12.0],
        "gemini-3-flash-preview": [0.50, 3.0],
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
        # Median judge score every category must reach (out of 10).
        "min_score": 8,
        "max_revisions": 2,
        "pitches_per_pitcher": 3,
        "tournament_size": 4,
        # Pitches ranked just below the winner are shelved for later if they score this well.
        "shelve_min_score": 7.5,
        "max_attempts_per_story": 2,
        # Simulated-viewer test: chunks of about 30 seconds of narration.
        "audience_chunk_words": 70,
        "audience_min_interest": 7,
        "max_chunks_to_rewrite": 5,
        "audience_personas": [
            "a 24-year-old horror fan who plays story channels at night and is bored by cliches",
            "a 17-year-old Shorts scroller who leaves within seconds unless something grabs them",
            "a 45-year-old who loves stories for their characters and emotional payoff, not jump scares",
            "a 30-year-old skeptic who notices plot holes, contrivances and cheap twists",
            "a 28-year-old Brazilian who understands English well but needs clear spoken language",
        ],
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

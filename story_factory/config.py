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
        # Most roles run on Amazon Bedrock (one AWS key). Gemini uses its own API key,
        # and research uses the direct Claude API because Bedrock has no web search.
        # Pitch competing ideas; several companies give genuinely different ideas.
        "pitchers": ["aws:anthropic.claude-sonnet-5-5@high", "google:gemini-3-pro-preview",
                     "aws:xai.grok-4.6"],
        # Score and rank pitches and drafts; never judge their own company's work.
        # Four companies, so almost every comparison gets two independent judges.
        "judges": ["aws:anthropic.claude-sonnet-5-5@high", "google:gemini-3-pro-preview",
                   "aws:openai.gpt-5.5@low", "aws:xai.grok-4.6"],
        # Each entry writes one competing full draft.
        "writers": ["aws:anthropic.claude-opus-5-5@high", "aws:xai.grok-4.6"],
        "reviser": ["aws:anthropic.claude-opus-5-5@high"],
        # Simulated viewers; personas are spread across these models.
        "audience": ["google:gemini-3-flash-preview", "aws:xai.grok-4.6"],
        "polisher": ["aws:anthropic.claude-opus-5-5@high"],
        # Needs Claude's web search, which only the direct Claude API offers.
        "researcher": ["anthropic:claude-sonnet-5-5@high"],
        "planner": ["aws:anthropic.claude-sonnet-5-5@medium"],
        "embedder": ["google:gemini-embedding-001"],
    },
    "routing": {"parallel_calls": 6},
    "providers": {
        "anthropic": {"max_tokens": 64000, "timeout_s": 900},
        "openai": {"max_tokens": 64000, "timeout_s": 900, "embedding_dimensions": None},
        "google": {"timeout_s": 600, "embedding_dimensions": 768},
        "xai": {"max_tokens": 64000, "timeout_s": 900},
        "aws": {
            # AWS_REGION in the environment overrides this.
            "region": "us-east-1",
            # Some Bedrock-served open models reject larger output limits.
            "max_tokens": 16000,
            "timeout_s": 900,
            # Bedrock's OpenAI-compatible endpoint (/v1 confirmed by a live run; a 404 is
            # still retried on /openai/v1).
            "openai_base_url": "https://bedrock-mantle.{region}.api.aws/v1",
            # Models served through the Responses API there; others use Chat Completions.
            "responses_api_models": ["openai.gpt-5.5"],
        },
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
        "grok-4.7": [2.0, 6.0],
        # Amazon Bedrock IDs; in-Region Bedrock prices can run about 10% higher.
        "anthropic.claude-opus-5-5": [4.0, 20.0],
        "anthropic.claude-sonnet-5-5": [2.0, 10.0],
        "openai.gpt-5.5": [5.0, 30.0],
        "xai.grok-4.6": [2.0, 6.0],
    },
    "backlog": {
        # Approved stories to keep ready for the production pipeline.
        "hollow_world": 4,
        "legends_retold": 2,
        # Shorts are switched off for now: no standalone Shorts are written.
        "sixty_seconds": 0,
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
        # Dedicated edits to bring a script inside the word range before each editor round.
        "length_fix_attempts": 3,
        # Alternative openings the hook lab writes and pits against the original.
        "hook_alternatives": 2,
        # Titles that go through the pairwise title contest.
        "title_finalists": 3,
        # Judges per pairwise comparison: full contests (pitches, drafts) and quick ones
        # (openings, titles, audience rewrites). Earlier judges in [roles] are used first.
        "judges_per_comparison": 3,
        "quick_judges": 2,
        # Long drafts are written in this many parts (one call each), then framed.
        "draft_sections": 3,
        # Share of story-bible promises the script must visibly deliver.
        "bible_min_coverage": 0.75,
        "bible_min_coverage_short": 0.5,
        # Dedicated line edits for machine-sounding phrasing before each editor round.
        "style_fix_attempts": 2,
        # Repair rounds when the final check finds must-fix issues.
        "final_check_repairs": 2,
        # Legends tried per attempt before giving up when judges find them unfit for horror.
        "legend_fit_tries": 4,
        # Dramatic pauses the polish step may mark for the narrator.
        "max_pauses_long": 12,
        "max_pauses_short": 3,
        # Packaging limits: a thumbnail must read at phone size.
        "thumbnail_brief_max_words": 45,
        "thumbnail_text_max_words": 4,
        # A 60-second teaser Short pointing to each long episode (off while Shorts are off).
        "teaser_shorts": False,
        # Recent heroes the story bible must differ from.
        "recent_heroes": 12,
        # Long episodes: every writer plans (story bible + outline) and judges pick the
        # better plan before any prose is written.
        "plan_contest": True,
        # Median plan-review score that approves a bible or outline (out of 10), and
        # the revisions it may get. The best-scoring version is always kept.
        "plan_min_score": 7,
        "plan_revisions": 2,
        # Shorts compete on breadth: this many complete drafts, a quick score, then a
        # head-to-head contest among the finalists.
        "short_drafts": 6,
        "short_finalists": 3,
        # The rewrite ladder (see `diagnose`): at most this many fresh starts (a new
        # draft, or a new plan and draft) per story, on top of `max_revisions`.
        "max_fresh_starts": 1,
        # A draft this far off is written again rather than patched.
        "redraft_below_coverage": 0.3,
        "redraft_at_or_below_score": 5,
        # A premise this unoriginal is dropped rather than rewritten.
        "kill_at_or_below_originality": 4,
        # Retold legends: most judges must recognize the legend's being in the script.
        "recognition_check": True,
        # Surprise test: readers guess the ending from the first part of the story.
        "surprise_test": True,
        "surprise_readers": 3,
        "surprise_cut": 0.7,
        # Alternative endings pitted against the original, like the hook lab.
        "ending_alternatives": 2,
        # Rejected originals whose premise wasn't the problem are parked with their best
        # draft for a stronger model; set True to develop parked stories again.
        "retry_parked": False,
        "audience_personas": [
            "a 24-year-old horror fan who plays story channels at night and is bored by cliches",
            "a 17-year-old Shorts scroller who leaves within seconds unless something grabs them",
            "a 45-year-old in Pune who grew up on her grandmother's ghost stories, loves characters and "
            "emotional payoff more than jump scares, and notices anything that rings false about Indian life",
            "a 30-year-old skeptic who notices plot holes, contrivances and cheap twists",
            "a 28-year-old Brazilian who understands English well but needs clear spoken language, knows "
            "little about India, and loses interest when a custom or word is not clear from the story",
        ],
    },
    "style": {
        # Free checks on the narration (no model call); thresholds suit spoken English
        # for a global audience.
        "max_avg_sentence_words": 17,
        "long_sentence_words": 32,
        "max_long_sentences": 3,
        "max_named_feelings": 2,
        "max_same_openings_in_a_row": 3,
        "words_per_dash": 60,
        "max_hard_word_share": 0.05,
    },
    # Sampling temperature per task, where the model accepts one (open models on
    # Bedrock, Grok, Gemini). Creative tasks run hot, judging runs cool.
    "temperature": {
        "pitches": 1.0, "story_bible": 0.9, "revise_bible": 0.8, "write_section": 0.85,
        "write_draft": 0.85, "write_frame": 0.8, "alternative_openings": 0.95, "packaging": 0.8,
        "rewrite_chunks": 0.8, "revise": 0.7, "line_edit": 0.6, "audience": 0.8,
        "score_pitch": 0.2, "compare_pitches": 0.2, "compare_plans": 0.2, "compare_endings": 0.2,
        "recognition": 0.1, "grade_guesses": 0.1, "predict_ending": 0.7, "annotate": 0.3,
        "rewrite_sections": 0.8, "alternative_endings": 0.95, "compare_drafts": 0.2, "compare_openings": 0.2,
        "compare_titles": 0.2, "review": 0.2, "review_outline": 0.2, "review_bible": 0.2,
        "bible_check": 0.1, "final_check": 0.1, "judge_duplicate": 0.1, "scene_plan": 0.3,
    },
    # Output cap per task for chat-completion models: a quick judgment that runs past
    # this is looping, so it fails fast and the panel carries on without it.
    "task_max_tokens": {
        "score_pitch": 6000, "compare_pitches": 6000, "compare_drafts": 6000, "compare_openings": 4000,
        "compare_titles": 4000, "judge_duplicate": 4000, "review_outline": 8000, "review_bible": 8000,
        "compare_plans": 6000, "compare_endings": 4000, "recognition": 6000, "grade_guesses": 4000,
        "predict_ending": 2000,
    },
    # Pipeline 2: the narrator's voice (Kokoro, open source, runs on the CPU). Kokoro
    # voices: am_michael, am_fenrir, am_onyx, bm_george, bm_lewis... lang "a" is US
    # English, "b" British English.
    "video": {
        "voice": "am_michael", "lang": "a", "speed": 0.92,
        # Characters read their own quoted lines (voices by kind of speaker; see production/voice.py).
        "character_voices": True,
    },
    # Pipeline 2, studio mode (with FAL_KEY): every shot a generated still picture (no video
    # clips), an expressive narrator, a scored soundtrack, and a broadcast edit.
    # Without the key, or with `enabled = false`, the drawn pipeline above runs instead.
    "studio": {
        "enabled": True,
        # Spending cap per video (estimates from `prices`); work past it falls back to free steps.
        # A typical 8-9 minute video costs about $20-25: ~110 pictures about $17, voices about $1,
        # music, room tone, effects and checks the rest.
        "max_usd": 30.0,
        "models": {
            "sheet": "fal-ai/nano-banana-pro",           # the style frame and character sheets
            "shot": "fal-ai/nano-banana-pro/edit",       # every shot, from the sheets
            "shot_cheap": "fal-ai/bytedance/seedream/v4.5/edit",  # the redraw when a picture fails
            "shot_alt": "fal-ai/nano-banana-2/edit",            # compared in the bake-off
            "voice": "fal-ai/elevenlabs/tts/eleven-v3",
            "sfx": "fal-ai/elevenlabs/sound-effects/v2",
            "music": "fal-ai/elevenlabs/music",
            "thumbnail": "fal-ai/nano-banana-pro/edit",
            "vision": "fal-ai/any-llm/vision",           # checks every picture
        },
        # Extra arguments per model role (checked against each model's inputs by `check`).
        "model_args": {
            "sheet": {"aspect_ratio": "16:9", "num_images": 1},
            "shot": {"aspect_ratio": "16:9", "num_images": 1, "resolution": "2K"},
            "shot_cheap": {"image_size": "landscape_16_9", "num_images": 1},
            "shot_alt": {"aspect_ratio": "16:9", "num_images": 1, "resolution": "2K"},
            "voice": {"stability": 0.5},
            "thumbnail": {"aspect_ratio": "16:9", "num_images": 1, "resolution": "2K"},
            "vision": {"model": "google/gemini-2.5-flash"},
        },
        # Estimated USD per picture, per second of audio, or per 1,000 characters of speech.
        "prices": {
            "fal-ai/nano-banana-pro": 0.15, "fal-ai/nano-banana-pro/edit": 0.15,
            "fal-ai/nano-banana-2/edit": 0.08, "fal-ai/bytedance/seedream/v4.5/edit": 0.04,
            "fal-ai/elevenlabs/tts/eleven-v3": 0.10,
            "fal-ai/elevenlabs/sound-effects/v2": 0.002, "fal-ai/elevenlabs/music": 0.01,
            "fal-ai/any-llm/vision": 0.002,
        },
        # Shots: 3-6.5 s each, every one a still picture with a slow camera move.
        "shot_min_seconds": 3.0, "shot_max_seconds": 6.5,
        "parallel": 6,
        # Narrator and cast voices on ElevenLabs v3 (names from `check`); Kokoro if a line fails.
        "narrator_voice": "Brian",
        "cast_voices": {"man": ["George", "Daniel", "Callum"], "woman": ["Charlotte", "Alice", "Lily"],
                        "old_man": ["Bill"], "old_woman": ["Dorothy"], "boy": ["Liam"], "girl": ["Laura"]},
        # The edit.
        "cold_open": True, "title_seconds": 2.0, "end_screen_seconds": 20.0,
        "burn_captions": False,  # long videos upload captions.srt instead; Shorts always burn them
        "shorts": 3, "short_max_seconds": 55.0,
        "thumbnails": 3,
        # Audio loudness for YouTube.
        "loudness_lufs": -14.0,
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
        # Research tops itself up automatically before `fill` when fewer unused legends remain.
        "min_unused_legends": 15,
        "wikipedia_page_limit": 100,
        # Also read pages from each category's subcategories (one level down).
        "wikipedia_subcategories": True,
        # Indian folklore only (checked on Wikipedia, November 2026). Legends are retold, and
        # used as sparks, only from these categories' regions: folklore that earlier research
        # brought in from elsewhere stays in the database, unused.
        "wikipedia_categories": [
            {"category": "Category:Indian legendary creatures", "region": "India"},
            {"category": "Category:Demons in Hinduism", "region": "India"},
            {"category": "Category:Indian folklore", "region": "India"},
            {"category": "Category:Superstitions of India", "region": "India"},
            {"category": "Category:Reportedly haunted locations in India", "region": "India"},
            {"category": "Category:Bengali folklore", "region": "Bengal"},
        ],
        # What horror viewers watch (titles only, as a demand signal for the pitchers).
        "youtube_queries": [
            "indian horror story animated",
            "indian folklore horror story",
            "indian urban legends",
            "haunted places in india story",
            "horror story hindi animated",
            "scary rules story",
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

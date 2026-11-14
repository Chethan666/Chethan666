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
            "a 45-year-old who loves stories for their characters and emotional payoff, not jump scares",
            "a 30-year-old skeptic who notices plot holes, contrivances and cheap twists",
            "a 28-year-old Brazilian who understands English well but needs clear spoken language",
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
    # Pipeline 2's free narrator (Kokoro, open source, runs on the CPU), behind every paid line.
    # Kokoro voices: am_michael, am_fenrir, am_onyx, bm_george, bm_lewis... lang "a" is US
    # English, "b" British English.
    "video": {"voice": "am_michael", "lang": "a", "speed": 0.92},
    # Pipeline 2: the story told in still pictures (cinematic film stills, each a shot under a
    # slow camera move), voices and music, made by a team of agents (production/stills). Runs
    # free without keys (Kokoro, free picture services, synthesized sound); with FAL_KEY the paid
    # models below are used.
    "stills": {
        # Spending cap per video in USD (estimates from `prices`); past it, free fallbacks.
        "max_usd": 10.0,
        # The channel's house look, sent with every picture: "cinematic" (stills from a dark horror
        # film) or "illustrated" (a painted graphic novel). A non-empty "style" replaces the look's text.
        "look": "cinematic",
        "style": "",
        "looks": {
            "cinematic": {
                "style": ("A still frame from a dark, slow-burn horror film, shot on 35mm: photorealistic real "
                          "people and places, natural skin texture, low-key lighting from one motivated source "
                          "(a lamp, a screen, a torch, moonlight through a window), deep shadows that swallow "
                          "the edges of the frame, shallow depth of field, muted, desaturated colour grade of "
                          "cold teal-greys and sodium amber, fine film grain, anamorphic 16:9 composition. "
                          "Quiet, real and unsettling, never gory, never glossy."),
                "noun": "film still",
                "people": ("real, ordinary people a casting director could cast (not models): believable faces, "
                           "worn everyday clothes"),
                "avoid": ("No plastic or airbrushed skin, no CGI or video-game look, no cartoon or painting, no "
                          "lens flare, no bright stock-photo lighting. Hands relaxed, half-hidden or out of frame."),
                "wrong": ("it looks like a cartoon, drawing, painting or 3D render instead of a real film frame; "
                          "skin is glossy, plastic or airbrushed like a stock photo"),
                "lettering": "bold, clean, condensed sans-serif capitals with a slight distressed edge",
                # The film finish, added in the edit so every painter's pictures match: fine grain laid
                # into each picture at the size it is shown, and a vignette fixed to the frame.
                "grain": 0.018,
                "vignette": 0.22,
                "halation": 0.35,  # film's warm glow around bright lights (lamps, screens, the red accent)
                # The film's brightness ceiling (mean luma, 0-255): above it the colorist prints every
                # picture down together. Night horror sits around 30-40.
                "max_brightness": 40,
                "synthetic_media": True,  # realistic made-up people and events: YouTube's AI label
            },
            "illustrated": {
                "style": ("A dark, painterly horror illustration in the style of a graphic novel: confident ink "
                          "linework, textured brushwork, deep shadows and heavy chiaroscuro, a muted, desaturated "
                          "palette of cold blue-greys, sickly greens and bone white, cinematic 16:9 composition "
                          "with a single motivated light source. Moody and unsettling, never gory."),
                "noun": "illustration",
                "people": "illustrated people with clear, repeatable shapes",
                "avoid": "",
                "wrong": "it is a photograph or looks photographic instead of an illustration",
                "lettering": "big bold hand-lettered words",
                "grain": 0.0, "vignette": 0.0,
                "synthetic_media": False,
            },
        },
        # Cameras a beat may be "seen through" when the story gives a reason (a security camera, a
        # phone, an old photo); otherwise every beat is a film frame. Only used with the cinematic look.
        "cameras": {
            "film": "",
            "cctv": ("Seen on a security camera at night: high fixed corner angle, wide lens distortion, grainy "
                     "dark monochrome night-vision, mostly black with dim grey shapes, soft focus, no timestamp "
                     "or text; as dark as the rest of the film, never bright or evenly lit"),
            "phone": ("A phone photo taken in the dark: harsh direct flash, blown highlights close to the lens, "
                      "deep black behind, slight motion blur, tilted framing"),
            "polaroid": ("An old instant photograph: faded warm colours, soft focus, light leaks at one edge, "
                         "shown full frame with no border"),
        },
        # "drift": every story picture gets a slow, steady camera move suited to its framing (wide shots
        # pan, close ones push in, a scare lunges and settles, a security camera holds still);
        # "none": every picture is held perfectly still.
        "motion": "drift",
        # Cinema bars: 2.39 frames every shot as anamorphic widescreen inside the 16:9 video; 0 = none.
        "letterbox": 0,
        # The video's frame. 1440p, not 1080p: YouTube gives 1440p uploads its better codec (VP9/AV1)
        # even when watched at 1080p, and dark, grainy pictures are where the plain one bands and
        # blocks. About twice the render time of 1080p.
        "video_size": [2560, 1440],
        # Pictures are painted and kept at this size until the edit: the 1440p frame plus the room a
        # camera move needs (a push-in ends on real detail, never on an enlargement).
        "picture_size": [2752, 1548],
        # What moves in the air of a shot (the storyboard decides per beat): rain, fog, a flickering light.
        "atmosphere": True,
        # The colorist: a picture whose brightness, contrast or colour cast lies outside the film's
        # tolerance is brought this far back to it; pictures inside are not touched (0 = off).
        "grade_strength": 0.75,
        # The set designer gives each place the story returns to a fixed design (free), and the
        # most-used places a location sheet the pictures set there are painted from (one picture each).
        "set_design": True, "location_sheets": 3,
        # How many scare beats are painted by the strongest painter (`chains.scare`, about $0.15 each).
        "scare_shots": 4,
        # Key beats: at least this important (1-5, from the storyboard), at most this share of the beats.
        "key_importance": 3, "key_share": 0.25,
        # The first try is meant to be right: retries and art-review redraws may spend at most this share
        # of max_usd; past it, every picture keeps its best try. `lessons`: the faults the critic refused
        # most often in earlier videos are named in every picture prompt (counted in the story database).
        "retry_share": 0.15, "lessons": True,
        # Each character also gets a large face portrait painted from their sheet; pictures are painted
        # from portrait + sheet, and the checker compares every face with its portrait.
        "portraits": True, "face_check": True,
        # The animation stage (production/stills/motion.py): chosen beats become clips made from their
        # finished pictures, checked like pictures; everything else stays a still. Off until an
        # image-to-video model is set in `models.clip`. beats: "scares", "key" (scares and key
        # beats) or "all". durations: the clip lengths each model accepts, in seconds. image_field:
        # the name a model gives its start picture ("image_url" for most, "start_image_url" for Kling).
        "animation": {"enabled": False, "beats": "scares", "max_clips": 12, "max_stretch": 1.25,
                   "durations": {"clip": [5, 10], "clip_alt": [5, 10]},
                   "image_field": {"clip": "image_url", "clip_alt": "image_url"},
                   "check_clips": True, "parallel": 3},
        "beat_min_seconds": 5.0, "beat_max_seconds": 11.0,
        "scene_dip_seconds": 0.35,
        # Painters tried in order until the critic passes a picture; each is told what the critic
        # refused in the one before. fal:<role> needs FAL_KEY; cloudflare needs CF_ACCOUNT_ID and
        # CF_API_TOKEN (free daily allowance); pollinations is keyless. The free painters take no
        # references, so they only paint when no paid picture could be made (no key, over the cap).
        "chains": {
            # Spent by importance (runs 96 and 98 compared: PRODUCTION.md, "Cost and quality").
            # A style frame or a sheet that fails its check is painted once more, told what was wrong.
            "reference": ["fal:reference", "fal:reference", "cloudflare", "pollinations"],
            # Sheets: Seedream 4.5 (it followed descriptions best), once more told what was wrong, then
            # Seedream 5.0 Lite: a different model may succeed where the first failed twice.
            "sheet": ["fal:sheet", "fal:sheet", "fal:sheet_alt", "cloudflare", "pollinations"],
            # Ordinary beats: Seedream, and a second Seedream try only for a serious fault (a picture
            # refused for a detail is kept).
            "beat": ["fal:shot", "fal:shot", "cloudflare", "pollinations"],
            # Key beats (`key_importance`, at most `key_share` of the story): Nano Banana 2 corrects
            # a refused picture far more often than Seedream does (about 2 in 3 against 1 in 7).
            "key": ["fal:shot", "fal:shot_alt", "cloudflare", "pollinations"],
            # The scares (at most `scare_shots`, most important first) are the frames a viewer
            # remembers and the ones a thumbnail is made from: the strongest painter first.
            "scare": ["fal:hero", "fal:shot", "cloudflare", "pollinations"],
            "thumbnail": ["fal:thumbnail"],
        },
        # The picture bake-off (`produce --bakeoff`, workflow overrides `bakeoff=...`): the same
        # references, then every beat painted by each setup, side by side, with a judge's scores.
        "bakeoff": {
            "max_usd": 7.0, "retry_share": 0.06,
            "setups": {
                "seedream_detail": {"label": "Seedream 4.5 (today's setup) + 4K detail pass", "detail": True},
                "flux2": {"label": "FLUX.2 pro, every picture",
                          "chains": {"beat": ["fal:flux", "fal:flux"], "key": ["fal:flux", "fal:flux"],
                                     "scare": ["fal:flux", "fal:flux"]}},
                "pro": {"label": "Nano Banana Pro, every picture",
                        "chains": {"beat": ["fal:hero", "fal:hero"], "key": ["fal:hero", "fal:hero"],
                                   "scare": ["fal:hero", "fal:hero"]}},
            },
        },
        "references": True,  # a style frame, character sheets and location sheets keep look, people and places the same
        "pollinations": True, "pollinations_gap_seconds": 16,
        "cloudflare_model": "@cf/black-forest-labs/flux-1-schnell", "cloudflare_steps": 4,  # 4 keeps ~170 pictures a day free
        "models": {
            "reference": "fal-ai/bytedance/seedream/v4.5/text-to-image",  # style frame (text only)
            "sheet": "fal-ai/bytedance/seedream/v4.5/edit",       # character and location sheets (paints children too)
            "sheet_alt": "fal-ai/bytedance/seedream/v5/lite/edit",
            "shot": "fal-ai/bytedance/seedream/v4.5/edit",        # every beat, from the references
            "shot_text": "fal-ai/bytedance/seedream/v4.5/text-to-image",
            "shot_alt": "fal-ai/nano-banana-2/edit",              # the corrector for key beats
            "shot_alt_text": "fal-ai/nano-banana-2",
            "hero": "fal-ai/nano-banana-pro/edit",                # the scare shots
            "hero_text": "fal-ai/nano-banana-pro",
            "thumbnail": "fal-ai/nano-banana-2/edit",
            "voice": "fal-ai/elevenlabs/tts/eleven-v3",
            "sfx": "fal-ai/elevenlabs/sound-effects/v2",
            # Music the sound library lacks: ACE-Step 1.5 (open source, ~$0.0003 a second). For
            # ElevenLabs instead: "fal-ai/elevenlabs/music" ($0.80 a minute, whole minutes).
            "music": "fal-ai/ace-step-1.5",
            "vision": "fal-ai/any-llm/vision",                   # the critic's eye
            # For the picture bake-off (and a later switch): FLUX.2 [pro] with references, and a
            # detail pass that upscales a finished picture to 4K.
            "flux": "fal-ai/flux-2-pro/edit", "flux_text": "fal-ai/flux-2-pro",
            "upscale": "fal-ai/seedvr/upscale/image",
            # Image-to-video models for the motion stage (off while empty): e.g. a Kling, Seedance or
            # Wan image-to-video endpoint on fal.ai, with its price per second in `prices`.
            "clip": "", "clip_alt": "",
        },
        # An edit model asked without references uses its text-to-image twin.
        "text_roles": {"shot": "shot_text", "shot_alt": "shot_alt_text", "sheet": "reference", "sheet_alt": "reference",
                       "thumbnail": "shot_alt_text", "hero": "hero_text", "flux": "flux_text"},
        # References sent to a model, longest side in pixels (FLUX.2 counts every input megapixel).
        "reference_px": {"flux": 1024},
        "model_args": {
            "reference": {"image_size": {"width": 2752, "height": 1548}, "num_images": 1},
            "sheet": {"image_size": {"width": 2560, "height": 1440}, "num_images": 1},
            "sheet_alt": {"image_size": {"width": 2560, "height": 1440}, "num_images": 1},
            # Pictures that go into the film are painted at 2K, a little larger than the 1440p frame,
            # so they stay sharp under a camera move. Seedream: a custom size (from 2560x1440 up). Nano
            # Banana 2 and Pro: "2K" (the default "1K" is 1376x768, which goes soft when enlarged).
            "shot": {"image_size": {"width": 2752, "height": 1548}, "num_images": 1},
            "shot_text": {"image_size": {"width": 2752, "height": 1548}, "num_images": 1},
            "shot_alt": {"aspect_ratio": "16:9", "resolution": "2K", "num_images": 1},
            "shot_alt_text": {"aspect_ratio": "16:9", "resolution": "2K", "num_images": 1},
            "hero": {"aspect_ratio": "16:9", "resolution": "2K", "num_images": 1},
            "hero_text": {"aspect_ratio": "16:9", "resolution": "2K", "num_images": 1},
            "thumbnail": {"aspect_ratio": "16:9", "num_images": 1},
            "voice": {"stability": 0.5},
            "vision": {"model": "google/gemini-2.5-flash"},
            "judge": {"model": "google/gemini-2.5-pro", "reasoning": True},  # this model refuses without it
            "flux": {"image_size": {"width": 2560, "height": 1440}, "output_format": "png", "safety_tolerance": "4"},
            "flux_text": {"image_size": {"width": 2560, "height": 1440}, "output_format": "png", "safety_tolerance": "4"},
            "upscale": {"upscale_mode": "target", "target_resolution": "2160p", "output_format": "png"},
        },
        # A role whose settings cost more than the model's base price (Nano Banana 2 at 2K: 1.5x).
        "model_units": {"shot_alt": 1.5, "shot_alt_text": 1.5},
        # Estimated USD per picture, per 1,000 characters of speech, or per second of audio.
        "prices": {
            "fal-ai/nano-banana-2": 0.08, "fal-ai/nano-banana-2/edit": 0.08,
            "fal-ai/nano-banana-pro": 0.15, "fal-ai/nano-banana-pro/edit": 0.15,  # 1K and 2K; 4K is double
            "fal-ai/bytedance/seedream/v5/lite/edit": 0.035,
            "fal-ai/bytedance/seedream/v4.5/edit": 0.04, "fal-ai/bytedance/seedream/v4.5/text-to-image": 0.04,
            "fal-ai/elevenlabs/tts/eleven-v3": 0.10, "fal-ai/elevenlabs/sound-effects/v2": 0.002,
            # ElevenLabs music on fal: $0.80 a minute, billed in whole minutes (the score rounds up).
            "fal-ai/elevenlabs/music": 0.80 / 60, "fal-ai/ace-step-1.5": 0.0003, "fal-ai/any-llm/vision": 0.002,
            # estimates, high side: FLUX.2 [pro] bills by megapixel in and out (a 2560x1440 picture
            # from up to 7 references at 1024 px); the SeedVR detail pass to 4K
            "fal-ai/flux-2-pro/edit": 0.14, "fal-ai/flux-2-pro": 0.08, "fal-ai/seedvr/upscale/image": 0.03,
        },
        # Voices. The host reads the intro and outro; a first-person story's teller gets a voice
        # matched to them; characters get distinct voices by kind. Kokoro names are the free fallback.
        # point_of_view: "" decides from the text; "first" or "third" forces it.
        "point_of_view": "", "teller_default": "man",
        # How the narration sounds. The performance director gives each sentence its delivery (a
        # v3 tag such as [whispers] or [nervously], most sentences plain); the narrator reads a
        # paragraph's sentences in one take so the voice flows instead of restarting flat at every
        # full stop; voice_pace speeds the reading up without changing pitch (1.0 = as recorded);
        # voice_gap_scale shortens the silences between paragraphs and at story pauses.
        "performance_director": True, "voice_takes": True, "voice_pace": 1.12, "voice_gap_scale": 0.7,
        # The listener transcribes the paid narration (Whisper, free, on the CPU) and has any take
        # that skipped or invented words read again; "" turns it off.
        "listener": "base.en",
        # Planned length of the story part of a video (the script check warns when the narration
        # would run well over or under); 0 = no target.
        "target_seconds": 0,
        "host_voice": {"elevenlabs": "Brian", "kokoro": "am_michael"},
        "teller_voices": {
            "elevenlabs": {"man": "Callum", "woman": "Charlotte", "old_man": "Bill", "old_woman": "Dorothy"},
            "kokoro": {"man": "am_fenrir", "woman": "af_heart", "old_man": "bm_lewis", "old_woman": "bf_isabella"},
        },
        "cast_voices": {
            "elevenlabs": {"man": ["George", "Daniel", "Callum"], "woman": ["Alice", "Lily", "Charlotte"],
                           "old_man": ["Bill"], "old_woman": ["Dorothy"], "boy": ["Liam"], "girl": ["Laura"]},
            "kokoro": {"man": ["am_echo", "bm_george", "am_liam", "am_fenrir"],
                       "woman": ["bf_emma", "af_nicole", "af_sarah", "af_heart"],
                       "old_man": ["bm_daniel", "bm_lewis"], "old_woman": ["bf_alice", "bf_isabella"],
                       "boy": ["am_puck"], "girl": ["af_sky", "af_bella"]},
        },
        "parallel": 4, "render_workers": 3, "max_effects": 30,
        "cold_open": True, "title_seconds": 3.0, "end_screen_seconds": 20.0,
        "burn_captions": False, "thumbnails": 3,
        # The channel's sound library (production/library.py; built by the sound-library workflow
        # from Kevin MacLeod, OpenGameArt CC0, the Free Music Archive, Sonniss packs and ACE-Step).
        # The sound director picks from it first; only what it lacks is generated, and saved back.
        "sound_library": {
            "enabled": True,
            "repo": "",  # owner/name of the GitHub repo holding the release; "" = this repository
            "release_tag": "sound-library",
            "local_dir": "",  # a folder with library.json and the files, instead of the release
            "min_score": 0.55,  # how well an effect must match its description to be used
            "generate_missing": False,  # True: the director may have missing sounds made on fal (ACE-Step, effects); False: library only, never pays for sound
            "save_generated": True,  # paid-for sounds go into the library after the run
        },
        # YouTube's "altered or synthetic content" flag: None follows the look (on for cinematic).
        "loudness_lufs": -14.0, "synthetic_media": None,
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
            {"category": "Category:Female ghosts", "region": "Global"},
            {"category": "Category:Japanese folklore", "region": "Japan"},
            {"category": "Category:Korean legendary creatures", "region": "Korea"},
            {"category": "Category:Chinese ghosts", "region": "China"},
            {"category": "Category:Thai folklore", "region": "Thailand"},
            {"category": "Category:Malaysian folklore", "region": "Malaysia"},
            {"category": "Category:Arabian legendary creatures", "region": "Middle East"},
            {"category": "Category:Germanic legendary creatures", "region": "Germany"},
            {"category": "Category:Irish folklore", "region": "Ireland"},
            {"category": "Category:English folklore", "region": "England"},
            {"category": "Category:Scandinavian folklore", "region": "Scandinavia"},
            {"category": "Category:Native American legendary creatures", "region": "North America"},
            {"category": "Category:Caribbean folklore", "region": "Caribbean"},
            {"category": "Category:Australian Aboriginal legendary creatures", "region": "Australia"},
            {"category": "Category:Polynesian legendary creatures", "region": "Polynesia"},
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

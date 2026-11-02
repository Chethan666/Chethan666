# Story Factory (pipeline 1)

The story factory is a writers' room made of AI models from several companies. It researches, pitches, writes, edits and tests horror stories. Approved stories go into a SQLite database, where the production pipeline (pipeline 2) claims them and turns them into videos.

Its goal is the best possible story, not the cheapest. Every step either creates competition (several pitches, several drafts) or adds an independent critic (editors, simulated viewers, a final checker).

## The flow

```
                  ┌───────────────────────── Research (weekly) ──────────────────────────┐
                  │ Wikipedia legends · YouTube demand titles · web-researched dossier    │
                  └──────────────────────────────────┬────────────────────────────────────┘
                                                     ▼
 1. PITCH CONTEST   3 pitchers (Claude, Gemini, Grok), each with its own fresh recipe, pitch 3 ideas
                    → 9-layer repeat check → judges from OTHER companies score every pitch
                    → top 4 go head-to-head, each pair judged in both orders → winner
                    → strong runners-up are SHELVED for a later run
 2. OUTLINE         head writer builds a beat sheet: hook, escalations, false relief, clue plants/payoffs
                    → independent editors approve it or send it back
 3. DRAFT CONTEST   2 writers (Claude, Grok) draft the full script → head-to-head judging
 4. EDITORS' LOOP   independent editors score 7 qualities; median of every one must reach 8/10
                    → head writer revises against their line notes (up to 2 rounds) or the story is rejected
 5. HOOK LAB        4 alternative openings compete with the original for the first ten seconds;
                    the original stays unless judges prefer a new one
 6. AUDIENCE TEST   5 simulated viewers rate every ~30 seconds and say where they'd leave
                    → weak parts are rewritten, kept only if judges agree the new version is better
 7. POLISH          read-aloud line edit for one narrator voice + pronunciation guide for the voice engine
                    (discarded if it changes the story)
 8. FINAL CHECK     rule violations, legend accuracy against the dossier, cultural sensitivity
 9. SCENE PLAN      exact narration spans, visuals, sound cues, a delivery cue for the narrator,
                    key moments that need new images, and a CAST with one fixed look per character
                    (recurring world characters always keep their first look)
10. PACKAGING       5 titles ranked head-to-head for click appeal, thumbnail brief and text,
                    description with credit, tags, pinned comment
                    → APPROVED, waiting for pipeline 2
```

Shorts go through the same room with shorter targets.

## Who does what

The default is a **balanced** setup on three accounts: **Amazon Bedrock** runs most roles (Claude, GPT and Grok through one AWS key), **Google's Gemini API** runs the Gemini roles (Bedrock doesn't offer Gemini), and the **direct Claude API** runs legend research, because only it offers Claude's web search.

| Role | Default model | Runs on | Price per million tokens (in / out) | Why |
|---|---|---|---|---|
| Head writer: outline, revisions, polish, packaging | Claude Opus 5.5 | AWS Bedrock | $4 / $20 | Top tier for fiction; the writing is the product, so this is where money goes |
| Draft writers | Claude Opus 5.5, Grok 4.6 | AWS Bedrock | $4 / $20, $2 / $6 | Two competing drafts from two companies; Grok has ranked near the top for fiction and its output is cheap |
| Pitchers | Claude Sonnet 5.5, Gemini 3 Pro, Grok 4.6 | Bedrock, Gemini API, Bedrock | $2 / $10, $2 / $12, $2 / $6 | Three companies give varied ideas |
| Judges and editors | Claude Sonnet 5.5, Gemini 3 Pro, GPT-5.5 (low effort), Grok 4.6 | Bedrock, Gemini API, Bedrock, Bedrock | as above, GPT-5.5 $5 / $30 | A judge never scores its own company's work; four companies give almost every comparison two independent judges |
| Simulated viewers | Gemini 3 Flash, Grok 4.6 | Gemini API, Bedrock | $0.50 / $3, $2 / $6 | Reacting like a viewer doesn't need a top model |
| Researcher | Claude Sonnet 5.5 with web search | Direct Claude API | $2 / $10, plus search fees | Builds a sourced fact file on each legend; Bedrock has no web search |
| Scene planner | Claude Sonnet 5.5 | AWS Bedrock | $2 / $10 | Long, exact structured output |
| Repeat detection | Gemini embedding model | Gemini API | very low | Top proprietary model on the MTEB embedding benchmark |

Providers: `aws` (Amazon Bedrock; write the model's maker in its ID, e.g. `aws:anthropic.claude-opus-5-5`), `anthropic`, `google`, `openai` and `xai`. Every role is set as `provider:model@effort`, so any model can be swapped. `config.example.toml` also has a **direct-API** preset (no AWS) and a **premium** preset.

On Bedrock, Claude doesn't get Anthropic's automatic refusal fallback; a declined request fails that one step and the pipeline carries on with the next story. Claude prices are Anthropic's list prices (Bedrock's in-Region prices can be about 10% higher); the other prices and the Bedrock model IDs come from AWS announcements and third-party trackers, so confirm them with `python -m story_factory models`.

## How repeats are prevented

| Layer | What it catches |
|---|---|
| Recipe signature | The same setting + fear + twist combination can never be used twice. A database constraint enforces it |
| Cooldowns | The same setting within 8 stories, twist within 4, or fear within 3 |
| Legend ledger | Each legend is retold once. A database constraint enforces it |
| Prompt memory | Pitchers see the 40 most recent titles and are told to differ from them |
| Title check | Near-identical titles |
| Wording check | Synopses sharing too many word pairs with an earlier one |
| Meaning check | Embeddings catch the same story in different words; borderline cases go to an AI judge |
| Same-round twins | Two pitchers pitching the same idea in one round |
| Script check | Near-copies of earlier scripts, and any 8-word run copied from research text |

Rejected and shelved ideas stay in the database and keep blocking repeats.

## Learning from real results

Once pipeline 2 publishes videos, it records their YouTube metrics:

```bash
python -m story_factory record-performance 12 --views 48000 --avg-view-percentage 61 --ctr 7.4
```

Two things then improve future stories:

- **Ingredient weights:** settings, fears, twists, formats and endings from videos that keep viewers watching longer are drawn more often. This starts after 5 measured videos.
- **Written lessons:** `python -m story_factory learn` has a judge compare the best and worst performers and write insights. This needs at least 12 measured videos. Every future pitch prompt sees the latest insights.

## Setup

1. Install Python 3.11+ and the SDKs:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your keys (never paste them into a chat):

   | Variable | Where it comes from | Used for |
   |---|---|---|
   | `AWS_BEARER_TOKEN_BEDROCK` | AWS console → Amazon Bedrock → API keys | Every `aws:` model |
   | `AWS_REGION` | The region where you enabled the models, e.g. `us-east-1` | Bedrock endpoint |
   | `GEMINI_API_KEY` | aistudio.google.com | Gemini roles and repeat detection |
   | `ANTHROPIC_API_KEY` | console.anthropic.com | Legend research with web search only (a few dollars a month) |
   | `YOUTUBE_API_KEY` | console.cloud.google.com → YouTube Data API v3 | Trend research (free) |

   `OPENAI_API_KEY` and `XAI_API_KEY` are only needed if you switch roles to the direct `openai:` or `xai:` providers.

   In the AWS console, also open Bedrock → **Model access** and enable Claude Opus 5.5, Claude Sonnet 5.5, GPT-5.5 and Grok 4.6 in your region. Anthropic models ask for a short use-case form, and new accounts may need a quota increase before the pipeline can run at full speed.
3. Copy the example config:
   ```bash
   cp config.example.toml config.toml
   ```
4. Check every configured model against what your keys can use:
   ```bash
   python -m story_factory models
   ```
5. Create the database:
   ```bash
   python -m story_factory init
   ```

## Weekly run

```bash
python -m story_factory research
```

```bash
python -m story_factory fill --max 10
```

```bash
python -m story_factory status
```

`research` collects legends from 27 folklore categories worldwide (and their subcategories) plus popular horror video titles. `fill` also tops up legends from Wikipedia by itself whenever fewer than 15 unused ones remain, so a weekly `fill` alone keeps everything running.

| Command | What it does |
|---|---|
| `generate SERIES` | Make one story for `hollow_world`, `legends_retold` or `sixty_seconds` |
| `show ID` | Print a story with its outline, reviews, audience report and packaging |
| `world` | List the shared story world's canon |
| `claim --kind long` | (Pipeline 2) take the oldest approved story |
| `set-status ID STATUS` | (Pipeline 2) record progress, such as `published --video-id abc123` |
| `record-performance ID ...` | (Pipeline 2) save a video's YouTube metrics |
| `learn` | Turn measured performance into lessons for future pitches |
| `stats` | Cost per approved story, spend by model, which models win the draft contest, and the most common rejection reasons |

## Cost

Rough estimates for the balanced default, not yet measured on real runs:

| | Per story | Per month (13 long, 17 Shorts) |
|---|---|---|
| Long episode | ~$3–6 | ~$40–80 |
| Short | ~$0.50–1 | ~$10–17 |
| **Total** | | **~$50–100** |

The premium preset is roughly $100–250 a month. There is no spending cap by default; set `monthly_cap_usd` for a hard stop. Every call's tokens and estimated price are logged in the `costs` table, so `status` shows the real spend after the first runs.

## Tests

```bash
python -m unittest discover -s tests -t .
```

The pipeline tests run offline with fake models and check: judges never score their own company's work; the best draft wins; runners-up are shelved and used first; repeats are blocked; weak audience moments are rewritten; a polish that changes the story is discarded; and failures don't leave stories stuck. With the SDKs installed, the provider tests also run the real Anthropic and OpenAI SDKs against a mocked network to check requests and response handling.

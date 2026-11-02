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

The default is a **balanced** setup: the strongest affordable writer writes the story, and cheaper models do the scoring, viewer testing and planning.

| Role | Default model | Price per million tokens (in / out) | Why |
|---|---|---|---|
| Head writer: outline, revisions, polish, packaging | Claude Opus 5.5 | $4 / $20 | Top tier for fiction at 2.5× less than Claude Fable 5.1. The writing is the product, so this is where money goes |
| Draft writers | Claude Opus 5.5, Grok 4.7 | $4 / $20, $2 / $6 | Two competing drafts from two companies. Grok has ranked near the top for fiction and has the cheapest output among strong models |
| Pitchers | Claude Sonnet 5.5, Gemini 3 Pro, Grok 4.7 | $2 / $10, $2 / $12, $2 / $6 | Pitches are short; three companies give varied ideas |
| Judges and editors | Claude Sonnet 5.5, Gemini 3 Pro, GPT-5 mini, Grok 4.7 | as above, GPT-5 mini $0.25 / $2 | A judge never scores its own company's work; with four companies, almost every comparison gets two independent judges |
| Simulated viewers | Gemini 3 Flash, GPT-5 mini | $0.50 / $3, $0.25 / $2 | Reacting like a viewer doesn't need a top model |
| Researcher (web search) | Claude Sonnet 5.5 | $2 / $10, plus search fees | Builds a sourced fact file on each legend |
| Scene planner | Claude Sonnet 5.5 | $2 / $10 | Long, exact structured output |
| Repeat detection | Gemini embedding model | very low | Top proprietary model on the MTEB embedding benchmark |

`config.example.toml` also has a **premium** preset (Claude Fable 5.1 and GPT-5.5 as writers, at roughly 2–3× the cost). Providers: `anthropic`, `openai`, `google` and `xai` (Grok). Every role is set as `provider:model@effort`, so any model can be swapped. Claude prices are Anthropic's list prices; the GPT and Gemini prices and model names come from third-party trackers, so check them with `python -m story_factory models` and on each provider's pricing page.

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
2. Set your API keys:
   ```bash
   export ANTHROPIC_API_KEY="..."
   ```
   ```bash
   export OPENAI_API_KEY="..."
   ```
   ```bash
   export GEMINI_API_KEY="..."
   ```
   ```bash
   export XAI_API_KEY="..."
   ```
   ```bash
   export YOUTUBE_API_KEY="..."
   ```
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

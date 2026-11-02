# Story Factory (pipeline 1)

The story factory is a writers' room made of AI models from several companies. It researches, pitches, writes, edits and tests horror stories. Approved stories go into a SQLite database, where the production pipeline (pipeline 2) claims them and turns them into videos.

Its goal is the best possible story, not the cheapest. Every step either creates competition (several pitches, several drafts) or adds an independent critic (editors, simulated viewers, a final checker).

## The flow

```
                  ┌───────────────────────── Research (weekly) ──────────────────────────┐
                  │ Wikipedia legends · YouTube demand titles · web-researched dossier    │
                  └──────────────────────────────────┬────────────────────────────────────┘
                                                     ▼
 1. PITCH CONTEST   3 pitchers (Claude, GPT, Gemini), each with its own fresh recipe, pitch 4 ideas
                    → 8-layer repeat check → judges from OTHER companies score every pitch
                    → top 4 go head-to-head, each pair judged in both orders → winner
                    → strong runners-up are SHELVED for a later run
 2. OUTLINE         head writer builds a beat sheet: hook, escalations, false relief, clue plants/payoffs
                    → independent editors approve it or send it back
 3. DRAFT CONTEST   3 writers draft the full script from the outline → head-to-head judging
 4. EDITORS' LOOP   independent editors score 7 qualities; median of every one must reach 8/10
                    → head writer revises against their line notes (up to 3 rounds) or the story is rejected
 5. AUDIENCE TEST   5 simulated viewers rate every ~30 seconds and say where they'd leave
                    → weak parts are rewritten, kept only if judges agree the new version is better
 6. POLISH          read-aloud line edit for one narrator voice + pronunciation guide for the voice engine
                    (discarded if it changes the story)
 7. FINAL CHECK     rule violations, legend accuracy against the dossier, cultural sensitivity
 8. SCENE PLAN      exact narration spans, visuals, sound cues, key moments that need new images
 9. PACKAGING       5 titles, thumbnail brief and text, description with credit, tags, pinned comment
                    → APPROVED, waiting for pipeline 2
```

Shorts go through the same room with shorter targets.

## Who does what

| Role | Default model | Why |
|---|---|---|
| Pitchers | Claude Fable 5.1, GPT-5.5, Gemini 3 Pro | Different companies' models have different instincts, so the ideas are genuinely varied |
| Head writer (outline, revisions, packaging) | Claude Fable 5.1 | Top of the creative-writing leaderboards with OpenAI's flagship, and Anthropic's most capable model |
| Draft writers | Claude Fable 5.1 (×2, at different effort levels), GPT-5.5 | Three competing drafts |
| Judges and editors | Claude Opus 5.5, GPT-5.5, Gemini 3 Pro | A judge never scores work from its own company, because models measurably favor their own family's writing |
| Simulated viewers | GPT-5.5, Gemini 3 Pro | Not the writer's family, so they react like outsiders |
| Polisher | Claude Fable 5.1 | Read-aloud line edit |
| Researcher | Claude Opus 5.5 with web search | Checks legends against real sources before anyone writes |
| Scene planner | Claude Opus 5.5 | Long, exact structured output |
| Repeat detection | Gemini embedding model | Top proprietary model on the MTEB embedding benchmark |

Every role is set in `config.toml` as `provider:model@effort`, so models can be swapped as better ones appear. The GPT and Gemini model names are the best I could confirm; check them with `python -m story_factory models` before the first run.

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

## Cost

Cost is no longer capped by default (`monthly_cap_usd = 0`); set a number to add a hard stop. A long episode makes roughly 80 model calls. A rough estimate, not measured on real runs, is $5–15 per long episode and $1–3 per Short. That works out to about $100–250 a month at 13 long episodes and 17 Shorts. Every call's tokens are logged in the `costs` table, so after the first runs `status` shows the real figure.

## Tests

```bash
python -m unittest discover -s tests -t .
```

The pipeline tests run offline with fake models and check: judges never score their own company's work; the best draft wins; runners-up are shelved and used first; repeats are blocked; weak audience moments are rewritten; a polish that changes the story is discarded; and failures don't leave stories stuck. With the SDKs installed, the provider tests also run the real Anthropic and OpenAI SDKs against a mocked network to check requests and response handling.

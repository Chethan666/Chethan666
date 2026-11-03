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
 2. STORY BIBLE     before any plotting, the head writer decides what a storyteller decides first:
                    a HERO (want, wound, flaw, false belief, one human detail), the THREAT's rules and
                    limits, the THEME and the question the ending answers, the hero's CHOICE, what the
                    viewer knows before the hero (dramatic irony), a breath moment, a REFRAIN that
                    returns darker, a sound motif, the meaning of the red accent, a deliberate
                    ambiguity, the AFTERTASTE that follows viewers home, and clues with payoffs
                    → free completeness check → independent editors approve it or it is revised
 3. OUTLINE         beat sheet built from the bible: every beat has a turn and leaves an open question;
                    planned REHOOKS (~30 s, ~2 min, midpoint), a midpoint reversal and a peak moment
                    → free structure check → independent editors approve it or send it back
 4. DRAFT CONTEST   2 writers draft the script; long episodes are written in 3 PARTS (each part sees the
                    story so far), then The Storyteller's frame: a COLD OPEN flashed forward from the
                    story, "The lantern is lit.", and an outro that lets the fear follow viewers home;
                    Shorts aim for a last line that loops back into the first
                    → head-to-head judging
 5. EDITORS' LOOP   each round: fit the length → FREE STYLE CHECKS (machine-sounding phrases, named
                    feelings, long sentences, repeated openings, dashes, hard words) with a dedicated
                    line edit (only machine phrasing blocks; the rest is advice for the editors)
                    → STORY-BIBLE CHECK (a judge must quote where each promise is delivered;
                    quotes are verified in code; broken threat rules block) → editors score 8 qualities
                    against a written scoring guide; the median of every one must reach the bar
                    → revisions use the editors' quoted line notes with fixes, or the story is rejected
 6. HOOK LAB        2 alternative openings compete with the original for the first ten seconds;
                    the original stays unless judges prefer a new one
 7. AUDIENCE TEST   5 simulated viewers rate every ~30 seconds and say where they'd leave
                    → weak parts are rewritten, kept only if judges agree the new version is better
 8. POLISH          read-aloud line edit for one narrator voice, a pronunciation guide and up to 12
                    DRAMATIC PAUSES for the voice engine (discarded if it changes the story)
 9. FINAL CHECK     rule violations, legend accuracy against the dossier, cultural sensitivity
10. SCENE PLAN      exact narration spans, visuals (red only for the story's red accent), sound cues
                    including the story's sound motif, a delivery cue for the narrator, key moments,
                    and a CAST with one fixed look per character (recurring characters keep their look)
11. PACKAGING       5 titles, the top 3 ranked head-to-head, a phone-readable thumbnail (one or two
                    figures, the red accent, at most 3 words) that adds to the title, description with
                    credit, tags, pinned comment, and an end-screen line inviting the next story
12. TEASER SHORT    every long episode also gets a 60-second teaser Short with its own scene plan,
                    stopping on a cliffhanger that sends viewers to the full episode
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

## The storyteller's craft, written down

Weaker models write well when the craft is spelled out instead of assumed, and stronger
models write even better from the same instructions. Every writer prompt carries:

- **What the audience needs**, in psychological terms: curiosity with new questions before old
  ones close, a hero we care about who makes choices, dread over surprise, lingering before the
  reveal, fear that breathes, fair play, the peak and the ending, a human meaning under the fear,
  an aftertaste that follows the viewer home, and one deliberate ambiguity to argue about.
- **A house style with examples**: three short original passages showing a hook, dramatic irony
  and an ending with an aftertaste, plus a "never like this" passage with its fix.
- **The Storyteller's voice and ritual**: warm, unhurried, speaks to "you", never explains;
  every long episode opens cold on a flash-forward, then "The lantern is lit." (added
  automatically if missing). Simulated viewers rate the intro too.
- **Every hero is new**: the story bible sees the heroes of the last 12 episodes and must
  differ in age, gender, background and culture, with an invented name no famous person has.
- **Advertiser-friendly**: strong profanity blocks a story.
- **Writing for the ear and a global audience**: plain words, short sentences, a refrain,
  writing for sound.

Judges get a **scoring guide** that says what a 4, 6, 8 and 10 look like for each quality
(hook, tension, character, emotional payoff, originality, prose for the ear, drawability,
following the recipe), so different models score on the same scale.

**Creativity per task**: where a model accepts it (the open models on Bedrock, Grok, Gemini),
creative tasks run at a higher temperature and judging at a low one (`[temperature]` in the
config). A model that rejects the setting is retried without it.

**Fewer judge calls, more writing**: quick comparisons (openings, titles, audience rewrites)
use 2 judges, full contests 3 (`quick_judges`, `judges_per_comparison`). The savings pay for
the story bible, writing in parts and the bible check.

**Between GitHub runs, the database persists**: each run restores the database the last run
saved (an Actions artifact kept 90 days and re-saved every run), so repeat prevention, the
shelf, the shared world and the hero history all carry over.

**Judges that loop fail fast**: quick judgments have a smaller output limit
(`[task_max_tokens]`), so a model stuck in a loop gives up in seconds and the panel carries on.

## Benchmarking model setups

Before switching models, prove the upgrade:

```bash
python -m story_factory --config config.aws-open.toml bench --against config.example.toml --pitches 2
```

Both setups write the same fixed pitches (story bible → outline → drafts → line edit →
polish). Judges from both setups compare the two scripts blind, in both orders. The table
shows wins, word counts and cost per setup. In GitHub Actions, run the workflow with
`mode = bench`.

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
| `report [ID]` | Markdown summary of a story: verdict, hero and theme, scores, audience, titles, cost |
| `bench --against FILE` | Blind-compare this config's writing against another config on fixed pitches |

## Cost

Rough estimates for the balanced default, not yet measured on real runs:

| | Per story | Per month (13 long, 17 Shorts) |
|---|---|---|
| Long episode | ~$3–6 | ~$40–80 |
| Short | ~$0.50–1 | ~$10–17 |
| **Total** | | **~$50–100** |

The open models on Bedrock (`config.aws-open.toml`) cost roughly a tenth of the balanced setup. The story bible, writing in parts and the bible check add roughly 10–15 calls per long story; capping quick comparisons at 2 judges, contests at 3, the hook lab at 2 alternatives and the title contest at 3 finalists removes roughly 100, so a long story now takes fewer calls than before.

The premium preset is roughly $100–250 a month. There is no spending cap by default; set `monthly_cap_usd` for a hard stop. Every call's tokens and estimated price are logged in the `costs` table, so `status` shows the real spend after the first runs.

## Tests

```bash
python -m unittest discover -s tests -t .
```

The pipeline tests run offline with fake models and check: judges never score their own company's work; the best draft wins; runners-up are shelved and used first; repeats are blocked; weak audience moments are rewritten; a polish that changes the story is discarded; and failures don't leave stories stuck. With the SDKs installed, the provider tests also run the real Anthropic and OpenAI SDKs against a mocked network to check requests and response handling.

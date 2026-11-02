# Story Factory (pipeline 1)

The story factory researches inspiration, writes original horror stories, rejects anything that repeats an earlier story, and stores approved stories in a SQLite database. The production pipeline (pipeline 2, not built yet) claims stories from that database and turns them into videos.

It uses only the Python standard library and the Gemini API free tier, so it costs $0 to run.

## Setup

1. Install Python 3.11 or newer.
2. Create a Gemini API key in Google AI Studio, then set it:
   ```bash
   export GEMINI_API_KEY="your-key"
   ```
3. Optional: create a YouTube Data API key in Google Cloud for audience-demand research:
   ```bash
   export YOUTUBE_API_KEY="your-key"
   ```
4. Copy the example config and adjust it if needed:
   ```bash
   cp config.example.toml config.toml
   ```
5. Check which Gemini models your key can use, and update `text_model` and `embedding_model` in `config.toml` if the defaults are not listed:
   ```bash
   python -m story_factory models
   ```
6. Create the database:
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

`research` collects folklore and popular horror video titles. `fill` generates stories until each series has its target number of approved stories waiting (4 *Tales from the Hollow*, 2 *Legends Retold*, 8 *60-Second Horror* by default). Schedule both with cron or Windows Task Scheduler.

Other commands:

| Command | What it does |
|---|---|
| `generate SERIES` | Make one story for `hollow_world`, `legends_retold` or `sixty_seconds` |
| `show ID` | Print a story as JSON |
| `world` | List the shared story world's canon |
| `claim --kind long` | (Pipeline 2) take the oldest approved story; marks it `in_production` |
| `set-status ID STATUS` | (Pipeline 2) record progress, such as `published --video-id abc123` |

## How a story is made

1. **Recipe.** A setting, core fear, protagonist, format, twist and ending are drawn from `story_factory/catalog.py`. There are millions of combinations.
2. **Research.**
   - *Legends Retold* takes one Wikipedia legend that has never been used.
   - *Tales from the Hollow* gets two folklore motifs as sparks, plus the shared world's canon.
   - Every series sees popular YouTube horror titles, as demand signals only.
3. **Premises.** Gemini pitches three premises. The first one that passes the repeat checks is kept.
4. **Script.** The script is written to a fixed structure: Storyteller intro, hook in the first sentence, rising dread, false relief, climax, twist or emotional ending, and an outro question. It has to deliver eight audience needs (curiosity, safe thrill, relatability, attachment, fair clues, an earned ending, an emotional core, something to discuss).
5. **Checks.**
   - Word count (1,200-1,500 for long episodes, about 8.5-10.5 minutes; 110-170 for Shorts).
   - No claims that the story is true.
   - No passage copied from the research text.
   - A strict editor review that scores seven qualities and flags rule breaks (gore, harm to children, real people, copyrighted characters and more).
   - Failing scripts are revised up to twice, then rejected.
6. **Scene plan.** The narration is split into scenes with exact text spans, locations, visuals, mood and sound cues. Key moments are marked because they need brand-new images. The plan is validated against the narration.
7. **Approved.** New characters and places join the shared world for future stories.

## How repeats are prevented

| Layer | What it catches |
|---|---|
| Recipe signature | The same setting + fear + twist combination can never be used twice. A database constraint enforces it |
| Cooldowns | The same setting within 8 stories, twist within 4, or fear within 3 |
| Legend ledger | Each legend is retold once. A database constraint enforces it |
| Prompt memory | Gemini sees the 40 most recent titles and loglines and is told to differ from them |
| Title check | Near-identical titles |
| Wording check | Synopses that share too many word pairs with an earlier one |
| Meaning check | Embeddings catch the same story told in different words. Clear matches are rejected; borderline ones go to an AI judge that compares the two stories directly |
| Script check | Near-copies of earlier scripts, and any 8-word run copied from research text |

Rejected ideas stay in the database and keep blocking repeats, so the factory never wastes calls pitching them again.

## Sources

- **Wikipedia:** legends and motifs, CC BY-SA. Used as research notes only; scripts are checked for copied passages, and legend episodes credit the tradition.
- **YouTube Data API:** titles and view counts of popular horror videos. These are demand signals; plots and titles are never copied.
- **Not used on purpose:** Reddit, creepypasta sites and other authors' stories, because those stories belong to their writers.

## Tests

```bash
python -m unittest discover -s tests -t .
```

The tests run offline with a fake model and cover the repeat checks, the legend ledger, copy detection, revisions, scene validation, the budget cap and claiming.

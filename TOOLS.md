# Tools and budget

> **Update (November 2026):** the production pipeline now runs on GitHub Actions with open-source tools: Kokoro for the narrator's voice, chalk stickman pictures drawn in code with art direction from an open model on Bedrock, and FFmpeg. See [PRODUCTION.md](PRODUCTION.md). The tool table below is the original plan.
>
> **Update:** the story pipeline (pipeline 1) now uses the best available models from Anthropic, OpenAI and Google instead of the Gemini free tier, and is no longer bound by the $20 limit. With the balanced default models it is estimated at about $50–100 a month; see [STORY_FACTORY.md](STORY_FACTORY.md). The table below still describes the production pipeline (pipeline 2), which hasn't been redesigned yet.

Original limit: **$20 per month**, with all paid calls stopping at **$18**.

Output: 3 long episodes per week (8+ minutes) and 4 Shorts per week, about 13 long videos and 17 Shorts a month, in English.

## Final tool stack

| # | Job | Tool | Runs on | Monthly cost |
|---|---|---|---|---|
| 1 | Orchestrates every step, saves checkpoints, resumes after failures | Our Python orchestrator | Your PC | $0 |
| 2 | Runs the pipeline on schedule | cron (macOS/Linux) or Task Scheduler (Windows) | Your PC | $0 |
| 3 | Story ideas, scripts, scene plans, titles, descriptions, repeat detection | Writers' room of Claude, GPT and Gemini models ([STORY_FACTORY.md](STORY_FACTORY.md)) | Anthropic, OpenAI, Google | ~$50–100 (estimate) |
| 3a | Story research: legends and motifs | Wikipedia API | Wikipedia | $0 |
| 3b | Story research: what horror viewers watch | YouTube Data API | Google | $0 |
| 3c | Story database shared by both pipelines | SQLite | Your PC | $0 |
| 4 | Narration (one fixed channel voice) | Gemini API, text-to-speech model | Google | ~$2–4 |
| 5 | Scene images and character poses (~265 new images a month) | Gemini API, image model, Batch mode | Google | ~$5–6 |
| 6 | Transparent backgrounds for images | rembg | Your PC | $0 |
| 7 | Word timing for captions | OpenAI Whisper (local) | Your PC | $0 |
| 8 | Horror ambience and sound effects | Procedurally generated sounds and CC0 sounds from Freesound | Your PC | $0 |
| 9 | Animation and video rendering | stickman-video-director (our fork with dark mode and sound support), Node.js 20+, FFmpeg | Your PC | $0 |
| 10 | Thumbnails | Generated in code with Pillow from character art and bold text | Your PC | $0 |
| 11 | Uploading and scheduling videos | YouTube Data API v3 | Google | $0 |
| 12 | Tracking performance to choose future stories | YouTube Analytics API | Google | $0 |
| 13 | Spending guard | Built into the orchestrator: logs every paid call, hard stop at $18 | Your PC | $0 |

One-time setup cost: about **$4–8** to generate the starting character and prop library (about 200 images).

## Monthly estimate

| Item | Estimate |
|---|---|
| Images: ~15 new per long episode, ~3 per original Short, plus ~20% retries (~265 images) | ~$5–6 |
| Narration (~110 min of long episodes + ~17 min of Shorts) | ~$2–4 |
| Everything else | $0 |
| **Total** | **~$8–10 per month** |

The image library is not a one-time cost. It is built once (~200 images for the host, recurring cast, common props and places) and reused, but every story still needs new images for its own characters, locations, key scary moments and thumbnail. The library lowers the count over time; it never reaches zero.

If Batch mode is not available for the chosen image model, images cost about twice as much and the total rises to about $13–15. That is still under the limit, and the $18 hard stop always applies.

Prices are estimates from third-party pricing sites, not Google's official page. Check current model prices in Google AI Studio before launch.

## Accounts and keys needed

- **Google Cloud project with billing enabled:** provides the Gemini API key (`GEMINI_API_KEY`).
- **YouTube channel:** authorize the YouTube Data and Analytics APIs once with OAuth.
- **Freesound account:** free API key for CC0 sounds.

## Not used

- Google AI Pro
- ChatGPT and Claude subscriptions
- Cloud servers
- Paid video editors
- Paid voice services such as ElevenLabs

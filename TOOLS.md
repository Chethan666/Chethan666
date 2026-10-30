# Tools and budget

Hard limit: **$20 per month**. The pipeline stops all paid calls at **$18** to leave a safety margin.

Output: 3 long episodes per week (8+ minutes) and 4 Shorts per week, about 13 long videos and 17 Shorts a month, in English.

## Final tool stack

| # | Job | Tool | Runs on | Monthly cost |
|---|---|---|---|---|
| 1 | Orchestrates every step, saves checkpoints, resumes after failures | Our Python orchestrator | Your PC | $0 |
| 2 | Runs the pipeline on schedule | cron (macOS/Linux) or Task Scheduler (Windows) | Your PC | $0 |
| 3 | Story ideas, scripts, shot plans, titles, descriptions | Gemini API, Flash text model, free tier | Google | $0 |
| 4 | Narration (one fixed channel voice) | Gemini API, text-to-speech model | Google | ~$2–4 |
| 5 | Scene images and character poses | Gemini API, image model, Batch mode | Google | ~$6–8 |
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
| Images (~22 per long episode, a few per original Short) | ~$6–8 |
| Narration (~110 min of long episodes + ~17 min of Shorts) | ~$2–4 |
| Retries and spare | ~$2 |
| Everything else | $0 |
| **Total** | **~$10–14 per month** |

If Batch mode is not available for the chosen image model, images cost about twice as much and the total rises to about $15–17. That is still under the limit, and the $18 hard stop always applies.

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

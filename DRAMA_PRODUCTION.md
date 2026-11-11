# Pipeline 4: drama videos

Pipeline 4 turns an approved episode from pipeline 3 into a vertical 1080x1920 video that
looks like a TV series. It is a clone of pipeline 2 (the stickman video pipeline) rebuilt
for drama, with its own voice and sound modules and its own settings (`[drama_video]`).

## The flow

```
REFERENCES   (fal.ai) one portrait per character and one establishing plate per location,
             made once per series and kept with the database (data/series/<id>/), so
             faces, clothes and places stay the same in every episode
SHOTS        (fal.ai, reference-based editing) every shot's picture from those references,
             framed as the script asks: establishing, close-up, over-the-shoulder, insert...
MOTION       (fal.ai, image-to-video) real moving clips for the key beats, at most
             `max_motion_shots`, within `max_usd_per_episode`
VOICES       one fixed voice per character, picked from Kokoro's voices by gender. With
             BOSON_API_KEY set, Boson Higgs Audio v3 speaks every line, cloned from a short
             clip of that voice (made once per series and kept with it), so each character
             sounds the same in every episode; a refused line is read by Kokoro instead.
             Thoughts are softer with a little room
EDIT         a cut on every line like TV; slow camera moves on stills by framing; a film look
             (grade, vignette, grain); big subtitles (thoughts in italics); a sound bed
→ episode.mp4, thumbnail.png, metadata.json (the episode is marked produced)

Paid pictures and clips for an episode are kept with the series until the episode is
made, so a failed run never pays for them twice.
```

## Keys

GitHub Actions secrets (Settings → Secrets and variables → Actions):

| Secret | Used for |
|---|---|
| `FAL_KEY` | references, shot pictures and moving clips |
| `BOSON_API_KEY` | character voices; without it, Kokoro voices are used |

## Commands

| Workflow mode | Command | What it does |
|---|---|---|
| `drama-check` | `python -m drama_production check` | each configured fal.ai model's inputs, and the Boson voices; no spend |
| `drama-video` | `python -m drama_production produce [--episode ID]` | the next approved episode's video, saved as the `drama-<run>` artifact for 14 days |

## Cost

Estimated from third-party price figures (check fal.ai's pricing pages): about $0.03 per
picture and $0.036 per second of moving clip. Every shot is a moving clip, so a 90-second
episode with ~20 shots is about $3.85 (capped by `max_usd_per_episode`, $4.50), plus about
$0.25 once per series for references.
Every call is logged in the `costs` table under the `drama` pipeline.

## Open models on a rented GPU (mode drama-vast)

`python -m drama_production vast` (workflow mode **drama-vast**) makes an episode with open
models only, on a GPU rented from Vast.ai with the `VAST_API_KEY` secret. With no `story_id`
it makes the ready 60-second script `drama_production/scripts/the-last-call.json`; with an
episode id it makes that episode from the database.

- **Here (free)**: a Kokoro reference clip of each character's fixed voice, Kokoro fallback
  lines, and afterwards the edit (a cut on every line, film look, big subtitles, sound bed).
- **On the GPU**: Chatterbox (MIT) speaks every line, cloned from its character's voice;
  Qwen-Image-Edit-2511 (Apache-2.0) draws a portrait per character (a double is drawn from the
  original's portrait), a plate per location and every shot from those references;
  Wan2.2-S2V-14B (Apache-2.0) films on-screen speakers lip-synced to their line; Wan 2.2 I2V
  A14B (Apache-2.0) films every other shot and any talking shot S2V couldn't make; MMAudio
  makes the room's ambience.
- **Money**: only hosts that charge **nothing** for internet traffic are rented (run 77 paid
  $6.72 for downloads alone), cheapest true cost first, with download time counted as rented
  time. The run stops before `drama_vast.max_usd` (default $15) or `drama_vast.max_hours` (5),
  keeps what it has made (missing clips become stills with camera moves), and the machine is
  destroyed in every case. `video-vast-check` shows the balance and offers without spending.

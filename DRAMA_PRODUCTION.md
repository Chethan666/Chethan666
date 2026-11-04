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
VOICES       one fixed voice per character: Boson Higgs TTS once its voices are listed in
             `boson_voices`, otherwise Kokoro (free); thoughts are softer with a little room
EDIT         a cut on every line like TV; slow camera moves on stills by framing; a film look
             (grade, vignette, grain); big subtitles (thoughts in italics); a sound bed
→ episode.mp4, thumbnail.png, metadata.json (the episode is marked produced)
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
picture and $0.036 per second of moving clip, so a 90-second episode with ~20 shots and
three 5-second clips is roughly $1.20, plus about $0.25 once per series for references.
Every call is logged in the `costs` table under the `drama` pipeline.

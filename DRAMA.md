# Short drama series

Vertical 9:16 episodes of 60–120 seconds, told in dialogue, in the style of the
micro-drama apps: a hook in the first three seconds, a reversal in every episode and a
cliffhanger at the end. Original stories only; the format and tropes are free to use,
other shows are not.

## The flow

```
SERIES BIBLE   (open models on Bedrock) title, logline, genre, 3-6 characters with a fixed
               look, wardrobe and voice, a secret each, 3-6 recurring locations, the arc,
               and every episode's hook, beats and cliffhanger
EPISODES       one script per episode, shot by shot: framing (close-up, over-the-shoulder,
               insert...), location, who is in frame, what we see, who speaks which line
               and how, inner thoughts, and up to four shots that must really move.
               Free checks (known cast and places, mostly dialogue, short lines), then
               judges from other companies score hook, conflict, dialogue, cliffhanger and
               continuity; one revision; the best version is kept, approved at 7+
REFERENCES     (fal.ai) one portrait per character and one plate per location, made once
               per series and kept with the database, so faces, clothes and places stay
               the same in every episode
SHOTS          (fal.ai) every shot's picture is generated from those references
MOTION         (fal.ai) image-to-video clips for the key beats, within a cost cap per episode
VOICES         one fixed voice per character: Boson Higgs TTS when its voices are
               configured, otherwise Kokoro (free); thoughts are softer with a little room
EDIT           1080x1920, a cut on every line like TV, slow camera moves on stills, a film
               look (grade, vignette, grain), big subtitles (thoughts in italics), a sound
               bed under the voices → episode.mp4, thumbnail.png, metadata.json
```

## Keys

Add these as GitHub Actions secrets (Settings → Secrets and variables → Actions):

| Secret | Used for |
|---|---|
| `FAL_KEY` | pictures and moving clips (fal.ai) |
| `BOSON_API_KEY` | character voices (Boson Higgs TTS); without it, Kokoro voices are used |
| `AWS_BEARER_TOKEN_BEDROCK` | already set: the open models that write and judge |

## Running it (workflow modes)

1. `drama-check`: lists every configured fal.ai model's inputs and the Boson voices,
   without spending anything. Fix model names or voice ids in `[drama]` if needed.
2. `drama-series`: writes a new series bible.
3. `drama-write` (story_id = how many episodes, default 3): writes the next episodes.
4. `drama-video` (story_id = an episode id, optional): makes the next approved episode's
   video, saved as the run's `drama-<run number>` artifact for 14 days.

One-off changes go in `overrides`, e.g. `drama.genre="revenge romance"`.

## Cost

Estimated, from third-party price figures (check fal.ai's pricing pages):
about $0.03 per picture and $0.036 per second of moving clip. A 90-second episode with
~20 shots and three 5-second clips is roughly $1.20, plus about $0.25 once per series for
the references. `max_usd_per_episode` (default $3) stops moving clips beyond the cap.
Every call is logged in the `costs` table under the `drama` pipeline.

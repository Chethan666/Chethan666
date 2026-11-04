# Production (pipeline 2)

Pipeline 2 turns an approved story into a finished video. It runs on GitHub Actions
(workflow mode `video`) with open-source tools and the same open models as the story
factory, so it costs nothing beyond the few art-direction calls. Uploading to YouTube
comes later; for now each run saves the video as a downloadable artifact.

## The flow

```
claim the next approved long story (or a chosen one)
 1. VOICE        Kokoro (82M, Apache-2.0) reads the narration on the CPU, chunk by chunk:
                 sentences, cut wherever the story asks for a dramatic pause. Each scene's
                 delivery cue sets the speed (slow, hushed, tense...), hard names are read
                 from their respellings, and intro, paragraph and outro breaks get longer
                 silences. Every chunk's start and end time is known exactly.
 2. TIMING       scenes are found in the narration and timed from their first word;
                 captions are split evenly within each spoken chunk
 3. DIRECTION    an open model (the story factory's planner role) turns each scene's plan
                 into a drawable composition from a fixed vocabulary: background (14),
                 light, camera move, up to three figures with pose (13) and place, up to
                 three props (40), and the one element drawn in red. If the model fails,
                 keywords in the scene plan decide
 4. DRAWING      chalk stickman pictures drawn in code: white chalk on black paper, one
                 red accent, each character with the same features in every scene (read
                 from the cast's fixed looks), unseen things as dark shapes with red eyes,
                 ghosts dashed. Three jittered versions per scene make the hand-drawn boil
 5. SOUND        a low drone and air under everything, darker in tense scenes and dipping
                 under the voice, plus synthesized cues the scene asks for (knock,
                 heartbeat, phone, footsteps, rain, wind, thunder, static, drip, creak, bell)
 6. ASSEMBLY     FFmpeg: each scene a clip with its camera move (push in, pull out, pan,
                 shake), flicker where the light flickers, fades through black; joined,
                 captions burned in, sound mixed
→ video.mp4 (1080p), thumbnail.png, captions.srt, narration.wav, metadata.json (titles,
  description, tags, pinned comment, end-screen line), direction.json
```

The story is marked `produced` with a summary in its `production` field. If anything
fails, a claimed story goes back to `approved` with the error recorded.

## Running it

In GitHub Actions: run the workflow with **mode = video** (optionally a **story_id**).
Download the result from the run's **Artifacts** (`video-<run number>`, kept 14 days).
This repository is public, so signed-in GitHub users can download those artifacts too.

Locally (Python 3.10–3.12, FFmpeg and espeak-ng installed):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install -r requirements-video.txt
```

```bash
python -m production --config config.aws-open.toml produce --story 12
```

`--voice silent` swaps the narrator for a placeholder tone, and `--no-direction-model`
directs from keywords only, for testing the rest without downloads or model calls.

The narrator is set in the config: `[video] voice = "am_michael"`, `lang = "a"` (US
English; `"b"` is British), `speed = 0.92`.

## Later

- Upload and scheduling on YouTube, then performance tracking (`record-performance`).
- Stronger voice and image models once paid models are available; the drawing step can
  take generated images per scene without changing the rest.
- Shorts, when they return.

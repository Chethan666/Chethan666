# Production (pipeline 2)

Pipeline 2 turns an approved story into a finished video. It has two modes:

- **Studio mode** (the default whenever the `FAL_KEY` secret is set): every shot is a generated
  chalk picture, the key shots are real clips, ElevenLabs performs the narration, a generated
  score and sound effects carry the scares, and the edit is cut like a film. About $30–45 per
  8–9 minute video, capped by `studio.max_usd`.
- **Drawn mode** (no key, `--drawn`, or `studio.enabled = false`): the free pipeline described
  further down, with pictures drawn in code and Kokoro's voice.

## Studio mode

```
claim the next approved long story (or a chosen one)
 1. VOICES      ElevenLabs v3 (on fal) reads every chunk at once, in parallel; each scene's
                delivery cue becomes an audio tag ([whispers], [nervously], [slowly]...), and each
                speaking character gets a fixed voice of their kind. A chunk that fails is read by
                Kokoro. Every chunk's start and end time is known exactly.
 2. SHOT LIST   the timed chunks are cut into shots of 3-6.5 s (about 110 for an 8.5-minute
                story). An open model stages each one like a storyboard artist: framing (wide,
                medium, close, insert, point of view, low or high angle), what the picture shows
                and how the stick figures act with their whole bodies, the camera move, what moves
                in a clip, one sound, whether it is a scare, and how much it matters (1-5). The
                most important shots become hero clips, the next ones medium clips, spread out
 3. LOOK        Nano Banana Pro draws a style frame (the main place in the channel's chalk look)
                and a character sheet per cast member (front, side, walking). Every shot is drawn
                from those references, so the style and the characters hold from the first shot
                to the last. Each picture is checked in code (mostly black, white chalk, no colour
                but the red accent) and by a vision model (the right style and the right moment);
                a failure is redrawn by Seedream 4.5, and the code renderer is the last resort
 4. CLIPS       hero shots: Seedance 2.0 (Kling 3 if it fails); medium shots: Seedance lite.
                Each clip is as long as its shot needs. Frames of every clip are checked; a clip
                that loses the chalk look is dropped and the shot stays a living still (the
                picture with the chalk boil and a slow camera move)
 5. SOUND       three generated score cues (unease, dread, terror) follow the story's tension and
                crossfade; generated room tone per place; one generated effect per shot that asks
                for it; foley made from the hero clips; a stinger on every scare, with the music
                and room tone dropping away for a beat before it. Everything dips under the voice
 6. EDIT        cold open (the scariest moment before the ending, 6-14 s), a 2 s title card, the
                story cut hard on its own sentence breaks, and a 20 s end screen for YouTube's
                end-screen elements. Loudness -14 LUFS. Captions as captions.srt (burned in only
                with studio.burn_captions)
 7. PACKAGING   three thumbnails (the victim, the threat, the place) for YouTube's Test &
                Compare; two or three vertical Shorts cut from the scariest stretches (never the
                ending) with big captions; chapters in the description
 8. REVIEW KIT  contact_sheet.jpg (every shot, marked clip or still) and preview_60s.mp4
→ video.mp4 (1080p), thumbnail_1-3.jpg, short_1-3.mp4, captions.srt, metadata.json (title,
  description with chapters, tags, comment, end-screen time), shots.json, report.json (what
  was made, what fell back, and the spend by step), narration.wav
```

Every paid step has a free fallback, and one cap covers the whole video: work is reserved
before it is paid for, most important first, so parallel jobs never overspend. Paid results
are cached by request within a run, so retries are free.

Typical spend for an 8.5-minute story (estimates in `studio.prices`; check fal.ai's pages):

| Step | About |
|---|---|
| Style frame, character sheets, ~110 shot pictures (Nano Banana Pro) | $17 |
| 20 hero clips (Seedance 2.0 fast, ~5 s) | $24 |
| 40 medium clips (Seedance lite) | $7 |
| Voices (ElevenLabs v3, ~9,000 characters) | $1 |
| Score, room tone, effects, foley, checks | $2–3 |

The cap (`studio.max_usd`, default 45) wins: past it, shots stay living stills and sounds are
synthesized. Lower `hero_shots` or switch `models.shot` to `fal-ai/nano-banana-2/edit`
(about half the price) for cheaper videos.

### Choosing models: the bake-off

`mode = video-bakeoff` (optionally with a `story_id`) runs four of a story's most important
shots through every picture model, the two most important through every motion model, the
opening through two ElevenLabs voices and Kokoro, plus a music cue and a stinger, side by side
with their checks and costs (capped at $10). `mode = video-check` lists what every configured
model accepts, at no cost.

### Review, then publish

Nothing goes to YouTube until a person has watched it:

1. Run **mode = video**. Download `video-<run number>` from the run's Artifacts. Watch
   `preview_60s.mp4`, look over `contact_sheet.jpg`, skim `video.mp4`.
2. Run **mode = video-approve** with the story's number (or **video-reject**, with the reason
   in `overrides`).
3. Run **mode = video-publish** with the story's number. It uploads the video as **private**
   with its title, description and chapters, tags, captions, first thumbnail and the comment
   question. Put a time in `overrides` (e.g. `2026-11-20T17:00:00Z`) to schedule it instead.
   In YouTube Studio: pin the comment, add the end-screen elements over the last 20 s, start
   a Test & Compare with the other two thumbnails, and upload the Shorts.
4. **mode = video-stats** reads every published video's views, likes, comments and average
   view percentage into the performance table that the story factory learns from.

Publishing needs three GitHub Actions secrets for the channel (Settings, Secrets and
variables, Actions): `YOUTUBE_CLIENT_ID`, `YOUTUBE_CLIENT_SECRET` and
`YOUTUBE_REFRESH_TOKEN`. To make them: in Google Cloud Console create a project, enable the
YouTube Data API v3 and the YouTube Analytics API, create an OAuth client (Desktop app), then
authorise it once for the channel's Google account with the scopes `youtube.upload`,
`youtube.force-ssl` and `yt-analytics.readonly` (for example with Google's OAuth Playground,
using your own client), and keep the refresh token. Never paste these into a chat or commit
them. Until the app is verified by Google, uploads from it stay private, which is what we
want anyway.

## Drawn mode

Drawn mode runs on GitHub Actions with open-source tools and the same open models as the
story factory, so it costs nothing beyond the few art-direction calls.

## The drawn flow

```
claim the next approved long story (or a chosen one)
 0. DIALOGUE     every quoted line is found; an open model says who speaks it (a cast
                 member, a minor speaker, or the first-person teller) and what kind of voice
                 they have; the words around the quote decide if the model fails
 1. VOICE        Kokoro (82M, Apache-2.0) reads the narration on the CPU, chunk by chunk:
                 sentences, cut wherever the story asks for a dramatic pause. Each scene's
                 delivery cue sets the speed (slow, hushed, tense...), hard names are read
                 from their respellings, and intro, paragraph and outro breaks get longer
                 silences. Every chunk's start and end time is known exactly. Each
                 character reads their own quoted lines in their own Kokoro voice (one per
                 speaker, never the narrator's); the teller's own lines stay with the narrator
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
 4b. MOTION      with FAL_KEY set, up to 8 scenes (key moments first, then the tensest)
                 become real moving clips: the chalk picture goes to an image-to-video model
                 (Seedance lite, 720p 16:9, 5 s) told to keep the chalk-on-black look. Capped at
                 $2 a video (an estimate from published prices); a failed clip leaves the scene
                 drawn. `--no-motion` or `video.motion_scenes = 0` turns it off
 5. SOUND        a low drone and air under everything, darker in tense scenes and dipping
                 under the voice, plus synthesized cues the scene asks for (knock,
                 heartbeat, phone, footsteps, rain, wind, thunder, static, drip, creak, bell)
 6. ASSEMBLY     FFmpeg: each scene a clip with its camera move (push in, pull out, pan,
                 shake), flicker where the light flickers, fades through black; a moving
                 scene plays its clip (slowed up to 1.4x) and the drawing carries on if the
                 scene is longer; joined,
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

- A daily scheduled run of `video-stats`, once videos are live.
- Uploading the Shorts automatically (they are made with every video; for now they are
  uploaded by hand from YouTube Studio or the app).

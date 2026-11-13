# Production (pipeline 2)

Pipeline 2 turns an approved story into a finished video. The default is **stills mode**: the
story told in still pictures that look like frames of a dark horror film, over voices and
music, with no moving clips and no stick figures. The older chalk modes are kept below (`--studio`, `--drawn`, or
`stills.enabled=false`).

## Stills mode: the agent team

A team of agents, each with one job, works on a shared board (`production/stills/`). The
showrunner runs them in order, times them, and keeps each one's report in `report.json`. An
agent that is not critical (art references, sound, packaging, QA) falls back instead of
stopping the video.

| # | Agent | Job | Paid (with `FAL_KEY`) | Free fallback |
|---|---|---|---|---|
| 1 | Casting | Who speaks: the host, the teller, each character; picks the voices | the planner model (Bedrock) | words around each quote |
| 2 | Voice | Reads everything chunk by chunk, so every line's time is exact; dramatic pauses | ElevenLabs v3 on fal | Kokoro |
| 3 | Storyboard | Cuts the timeline into beats of 5-11 s and stages each as one picture | the planner model | the scene plan |
| 4 | Art director | One house look; a design per character; a style frame and character sheets | Nano Banana 2 | Cloudflare FLUX, Pollinations |
| 5 | Illustrator + critic | One picture per beat from the references; each checked in code and by a vision model, a rejected one redrawn by the next painter | Seedream 4.5 edit, then Nano Banana 2 edit | Cloudflare FLUX, Pollinations, a neighbour's picture |
| 6 | Edit plan | Cold open, title card, the story, end screen; captions | | |
| 7 | Sound | Score cues that follow the tension, room tone, effects, a stinger per scare | ElevenLabs music and effects | synthesized drone and cues |
| 8 | Edit | Every still for exactly its frames (held still, or `stills.motion=drift`), a dip to black between scenes, -14 LUFS | | |
| 9 | Packager | Three thumbnails, Shorts, chapters, metadata, contact sheet, 60 s preview | Nano Banana 2 edit | the key picture with lettering |
| 10 | QA | Checks length, streams, missing files, fallbacks; `pass` or `check` with the reasons | | |

**Voices.** The Storyteller (host) reads the intro and outro in the third person, always in the
same voice. A first-person story (decided from the text) is told by its teller in a voice of
their own, matched to them by the planner (a man, a woman, an older person). Every quoted line
is read by its speaker in a distinct voice. A third-person story is told by the host throughout.

**Pictures.** One house look for the channel, `stills.look`:
- `cinematic` (the default): frames of a slow-burn horror film shot on 35mm. Real-looking
  people and places, low light from one source, a muted cold grade, one blood-red accent. Every
  picture then gets the same film grain and vignette in code, so pictures from different
  painters match. YouTube's "altered or synthetic content" box is ticked in `metadata.json`.
- `illustrated`: the dark painted graphic-novel look used before November 2026.

`stills.style` replaces the look's text when set. The storyboard can show a beat through a
security camera, a phone flash photo or an old instant photo (`stills.cameras`) when the story
itself looks through one.

Character designs fix every feature for the whole story (accessories are "always" or "never",
so glasses don't come and go). Each beat's prompt asks for the same face, hair, clothes and
accessories as the character sheet, and the critic rejects a picture where a named person
clearly contradicts their design, a cartoon or plastic-skinned picture, broken hands or faces,
or any written text. Older stories' stick-figure cast notes are turned into real designs. Painters are tried in order (`stills.chains`) until
the critic passes a picture: Seedream 4.5 edit with the style frame and character sheets
(about $0.04), Nano Banana 2 edit as the redraw (about $0.08), then the free ones. The free
painters need no fal key: Cloudflare Workers AI's FLUX.1 schnell (secrets `CF_ACCOUNT_ID` and
`CF_API_TOKEN`; 10,000 free neurons a day, roughly 170 pictures; square output, cropped to
16:9; text only, so characters vary more) and Pollinations (keyless, slow, no guarantee).

**Sound library.** Music, room tones, effects and stings come first from the channel's own
library (`production/library.py`), published as the GitHub release `sound-library`: a catalogue
(`library.json`, one entry per sound with its kind, moods, intensity, tags, length, license,
source page and credit line) and one Ogg file per sound. The sound director lists every need
(a cue per mood, a room tone per place, an effect per beat, the sting), searches the catalogue
for a short list per need, and the planner model picks one or none from each list. Picks cost
nothing; only what the library lacks is generated, and those sounds are added to the library
after the run (the workflow's "Save this video's new sounds" step), so each sound is paid for
once. The least-used tracks are preferred, so videos don't all sound alike. CC BY credit lines
(Kevin MacLeod's) go into the video description automatically.

Build or extend it with the workflow's `sound-library` mode (free; about 20-40 minutes) or
`sound-library-music` (also 9 ACE-Step score cues on fal.ai, about $0.25). Sources are listed in
`sounds/sources.json`:

| Source | What | License |
|---|---|---|
| Kevin MacLeod (incompetech) | dark, suspenseful and intense tracks | CC BY 4.0, credit line added automatically |
| OpenGameArt | horror music, ambiences and effects, hand-picked pages and searches | CC0 or CC BY only (checked per page) |
| Free Music Archive (Hugging Face dataset) | instrumental tracks in dark genres | CC0 or CC BY only (checked per track) |
| Sonniss GDC bundle part (optional) | professional effects and ambiences | royalty-free, no credit; paste a part's zip URL and enable it |
| ACE-Step on fal.ai (optional) | score cues for each mood | generated, ours |

Non-commercial, no-derivatives and share-alike files are never added. `python -m production
library show` lists what the library holds. Settings: `stills.sound_library` (turn it off,
use a local folder, never generate, how close an effect must match).

**Cost.** About $4-6 for an 8.5-minute story with `FAL_KEY` (about 50-70 beats at $0.04-0.08,
references $0.5, voices about $1, checks a few cents, and music and effects about $0-0.5 from the
sound library; generating all of it instead costs about $7-8 more, mostly music), capped by
`stills.max_usd` (default $10). Free without any key, at lower quality.

**Outputs:** `video.mp4` (1080p), `thumbnail_1-3.jpg`, `short_1-2.mp4`, `captions.srt`,
`metadata.json`, `beats.json`, `contact_sheet.jpg`, `preview_60s.mp4`, `narration.wav`,
`report.json` (every agent's report, the spend by step, and QA's verdict). Review, approve and
publish work exactly as below.

**Keys for a first test** (GitHub, Settings, Secrets and variables, Actions): `FAL_KEY` (fal.ai:
pictures, voices, music, the critic), `AWS_BEARER_TOKEN_BEDROCK` (already set: the planner),
and optionally `CF_ACCOUNT_ID` + `CF_API_TOKEN` for the free picture fallback. Run
`mode = video-check` first (lists what every model expects, no spend), then `mode = video`.

## The older chalk modes

The chalk modes below are kept for comparison:

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

The first bake-off (run 71, story 18, $7.43) settled the defaults: Nano Banana Pro draws the
cleanest chalk-on-blackboard pictures with expressive stick figures; Nano Banana 2 crowds
the frame; Seedream 4.5 drifts into grey, shaded realism (it stays only as the redraw).
Seedance 2.0 and Kling 3 both keep the chalk look in motion (a tear running down a cheek, a
pupil pulsing red); Seedance 2.0 also adds the camera move, so it leads and Kling backs it up.

Every video and bake-off run also pushes small review images (comparison sheets, the
contact sheet, thumbnails, a frame every 20 s, clip strips, the reports) to the
`review-media` branch under `run-<number>/`, so a run can be judged without downloading
the artifact; `mode = video-share` with a run id does the same for an earlier run.

### Review, then publish

Nothing goes to YouTube until a person has watched it:

1. Run **mode = video**. Download `video-<run number>` from the run's Artifacts. Watch
   `preview_60s.mp4`, look over `contact_sheet.jpg`, skim `video.mp4`.
2. Run **mode = video-approve** with the story's number (or **video-reject**, with the reason
   in `overrides`).
3. Run **mode = video-publish** with the story's number. It uploads the video as **private**
   with its title, description and chapters, tags, captions, first thumbnail and the comment
   question, and the Shorts (private too, each linking to the full video). Put a time in
   `overrides` (e.g. `2026-11-20T17:00:00Z`) to schedule them instead. In YouTube Studio: pin
   the comment, add the end-screen elements over the last 20 s, and start a Test & Compare
   with the other two thumbnails.
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

## Open models on a rented GPU (Vast.ai)

**mode = video-vast** (optionally a `story_id`) makes a 40–60 s piece with open models only,
on a GPU rented from Vast.ai with the `VAST_API_KEY` secret:

- **The excerpt**: the stretch of the story that builds up to its first key scare (never the
  ending), so it plays as a teaser that ends on the hook, with a "what happened next?" card.
- **Here (free)**: Kokoro narration, the open planner's shot list, the edit, burned captions,
  a vertical version and the contact sheet.
- **On the GPU**: Qwen-Image-Edit-2509 draws a style frame, character sheets and every shot
  (best of three by the chalk check); Wan 2.2 I2V A14B animates every shot at 720p (most
  important first); MMAudio makes foley from each clip, room tone, score beds and a stinger.
- **Money**: the cheapest verified x86 machine with 80 GB+ of GPU memory is rented (the first
  check found an RTX PRO 6000 at $1.21/h). The job stops before `vast.max_usd` (default $12;
  set e.g. `vast.max_usd=6` in `overrides`) or `vast.max_hours`, keeps whatever it has made,
  and the machine is destroyed in every case (and again by an always-run cleanup step).
- **mode = video-vast-check** shows the balance and the best offers without spending.

All of these models are open (Apache-2.0 for Qwen-Image and Wan 2.2, MIT for MMAudio's code;
check MMAudio's weights licence before commercial use).

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

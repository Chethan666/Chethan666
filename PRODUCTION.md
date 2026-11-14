# Production (pipeline 2)

Pipeline 2 turns an approved story into a finished video. The default is **stills mode**: the
story told in still pictures that look like frames of a dark horror film, each one a shot
under a slow camera move, over voices and music. No generated video clips and no stick
figures. (The older chalk modes, the bake-off and the Vast.ai open-model path were removed in
November 2026; they are in the git history.)

## The agent team

A team of agents, each with one job, works on a shared board (`production/stills/`). The
showrunner runs them in order, times them, and keeps each one's report in `report.json`. An
agent that is not critical (art references, sound, packaging, QA) falls back instead of
stopping the video.

Every step that can go wrong has a **check gate** right after it, and a gate fixes what it
can before the next agent builds on it (✓ = checks; ↻ = fixes itself):

| # | Agent | Job | Gate after it | Paid (with `FAL_KEY`) | Free fallback |
|---|---|---|---|---|---|
| 0 | Script check | Every scene and pause is in the narration, quotes are closed, the length suits `target_seconds` | ✓ stops only a script that cannot be timed; the rest is reported | | |
| 1 | Casting | Who speaks: the host, the teller, each character; picks the voices | ✓ a speaker name must look like a name | the planner model (Bedrock) | words around each quote |
| 2 | Voice | The performance director gives each sentence its delivery; the narrator reads each paragraph in one flowing take, a little quicker than recorded | ✓↻ the **listener** (Whisper, free) transcribes every take: a take that skipped or invented words is read again; good takes are cut at exact word times | ElevenLabs v3 on fal (+ the planner model) | Kokoro |
| 3 | Storyboard | Cuts the timeline into beats (`beat_min_seconds`-`beat_max_seconds`) and stages each as one picture | ✓↻ each staged picture is matched to the line it quotes (never by its number alone); skipped beats are asked for again; then the **storyboard check** holds every picture description against its own line and the cast's fixed looks and rewrites the ones that don't match, before any picture is paid for | the planner model | the scene plan |
| 4 | Art director | One house look; a design per character and per place the story returns to; a style frame (the main place), character sheets and location sheets | ✓↻ the **reference check**: the style frame and every sheet are held against their design by a vision model before any picture is painted from them; one that fails is painted once more, told what was wrong | Nano Banana 2 (+ the planner model) | Cloudflare FLUX, Pollinations |
| 5 | Illustrator + critic | One picture per beat, painted at 2K from its place's sheet and its people's sheets; the scare shots by the strongest painter | ✓↻ each picture checked in code and by a vision model against its description and the narrated words; a refused picture goes to the next painter, which is told what was wrong; a refused paid picture is never replaced by a free painter's | Seedream 4.5 edit, then Nano Banana 2 edit | Cloudflare FLUX, Pollinations, a neighbour's picture |
| 6 | Art review | All the pictures side by side with the narration: the same person and the same place look the same, the order tells the story, the scare lands | ✓↻ the worst `art_review_fixes` (2) are redrawn with its note (the critic checks the redraw) | vision model, ~$0.002 | |
| 7 | Colorist | Measures every picture against the film's median brightness, contrast and colour cast | ✓↻ a picture outside the tolerance (one that would jump at the cut) is brought most of the way back; the rest are not touched | | |
| 8 | Edit plan | Cold open, title card, the story, end screen; captions | | | |
| 9 | Sound | The sound director picks music, room tone, effects and stingers from the channel's sound library (or has them made, if `sound_library.generate_missing`) | ✓↻ the **mix check**: under speech the voice must stand `voice_over_bed_db` (12 dB) above everything else, or the bed is lowered | library (free) | synthesized drone and cues |
| 10 | Edit | Every still becomes a shot at 1440p: a slow, steady camera move suited to its framing, rain, fog or a failing light where the storyboard saw them, fine film grain, a vignette, HD colours; exactly its frames, a dip to black between scenes, -14 LUFS | | | |
| 11 | Packager | Thumbnails, Shorts, chapters, metadata, contact sheet, 60 s preview | ✓↻ each thumbnail's words are read back by a vision model; misspelled or unreadable ones are repainted once | Nano Banana 2 edit | the key picture with lettering |
| 12 | QA | Length, streams, files, fallbacks, a stretch of black picture, the colour label, loudness and true peak, and every gate's findings in one list | ✓ `pass`, or `check` with the reasons | | |

By default only pictures and the narrator's voice are paid for: music and sounds come from the
library (`sound_library.generate_missing` false). A 60-second test: workflow mode `video-test`
with overrides `script=one_minute.json`.

**Voices.** The Storyteller (host) reads the intro and outro in the third person, always in the
same voice. A first-person story (decided from the text) is told by its teller in a voice of
their own, matched to them by the planner (a man, a woman, an older person). Every quoted line
is read by its speaker in a distinct voice. A third-person story is told by the host throughout.

**Delivery.** A narrator that reads every sentence on its own sounds flat: each one starts again at the same pitch. So the narrator reads a paragraph's sentences in one take (story pauses still break it), and the performance director (the planner model) gives each sentence a v3 tag such as [nervously], [whispers] or [ominously], keeping most sentences plain and whispers rare. Settings: `performance_director`, `voice_takes`, `voice_pace` (1.12: 12% quicker, same pitch) and `voice_gap_scale` (0.7: shorter silences between paragraphs and at story pauses).

**Pictures.** One house look for the channel, `stills.look`:
- `cinematic` (the default): frames of a slow-burn horror film shot on 35mm. Real-looking
  people and places, low light from one source, a muted cold grade, one blood-red accent.
  YouTube's "altered or synthetic content" box is ticked in `metadata.json`.
- `illustrated`: the dark painted graphic-novel look used before November 2026.

`stills.style` replaces the look's text when set. The storyboard can show a beat through a
security camera, a phone flash photo or an old instant photo (`stills.cameras`) when the story
itself looks through one.

Each beat's prompt leads with the shot itself (framing, who does what, where), then what is
fixed (the place, the people), what each reference picture is for, the look, and what to avoid.
A model follows best what it reads first; with the look in front, run 94's first tries ignored
their description more than a third of the time.

What has to stay the same from picture to picture is designed once and painted from references:
- **People.** Character designs fix every feature for the whole story (accessories are "always"
  or "never", so glasses don't come and go). The script's own notes are kept word for word;
  the designer only adds what they leave out. Each character gets a sheet, and each beat's
  prompt asks for the same face, hair, clothes and accessories as the sheet. Older stories'
  stick-figure cast notes are turned into real designs. With the cinematic look, a sheet is
  asked for an unknown, ordinary face that resembles no actor or public figure (a photoreal
  face can drift toward a famous one).
- **Places.** The set designer (the planner model, free) groups the scenes into the places
  the story returns to and fixes what the script leaves open: what the place is, its colours
  and materials, its fixtures, its lights. The most-used places (`location_sheets`, 3) get a
  location sheet, with an outside and an inside view when pictures are set in both. Every
  picture set there is painted from that sheet and carries the design in its prompt. Without
  this, each picture invents the place again: run 94 showed a city bus in one shot and a
  yellow school bus in the next. `set_design: false` turns it off.
- **The look.** The style frame is a wide view of the story's main place (the one with the
  most pictures); it sets the look and is the title card's background.
- **The reference check.** Every picture is painted from these references, so a wrong sheet is
  a wrong film. A vision model holds the style frame and each sheet against its design (the
  same person in every view, a face that must stay hidden is hidden, the right kind of vehicle
  or room, no text); a failed one is painted once more with the reason in front. One still
  refused is used, and QA lists it (`reference: ...`), except a free painter's: that is worse
  than none (run 96: an empty room as a child's sheet). Sheets have a second paid painter,
  Seedream, after Nano Banana. Places are painted at night, in their darkest light.

**Painters, spent by importance.** Each picture is asked of the painters in its chain
(`stills.chains`) until the critic passes one. Where the money goes depends on how much the
picture matters (see "Cost and quality" below for the evidence):

| Pictures | Chain | Price per try |
|---|---|---|
| Style frame | Seedream 4.5 text-to-image, then once more if refused | $0.04 |
| Character and place sheets | Seedream 4.5 edit, again, then Seedream 5.0 Lite | $0.04 / $0.035 |
| Ordinary beats | Seedream 4.5 edit; a second Seedream try only for a serious fault | $0.04 |
| Key beats (`key_importance` 4+, at most `key_share` 25% of the story) | Seedream 4.5, then Nano Banana 2 edit at 2K as the corrector | $0.04 / $0.12 |
| Scares (`scare_shots`, at most 4) | Nano Banana Pro edit at 2K, then Seedream 4.5 | $0.15 / $0.04 |
| Thumbnail | Nano Banana 2 edit (clean lettering) | $0.08 |

The critic rejects a picture where a named person clearly contradicts their design, a cartoon
or plastic-skinned picture, broken hands or faces, written text, or a different action than
the narrated words, and says whether the fault is **serious** (a viewer would notice it in the
five seconds the picture is on screen) or a **detail** (a prop's state, the exact framing, which
hand holds what). An ordinary beat refused only for a detail keeps its picture; no money is
spent correcting what nobody would see. The next painter is told what was refused (the fault
leads its prompt), so a redraw is a correction. If every try is refused, the first picture
whose fault was only a detail is kept, so the strongest painter's picture is not lost to a
correction that broke something worse (run 96: a split screen); it is counted as doubted, and
QA flags a doubted scare. The free painters
take no references, so they paint only when no paid picture could be made (no key, or over
the cap): Cloudflare Workers AI's FLUX.1 schnell (secrets `CF_ACCOUNT_ID` and `CF_API_TOKEN`;
10,000 free neurons a day, roughly 170 pictures; square output, cropped to 16:9) and
Pollinations (keyless, slow, no guarantee). References are sent to the models as JPEGs of at
most 2048 px, a tenth of the upload of the full-size pictures.

**Sharpness and frame size.** The video is 1440p (`video_size`, 2560x1440): YouTube gives
1440p uploads its better codecs (VP9, AV1) even for viewers watching at 1080p, and dark,
grainy pictures are exactly where the plain codec bands and blocks. Rendering takes about
twice as long as 1080p. Pictures are painted at 2K (`model_args.shot`: 2752x1548;
`shot_alt` and `hero`: "2K") and kept at `picture_size` (2752x1548, the frame plus the room a
move needs) until the edit, so a push-in ends on real detail and nothing is enlarged. The
report's `illustrator.painter_sizes` shows the size each painter really sent back.

**The colorist.** Each picture comes from a model that sees only its own prompt, so one can
come back twice as bright as its neighbours or with another colour cast. The colorist measures
every picture against the film's median and corrects only the ones outside a tolerance. A film
whose pictures all came back lit like day is first printed down as a whole, to the look's
`max_brightness` (40 of 255 for the cinematic look), so night looks like night
(`grade_strength`, 0.75 of the way back to it): brightness with a gamma curve, so black stays
black; contrast only raised, for a flat grey frame; colour cast only for film frames (a
security-camera view keeps its own colour). The report lists each change.

**The camera and the film finish** (`production/stills/camera.py`). With `motion: "drift"` (the
default) every story picture gets a slow, steady move suited to its framing: wide shots pan,
close ones push in, a scare lunges and settles, a security camera holds still, the end card
holds still. A move runs at an even speed for the whole shot, a little further for a longer
shot, so the picture is moving when it cuts in and when it cuts out. Moves are rendered with
sub-pixel accuracy (FFmpeg's `perspective` filter per frame): the usual `zoompan` filter steps
by 0.2-0.4 px from frame to frame on a slow zoom, which shows as a faint shake. `motion: "none"`
holds every picture still. Every shot then gets the same finish, whichever painter made it:
- fine film grain (`looks.cinematic.grain`), laid into the picture at the size it is shown.
  It is not animated: grain that changes every frame multiplies the file size and is the first
  thing YouTube's encoder smears;
- a vignette (`vignette`) fixed to the frame, so it stays put when a shot pans;
- colours converted for HD video (BT.709) and labelled. FFmpeg's default is the
  standard-definition matrix, which showed the channel's blood red as orange-red in HD players;
- an encode that spends more of its bitrate on dark areas (x264 `aq-mode=3`, CRF 18).

**Atmosphere.** The storyboard says what moves in the air of each picture, and the edit
animates it (`atmosphere`, on):
- rain: streaks falling at a slant in front of everything, only when rain falls between the
  camera and the scene (never over an interior, never behind glass);
- fog: soft banks of mist drifting slowly across the frame, thicker low down;
- flicker: a failing lamp, a guttering candle or a stuttering screen: the picture dips darker
  for a frame or two, irregularly.
Rain and fog are textures made once per video and laid over the moving picture, fixed to the
frame; flicker is a brightness curve. A shot's bitrate is capped at 24 Mbit/s, since rain
changes every pixel of every frame.

Shorts keep the film's 24 frames a second, so its camera moves stay smooth in them.

**Sound library.** Music, room tones, effects and stings come first from the channel's own
library (`production/library.py`), published as the GitHub release `sound-library`: a catalogue
(`library.json`, one entry per sound with its kind, moods, intensity, tags, length, license,
source page and credit line) and one Ogg file per sound. The sound director lists every need
(a cue per mood, a room tone per place, an effect per beat, the sting), searches the catalogue
for a short list per need, and the planner model, as sound director, decides each one: use a
library sound (free), have a new one made in its own words when nothing fits (music by ACE-Step
1.5 on fal.ai, about $0.05 a cue; other sounds by the sound-effects model, about $0.01), or leave
an effect out. Sounds made are added to the library
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

Non-commercial, no-derivatives and share-alike files are never added. Keep the repository
private: the release is then private too, and runs read it with the workflow's own token. `python -m production
library show` lists what the library holds. Settings: `stills.sound_library` (turn it off,
use a local folder, never generate, how close an effect must match).

**Cost.** About $4-6 for an 8.5-minute story with `FAL_KEY` (about 50-70 beats at $0.04-0.08,
references $0.5, voices about $1, checks a few cents, and music and effects about $0-0.5 from the
sound library; generating all of it instead costs about $7-8 more, mostly music), capped by
`stills.max_usd` (default $10). Free without any key, at lower quality.

**Outputs:** `video.mp4` (1440p), `thumbnail_1-3.jpg`, `short_1-2.mp4`, `captions.srt`,
`metadata.json`, `beats.json`, `contact_sheet.jpg`, `preview_60s.mp4`, `narration.wav`,
`report.json` (every agent's report, the spend by step, and QA's verdict). Review, approve and
publish work exactly as below.

**Keys for a first test** (GitHub, Settings, Secrets and variables, Actions): `FAL_KEY` (fal.ai:
pictures, voices, music, the critic), `AWS_BEARER_TOKEN_BEDROCK` (already set: the planner),
and optionally `CF_ACCOUNT_ID` + `CF_API_TOKEN` for the free picture fallback. Run
`mode = video-check` first (lists what every model expects, no spend), then `mode = video`.

Every video run also pushes small review images (the contact sheet, thumbnails, a frame every
20 s, four full-size frames, the reports) to the `review-media` branch under `run-<number>/`,
so a run can be judged without downloading the artifact; `mode = video-share` with a run id
does the same for an earlier run.

## Cost and quality

The same 60-second story was made twice in November 2026 with different picture models (runs
96 and 98; the narration, sound and edit were the same):

| | Run 96: premium | Run 98: cheap |
|---|---|---|
| Models | Seedream 4.5, Nano Banana 2 (2K redraws, sheets, style frame, thumbnail), Nano Banana Pro (scares) | Seedream 5.0 Lite and 4.5 for everything |
| Total | $2.20 (pictures $2.06) | $1.41 (pictures $1.27) |
| Pictures refused | 9 | 18 |
| Pictures used though no check passed them | about 2 of 11 | 7 of 11 |
| First try passed | Seedream 4.5: about 1 in 2 | Seedream 5.0 Lite: about 1 in 3 |
| A refused picture corrected | Nano Banana 2: about 2 in 3 | Seedream 4.5: about 1 in 7 |
| Scares | Nano Banana Pro: 1 of 2 passed | Seedream 4.5: 0 of 3 passed |
| Sharpness, same rooms | full 2K; the same rooms throughout | the same |
| Acting | fear reads | often a neutral face where the line is terror |
| Thumbnail | clean | a fake logo and misspelled words on the first try |

What it showed: the cheap models keep the look, the sharpness and the continuity, but follow
direction less well; Nano Banana is worth its price as a corrector and for faces that must act,
not as the first painter of every picture; and about half of all refusals were for details no
viewer would notice, each paid for with a redraw. Hence the chains above: Seedream paints
almost everything, the corrector and the strongest painter are kept for the key beats and the
scares, and a detail no longer buys a redraw.

Estimates for the balanced setup (the budget's prices, not fal's invoice):

| | 60-second test | 8-10 minute episode (about 100 beats, 9,000 characters) |
|---|---|---|
| Premium (run 96's setup) | $2.20 | about $10-11 |
| Balanced (the default) | about $1.50 | about $7-8 |
| Cheap (run 98's setup, `production/samples/one_minute_cheap.json`) | $1.41 | about $6-7 |

The balanced estimate is worked out from the two runs' pass rates; the first run with it will
show the real figure. Every other cost is small: the narration about $0.10 a minute, the vision
checks about $0.002 each, the sound library free.

## Review, then publish

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

## Running it

In GitHub Actions: run the workflow with **mode = video** (optionally a **story_id**).
Download the result from the run's **Artifacts** (`video-<run number>`, kept 14 days).
Keep this repository private: artifacts and review images are then visible only to you.

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

`--script production/samples/one_minute.json` makes a video from a script file instead of the
database. `--voice silent` swaps the free narrator for a placeholder tone, and
`--no-direction-model` makes no planner calls, for testing the rest without downloads or
model calls.

The story is marked `produced` with a summary in its `production` field. If anything fails,
a claimed story goes back to `approved` with the error recorded.

The free narrator (Kokoro, behind every paid line) is set in the config: `[video] voice =
"am_michael"`, `lang = "a"` (US English; `"b"` is British), `speed = 0.92`.

## Later

- A daily scheduled run of `video-stats`, once videos are live.

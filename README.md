# Hollow Lines (working title)

Hollow Lines is a faceless YouTube horror storytelling channel. Every video is made of still
pictures, one narrator and music: a new chalk-style picture every 3 to 6 seconds (white chalk
stick figures on black with a single blood-red accent, each under a slow camera move), the calm
voice of a recurring lantern-carrying host called The Storyteller, a horror score and sound
effects. There are no video clips. About three-quarters of the stories are original, set in one
shared fictional world, so recurring places, characters and hidden clues reward returning
viewers who theorize across episodes. The rest retell folklore, which brings in search traffic.
Each story is built for suspense rather than gore: a hook in the first ten seconds, everyday
settings, rising dread, and endings that land as terror, a twist, justice or heartbreak. Every
video is clearly presented as fiction or folklore and marked "not made for kids". The plan is
three long episodes a week (8+ minutes), in English; Shorts are paused for now.

## The two pipelines

| # | Pipeline | What it makes | Code | Docs |
|---|---|---|---|---|
| 1 | Story factory | the stories: a writers' room of AI models from several companies (today open-weight models on Amazon Bedrock) pitches, plans, writes, edits and audience-tests every story | `story_factory/` | [STORY_FACTORY.md](STORY_FACTORY.md) |
| 2 | Video production | the videos: narration, a still picture per shot, music and sound, captions, thumbnails, and a review before anything is published | `production/` | [PRODUCTION.md](PRODUCTION.md) |

Both run in one GitHub Actions workflow (Actions tab, "Story factory", Run workflow; one mode
per run) and share the model access, the story database and the cost log. The channel's
decisions (name, narrator, schedule, thumbnails) are in [DECISIONS.md](DECISIONS.md).

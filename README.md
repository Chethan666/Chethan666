# Hollow Lines (working title)

Hollow Lines is a faceless YouTube horror storytelling channel that tells horror stories from
India, in English, for viewers everywhere: Indian legends (the churel, the vetala, Naale Ba) and
original stories set in today's India (a night bus on a ghat road, a joint-family house in a
power cut, a new paying-guest room in Bengaluru). Many stories turn on a rule from folk belief
that the viewer learns before the hero breaks it, and every spirit that stays has a reason.
Every video is made of still pictures, one narrator and music: a new chalk-style picture every 3
to 6 seconds (white chalk stick figures on black with a single blood-red accent, each under a
slow camera move), the calm voice of a recurring lantern-carrying host called The Storyteller,
an Indian folk-horror score (tanpura drone, temple bell, sarangi, dhol) and sound effects. There
are no video clips. About three-quarters of the stories are original, set in one shared
fictional world, so recurring places, characters and hidden clues reward returning viewers who
theorize across episodes; the rest retell legends, which brings in search traffic. Each story is
built for suspense rather than gore: a hook in the first ten seconds, everyday settings, rising
dread, and endings that land as terror, a twist, justice or heartbreak. The Storyteller closes
every episode with one open question, like Betaal's riddles to King Vikram, for the comments.
Every video is clearly presented as fiction or folklore and marked "not made for kids". The plan
is three long episodes a week (8+ minutes); Shorts are paused for now.

## The two pipelines

| # | Pipeline | What it makes | Code | Docs |
|---|---|---|---|---|
| 1 | Story factory | the stories: a writers' room of AI models from several companies (today open-weight models on Amazon Bedrock) pitches, plans, writes, edits and audience-tests every story | `story_factory/` | [STORY_FACTORY.md](STORY_FACTORY.md) |
| 2 | Video production | the videos: narration, a still picture per shot, music and sound, captions, thumbnails, and a review before anything is published | `production/` | [PRODUCTION.md](PRODUCTION.md) |

Both run in one GitHub Actions workflow (Actions tab, "Story factory", Run workflow; one mode
per run) and share the model access, the story database and the cost log. The channel's
decisions (name, narrator, schedule, thumbnails) are in [DECISIONS.md](DECISIONS.md).

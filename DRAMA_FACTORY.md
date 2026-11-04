# Pipeline 3: the drama writers' room

Pipeline 3 writes vertical short-drama series: 60–120-second episodes told in dialogue,
with a hook in the first three seconds, a reversal in every episode and a cliffhanger at
the end. It is a clone of pipeline 1 (the stickman story factory) rebuilt for series. It
uses the same models, judges, database and cost log, but has its own stages and settings
(`[drama]` in the config), so changing it never touches the stickman channel.

## The flow

```
SERIES
 1. PITCH CONTEST   every pitcher pitches against its own trope recipe (trope, setting,
                    twist; never one an earlier series used); repeats of earlier series
                    dropped; judges from other companies score hook, bingeability,
                    originality, emotional core and clarity; the top 4 go head to head
                    in both orders; strong runners-up are SHELVED and used first next time
 2. PLAN CONTEST    every writer writes a series bible from the winning pitch: 3-6
                    characters with a precise look, wardrobe, voice, want and secret;
                    3-6 recurring locations; the arc; every episode's hook, beats and
                    cliffhanger. Judges pick one head to head
 3. BIBLE REVIEW    free checks, then judges score the plan; revisions are reviewed again
                    and the best-scoring version is kept

EPISODE
 4. DRAFT CONTEST   every writer drafts the episode shot by shot (framing, location, who
                    is in frame, what we see, who says which line and how, inner
                    thoughts, the few shots that must really move); judges pick one
 5. EDITORS' LOOP   free checks (known cast and places, mostly dialogue, short lines),
                    then judges score hook, conflict, character, dialogue, cliffhanger
                    and continuity and leave notes on shot numbers. KEEP THE BEST: a
                    worse version is set aside. THE SHOWRUNNER picks the fix:
                      free checks failed                    → repair
                      a score of 5 or less                  → a fresh draft contest
                      conflict/character/cliffhanger still low after a revision
                                                            → rethink the episode's outline,
                                                              then a fresh draft contest
                      only the hook (or cliffhanger) weak   → the hook (or cliffhanger) lab
                      notes on shots                        → rewrite only those shots
                      otherwise                             → revise the whole episode
 6. HOOK LAB        alternative first two shots compete with the original
 7. CLIFFHANGER LAB alternative last two shots compete with the original
 8. AUDIENCE TEST   simulated viewers say where they would swipe away and how badly they
                    need the next episode (unlock_next, 1-10)
 9. FINAL CHECK     safety flags; one repair, then rejected if a flag remains
10. PACKAGING       title, caption, hashtags and a pinned comment for TikTok, Shorts, Reels
→ APPROVED, waiting for pipeline 4
```

A rejected episode is written again the next time, so a series never skips a number.

## Commands

| Workflow mode | Command | What it does |
|---|---|---|
| `drama-series` | `python -m drama_factory series` | a new series (from the shelf, or a fresh pitch contest) |
| `drama-write` | `python -m drama_factory write --count 3` | the next episodes of the newest series |
| | `python -m drama_factory status` | every series and episode |
| | `python -m drama_factory report [ID]` | one episode's scores, viewers and packaging |

Settings live in `[drama]`: `genre` (default "supernatural thriller"), `episodes_per_series`,
`episode_seconds`, `min_score`, contest sizes, revision rounds and the simulated viewers.
One-off changes: workflow input `overrides`, e.g. `drama.genre="revenge romance"`.

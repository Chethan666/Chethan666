"""Pipeline 2, stills mode: a horror story told in still illustrations, voices and music.

No moving clips and no stick figures. A team of agents, each with one job, works on a
shared board, and the showrunner runs them in order:

    casting      who speaks: the Storyteller host (intro and outro, third person), the teller
                 (the story body; a first-person story gets its own voice, matched to the
                 teller) and every character with a quoted line (their own voice each)
    voice        reads it all, chunk by chunk, so every line's time is exact
    storyboard   cuts the timeline into beats of 5-11 s and stages each one as a picture
    art          one house look for the channel: a style frame and a character sheet per
                 character (the host included), so every picture matches
    illustrator  one picture per beat from those references; the critic checks each one
                 (in code, then by a vision model) and a failed picture is redrawn by the
                 next painter in the chain (paid models first, then free ones)
    edit_plan    cold open, title card, the story, end screen; captions
    sound        score cues that follow the tension, room tone, effects, a stinger per scare
    edit         renders the stills (static, or a slow drift) with dips to black between scenes
    packager     thumbnails, Shorts, chapters, metadata, contact sheet, 60 s preview
    qa           checks the finished files before a person reviews them

One spending cap covers the whole video and every paid step has a free fallback.
"""

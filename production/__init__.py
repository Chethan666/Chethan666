"""Pipeline 2: turns an approved story into a finished video of still pictures, narration and sound.

    claim an approved story -> the narrator's voice with the story's pauses and
    pronunciations, timed chunk by chunk -> a shot list (3-6 s shots, staged by an open
    model) -> one still chalk picture per shot -> score, room tone, effects and stingers
    -> the edit (cold open, title, a slow camera move over every picture, end screen)
    -> video, captions, thumbnails, Shorts cut from the video, upload metadata, review kit

Studio mode (with FAL_KEY) generates the pictures and the voice on fal.ai; drawn mode is
free (pictures drawn in code, Kokoro's voice). No video clips in either mode. A person
approves every video before `publish` uploads it to YouTube.
"""

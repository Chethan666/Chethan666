"""Pipeline 2: turns an approved story into a finished video.

    claim an approved story -> narrator voice (Kokoro, open source, on the CPU) with the
    story's pauses and pronunciations -> word timings -> art direction per scene (an open
    model picks background, figures, poses, props, camera) -> chalk stickman frames drawn
    in code -> ambience and sound cues -> FFmpeg assembly with burned-in captions
    -> video, thumbnail, captions and upload metadata

Everything runs on a GitHub Actions runner. Uploading to YouTube comes later.
"""

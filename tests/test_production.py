"""Pipeline 2: timing, direction, drawing, sound and a whole video with a placeholder voice."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import PIL  # noqa: F401
    import soundfile  # noqa: F401
    HAVE_VIDEO_LIBS = True
except ImportError:  # the story factory's own tests run without the video libraries
    HAVE_VIDEO_LIBS = False

from production import timing

NARRATION = ("The phone rings at 3:07. The lantern is lit. Listen closely.\n\n"
             "Marisol drove the night bus for nine years, and nobody ever pressed the bell between Calder "
             "Road and the reservoir. Tonight the bell rang. She checked the mirror. Every seat was empty.\n\n"
             "Some doors are better left closed. Would you have opened the door?")
SCENES = [
    {"text": "The phone rings at 3:07. The lantern is lit. Listen closely.", "visual": "The Storyteller lifts a lantern",
     "location": "a dark road", "characters": ["The Storyteller"], "mood": "eerie", "delivery": "slow",
     "sfx": ["phone ringing"], "key_moment": False},
    {"text": "Marisol drove the night bus for nine years, and nobody ever pressed the bell between Calder Road and "
             "the reservoir.", "visual": "A bus on an empty road at night", "location": "a road",
     "characters": ["Marisol"], "mood": "calm", "delivery": "steady", "sfx": [], "key_moment": False},
    {"text": "Tonight the bell rang. She checked the mirror. Every seat was empty.", "visual": "Marisol at the "
             "wheel looks at the mirror", "location": "inside the bus", "characters": ["Marisol"],
     "mood": "tense", "delivery": "hushed", "sfx": ["bell"], "key_moment": True},
    {"text": "Some doors are better left closed. Would you have opened the door?", "visual": "The Storyteller "
             "closes a door", "location": "a room", "characters": ["The Storyteller"], "mood": "eerie",
     "delivery": "slow", "sfx": ["door creak"], "key_moment": False},
]


class TimingTests(unittest.TestCase):
    def test_chunks_cover_sentences_with_longer_silences_at_breaks_and_pauses(self):
        chunks = timing.split_chunks(NARRATION, [{"after": "Tonight the bell rang", "ms": 1500}])
        texts = [c.text for c in chunks]
        self.assertIn("Tonight the bell rang.", texts)
        self.assertEqual(next(c for c in chunks if c.text == "Tonight the bell rang.").gap_ms, 1500)
        self.assertEqual(next(c for c in chunks if c.text == "Listen closely.").gap_ms, 1100)  # end of the intro
        self.assertEqual(chunks[-1].gap_ms, 0)
        for c in chunks:
            self.assertEqual(NARRATION[c.start:c.end], c.text)

    def test_a_pause_inside_a_sentence_splits_it(self):
        chunks = timing.split_chunks(NARRATION, [{"after": "and nobody ever pressed", "ms": 900}])
        cut = next(c for c in chunks if c.text.endswith("and nobody ever pressed"))
        self.assertEqual(cut.gap_ms, 900)
        self.assertTrue(chunks[chunks.index(cut) + 1].text.startswith("the bell between"))

    def test_scene_spans_follow_the_text_and_cover_everything(self):
        spans = timing.scene_spans(NARRATION, SCENES)
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], len(NARRATION))
        self.assertTrue(NARRATION[spans[2][0]:].startswith("Tonight"))
        drifted = [dict(s) for s in SCENES]
        drifted[2]["text"] = "Tonight, the bell rang!  She checked the mirror."
        self.assertTrue(NARRATION[timing.scene_spans(NARRATION, drifted)[2][0]:].startswith("Tonight"))

    def test_captions_are_even_and_never_leave_a_lonely_word(self):
        timeline = timing.Timeline(NARRATION, [timing.Chunk(
            "The phone rings at 3:07 and the voice is his own.", 0, 50, 0, 0.0, 3.0)], 3.0)
        cues = timing.caption_cues(timeline, max_chars=44)
        self.assertEqual(len(cues), 2)
        self.assertGreater(min(len(c.text.split()) for c in cues), 2)
        self.assertIn("-->", timing.srt(cues))
        self.assertIn("Dialogue:", timing.ass(cues, 1920, 1080))


@unittest.skipUnless(HAVE_VIDEO_LIBS, "video extras not installed")
class DirectionAndDrawingTests(unittest.TestCase):
    def test_keywords_direct_a_scene_when_no_model_answers(self):
        from production import director

        spec = director.guess(SCENES[2], ["Marisol"])
        self.assertEqual(spec["background"], "vehicle")
        self.assertIn("mirror", [p["kind"] for p in spec["props"]])
        self.assertEqual(spec["camera"], "push_in")

    def test_a_models_answer_is_cleaned_before_drawing(self):
        from production import director

        spec = director.clean({"background": "spaceship", "camera": "push_in", "figures": [
            {"name": "Marisol", "pose": "moonwalk", "x": 7, "facing": "left", "ghost": False}],
            "props": [{"kind": "laser", "x": 0.4}, {"kind": "door", "x": -3}], "red": "prop:door"}, SCENES[2], [])
        self.assertEqual(spec["background"], "vehicle")
        self.assertEqual(spec["figures"][0]["pose"], "stand")
        self.assertEqual(spec["figures"][0]["x"], 0.92)
        self.assertEqual([p["kind"] for p in spec["props"]], ["door"])

    def test_direction_falls_back_per_batch_when_the_model_fails(self):
        from production import director
        from story_factory.llm import LLMError

        class Broken:
            def one(self, role):
                return self

            def generate_json(self, *a):
                raise LLMError("down")

        specs, fallbacks = director.direct(Broken(), SCENES, [{"name": "Marisol", "look": "x"}], "the bell")
        self.assertEqual((len(specs), fallbacks), (4, 4))

    def test_looks_read_whole_words(self):
        from production import director

        self.assertEqual(director.features("X", "Invented look that canon must override")["headwear"], "none")
        look = director.features("Leo", "Tall, long grey coat, flat cap, glasses")
        self.assertEqual((look["headwear"], look["outfit"], look["accessory"], look["height"]),
                         ("cap", "coat", "glasses", 1.12))

    def test_every_pose_prop_and_background_draws(self):
        from production import director, draw

        paper, grain = draw.make_paper(320, 180)
        looks = {"A": director.features("A", "long hair, dress, lantern")}
        for bg in director.BACKGROUNDS:
            spec = {"background": bg, "light": "moonlight", "camera": "still", "red": "prop:door",
                    "figures": [{"name": "A", "pose": pose, "x": 0.5, "facing": "left", "ghost": False}
                                for pose in director.POSES[:2]] + [{"name": "Shadow", "pose": "stand", "x": 0.8,
                                                                     "facing": "left", "ghost": True}],
                    "props": [{"kind": "door", "x": 0.3}]}
            frames = draw.scene_frames(spec, looks, (320, 180), paper, grain, seed=1, variants=2)
            self.assertEqual(len(frames), 2)
        c = draw.Canvas(320, 180, 1)
        for kind in director.PROPS:
            draw.prop(c, kind, 160, 140, 20)
        for pose in director.POSES:
            draw.figure(c, 160, 140, 70, pose, "right", looks["A"])
        image = c.render(paper, grain)
        self.assertGreater(np.asarray(image).max(), 150)  # chalk is visible on the paper


@unittest.skipUnless(HAVE_VIDEO_LIBS, "video extras not installed")
class SoundTests(unittest.TestCase):
    def test_cues_are_found_and_synthesized(self):
        from production import sound

        self.assertEqual(sound.cues_for(["a phone ringing", "door creak"]), ["ring", "creak"])
        rng = np.random.default_rng(0)
        for name in set(sound.CUE_WORDS.values()):
            self.assertGreater(sound.cue(name, 8000, rng).size, 0, name)

    def test_mix_keeps_the_voice_on_top(self):
        from production import sound

        rate = 8000
        voice = np.zeros(rate * 4, dtype=np.float32)
        voice[rate:rate * 2] = 0.5 * np.sin(np.linspace(0, 400, rate))
        out = sound.mix(voice, rate, SCENES[:2], [(0, 2), (2, 4)], [(1, 2)])
        self.assertEqual(out.size, voice.size)
        self.assertLessEqual(float(np.abs(out).max()), 0.951)
        speech = np.abs(out[rate:rate * 2]).mean()
        quiet = np.abs(out[int(rate * 2.6):int(rate * 3.4)]).mean()
        self.assertGreater(speech, 4 * quiet)


@unittest.skipUnless(HAVE_VIDEO_LIBS and shutil.which("ffmpeg"), "video extras or ffmpeg not installed")
class WholeVideoTests(unittest.TestCase):
    def test_a_story_becomes_a_video_with_captions_thumbnail_and_metadata(self):
        from production import run, voice

        story = {"id": 7, "title": "Night Bus", "narration": NARRATION, "scenes": json.dumps(SCENES),
                 "cast": json.dumps([{"name": "Marisol", "look": "short, long hair, bus driver's cap"}]),
                 "bible": json.dumps({"red_accent": "the stop bell"}),
                 "pauses": json.dumps([{"after": "Tonight the bell rang", "ms": 1200}]),
                 "pronunciations": json.dumps([{"word": "Marisol", "say_as": "MAH-ree-sol"}]),
                 "packaging": json.dumps({"titles": ["The Extra Passenger"], "thumbnail_text": "WHO RANG",
                                          "description": "A story.", "tags": ["horror"]}),
                 "comment_question": "Would you have opened the door?",
                 "description_credit": "This is an original work of fiction."}
        out = Path(tempfile.mkdtemp())
        try:
            summary = run.produce(story, out, voice.SilentVoice(), None, "360")
            self.assertEqual(summary["scenes"], 4)
            for name in ("video.mp4", "thumbnail.png", "captions.srt", "metadata.json", "narration.wav"):
                self.assertTrue((out / name).stat().st_size > 0, name)
            meta = json.loads((out / "metadata.json").read_text())
            self.assertEqual(meta["title"], "The Extra Passenger")
            self.assertFalse(meta["made_for_kids"])
            self.assertIn("Marisol", (out / "captions.srt").read_text())  # captions keep the real spelling
            self.assertFalse((out / "work").exists())
        finally:
            shutil.rmtree(out, ignore_errors=True)


    def test_a_video_with_a_characters_line(self):
        from production import run, voice

        narration = NARRATION.replace("Tonight the bell rang.", "Tonight the bell rang. \"Who's there?\" Marisol said.")
        story = {"id": 8, "title": "Night Bus", "narration": narration, "scenes": json.dumps(SCENES),
                 "cast": json.dumps([{"name": "Marisol", "look": "bus driver's cap"}])}
        out = Path(tempfile.mkdtemp())

        import os
        here = os.getcwd()
        try:
            os.chdir(out)  # relative folders, as in the workflow
            summary = run.produce(story, Path("made"), voice.SilentVoice(), None, "360")
            out = out / "made"
            self.assertEqual(summary["quoted_lines"], 1)
            self.assertIn("Marisol", summary["character_voices"])
            self.assertTrue((out / "video.mp4").stat().st_size > 0)
        finally:
            os.chdir(here)
            shutil.rmtree(out.parent if out.name == "made" else out, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

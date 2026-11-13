"""Pipeline 2: narration timing and captions, and the synthesized sound behind every paid one."""

import unittest

try:
    import numpy as np
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



if __name__ == "__main__":
    unittest.main()

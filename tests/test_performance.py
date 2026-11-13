import unittest

import numpy as np

from production import voice
from production.stills import performance


class FakePlanner:
    def __init__(self, answer):
        self.answer, self.prompts = answer, []

    def one(self, role):
        return self

    def generate_json(self, task, system, prompt, schema):
        self.prompts.append(prompt)
        return self.answer


class TakeVoice:
    """An expressive voice that reads whole takes: 0.3 s of tone per word, 0.25 s pause per full stop."""
    sample_rate = 8000
    expressive = True
    takes = True

    def __init__(self):
        self.read = []

    def prepare(self, texts):
        pass

    def say(self, text, speed):
        self.read.append(text)
        parts = []
        for sentence in [s for s in text.replace("?", ".").split(".") if s.strip()]:
            words = len([w for w in sentence.split() if not w.startswith("[")])
            t = np.arange(int(0.3 * words * self.sample_rate)) / self.sample_rate
            parts += [0.3 * np.sin(2 * np.pi * 220 * t), np.zeros(int(0.25 * self.sample_rate))]
        return np.concatenate(parts[:-1]).astype(np.float32)


STORY = "I work nights. Line nine lit up. Nobody uses line nine.\n\nI answered anyway. Nobody spoke."


class Takes(unittest.TestCase):
    def test_paragraph_read_in_one_take_and_cut_at_its_pauses(self):
        v = TakeVoice()
        scenes = [{"delivery": "steady"}]
        audio, tl = voice.narrate(v, STORY, [], [], scenes, [(0, len(STORY))])
        self.assertEqual(len(v.read), 2)  # one take per paragraph, not one call per sentence
        first = tl.chunks[:3]
        self.assertEqual([c.text for c in first], ["I work nights.", "Line nine lit up.", "Nobody uses line nine."])
        # each sentence's time span holds its own words: 3, 4 and 4 words at 0.3 s
        lengths = [c.t1 - c.t0 for c in first]
        self.assertAlmostEqual(lengths[0], 0.9 + 0.25, delta=0.2)
        self.assertAlmostEqual(lengths[1], 1.2 + 0.25, delta=0.2)
        self.assertAlmostEqual(first[0].t1, first[1].t0)

    def test_story_pause_breaks_the_take(self):
        v = TakeVoice()
        voice.narrate(v, STORY, [{"after": "Line nine lit up.", "ms": 900}], [], [{}], [(0, len(STORY))])
        self.assertEqual(len(v.read), 3)

    def test_director_tags_go_into_the_take_once_per_change(self):
        v = TakeVoice()
        chunks = performance.sentences(STORY, [], [])
        marks = {chunks[0].start: "", chunks[1].start: "[nervously]", chunks[2].start: "[nervously]"}
        voice.narrate(v, STORY, [], [], [{}], [(0, len(STORY))], marks=marks)
        self.assertEqual(v.read[0], "I work nights. [nervously] Line nine lit up. Nobody uses line nine.")

    def test_split_take_falls_back_to_text_length_without_pauses(self):
        audio = np.ones(1000, dtype=np.float32) * 0.3
        self.assertEqual(voice.split_take(audio, 8000, [1, 1]), [500])


class Director(unittest.TestCase):
    def test_marks_follow_the_model_within_the_rules(self):
        answer = {"marks": [{"n": 1, "tag": ""}, {"n": 2, "tag": "[whispers]"}, {"n": 3, "tag": "[whispers]"},
                            {"n": 4, "tag": "[shouts]"}, {"n": 5, "tag": "[Quietly]"}]}
        router = FakePlanner(answer)
        scenes = [{"mood": "dread", "delivery": "hushed"}]
        marks, report = performance.direct(router, {"title": "Line Nine"}, STORY, [], [], scenes, [(0, len(STORY))])
        chunks = performance.sentences(STORY, [], [])
        tags = [marks[c.start] for c in chunks]
        self.assertEqual(tags, ["", "[whispers]", "[quietly]", "", "[quietly]"])
        self.assertTrue(report["directed"])
        self.assertIn("1. [dread/hushed] I work nights.", router.prompts[0])

    def test_no_model_or_a_bad_answer_keeps_scene_cues(self):
        self.assertEqual(performance.direct(None, {}, STORY, [], [], [{}], [(0, len(STORY))])[0], {})
        marks, report = performance.direct(FakePlanner({"oops": 1}), {}, STORY, [], [], [{}], [(0, len(STORY))])
        self.assertEqual(marks, {})
        self.assertFalse(report["directed"])

    def test_rules(self):
        self.assertEqual(performance.rules(["[sighs]", "[sighs]", "[nervously]", "[nervously]", "[nervously]"]),
                         ["[sighs]", "", "[nervously]", "[nervously]", ""])


if __name__ == "__main__":
    unittest.main()

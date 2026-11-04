"""Pipeline 2: characters read their own quoted lines."""

import unittest

try:
    import numpy as np  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

from production import dialogue, timing

STORY = ("Delano found the tape on the porch.\n\n"
         "\"Don't play it,\" Marisol said. I laughed. \"It's only a tape,\" I said.\n\n"
         "She stared at the window. \"Then why is it labelled with tomorrow's date?\" she whispered.")
CAST = [{"name": "Delano Marchetti", "look": "tall"}, {"name": "Marisol", "look": "cap"}]


class FakeRouter:
    def __init__(self, answer=None, fail=False):
        self.answer, self.fail, self.prompts = answer, fail, []

    def one(self, role):
        return self

    def generate_json(self, task, system, prompt, schema):
        from story_factory.llm import LLMError
        self.prompts.append(prompt)
        if self.fail:
            raise LLMError("down")
        return self.answer


class DialogueTests(unittest.TestCase):
    def test_quotes_are_found_without_their_marks(self):
        lines = dialogue.find(STORY)
        self.assertEqual([ln.text for ln in lines],
                         ["Don't play it,", "It's only a tape,", "Then why is it labelled with tomorrow's date?"])
        self.assertEqual(STORY[lines[0].start], '"')

    def test_the_words_around_a_quote_say_who_speaks_when_the_model_fails(self):
        lines, kinds = dialogue.attribute(FakeRouter(fail=True), STORY, CAST)
        self.assertEqual([ln.speaker for ln in lines], ["Marisol", "narrator", "Marisol"])  # she = Marisol
        self.assertEqual(kinds["Marisol"], "man")  # unknown kind: a default voice

    def test_the_models_answer_is_snapped_to_the_cast(self):
        answer = {"lines": [{"index": 1, "speaker": "Marisol"}, {"index": 2, "speaker": "I"},
                            {"index": 3, "speaker": "marisol"}],
                  "speakers": [{"name": "Marisol", "voice": "woman"}]}
        router = FakeRouter(answer)
        lines, kinds = dialogue.attribute(router, STORY, CAST)
        self.assertEqual([ln.speaker for ln in lines], ["Marisol", "narrator", "Marisol"])
        self.assertEqual(kinds["Marisol"], "woman")
        self.assertIn("[[\"Don't play it,\"]]", router.prompts[0])

    def test_chunks_are_cut_at_quotes_and_tagged_with_the_speaker(self):
        lines, _ = dialogue.attribute(FakeRouter(fail=True), STORY, CAST)
        chunks = timing.split_lines(STORY, timing.split_chunks(STORY), lines)
        said = [(c.speaker, c.text) for c in chunks]
        self.assertIn(("Marisol", "\"Don't play it,\""), said)
        self.assertIn(("narrator", "Marisol said."), said)
        mine = [c for c in chunks if "only a tape" in c.text][0]
        self.assertEqual((mine.speaker, mine.text), ("narrator", "\"It's only a tape,\" I said."))
        # every character of the story is still covered, in order
        self.assertEqual([c.start for c in chunks], sorted(c.start for c in chunks))
        self.assertTrue(all(STORY[c.start:c.end] == c.text for c in chunks))


@unittest.skipUnless(HAVE_NUMPY, "needs numpy")
class VoiceTests(unittest.TestCase):
    def test_each_speaker_gets_their_own_voice_and_reads_their_line(self):
        from production import voice

        names = voice.pick_voices({"Marisol": "woman", "The Operator": "woman", "narrator": "narrator"},
                                  voice.KOKORO_CAST, "am_michael")
        self.assertEqual(len(set(names.values())), 2)
        self.assertNotIn("am_michael", names.values())

        heard = []

        class Recorder(voice.SilentVoice):
            def say(self, text, speed):
                heard.append((getattr(self, "voice", "narrator"), text))
                return super().say(text, speed)

        narrator = Recorder()
        lines = dialogue.find(STORY)
        lines[0].speaker = "Marisol"
        cast = {"Marisol": Recorder()}
        cast["Marisol"].voice = "af_heart"
        audio, timeline = voice.narrate(narrator, STORY, [], [], [], [(0, len(STORY))], lines, cast)
        self.assertIn(("af_heart", "Don't play it,"), heard)  # no quote marks read aloud
        self.assertTrue(any(c.speaker == "Marisol" for c in timeline.chunks))
        self.assertGreater(timeline.duration, 3)


if __name__ == "__main__":
    unittest.main()

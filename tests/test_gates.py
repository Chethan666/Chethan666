"""The check gates between the agents: script check, listener, art review, mix check, thumbnails."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from production import listen, voice
from production.paid import Budget
from production.shots import Shot
from production.stills import checks, sounds
from production.stills.board import Board
from production.stills.post import voice_over_bed

STORY = "I drive the last bus. Twelve stops, then the depot.\n\nSomeone rang the bell. Nobody was there."


class Align(unittest.TestCase):
    def test_good_take_gives_word_time_cuts(self):
        heard = [(" I", 0.0, 0.1), (" drive", 0.1, 0.4), (" the", 0.4, 0.5), (" last", 0.5, 0.8),
                 (" bus.", 0.8, 1.1), (" Twelve", 1.6, 1.9), (" stops,", 1.9, 2.2), (" then", 2.2, 2.4),
                 (" the", 2.4, 2.5), (" depot.", 2.5, 3.0)]
        score, cuts = listen.align(["I drive the last bus.", "Twelve stops, then the depot."], heard, 1000, 3000)
        self.assertEqual(score, 1.0)
        self.assertEqual(cuts, [1350])  # halfway through the pause between "bus" and "Twelve"

    def test_skipped_words_score_low(self):
        heard = [(" I", 0.0, 0.1), (" drive", 0.1, 0.4), (" depot.", 2.5, 3.0)]
        score, _ = listen.align(["I drive the last bus.", "Twelve stops, then the depot."], heard, 1000, 3000)
        self.assertLess(score, 0.5)

    def test_tags_and_punctuation_are_not_words(self):
        self.assertEqual(listen.tokens("[whispers] Don't — look!"), ["don't", "look"])


class FakeListener:
    """Hears every take wrong the first time and right after that."""

    def __init__(self):
        self.report = {"takes": 0, "reread": 0, "worst_match": 1.0}
        self.calls = 0

    def check(self, audio, rate, sentences):
        self.calls += 1
        self.report["takes"] += 1
        return (0.4 if self.calls == 1 else 1.0), None


class ExpressiveVoice:
    sample_rate = 8000
    expressive = True
    takes = True

    def __init__(self):
        self.read = []

    def prepare(self, texts):
        pass

    def say(self, text, speed, attempt=0):
        self.read.append(text)
        return np.full(int(0.2 * len(text.split()) * self.sample_rate), 0.3, dtype=np.float32)


class Listener(unittest.TestCase):
    def test_a_bad_take_is_read_again_sentence_by_sentence(self):
        v, heard = ExpressiveVoice(), FakeListener()
        _, tl = voice.narrate(v, STORY, [], [], [{}], [(0, len(STORY))], listener=heard)
        self.assertEqual(v.read[:3], ["I drive the last bus. Twelve stops, then the depot.",
                                      "I drive the last bus.", "Twelve stops, then the depot."])
        self.assertEqual(heard.report["reread"], 1)
        first, second = tl.chunks[0], tl.chunks[1]
        self.assertLess(first.t0, first.t1)
        self.assertAlmostEqual(first.t1, second.t0)


def board(tmp, **data):
    b = Board(story={"title": "Route 14", "id": 1}, out=Path(tmp), work=Path(tmp), cfg={})
    b.data.update(data)
    return b


class ScriptCheck(unittest.TestCase):
    def test_reports_what_would_go_wrong(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = board(tmp)
            b.story.update(narration=STORY + ' He said "wait.', scenes=[
                {"text": "I drive the last bus.", "visual": "a bus"}, {"text": "A scene not there.", "visual": ""}],
                pauses=[{"after": "Nowhere in it.", "ms": 500}])
            b.cfg.update(target_seconds=60, voice_pace=1.0)
            report = checks.ScriptCheckAgent().run(b)
            text = " ".join(report["issues"])
            for part in ("scene 2's text", "scene 2 has no visual", "Nowhere in it", "quotation mark", "60 s"):
                self.assertIn(part, text)

    def test_no_narration_stops_the_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                checks.ScriptCheckAgent().run(board(tmp))


class FakeVision:
    def __init__(self, answers):
        self.answers, self.prompts = list(answers), []

    def run(self, model, args):
        self.prompts.append(args["prompt"])
        return {"output": json.dumps(self.answers.pop(0))}


def studio(answers):
    return SimpleNamespace(models={"vision": "vision-model"}, args={}, fal=FakeVision(answers),
                           budget=Budget(1.0, {"vision-model": 0.002}))


def picture(path):
    from PIL import Image
    Image.new("RGB", (320, 180), (20, 20, 20)).save(path)
    return path


class ArtReview(unittest.TestCase):
    def test_flagged_pictures_are_redrawn_with_the_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            beats = [Shot(index=i, scene=0, t0=i, t1=i + 1, text=f"line {i}") for i in range(3)]
            pics = {i: picture(Path(tmp) / f"p{i}.png") for i in range(3)}
            asked = []

            def repaint(beat, fix):
                asked.append((beat.index, fix))
                return picture(Path(tmp) / f"fixed{beat.index}.png")
            b = board(tmp, beats=beats, pictures=pics, repaint=repaint,
                      cast=[{"name": "Tomas", "design": "grey stubble, navy jacket"}])
            b.studio = studio([{"score": 5, "fix": [{"picture": 2, "problem": "Tomas has a beard here"},
                                                    {"picture": 3, "problem": "no bus in it"},
                                                    {"picture": 1, "problem": "too bright"}]}])
            report = checks.ArtReviewAgent().run(b)
            self.assertEqual(asked, [(1, "Tomas has a beard here"), (2, "no bus in it")])  # at most 2
            self.assertEqual(b.data["pictures"][1].name, "fixed1.png")
            self.assertEqual(len(report["flagged"]), 3)
            self.assertIn("Tomas: grey stubble", b.studio.fal.prompts[0])

    def test_without_a_vision_model_nothing_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            beats = [Shot(index=0, scene=0, t0=0, t1=1, text="x")]
            b = board(tmp, beats=beats, pictures={0: picture(Path(tmp) / "p.png")})
            self.assertEqual(checks.ArtReviewAgent().run(b), {"reviewed": False})


class Thumbnail(unittest.TestCase):
    def test_reads_the_words(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = studio([{"ok": False, "reason": "says LAST SOTP"}])
            ok, why = checks.thumbnail_ok(st, picture(Path(tmp) / "t.png"), "last stop")
            self.assertFalse(ok)
            self.assertIn('"LAST STOP"', st.fal.prompts[0])


class Mix(unittest.TestCase):
    def test_a_loud_bed_is_lowered_under_the_voice(self):
        rate = 1000
        voice_track = np.zeros(4000, dtype=np.float32)
        voice_track[1000:2000] = 0.5
        bed = np.full(4000, 0.25, dtype=np.float32)
        mixed, report = voice_over_bed(voice_track + bed, voice_track, [(1.0, 2.0)], rate, 12.0)
        self.assertAlmostEqual(report["voice_over_bed_db"], 6.0, delta=0.1)
        self.assertAlmostEqual(report["after_db"], 12.0, delta=0.1)
        np.testing.assert_allclose(mixed - voice_track, bed * 10 ** (-6 / 20), atol=1e-3)

    def test_a_quiet_bed_is_left_alone(self):
        v = np.full(1000, 0.5, dtype=np.float32)
        mixed, report = voice_over_bed(v + 0.01, v, [(0.0, 1.0)], 1000, 12.0)
        self.assertEqual(report["bed_lowered_db"], 0)


class LibraryOnly(unittest.TestCase):
    def test_the_director_is_not_offered_new_sounds(self):
        text = sounds.prompt("Route 14", [({"need": "music_dread", "kind": "music", "text": "dread"}, [])],
                             None, can_make=False)
        self.assertIn("library only", text)
        self.assertNotIn("ACE-Step", text)


if __name__ == "__main__":
    unittest.main()

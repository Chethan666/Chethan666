"""Short drama series: bible, episodes, judging, and a whole vertical episode with fake services."""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from drama import writing
from tests.fakes import make_context

try:
    import numpy  # noqa: F401
    import PIL  # noqa: F401
    import soundfile  # noqa: F401
    HAVE_VIDEO = bool(shutil.which("ffmpeg"))
except ImportError:
    HAVE_VIDEO = False

BIBLE = {
    "title": "The Night Shift Bride", "logline": "A nurse marries a stranger to save her brother, and he isn't human.",
    "genre": "supernatural thriller", "tone": "tense", "world": "a hospital in Manila, today",
    "hook": "A ring slides onto a hand in a dark ward.", "central_secret": "The groom died ten years ago.",
    "cast": [
        {"name": "Lena", "role": "nurse", "age": 27, "gender": "female", "look": "short black hair, tired eyes",
         "wardrobe": "blue scrubs", "voice": "soft, quick", "want": "save her brother", "secret": "she forged a signature"},
        {"name": "Adrian", "role": "groom", "age": 34, "gender": "male", "look": "pale, sharp jaw, grey eyes",
         "wardrobe": "black suit", "voice": "low, calm", "want": "a living wife", "secret": "he is dead"},
    ],
    "locations": [{"name": "Ward 7", "look": "a dim hospital ward"}, {"name": "Chapel", "look": "a small candlelit chapel"}],
    "arc": "deal, doubt, the truth",
    "episodes": [{"number": n, "title": f"Episode {n}", "hook": "a ring", "beats": ["deal"], "cliffhanger": "he smiles"}
                 for n in range(1, 4)],
}


def shots(n: int = 14, bad_speaker: bool = False) -> list[dict]:
    out = []
    for i in range(n):
        speaker = ("Stranger" if bad_speaker and i == 2 else ["Lena", "Adrian"][i % 2]) if i % 4 != 3 else ""
        out.append({"shot": ["close_up", "over_shoulder", "medium", "insert"][i % 4],
                    "location": ["Ward 7", "Chapel"][i % 2], "characters": ["Lena", "Adrian"] if i % 4 != 3 else [],
                    "action": "Lena stares at the ring", "speaker": speaker,
                    "line": f"Line number {i} is said here." if speaker else "",
                    "emotion": "tense", "inner_voice": i == 2, "motion": i == 5})
    return out


def episode(n: int = 14, **kw) -> dict:
    s = shots(n, **kw)
    return {"title": "The Ring", "hook_line": s[0]["line"], "cliffhanger": "You married a dead man.", "shots": s}


def scores(value: int) -> dict:
    return {"scores": {k: value for k in ("hook", "conflict", "dialogue", "cliffhanger", "continuity")},
            "problems": ["the middle drags"] if value < 7 else []}


def drama_context(review=8, first=None):
    ctx, brain = make_context()
    brain.overrides["drama_bible"] = lambda p, s: BIBLE
    brain.overrides["drama_episode"] = lambda p, s: first or episode()
    brain.overrides["drama_revise"] = lambda p, s: episode()
    brain.overrides["drama_review"] = lambda p, s: scores(review)
    return ctx, brain


class WritingTests(unittest.TestCase):
    def test_a_series_then_episodes_in_order(self):
        ctx, brain = drama_context()
        series_id = writing.create_series(ctx, "supernatural thriller", 3)
        first, status = writing.write_episode(ctx, series_id)
        second, _ = writing.write_episode(ctx, series_id)
        self.assertEqual(status, "approved")
        numbers = [r["number"] for r in ctx.conn.execute("SELECT number FROM episodes ORDER BY id")]
        self.assertEqual(numbers, [1, 2])
        prompt = [p for p in brain.calls if p[0] == "drama_episode"]
        self.assertEqual(len(prompt), 2)

    def test_free_checks_catch_strangers_and_monologues(self):
        problems = writing.episode_issues(episode(bad_speaker=True), BIBLE)
        self.assertTrue(any("cast member" in p for p in problems))
        silent = episode()
        for s in silent["shots"]:
            s["line"] = ""
        self.assertTrue(any("dialogue drama" in p for p in writing.episode_issues(silent, BIBLE)))
        self.assertEqual(writing.episode_issues(episode(), BIBLE), [])

    def test_a_weak_episode_is_revised_once_then_kept_or_rejected(self):
        ctx, brain = drama_context(review=5)
        series_id = writing.create_series(ctx, "thriller", 3)
        _, status = writing.write_episode(ctx, series_id)
        self.assertEqual(status, "rejected")
        self.assertEqual(len(brain.tasks("drama_revise")), 1)
        # a rejected episode is written again next time, not skipped
        brain.overrides["drama_review"] = lambda p, s: scores(8)
        _, status = writing.write_episode(ctx, series_id)
        self.assertEqual(status, "approved")
        self.assertEqual([r["number"] for r in ctx.conn.execute("SELECT number FROM episodes")], [1])

    def test_a_broken_first_draft_is_revised_before_judging(self):
        ctx, brain = drama_context(first=episode(n=4))
        series_id = writing.create_series(ctx, "thriller", 3)
        _, status = writing.write_episode(ctx, series_id)
        self.assertEqual(status, "approved")
        self.assertEqual(len(brain.tasks("drama_revise")), 1)


class FakeFal:
    """Returns local files as if fal.ai had made them."""

    def __init__(self, folder: Path):
        from PIL import Image

        self.calls = []
        self.image = folder / "fake.png"
        Image.new("RGB", (540, 960), (40, 30, 30)).save(self.image)
        self.video = folder / "fake.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=540x960:d=2",
                        "-r", "30", str(self.video)], check=True)

    def run(self, endpoint, args):
        self.calls.append((endpoint, sorted(args)))
        if "image-to-video" in endpoint:
            return {"video": {"url": self.video.as_uri()}}
        return {"images": [{"url": self.image.as_uri()}]}


@unittest.skipUnless(HAVE_VIDEO, "video libraries or ffmpeg not installed")
class ProductionTests(unittest.TestCase):
    def test_an_episode_becomes_a_vertical_video_with_references_reused(self):
        from drama.produce import Producer
        from production.voice import SilentVoice

        ctx, brain = drama_context()
        series_id = writing.create_series(ctx, "thriller", 3)
        episode_id, _ = writing.write_episode(ctx, series_id)
        root = Path(tempfile.mkdtemp())
        try:
            fal = FakeFal(root)
            producer = Producer(ctx.conn, ctx.cfg, root / "assets", fal, SilentVoice(), size=(270, 480))
            meta = producer.produce(episode_id, root / "out")
            video = root / "out" / f"series-{series_id}-ep-01" / "episode.mp4"
            probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                    "stream=width,height", "-of", "csv=p=0", str(video)], capture_output=True, text=True)
            self.assertEqual(probe.stdout.strip(), "270,480")  # vertical 9:16
            self.assertEqual(meta["shots"], 14)
            self.assertEqual(meta["moving_shots"], 1)
            self.assertEqual(set(meta["voices"]), {"Lena", "Adrian"})
            self.assertNotEqual(meta["voices"]["Lena"], meta["voices"]["Adrian"])
            refs = [c for c in fal.calls if "text-to-image" in c[0]]
            self.assertEqual(len(refs), 4)  # two characters, two locations
            edits = [c for c in fal.calls if c[0].endswith("/edit")]
            self.assertTrue(all("image_urls" in c[1] for c in edits))
            # a second episode reuses the same references instead of making new ones
            second, _ = writing.write_episode(ctx, series_id)
            fal.calls.clear()
            producer.produce(second, root / "out")
            self.assertFalse([c for c in fal.calls if "text-to-image" in c[0]])
            spent = ctx.conn.execute("SELECT SUM(usd) FROM costs WHERE pipeline = 'drama'").fetchone()[0]
            self.assertGreater(spent, 0)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_subtitles_mark_thoughts(self):
        from drama.produce import subtitles

        text = subtitles(shots(4), [0, 1, 2, 3], [1, 1, 1, 1])
        events = [line for line in text.splitlines() if line.startswith("Dialogue:")]
        self.assertEqual(len(events), 3)  # the silent insert has no subtitle
        self.assertIn(",Thought,", events[2])  # shot 2 is an inner thought


if __name__ == "__main__":
    unittest.main()

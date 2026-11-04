"""Pipeline 4: a whole vertical episode with fake fal.ai and a placeholder voice."""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from story_factory import db
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


def add_episode(conn, bible=BIBLE, number=1) -> tuple[int, int]:
    """A series and an approved episode, written straight into the database."""
    now = db.now()
    row = conn.execute("SELECT id FROM series WHERE title = ?", (bible["title"],)).fetchone()
    if row is None:
        series_id = conn.execute(
            "INSERT INTO series (title, genre, status, bible, created_at, updated_at) VALUES (?, ?, 'active', ?, ?, ?)",
            (bible["title"], bible["genre"], json.dumps(bible), now, now)).lastrowid
    else:
        series_id = row["id"]
    episode_id = conn.execute(
        "INSERT INTO episodes (series_id, number, title, status, script, created_at, updated_at) "
        "VALUES (?, ?, 'The Ring', 'approved', ?, ?, ?)", (series_id, number, json.dumps(episode()), now, now)).lastrowid
    return series_id, episode_id


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
        from drama_production.produce import Producer
        from drama_production.voices import SilentVoice

        ctx, _ = make_context()
        series_id, episode_id = add_episode(ctx.conn)
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
            _, second = add_episode(ctx.conn, number=2)
            fal.calls.clear()
            producer.produce(second, root / "out")
            self.assertFalse([c for c in fal.calls if "text-to-image" in c[0]])
            spent = ctx.conn.execute("SELECT SUM(usd) FROM costs WHERE pipeline = 'drama'").fetchone()[0]
            self.assertGreater(spent, 0)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_relative_folders_work_and_a_retry_after_a_failure_pays_for_nothing_again(self):
        import os
        from unittest import mock

        from drama_production.produce import Producer
        from drama_production.voices import SilentVoice

        ctx, _ = make_context()
        _, episode_id = add_episode(ctx.conn)
        root = Path(tempfile.mkdtemp())
        here = os.getcwd()
        try:
            os.chdir(root)  # the workflow passes relative folders
            fal = FakeFal(root)
            producer = Producer(ctx.conn, ctx.cfg, Path("data/series/1"), fal, SilentVoice(), size=(270, 480))
            with mock.patch.object(Producer, "clip", side_effect=RuntimeError("the edit broke")):
                with self.assertRaises(RuntimeError):
                    producer.produce(episode_id, Path("out"))
            paid = len(fal.calls)
            self.assertGreater(paid, 0)
            producer.produce(episode_id, Path("out"))
            self.assertEqual(len(fal.calls), paid)  # pictures and clips came from the episode's cache
            self.assertTrue((root / "out" / "series-1-ep-01" / "episode.mp4").exists())
            self.assertFalse((root / "data/series/1/episodes/01").exists())  # cleared once the episode is made
        finally:
            os.chdir(here)
            shutil.rmtree(root, ignore_errors=True)

    def test_boson_speaks_every_line_cloned_from_each_characters_fixed_voice(self):
        import numpy as np
        import soundfile

        from drama_production.media import MediaError
        from drama_production.produce import REFERENCE_TEXT, Producer
        from drama_production.voices import SilentVoice

        class FakeBoson:
            def __init__(self):
                self.calls = []

            def speak(self, text, voice, path, reference=None):
                self.calls.append((text, reference))
                if len(self.calls) == 3:
                    raise MediaError("HTTP 429: slow down")
                soundfile.write(path, np.zeros(8000, dtype=np.float32), 16000)  # a different rate
                return path

        ctx, _ = make_context()
        _, episode_id = add_episode(ctx.conn)
        root = Path(tempfile.mkdtemp())
        try:
            boson = FakeBoson()
            producer = Producer(ctx.conn, ctx.cfg, root / "assets", FakeFal(root), SilentVoice(), size=(270, 480),
                                boson=boson)
            meta = producer.produce(episode_id, root / "out")
            spoken = [s for s in episode()["shots"] if s["line"]]
            self.assertEqual(len(boson.calls), len(spoken))
            references = {ref[0].name for _, ref in boson.calls}
            self.assertEqual(len(references), 2)  # one clip per character, made once
            self.assertTrue(all(ref[1] == REFERENCE_TEXT for _, ref in boson.calls))
            self.assertEqual(meta["boson_lines_refused"], 1)  # that line was read by the fixed voice
            self.assertEqual(meta["voice_engine"], "boson (cloned)")
            self.assertTrue((root / "assets" / "voices").is_dir())
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_subtitles_mark_thoughts(self):
        from drama_production.produce import subtitles

        text = subtitles(shots(4), [0, 1, 2, 3], [1, 1, 1, 1])
        events = [line for line in text.splitlines() if line.startswith("Dialogue:")]
        self.assertEqual(len(events), 3)  # the silent insert has no subtitle
        self.assertIn(",Thought,", events[2])  # shot 2 is an inner thought


if __name__ == "__main__":
    unittest.main()

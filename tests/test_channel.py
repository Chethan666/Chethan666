"""What the agent team shares: the beats, chapters, the spending cap, and the channel (YouTube, approval)."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from production import shots as shotlist
from production.timing import Chunk


def chunk(t0, t1, text="word " * 8, start=0):
    c = Chunk(text.strip(), start, start + len(text), 280)
    c.t0, c.t1 = t0, t1
    return c


class BeatTests(unittest.TestCase):
    def test_beats_cover_the_whole_narration_without_gaps(self):
        chunks = [chunk(0.6, 2.0), chunk(2.3, 4.1), chunk(4.4, 15.0), chunk(15.3, 18.0), chunk(18.2, 19.0)]
        shots = shotlist.cut(chunks, [0, 0, 0, 1, 1], 20.5)
        self.assertEqual(shots[0].t0, 0.0)
        self.assertEqual(shots[-1].t1, 20.5)
        for a, b in zip(shots, shots[1:]):
            self.assertAlmostEqual(a.t1, b.t0)  # no gaps: every moment has a picture
        self.assertTrue(all(s.seconds <= 8.0 for s in shots), [s.seconds for s in shots])
        long_sentence = [s for s in shots if 4.4 <= s.t0 < 15.0]
        self.assertGreaterEqual(len(long_sentence), 1)
        self.assertNotIn(0, [s.scene for s in shots if s.t0 >= 15.3])  # scene borders are respected

    def test_a_beat_the_model_skipped_is_staged_from_its_scene(self):
        beats = shotlist.cut([chunk(0.6, 4.0), chunk(4.2, 8.0)], [0, 0], 9.0)
        scene = {"visual": "A bus on an empty road", "characters": ["Marisol"], "sfx": ["engine hum"],
                 "key_moment": True, "mood": "terror"}
        for b in beats:
            shotlist.fallback(b, scene, beats)
        self.assertIn("A bus on an empty road", beats[0].picture)
        self.assertEqual(beats[0].characters, ["Marisol"])
        self.assertEqual(beats[0].sound, "engine hum")
        self.assertTrue(beats[0].scare)

    def test_chapters_start_at_zero_and_are_spaced(self):
        shots = [shotlist.Shot(i, i, i * 50.0, i * 50.0 + 50, "x") for i in range(5)]
        scenes = [{"location": f"place {i}"} for i in range(5)]
        chapters = shotlist.chapters(None, scenes, shots, offset=10.0)
        self.assertEqual(chapters[0][0], 0.0)
        self.assertGreaterEqual(len(chapters), 3)
        self.assertTrue(all(b[0] - a[0] >= 45 for a, b in zip(chapters, chapters[1:])))
        self.assertEqual(shotlist.stamp(3725), "1:02:05")


class BudgetTests(unittest.TestCase):
    def test_budget_stops_work_at_the_cap_and_failed_jobs_cost_nothing(self):
        from production.paid import Budget, OverBudget

        b = Budget(1.0, {"m": 0.4})
        held = b.reserve("m")
        b.settle(held, "t", "m", ok=False)
        self.assertEqual(b.spent, 0.0)
        b.settle(b.reserve("m"), "t", "m", ok=True)
        b.settle(b.reserve("m"), "t", "m", ok=True)
        with self.assertRaises(OverBudget):
            b.reserve("m")
        self.assertAlmostEqual(b.spent, 0.8)


class YouTubeTests(unittest.TestCase):
    def setUp(self):
        import os

        self.env = {k: os.environ.get(k) for k in ("YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET", "YOUTUBE_REFRESH_TOKEN")}
        os.environ.update(YOUTUBE_CLIENT_ID="id", YOUTUBE_CLIENT_SECRET="secret", YOUTUBE_REFRESH_TOKEN="refresh")

    def tearDown(self):
        import os

        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_publish_uploads_private_with_thumbnail_captions_and_comment(self):
        from production import youtube

        calls = []

        def http(url, method="GET", headers=None, data=None, timeout=0):
            calls.append((method, url.split("?")[0], headers or {}, data))
            if "oauth2" in url:
                return 200, {}, b'{"access_token": "tok"}'
            if "uploadType=resumable" in url:
                return 200, {"Location": "https://upload.example/session"}, b""
            if url.startswith("https://upload.example"):
                return 200, {}, b'{"id": "abc123"}'
            return 200, {}, b"{}"

        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        (folder / "video.mp4").write_bytes(b"video")
        (folder / "captions.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n")
        (folder / "thumbnail_1.jpg").write_bytes(b"jpg")
        (folder / "metadata.json").write_text(json.dumps({"title": "T", "description": "D", "tags": ["a"],
                                                          "pinned_comment": "Would you?"}))
        done = youtube.publish(youtube.YouTube(http=http), folder)
        self.assertEqual(done["video_id"], "abc123")
        self.assertEqual(done["steps"], ["thumbnail", "captions", "comment"])
        self.assertNotIn("shorts", done)
        body = json.loads(next(d for m, u, h, d in calls if u.endswith("/videos")))
        self.assertEqual(body["status"]["privacyStatus"], "private")
        self.assertEqual(body["snippet"]["title"], "T")
        scheduled = youtube.publish(youtube.YouTube(http=http), folder, publish_at="2026-12-01T17:00:00Z")
        self.assertEqual(scheduled["privacy"], "scheduled")

    def test_nothing_is_published_before_a_person_approves_it(self):
        import argparse

        from production.__main__ import channel
        from story_factory import db

        conn = db.connect(":memory:")
        sid = db.insert_story(conn, series="hollow_world", kind="long", title="T", title_norm="t", status="produced",
                              production=json.dumps({"review": "pending"}))
        args = argparse.Namespace(command="publish", story=sid, dir="nowhere", privacy="private", at=None)
        self.assertEqual(channel(args, conn), 1)
        channel(argparse.Namespace(command="approve", story=sid, reject=None), conn)
        self.assertEqual(json.loads(db.get_story(conn, sid)["production"])["review"], "approved")


if __name__ == "__main__":
    unittest.main()

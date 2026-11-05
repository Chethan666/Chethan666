"""Pipeline 2 studio mode: shot list, pictures, clips, voices, sound and the edit, with a fake fal.ai."""

import copy
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    from PIL import Image, ImageDraw
    import soundfile
    HAVE = bool(shutil.which("ffmpeg"))
except ImportError:
    HAVE = False

from production import shots as shotlist
from production.timing import Chunk
from tests.test_production import NARRATION, SCENES


def chunk(t0, t1, text="word " * 8, start=0):
    c = Chunk(text.strip(), start, start + len(text), 280)
    c.t0, c.t1 = t0, t1
    return c


class ShotListTests(unittest.TestCase):
    def test_shots_are_three_to_six_seconds_and_cover_the_whole_narration(self):
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

    def test_tiers_put_hero_clips_on_the_important_shots_and_spread_them(self):
        shots = [shotlist.Shot(i, 0, i * 4.0, i * 4.0 + 4, "x", importance=5 if i in (3, 4, 5, 9) else 1)
                 for i in range(12)]
        shotlist.assign_tiers(shots, [{"key_moment": False}], hero=4, medium=3)
        heroes = [s.index for s in shots if s.tier == "hero"]
        self.assertEqual(len(heroes), 4)
        self.assertIn(9, heroes)
        self.assertEqual(sum(1 for s in shots if s.tier == "medium"), 3)
        self.assertFalse({3, 4, 5} <= set(heroes))  # never three hero shots in a row

    def test_the_model_stages_shots_and_bad_answers_fall_back_to_the_scene_plan(self):
        shots = shotlist.cut([chunk(0.6, 4.0), chunk(4.2, 8.0)], [0, 1], 9.0)

        class Planner:
            def generate_json(self, task, system, prompt, schema):
                self.prompt = prompt
                return {"shots": [{"index": 0, "framing": "close", "picture": "Marisol grips the wheel",
                                   "characters": ["marisol"], "camera": "push_in", "motion": "her head turns",
                                   "sound": "a bell dings", "scare": True, "importance": 9},
                                  {"index": 1, "framing": "nonsense", "picture": ""}]}

        class Router:
            planner = Planner()

            def one(self, role):
                return self.planner

        got = shotlist.direct(Router(), shots, SCENES, [{"name": "Marisol", "look": "cap"}], "the bell")
        self.assertEqual(got, 1)
        self.assertEqual((shots[0].framing, shots[0].characters, shots[0].importance), ("close", ["Marisol"], 5))
        self.assertTrue(shots[0].scare)
        self.assertTrue(shots[1].picture)  # filled in from the scene plan
        self.assertIn("SCENE 1", Router.planner.prompt)

    def test_chapters_start_at_zero_and_are_spaced(self):
        shots = [shotlist.Shot(i, i, i * 50.0, i * 50.0 + 50, "x") for i in range(5)]
        scenes = [{"location": f"place {i}"} for i in range(5)]
        chapters = shotlist.chapters(None, scenes, shots, offset=10.0)
        self.assertEqual(chapters[0][0], 0.0)
        self.assertGreaterEqual(len(chapters), 3)
        self.assertTrue(all(b[0] - a[0] >= 45 for a, b in zip(chapters, chapters[1:])))
        self.assertEqual(shotlist.stamp(3725), "1:02:05")


@unittest.skipUnless(HAVE, "video libraries or ffmpeg not installed")
class ChecksTests(unittest.TestCase):
    def test_chalk_check_passes_chalk_and_fails_colour_and_bright_pictures(self):
        from production.look import chalk_check

        root = Path(tempfile.mkdtemp())
        try:
            good = root / "good.png"
            chalk_picture(good)
            self.assertTrue(chalk_check(good)[0])
            colour = root / "colour.png"
            im = Image.open(good).copy()
            ImageDraw.Draw(im).rectangle((0, 0, 500, 300), fill=(40, 90, 200))
            im.save(colour)
            self.assertFalse(chalk_check(colour)[0])
            bright = root / "bright.png"
            Image.new("RGB", (640, 360), (230, 230, 230)).save(bright)
            self.assertFalse(chalk_check(bright)[0])
        finally:
            shutil.rmtree(root, ignore_errors=True)

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

    def test_clip_length_is_the_shortest_that_covers_the_shot(self):
        from production.clips import length

        self.assertEqual(length(4.9, [4, 5, 6]), 4)  # 4 s slowed to 1.25x covers 5 s
        self.assertEqual(length(6.0, [4, 5, 6]), 5)
        self.assertEqual(length(30.0, [4, 5, 6]), 6)


def chalk_picture(path, size=(640, 360), colour=False):
    im = Image.new("RGB", size, (6, 6, 6))
    d = ImageDraw.Draw(im)
    d.ellipse((300, 80, 340, 120), outline=(235, 235, 230), width=4)
    d.line((320, 120, 320, 220), fill=(235, 235, 230), width=4)
    d.line((320, 220, 290, 290), fill=(235, 235, 230), width=4)
    d.line((320, 220, 350, 290), fill=(235, 235, 230), width=4)
    d.rectangle((420, 200, 440, 290), outline=(200, 20, 25), width=4)
    if colour:
        d.rectangle((0, 0, size[0], size[1] // 2), fill=(30, 120, 200))
    im.save(path)


class FakeFal:
    """Answers every studio model with local files, as if fal.ai had made them."""

    def __init__(self, folder: Path, colour_clips=False, fail_hero=False):
        self.calls = []
        self.fail_hero = fail_hero
        self.image = folder / "fake.png"
        chalk_picture(self.image)
        self.video = folder / "fake.mp4"
        clip_picture = folder / "clip.png"
        chalk_picture(clip_picture, colour=colour_clips)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-loop", "1", "-i", str(clip_picture), "-t", "3",
                        "-r", "24", "-pix_fmt", "yuv420p", str(self.video)], check=True)
        self.audio = folder / "fake.wav"
        t = np.arange(44100 * 2) / 44100
        soundfile.write(self.audio, (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 44100)

    def run(self, endpoint, args):
        from drama_production.media import MediaError

        self.calls.append((endpoint, dict((k, v) for k, v in args.items() if not str(v).startswith("data:"))))
        if "any-llm" in endpoint:
            return {"output": '{"ok": true, "reason": "fine"}'}
        if "seedance-2.0" in endpoint and self.fail_hero:
            raise MediaError("HTTP 500: busy")
        if "video" in endpoint:
            return {"video": {"url": self.video.as_uri()}}
        if "mmaudio" in endpoint:
            return {"video": {"url": self.video.as_uri()}}
        if "elevenlabs" in endpoint:
            return {"audio": {"url": self.audio.as_uri()}}
        return {"images": [{"url": self.image.as_uri()}]}


def story():
    return {"id": 7, "title": "Night Bus", "narration": NARRATION, "scenes": json.dumps(SCENES),
            "cast": json.dumps([{"name": "Marisol", "look": "short, long hair, bus driver's cap"}]),
            "bible": json.dumps({"red_accent": "the stop bell"}),
            "pauses": json.dumps([{"after": "Tonight the bell rang", "ms": 1200}]),
            "packaging": json.dumps({"titles": ["The Extra Passenger"], "thumbnail_text": "WHO RANG",
                                     "description": "A story.", "tags": ["horror"],
                                     "end_screen_line": "The next bus leaves at midnight."}),
            "comment_question": "Would you have opened the door?"}


def studio_cfg(**over):
    from story_factory.config import DEFAULTS

    cfg = copy.deepcopy(DEFAULTS["studio"])
    cfg.update({"end_screen_seconds": 3.0, "hero_shots": 2, "medium_shots": 2, "max_usd": 10.0, "parallel": 2},
               **over)
    return cfg


@unittest.skipUnless(HAVE, "video libraries or ffmpeg not installed")
class StudioTests(unittest.TestCase):
    def run_studio(self, fal, cfg=None):
        from production import studio, voice

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        out = root / "out"
        summary = studio.produce(story(), out, voice.SilentVoice(), None, cfg or studio_cfg(), fal(root), None)
        return summary, out

    def test_a_story_becomes_a_packaged_video(self):
        made = {}

        def fal(root):
            made["fal"] = FakeFal(root)
            return made["fal"]

        summary, out = self.run_studio(fal)
        for name in ("video.mp4", "captions.srt", "metadata.json", "shots.json", "report.json", "contact_sheet.jpg",
                     "preview_60s.mp4", "thumbnail_1.jpg", "thumbnail_3.jpg", "narration.wav"):
            self.assertTrue((out / name).exists(), name)
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=width,height",
                                "-of", "json", str(out / "video.mp4")], capture_output=True, text=True)
        info = json.loads(probe.stdout)
        self.assertEqual((info["streams"][0]["width"], info["streams"][0]["height"]), (1920, 1080))
        self.assertAlmostEqual(float(info["format"]["duration"]), summary["seconds"], delta=0.6)
        self.assertEqual(summary["voice"]["engine"], "elevenlabs-v3")
        self.assertEqual(summary["clips"], 4)  # 2 hero + 2 medium
        self.assertGreater(summary["usd"], 0)
        self.assertLessEqual(summary["usd"], 10.0)
        self.assertTrue(summary["cold_open_shots"])
        endpoints = [e for e, _ in made["fal"].calls]
        self.assertIn("fal-ai/nano-banana-pro", endpoints)  # the style frame
        self.assertIn("bytedance/seedance-2.0/fast/image-to-video", endpoints)
        self.assertIn("fal-ai/elevenlabs/music", endpoints)
        voiced = [a.get("text", "") for e, a in made["fal"].calls if e.endswith("eleven-v3")]
        self.assertTrue(any(t.startswith("[whispers]") for t in voiced))  # the hushed scene is performed
        meta = json.loads((out / "metadata.json").read_text())
        self.assertEqual(meta["title"], "The Extra Passenger")
        self.assertFalse((out / "work").exists())
        srt = (out / "captions.srt").read_text()
        self.assertIn("Tonight the bell rang", srt)

    def test_clips_that_lose_the_chalk_look_are_dropped_and_failed_heroes_try_kling(self):
        made = {}

        def fal(root):
            made["fal"] = FakeFal(root, colour_clips=True, fail_hero=True)
            return made["fal"]

        summary, _ = self.run_studio(fal)
        self.assertEqual(summary["clips"], 0)
        self.assertTrue(summary["clips_rejected"])
        self.assertIn("fal-ai/kling-video/v3/pro/image-to-video", [e for e, _ in made["fal"].calls])

    def test_the_cap_holds_and_the_video_is_still_made(self):
        summary, out = self.run_studio(FakeFal, studio_cfg(max_usd=0.5))
        self.assertLessEqual(summary["usd"], 0.5)
        self.assertTrue((out / "video.mp4").exists())
        self.assertGreater(summary["pictures"]["drawn_in_code"], 0)


    def test_the_bake_off_compares_every_model_on_the_same_shots(self):
        from production import bakeoff, voice

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        report = bakeoff.run(story(), root / "bake", voice.SilentVoice(), None, studio_cfg(), FakeFal(root))
        names = {p.name for p in (root / "bake").iterdir()}
        self.assertIn("compare_pictures.jpg", names)
        self.assertTrue(any(n.endswith("_shot_alt.png") for n in names))
        self.assertTrue(any(n.startswith("clip_") and n.endswith("_hero_alt.mp4") for n in names))
        self.assertIn("voice_eleven_Brian.mp3", names)
        self.assertGreater(report["usd"], 0)


if __name__ == "__main__":
    unittest.main()


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
        (folder / "short_1.mp4").write_bytes(b"short")
        (folder / "metadata.json").write_text(json.dumps({"title": "T", "description": "D", "tags": ["a"],
                                                          "pinned_comment": "Would you?", "shorts": ["short_1.mp4"]}))
        done = youtube.publish(youtube.YouTube(http=http), folder)
        self.assertEqual(done["video_id"], "abc123")
        self.assertEqual(done["steps"], ["thumbnail", "captions", "comment"])
        self.assertEqual(done["shorts"], ["abc123"])
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

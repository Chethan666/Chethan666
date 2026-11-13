"""Pipeline 2, stills mode: the agent team, the painter chain and the critic, with fake services."""

import base64
import copy
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import soundfile
    from PIL import Image, ImageDraw
    HAVE = bool(shutil.which("ffmpeg"))
except ImportError:
    HAVE = False

from tests.test_production import NARRATION, SCENES

INTRO = "I keep the lantern lit for stories like this one. The lantern is lit."
BODY = ("I drove the night bus for nine years, and my route never changed. Nobody ever pressed the bell after "
        "midnight. Tonight the bell rang. I checked my mirror and every seat was empty. \"Who's there?\" I said. "
        "\"Only the last stop,\" the old man whispered from the back.")
OUTRO = "Some routes should end early. Would you have stopped the bus?"
FIRST_SCENES = [
    {"text": INTRO, "visual": "The Storyteller with his lantern", "location": "the hollow", "characters": [],
     "mood": "eerie", "delivery": "slow", "sfx": [], "key_moment": False},
    {"text": "I drove the night bus for nine years, and my route never changed. Nobody ever pressed the bell "
             "after midnight.", "visual": "A bus on an empty road", "location": "a road", "characters": ["Dana"],
     "mood": "calm", "delivery": "steady", "sfx": ["engine hum"], "key_moment": False},
    {"text": "Tonight the bell rang. I checked my mirror and every seat was empty.", "visual": "Dana looks in the "
     "mirror", "location": "inside the bus", "characters": ["Dana"], "mood": "terror", "delivery": "hushed",
     "sfx": ["bell"], "key_moment": True},
    {"text": "\"Who's there?\" I said. \"Only the last stop,\" the old man whispered from the back.",
     "visual": "A thin old man in the back seat", "location": "inside the bus", "characters": ["Dana"],
     "mood": "dread", "delivery": "hushed", "sfx": [], "key_moment": True},
    {"text": OUTRO, "visual": "The Storyteller closes his lantern", "location": "the hollow", "characters": [],
     "mood": "eerie", "delivery": "slow", "sfx": [], "key_moment": False},
]


def painted(path, bright=False, seed=0):
    """A stand-in illustration: a dark, textured scene with a lit figure (or a bright, cheerful one)."""
    rng = np.random.default_rng(seed)
    base = 200 if bright else 30
    v = np.clip(base + rng.normal(0, 18, (360, 640)), 0, 255)
    im = Image.fromarray(np.stack([v * 0.8, v * 0.9, v], -1).astype("uint8"))
    d = ImageDraw.Draw(im)
    d.ellipse((280, 80, 360, 160), fill=(190, 180, 170))
    d.rectangle((295, 160, 345, 300), fill=(120, 110, 100))
    d.rectangle((450, 200, 470, 290), fill=(150, 20, 25))
    im.save(path)
    return path


def story(first_person=False):
    if first_person:
        narration = f"{INTRO}\n\n{BODY}\n\n{OUTRO}"
        return {"id": 9, "title": "Last Stop", "narration": narration, "intro": INTRO, "body": BODY,
                "outro": OUTRO, "scenes": json.dumps(FIRST_SCENES),
                "cast": json.dumps([{"name": "Dana", "look": "stick figure, short hair, driver's cap"}]),
                "bible": json.dumps({"red_accent": "the stop bell", "hero": {"name": "Dana", "who": "a bus driver"}}),
                "packaging": json.dumps({"titles": ["The Last Stop"], "thumbnail_text": "LAST STOP",
                                         "description": "A story.", "tags": ["horror"]})}
    intro, rest = NARRATION.split("\n\n", 1)
    body, outro = rest.rsplit("\n\n", 1)
    return {"id": 7, "title": "Night Bus", "narration": NARRATION, "intro": intro, "body": body, "outro": outro,
            "scenes": json.dumps(SCENES), "cast": json.dumps([{"name": "Marisol", "look": "bus driver's cap"}]),
            "bible": json.dumps({"red_accent": "the stop bell"}),
            "pauses": json.dumps([{"after": "Tonight the bell rang", "ms": 1200}]),
            "packaging": json.dumps({"titles": ["The Extra Passenger"], "thumbnail_text": "WHO RANG",
                                     "description": "A story.", "tags": ["horror"],
                                     "end_screen_line": "The next bus leaves at midnight."}),
            "comment_question": "Would you have opened the door?"}


def cfg(**over):
    from story_factory.config import DEFAULTS

    c = copy.deepcopy(DEFAULTS["stills"])
    c["sound_library"] = dict(c["sound_library"], enabled=False)  # no release downloads in tests
    c["listener"] = ""  # no Whisper download in tests (the listener has its own tests)
    c.update({"end_screen_seconds": 3.0, "title_seconds": 2.0, "max_usd": 5.0, "parallel": 2, "pollinations": False,
              "beat_min_seconds": 2.0, "beat_max_seconds": 5.0}, **over)
    return c


class FakeFal:
    """Answers every stills model with local files, as if fal.ai had made them."""

    def __init__(self, folder, bright_shots=0, reject_vision=0):
        self.calls = []
        self.good = painted(folder / "good.png")
        self.bright = painted(folder / "bright.png", bright=True)
        self.bright_shots = bright_shots
        self.reject_vision = reject_vision
        self.audio = folder / "fake.wav"
        t = np.arange(44100 * 2) / 44100
        soundfile.write(self.audio, (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 44100)

    def run(self, endpoint, args):
        self.calls.append((endpoint, {k: v for k, v in args.items() if not str(v).startswith("data:")
                                      and k != "image_urls"}))
        if "any-llm" in endpoint:
            if self.reject_vision > 0:
                self.reject_vision -= 1
                return {"output": '{"ok": false, "reason": "a watermark"}'}
            return {"output": '{"ok": true, "reason": "fine"}'}
        if "elevenlabs" in endpoint or "ace-step" in endpoint:
            return {"audio": {"url": self.audio.as_uri()}}
        if "seedream" in endpoint and self.bright_shots > 0:
            self.bright_shots -= 1
            return {"images": [{"url": self.bright.as_uri()}]}
        return {"images": [{"url": self.good.as_uri()}]}


class Planner:
    """The planner role: casts the teller and stages beats."""

    def __init__(self):
        self.tasks = []

    def generate_json(self, task, system, prompt, schema):
        self.tasks.append(task)
        if task == "teller_voice":
            return {"kind": "woman", "reason": "Dana is a woman"}
        if task == "dialogue_speakers":
            return {"lines": [{"index": 1, "speaker": "narrator"}, {"index": 2, "speaker": "the old man"}],
                    "speakers": [{"name": "the old man", "voice": "old_man"}]}
        if task == "storyboard":
            import re
            return {"beats": [{"index": int(i), "framing": "close", "picture": f"Beat {i}: Dana grips the wheel",
                               "characters": ["dana"], "sound": "", "scare": i == "3", "importance": 4,
                               "view": "cctv" if i == "3" else "film"}
                              for i in re.findall(r"beat (\d+) \(", prompt)]}
        if task == "character_design":
            return {"characters": [{"name": "Dana", "design": "a woman of forty in a navy driver's jacket"}]}
        from story_factory.llm import LLMError
        raise LLMError("not needed")


class Router:
    def __init__(self):
        self.planner = Planner()

    def one(self, role):
        return self.planner


class UnitTests(unittest.TestCase):
    def test_point_of_view_counts_the_teller_not_the_quotes(self):
        from production.stills.voices import point_of_view

        self.assertEqual(point_of_view(BODY), "first")
        self.assertEqual(point_of_view('Marisol drove the bus. "I am here," the man said. "My stop."'), "third")

    def test_the_host_frames_are_found_in_the_narration(self):
        from production.stills.voices import frame_spans

        s = story(first_person=True)
        spans = frame_spans(s, s["narration"])
        self.assertEqual(s["narration"][slice(*spans[0])], INTRO)
        self.assertEqual(s["narration"][slice(*spans[1])], OUTRO)


@unittest.skipUnless(HAVE, "video libraries or ffmpeg not installed")
class LookTests(unittest.TestCase):
    def test_the_default_look_is_cinematic_and_a_style_overrides_its_text(self):
        from production.stills.images import house_look

        look = house_look(cfg())
        self.assertEqual(look["name"], "cinematic")
        self.assertIn("35mm", look["style"])
        self.assertIn("cartoon", look["wrong"])
        self.assertTrue(look["synthetic_media"])
        self.assertIn("photograph", house_look(cfg(look="illustrated"))["wrong"])
        self.assertEqual(house_look(cfg(style="My own look."))["style"], "My own look.")

    @unittest.skipUnless(HAVE, "video libraries or ffmpeg not installed")
    def test_film_finish_adds_grain_and_darkens_the_corners(self):
        from production.stills.images import finish

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        Image.new("RGB", (320, 180), (90, 90, 90)).save(root / "flat.png")
        finish(root / "flat.png", root / "film.png", 0.04, 0.3, 1)
        v = np.asarray(Image.open(root / "film.png").convert("L"), dtype=np.float32)
        self.assertGreater(v[80:100, 150:170].std(), 2.0)  # grain
        self.assertLess(v[:20, :20].mean(), v[80:100, 150:170].mean() - 10)  # vignette
        finish(root / "flat.png", root / "same.png", 0.0, 0.0, 1)
        self.assertEqual((root / "same.png").read_bytes(), (root / "flat.png").read_bytes())

    def test_a_beat_prompt_locks_each_person_to_their_sheet(self):
        from production.shots import Shot
        from production.stills.art import IllustratorAgent
        from production.stills.images import house_look

        beat = Shot(index=3, scene=1, t0=0, t1=5, text="", framing="close", picture="Dana looks at the mirror.",
                    characters=["Dana"])
        cast = {"Dana": "a woman of forty, always wears thin round glasses"}
        text = IllustratorAgent().prompt(beat, "HOUSE.", cast, ["Dana"], True, house_look(cfg()), "Seen on CCTV")
        self.assertIn("identical face", text)
        self.assertIn("Seen on CCTV", text)
        self.assertIn("plastic", text)  # what the realistic look avoids
        self.assertIn("film still", text)
        self.assertIn("always wears thin round glasses", IllustratorAgent.check_text(beat, cast))


class PainterTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_the_critic_fails_bright_and_blank_pictures(self):
        from production.stills.images import still_check

        self.assertTrue(still_check(painted(self.root / "a.png"))[0])
        self.assertFalse(still_check(painted(self.root / "b.png", bright=True))[0])
        Image.new("RGB", (640, 360), (20, 20, 20)).save(self.root / "c.png")
        self.assertFalse(still_check(self.root / "c.png")[0])

    def test_a_rejected_picture_goes_to_the_next_painter(self):
        from production.paid import Budget, Studio
        from production.stills.images import Critic, Painter

        fal = FakeFal(self.root, bright_shots=1)
        c = cfg()
        studio = Studio(fal, Budget(5.0, c["prices"]), self.root / "cache", c)
        painter = Painter(studio, c, self.root / "art", Critic(studio, "dark"))
        ref = painted(self.root / "ref.png")
        out, source = painter.paint("a bus", [ref], self.root / "beat.png", "k", c["chains"]["beat"], check="a bus")
        self.assertEqual(source, "fal:shot_alt")
        self.assertTrue(out.exists())
        self.assertIn("too bright", painter.rejections[0])
        with Image.open(out) as im:
            self.assertEqual(im.size, (1920, 1080))

    def test_free_painters_without_any_paid_key(self):
        import os
        from unittest import mock

        from production.stills.images import Critic, Painter

        buf = io.BytesIO()
        Image.open(painted(self.root / "p.png")).save(buf, "PNG")
        png = buf.getvalue()
        seen = []

        def http(url, method="GET", headers=None, body=None, timeout=0, raw=False):
            seen.append(url)
            if "cloudflare" in url:
                return {"result": {"image": base64.b64encode(png).decode()}, "success": True}
            return png

        c = cfg(pollinations=True, pollinations_gap_seconds=0)
        with mock.patch.dict(os.environ, {"CF_ACCOUNT_ID": "acc", "CF_API_TOKEN": "tok"}):
            painter = Painter(None, c, self.root / "art", Critic(None, "dark"), http=http)
            out, source = painter.paint("a bus", [], self.root / "x.png", "k", c["chains"]["beat"], check="a bus")
        self.assertEqual(source, "cloudflare")
        self.assertIn("/accounts/acc/ai/run/@cf/black-forest-labs/flux-1-schnell", seen[0])
        painter = Painter(None, c, self.root / "art2", Critic(None, "dark"), http=http)
        out, source = painter.paint("a bus", [], self.root / "y.png", "k", c["chains"]["beat"], check="a bus")
        self.assertEqual(source, "pollinations")
        self.assertTrue(seen[-1].startswith("https://image.pollinations.ai/prompt/a%20bus"))


@unittest.skipUnless(HAVE, "video libraries or ffmpeg not installed")
class TeamTests(unittest.TestCase):
    def make(self, fal=None, router=None, first_person=False, **over):
        from production import voice
        from production.stills import pipeline

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        made = fal(root) if fal else None
        summary = pipeline.produce(story(first_person), root / "out", voice.SilentVoice(), router, cfg(**over), made)
        return summary, root / "out", made

    def test_a_story_becomes_a_packaged_stills_video(self):
        from story_factory.config import DEFAULTS
        lib = dict(DEFAULTS["stills"]["sound_library"], enabled=False, generate_missing=True)
        summary, out, fal = self.make(FakeFal, sound_library=lib)  # sounds may be made: the paid path
        for name in ("video.mp4", "captions.srt", "metadata.json", "beats.json", "report.json", "contact_sheet.jpg",
                     "preview_60s.mp4", "thumbnail_1.jpg", "thumbnail_3.jpg", "narration.wav"):
            self.assertTrue((out / name).exists(), name)
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=width,height",
                                "-of", "json", str(out / "video.mp4")], capture_output=True, text=True)
        info = json.loads(probe.stdout)
        self.assertEqual((info["streams"][0]["width"], info["streams"][0]["height"]), (1920, 1080))
        self.assertAlmostEqual(float(info["format"]["duration"]), summary["seconds"], delta=0.6)
        self.assertEqual(summary["qa"]["status"], "pass", summary["qa"])
        self.assertEqual(summary["point_of_view"], "third")
        endpoints = [e for e, _ in fal.calls]
        self.assertIn("fal-ai/nano-banana-2", endpoints)  # the style frame
        self.assertIn("fal-ai/bytedance/seedream/v4.5/edit", endpoints)  # the beats, from references
        self.assertIn("fal-ai/ace-step-1.5", endpoints)  # music the library lacked, when making is allowed
        self.assertIn("art_review", summary["agents"])
        self.assertIn("script_check", summary["agents"])
        self.assertTrue(summary["agents"]["sound"]["mix"]["checked"])
        self.assertFalse(any("video" in e for e in endpoints))  # no moving clips at all
        self.assertGreater(summary["usd"], 0)
        self.assertLessEqual(summary["usd"], 5.0)
        self.assertFalse((out / "work").exists())
        self.assertEqual(json.loads((out / "metadata.json").read_text())["title"], "The Extra Passenger")
        self.assertEqual(list(summary["agents"]), ["script_check", "casting", "voice", "storyboard", "art",
                                                   "illustrator", "art_review", "edit_plan", "sound", "edit",
                                                   "packager", "qa"])

    def test_a_first_person_story_has_a_host_a_teller_and_a_character_voice(self):
        summary, out, fal = self.make(FakeFal, Router(), first_person=True)
        self.assertEqual(summary["point_of_view"], "first")
        self.assertEqual(summary["voices"]["teller"], "Charlotte")  # the planner said the teller is a woman
        voiced = {(a.get("voice"), a.get("text", "")[:30]) for e, a in fal.calls if e.endswith("eleven-v3")}
        by_voice = {}
        for name, text in voiced:
            by_voice.setdefault(name, []).append(text)
        self.assertTrue(any("lantern" in t for t in by_voice.get("Brian", [])), by_voice)  # the host's intro
        self.assertTrue(any("night bus" in t for t in by_voice.get("Charlotte", [])), by_voice)
        self.assertTrue(any("last stop" in t for t in by_voice.get("Bill", [])), by_voice)  # the old man
        self.assertGreater(summary["agents"]["storyboard"]["host_beats"], 0)
        self.assertGreater(summary["agents"]["storyboard"]["directed_by_model"], 0)
        prompts = [a.get("prompt", "") for e, a in fal.calls if "seedream" in e]
        self.assertTrue(any("navy driver's jacket" in p for p in prompts))  # the designed character
        self.assertFalse(any("stick figure" in p for p in prompts))
        self.assertTrue(any("security camera" in p for p in prompts))  # beat 3 is seen on a camera
        self.assertTrue(all("identical face" in p for p in prompts if "character sheet of Dana" in p))
        self.assertEqual(summary["agents"]["illustrator"]["look"], "cinematic")
        self.assertTrue(json.loads((out / "metadata.json").read_text())["synthetic_media"])

    def test_by_default_only_pictures_voice_and_checks_are_paid_for(self):
        summary, _, fal = self.make(FakeFal)
        endpoints = {e for e, _ in fal.calls}
        self.assertFalse(any("ace-step" in e or "music" in e or "sound-effects" in e for e in endpoints), endpoints)
        self.assertEqual(summary["agents"]["sound"]["generated"], 0)

    def test_without_any_key_the_video_is_still_made_and_qa_says_what_to_check(self):
        summary, out, _ = self.make()
        self.assertTrue((out / "video.mp4").exists())
        self.assertEqual(summary["voices"]["engine"], "kokoro")
        self.assertEqual(summary["qa"]["status"], "check")
        self.assertTrue(any("placeholder" in i for i in summary["qa"]["issues"]))
        self.assertEqual(summary["usd"], 0)

    def test_the_illustrated_look_is_still_available(self):
        summary, out, fal = self.make(FakeFal, Router(), look="illustrated")
        prompts = [a.get("prompt", "") for e, a in fal.calls if "seedream" in e]
        self.assertTrue(prompts and all("graphic novel" in p for p in prompts))
        self.assertFalse(any("security camera" in p for p in prompts))  # cameras belong to the film look
        self.assertFalse(json.loads((out / "metadata.json").read_text())["synthetic_media"])

    def test_drift_motion_and_the_cap(self):
        summary, out, _ = self.make(FakeFal, motion="drift", max_usd=0.3)
        self.assertLessEqual(summary["usd"], 0.3)
        self.assertTrue((out / "video.mp4").exists())
        self.assertEqual(summary["agents"]["edit"]["motion"], "drift")


if __name__ == "__main__":
    unittest.main()

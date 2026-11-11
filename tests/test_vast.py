"""The open-model Vast.ai path: excerpt choice, the GPU job, the CLI wrapper, and a whole video with a fake GPU."""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.test_studio import HAVE, SCENES, chalk_picture, story

if HAVE:
    import numpy as np
    import soundfile


class ExcerptTests(unittest.TestCase):
    def test_the_excerpt_builds_up_to_the_first_key_scare_and_never_includes_the_ending(self):
        from production.vastvideo import excerpt

        scenes = [{"text": "w " * 30, "key_moment": False} for _ in range(8)]
        scenes[5]["key_moment"] = True
        scenes[7]["key_moment"] = True  # the ending is never used
        picked = excerpt(scenes)
        self.assertEqual(picked[-1], 5)
        self.assertEqual(picked, sorted(picked))
        self.assertTrue(75 <= 30 * len(picked) <= 128)
        self.assertNotIn(7, picked)

    def test_the_job_names_every_sound_the_mixer_will_ask_for(self):
        from production import shots as shotlist
        from production.paid import digest
        from production.vastvideo import build_job

        shots = [shotlist.Shot(0, 0, 0, 4, "x", picture="Marisol at the wheel", characters=["Marisol"]),
                 shotlist.Shot(1, 1, 4, 8, "y", picture="the empty bus")]
        job = build_job(shots, SCENES[1:3], [{"name": "Marisol", "look": "cap"}], "the bell", {})
        names = {a["name"] for a in job["audio"]}
        self.assertTrue({"foley_000", "foley_001", "music_unease", "music_dread", "music_terror", "sting",
                         f"room_{digest('a road')}"} <= names)
        self.assertIn("{sheet:Marisol}", job["shots"][0]["picture_prompt"])
        self.assertIn("No people", job["shots"][1]["picture_prompt"])
        self.assertEqual(job["sheets"][0]["name"], "Marisol")


class RemoteScriptTests(unittest.TestCase):
    def test_the_remote_script_names_both_models_and_keeps_its_shell_syntax(self):
        from production.vast import remote_script

        text = remote_script({"picture_model": "Qwen/Qwen-Image-Edit-2509", "clip_model": "Wan-AI/W"})
        self.assertIn("d('Qwen/Qwen-Image-Edit-2509')", text)
        self.assertIn("d('Wan-AI/W')", text)
        self.assertIn('step() { echo "$(date +%H:%M:%S) == $*"; }', text)
        check = subprocess.run(["bash", "-n"], input=text, capture_output=True, text=True)
        self.assertEqual(check.returncode, 0, check.stderr)


class FakeRun:
    """Answers `vastai` commands like the real CLI would."""

    def __init__(self):
        self.calls = []

    def __call__(self, cmd, capture_output=True, text=True, timeout=0):
        self.calls.append(cmd)
        args = cmd[1:cmd.index("--api-key")]
        out = ""
        if args[:2] == ["search", "offers"]:
            out = json.dumps([{"id": 2, "gpu_name": "H100 SXM", "dph_total": 2.4, "cpu_arch": "amd64"},
                              {"id": 3, "gpu_name": "GB10", "dph_total": 0.5, "cpu_arch": "arm64"},
                              {"id": 4, "gpu_name": "RTX PRO 6000", "dph_total": 1.2, "cpu_arch": "amd64",
                               "internet_down_cost_per_tb": 20.0},
                              {"id": 5, "gpu_name": "RTX PRO 6000", "dph_total": 1.3, "cpu_arch": "amd64",
                               "internet_down_cost_per_tb": 4.0},
                              {"id": 1, "gpu_name": "H100 NVL", "dph_total": 1.9, "cpu_arch": "amd64"}])
        elif args[:2] == ["create", "instance"]:
            out = "Started. " + json.dumps({"success": True, "new_contract": 777})
        elif args[:1] == ["show"] and args[1] == "user":
            out = json.dumps({"credit": 19.5})
        elif args[:2] == ["show", "instances"]:
            out = json.dumps([{"id": 777, "label": "hollow-lines-9"}, {"id": 5, "label": "other"}])
        return subprocess.CompletedProcess(cmd, 0, out, "")


class VastCliTests(unittest.TestCase):
    def test_offers_rent_balance_and_cleanup(self):
        from production.vast import Vast

        run = FakeRun()
        vast = Vast(key="k", run=run, label="hollow-lines-9")
        # ARM boxes and every host that charges anything for internet traffic are left out (run 77
        # paid $6.72 for downloads); the rest by true cost
        self.assertEqual([o["id"] for o in vast.offers()], [1, 2])
        self.assertEqual(vast.balance(), 19.5)
        machine = vast.rent({"id": 1, "gpu_name": "H100 NVL", "dph_total": 1.9})
        self.assertEqual((machine.id, machine.price), (777, 1.9))
        self.assertEqual(vast.cleanup_label(), 1)
        self.assertIn(["vastai", "destroy", "instance", "777", "-y", "--api-key", "k"], run.calls)
        self.assertTrue(all(c[-2:] == ["--api-key", "k"] for c in run.calls))

    def test_an_empty_account_stops_before_anything_is_rented(self):
        from production.vast import Vast, VastError, run_job

        run = FakeRun()
        vast = Vast(key="k", run=run)
        vast.balance = lambda: 0.0
        with self.assertRaises(VastError) as caught:
            run_job(vast, {}, Path("w"), Path("o"), 6.0)
        self.assertIn("add credit", str(caught.exception))
        self.assertFalse(any("create" in c for c in run.calls))


@unittest.skipUnless(HAVE, "video libraries or ffmpeg not installed")
class VastVideoTests(unittest.TestCase):
    def test_a_story_excerpt_becomes_a_video_from_the_gpus_pictures_clips_and_sounds(self):
        from production import vastvideo, voice

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        seen = {}

        def fake_job(vast, job, worker, local, max_usd, max_hours=3.4):
            seen["job"] = job
            out = local / "out"
            out.mkdir(parents=True)
            chalk_picture(out / "style_frame.png", size=(1664, 928))
            for k, shot in enumerate(job["shots"]):
                if k == 1:
                    continue  # one picture failed on the GPU: a neighbour stands in
                chalk_picture(out / f"shot_{shot['index']:03d}.png")
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-loop", "1", "-i",
                                str(out / f"shot_{shot['index']:03d}.png"), "-t", "5", "-r", "16", "-vf",
                                "scale=1280:720", "-pix_fmt", "yuv420p", str(out / f"clip_{shot['index']:03d}.mp4")],
                               check=True)
            t = np.arange(44100 * 3) / 44100
            for item in job["audio"]:
                soundfile.write(out / f"{item['name']}.flac", (0.1 * np.sin(2 * np.pi * 300 * t)).astype(np.float32),
                                44100)
            return {"gpu": "H100", "usd": 3.2, "hours": 1.4, "log": "ok"}

        s = story()
        scenes = json.loads(s["scenes"])
        scenes[2]["text"] = scenes[2]["text"] + " " + "She listened to the engine and the rain. " * 6
        s["narration"] = "\n\n".join(x["text"] for x in scenes)
        s["scenes"] = json.dumps(scenes)
        summary = vastvideo.produce(s, root / "out", voice.SilentVoice(), None, {"end_seconds": 2.0}, None,
                                    Path("unused"), run_job=fake_job)
        out = root / "out"
        for name in ("video.mp4", "short_1.mp4", "captions.srt", "report.json", "contact_sheet.jpg", "style_frame.png"):
            self.assertTrue((out / name).exists(), name)
        self.assertEqual(summary["clips"], summary["shots"] - 1)
        self.assertEqual(summary["pictures_drawn_in_code"], 0)
        self.assertGreater(summary["sound"]["foley"], 0)
        self.assertEqual(summary["sound"]["synthesized"], [])
        self.assertEqual(summary["usd"], 3.2)
        self.assertNotIn(3, summary["scenes"])  # the ending stays for the full video
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(out / "video.mp4")], capture_output=True, text=True)
        self.assertAlmostEqual(float(probe.stdout), summary["seconds"], delta=0.6)


class WorkerTests(unittest.TestCase):
    @unittest.skipUnless(HAVE, "video libraries not installed")
    def test_the_worker_chalk_check_matches_the_studio_one(self):
        from PIL import Image

        from production.vast_worker import chalk_check

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        chalk_picture(root / "a.png")
        chalk_picture(root / "b.png", colour=True)
        self.assertTrue(chalk_check(Image.open(root / "a.png"))[0])
        self.assertFalse(chalk_check(Image.open(root / "b.png"))[0])


if __name__ == "__main__":
    unittest.main()

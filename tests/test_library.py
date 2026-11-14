"""The sound library: licenses, search, the builder's sources, sync, and the sound director's picks."""

import io
import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

try:
    import numpy as np
    import soundfile
    HAVE = bool(shutil.which("ffmpeg"))
except ImportError:
    HAVE = False

from production.library import Library, Sound, license_of, merge


def tone(path, seconds=2.0, freq=220.0, rate=22050):
    t = np.arange(int(seconds * rate)) / rate
    soundfile.write(path, (0.3 * np.sin(2 * np.pi * freq * t)).astype("float32"), rate)
    return path


def wav_bytes(seconds=2.0, freq=220.0):
    buf = io.BytesIO()
    t = np.arange(int(seconds * 22050)) / 22050
    soundfile.write(buf, (0.3 * np.sin(2 * np.pi * freq * t)).astype("float32"), 22050, format="WAV")
    return buf.getvalue()


def sound(id_, kind, title, **kw):
    return Sound(id=id_, kind=kind, title=title, file=f"{id_}.wav", **kw)


class LicenseTests(unittest.TestCase):
    def test_only_commercially_usable_licenses_pass(self):
        self.assertEqual(license_of("http://creativecommons.org/publicdomain/zero/1.0/"), "cc0")
        self.assertEqual(license_of("CC0"), "cc0")
        self.assertEqual(license_of("Attribution 4.0 International"), "cc-by-4.0")
        self.assertEqual(license_of("creativecommons.org/licenses/by/3.0"), "cc-by-3.0")
        for bad in ("Attribution-NonCommercial 4.0", "creativecommons.org/licenses/by-sa/3.0", "CC BY-ND 4.0",
                    "GPL 3.0", "", "all rights reserved"):
            self.assertEqual(license_of(bad), "", bad)


class SourceRuleTests(unittest.TestCase):
    def test_fma_labels_stored_as_numbers_are_decoded(self):
        from production.library_build import class_labels, fma_reason

        meta = {b"huggingface": json.dumps({"info": {"features": {
            "license": {"_type": "ClassLabel", "names": ["Attribution-NonCommercial 4.0", "CC0 1.0 Universal"]},
            "instrumental": {"_type": "ClassLabel", "names": ["No", "Yes"]},
            "genres": {"feature": {"_type": "ClassLabel", "names": ["Pop", "Ambient", "Drone"]}}}}}).encode()}
        labels = class_labels(meta)
        self.assertEqual(labels["license"][1], "CC0 1.0 Universal")
        ok = {"license": 1, "instrumental": 1, "genres": [1, 2]}
        self.assertEqual(fma_reason(ok, labels, ["ambient"]), "ok")
        self.assertEqual(fma_reason(dict(ok, license=0), labels, ["ambient"]), "license")
        self.assertEqual(fma_reason(dict(ok, instrumental=0), labels, ["ambient"]), "vocals")
        self.assertEqual(fma_reason(dict(ok, genres=[0]), labels, ["ambient"]), "genre")

    def test_off_topic_pages_and_files_are_left_out(self):
        from production.library_build import relevant

        self.assertTrue(relevant("Creepy night ambience loop", "horror"))
        self.assertFalse(relevant("Happy JRPG town theme", "horror"))
        self.assertTrue(relevant("DOOR Wood Creak Open", "sfx_file"))
        self.assertFalse(relevant("ui click 3", "sfx_file"))
        self.assertFalse(relevant("Rabbit eating salad", "sfx_file"))
        self.assertFalse(relevant("Free Crowd Cheering Sounds ambience", "sfx"))
        self.assertFalse(relevant("Cozy Farm SFX door", "sfx_file"))


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.lib = Library([
            sound("door", "sfx", "Old wooden door creak", tags=["door", "creak"]),
            sound("steps", "sfx", "Footsteps on gravel", tags=["footsteps"]),
            sound("phone", "sfx", "Telephone ringing", description="an old phone rings"),
            sound("calm", "music", "Long Note One", moods=["unease"], seconds=300),
            sound("dread1", "music", "Gathering Darkness", moods=["dread"], seconds=200, uses=6),
            sound("dread2", "music", "Lightless Dawn", moods=["dread"], seconds=200, uses=0),
            sound("nc", "music", "Not allowed", moods=["dread"], license="cc-by-nc"),
        ], Path("."), Path(tempfile.mkdtemp()))

    def test_effects_are_found_by_meaning_and_synonyms(self):
        self.assertEqual(self.lib.search("sfx", "a door creaks open slowly")[0][0].id, "door")
        self.assertEqual(self.lib.search("sfx", "the phone rang")[0][0].id, "phone")
        self.assertEqual(self.lib.search("sfx", "slow steps behind her")[0][0].id, "steps")

    def test_music_by_mood_prefers_the_less_used_and_never_a_non_commercial_track(self):
        top = self.lib.search("music", "", "dread")
        self.assertEqual(top[0][0].id, "dread2")
        self.assertNotIn("nc", [s.id for s, _ in top])
        self.assertEqual(self.lib.search("music", "", "unease")[0][0].id, "calm")

    def test_credits_and_merge(self):
        lib = Library([sound("a", "music", "A", credit="\"A\" Kevin MacLeod", license="cc-by-4.0"),
                       sound("b", "sfx", "B")], Path("."), Path("."))
        self.assertEqual(lib.credits(["a", "b", "a"]), ["\"A\" Kevin MacLeod"])
        merged = merge([sound("a", "sfx", "A", uses=2)], [sound("b", "sfx", "B")], used=["a", "a", "b"])
        self.assertEqual({s.id: s.uses for s in merged}, {"a": 4, "b": 1})


OGA_CC0 = """<html><title>Creepy Night Loop | OpenGameArt.org</title>
<meta name="description" content="A slow, eerie ambient loop for horror games.">
<a href="/users/Some%20Author">Some Author</a>
<div class="license-name">CC0</div><a href="http://creativecommons.org/publicdomain/zero/1.0/">CC0</a>
<a href="https://opengameart.org/sites/default/files/creepy_loop.wav">creepy_loop.wav</a></html>"""
OGA_SA = OGA_CC0.replace("publicdomain/zero/1.0", "licenses/by-sa/3.0").replace(">CC0<", ">CC-BY-SA 3.0<")


@unittest.skipUnless(HAVE, "audio libraries or ffmpeg not installed")
class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def build(self, sources, pages):
        from production import library_build

        src = self.root / "sources.json"
        src.write_text(json.dumps({"sources": sources}))

        def http(url):
            if url not in pages:
                raise OSError(f"404 {url}")
            return pages[url]
        return library_build.build(src, self.root / "lib", None, None, None, 1.0, http=http)

    def test_opengameart_keeps_cc0_and_skips_share_alike(self):
        pages = {"https://oga/cc0": OGA_CC0.encode(), "https://oga/sa": OGA_SA.encode(),
                 "https://opengameart.org/sites/default/files/creepy_loop.wav": wav_bytes(40)}
        summary = self.build([{"type": "opengameart", "kind": "music", "pages": ["https://oga/cc0", "https://oga/sa"]}],
                             pages)
        self.assertEqual(summary["added"], 1)
        cat = json.loads((self.root / "lib" / "library.json").read_text())["sounds"]
        self.assertEqual(cat[0]["license"], "cc0")
        self.assertEqual(cat[0]["credit"], "")
        self.assertIn("unease", cat[0]["moods"])  # "eerie"
        self.assertTrue((self.root / "lib" / cat[0]["file"]).exists())
        self.assertEqual(summary["sources"]["opengameart"]["skipped_license"], 1)

    def test_a_hand_picked_pack_of_numbered_files_is_kept(self):
        page = OGA_CC0.replace("Creepy Night Loop", "Horror Hit Soundpack").replace("creepy_loop.wav", "hits.zip")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for k in range(3):
                z.writestr(f"{k + 1}.wav", wav_bytes(2, 300 + 80 * k))
        summary = self.build([{"type": "opengameart", "kind": "sfx", "pages": ["https://oga/hits"],
                               "require": "sfx", "require_file": "sfx_file"}],
                             {"https://oga/hits": page.encode(),
                              "https://opengameart.org/sites/default/files/hits.zip": buf.getvalue()})
        self.assertEqual(summary["by_kind"]["sting"], 3)

    def test_kevin_macleod_tracks_carry_his_credit_line(self):
        pieces = [{"title": "Gathering Darkness", "feel": "Dark, Suspenseful", "genre": "Horror"},
                  {"title": "Happy Bee", "feel": "Bright, Bouncy", "genre": "Pop"}]
        base = "https://inc/mp3/"
        pages = {"https://inc/pieces.json": json.dumps(pieces).encode(),
                 base + "Gathering%20Darkness.mp3": wav_bytes(60)}
        summary = self.build([{"type": "incompetech", "catalogue": "https://inc/pieces.json", "files": base,
                               "feels": ["dark"], "exclude_feels": ["bright"]}], pages)
        self.assertEqual(summary["added"], 1)
        s = json.loads((self.root / "lib" / "library.json").read_text())["sounds"][0]
        self.assertIn("Kevin MacLeod (incompetech.com)", s["credit"])
        self.assertEqual(s["license"], "cc-by-4.0")
        self.assertIn("dread", s["moods"])

    def test_a_pack_of_equal_length_hits_is_not_mistaken_for_duplicates(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for k in range(4):
                z.writestr(f"hits/horror_hit_{k:02d}.wav", wav_bytes(2, 200 + 50 * k))
        summary = self.build([{"type": "zip", "url": "https://pack/h.zip", "license": "cc0", "kind": "sfx"}],
                             {"https://pack/h.zip": buf.getvalue()})
        self.assertEqual(summary["added"], 4)
        self.assertEqual(summary["by_kind"]["sting"], 4)

    def test_a_zip_pack_becomes_effects_and_ambiences(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("Doors/DOOR_Wood_Creak_Open_01.wav", wav_bytes(2))
            z.writestr("Amb/AMB_Forest_Night_Wind_Loop.wav", wav_bytes(12))
            z.writestr("Music/Happy_Tune.wav", wav_bytes(3))
        summary = self.build([{"type": "zip", "url": "https://pack/x.zip", "license": "sonniss", "kind": "sfx",
                               "include": "door|forest"}], {"https://pack/x.zip": buf.getvalue()})
        cat = {s["title"]: s for s in json.loads((self.root / "lib" / "library.json").read_text())["sounds"]}
        self.assertEqual(summary["added"], 2)
        self.assertEqual(cat["DOOR Wood Creak Open"]["kind"], "sfx")
        self.assertEqual(cat["AMB Forest Night Wind Loop"]["kind"], "ambience")

    def test_sync_adds_generated_sounds_and_counts_uses(self):
        from production import library_build
        from production.library import read_catalogue, write_catalogue

        run = self.root / "run"
        (run / "library_new").mkdir(parents=True)
        tone(run / "library_new" / "gen-sfx-1.wav")
        (run / "library_new" / "entries.json").write_text(json.dumps(
            [{"file": "gen-sfx-1.wav", "kind": "sfx", "text": "a bus bell rings", "mood": "", "model": "m"}]))
        (run / "sounds_used.json").write_text(json.dumps(["old"]))
        existing = self.root / "current.json"
        write_catalogue(existing, [sound("old", "sfx", "Old", uses=1)])
        summary = library_build.sync([run], existing, self.root / "sync")
        cat = {s.id: s for s in read_catalogue(self.root / "sync" / "library.json")}
        self.assertEqual(summary["added"], 1)
        self.assertEqual(cat["old"].uses, 2)
        new = [s for s in cat.values() if s.origin == "generated"][0]
        self.assertEqual(new.license, "generated")
        self.assertTrue((self.root / "sync" / new.file).exists())


class MadeMusicTests(unittest.TestCase):
    @unittest.skipUnless(HAVE, "audio libraries or ffmpeg not installed")
    def test_music_the_director_asks_for_is_made_by_ace_step_with_its_words(self):
        from production.paid import Budget, Studio
        from production.score import Score

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        audio = tone(root / "a.wav")
        calls = []

        class Fal:
            def run(self, endpoint, args):
                calls.append((endpoint, args))
                return {"audio": {"url": audio.as_uri()}}

        studio = Studio(Fal(), Budget(5.0, {"fal-ai/ace-step-1.5": 0.0003}), root / "cache",
                        {"models": {"music": "fal-ai/ace-step-1.5"}})
        score = Score(studio, root / "w", 22050)
        score.cue_names = ["dread"]
        score.prompts = {"music_dread": "slow cello drone, instrumental"}
        score.music(90)
        endpoint, args = calls[0]
        self.assertEqual(endpoint, "fal-ai/ace-step-1.5")
        self.assertEqual(args["prompt"], "slow cello drone, instrumental")
        self.assertEqual(args["lyrics"], "[Instrumental]")
        self.assertEqual(args["duration"], 90.0)
        self.assertAlmostEqual(studio.budget.spent, 0.027, places=3)  # 90 s at $0.0003
        self.assertEqual(score.generated[0]["name"], "music_dread")


class PickTests(unittest.TestCase):
    def test_the_planner_picks_from_the_shortlist_and_invented_ids_are_ignored(self):
        from types import SimpleNamespace
        from production.shots import Shot
        from production.stills import sounds as director

        lib = Library([sound("door", "sfx", "Old wooden door creak"), sound("door2", "sfx", "Heavy door slam"),
                       sound("m", "music", "Dark drone", moods=["dread"]),
                       sound("m2", "music", "Other drone", moods=["dread"])], Path("."), Path("."))

        class Planner:
            def generate_json(self, task, system, prompt, schema):
                assert task == "sound_pick" and "door2" in prompt
                return {"picks": [{"need": "fx_001", "id": "door2"}, {"need": "music_dread", "id": "made-up"},
                                  {"need": "music_terror", "id": "", "generate": "frantic horror strings, instrumental"},
                                  {"need": "fx_002", "id": "", "generate": ""}]}

        board = SimpleNamespace(router=SimpleNamespace(one=lambda role: Planner()), story={"title": "T"}, data={
            "scenes": [{"mood": "dread", "location": ""}],
            "beats": [Shot(index=1, scene=0, t0=0, t1=5, text="It opened.", sound="a door creaks open"),
                      Shot(index=2, scene=0, t0=5, t1=9, text="Quiet.", sound="a clock ticks")]})
        d = director.pick(board, lib, ["dread", "terror"], 10)
        self.assertEqual(d.chosen["fx_001"].id, "door2")  # the director's choice, not the top text match
        self.assertIn(d.chosen["music_dread"].id, ("m", "m2"))  # invented id ignored: best match instead
        self.assertEqual(d.make["music_terror"], "frantic horror strings, instrumental")  # made by ACE-Step
        self.assertIn("fx_002", d.skip)  # left out on purpose
        self.assertEqual(d.report["decided_by_model"], 3)
        # music never made (generate_kinds ["sfx"]): the director's "make" is dropped, the closest track used
        d2 = director.pick(board, lib, ["dread", "terror"], 10, music_made=False)
        self.assertNotIn("music_terror", d2.make)
        self.assertEqual(d2.make, {})
        self.assertIn(d2.chosen["music_dread"].id, ("m", "m2"))  # music comes from the library

    def test_only_the_allowed_kinds_of_sound_are_paid_for(self):
        import numpy as np
        from production.paid import Budget, Studio
        from production.score import Score

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        asked = []

        class Fal:
            def run(self, endpoint, args):
                asked.append(endpoint)
                raise AssertionError("nothing should be generated")

        studio = Studio(Fal(), Budget(1.0, {}), root / "cache", {"models": {"music": "fal-ai/ace-step-1.5"}})
        score = Score(studio, root / "w", 24000)
        score.make_kinds = {"sfx"}  # effects may be made, music never
        self.assertIsNone(score._gen("music", {"prompt": "drone"}, "music_dread"))
        self.assertEqual(asked, [])
        self.assertIsInstance(np.zeros(1), np.ndarray)


@unittest.skipUnless(HAVE, "audio libraries or ffmpeg not installed")
class DirectorTests(unittest.TestCase):
    def test_the_sound_director_uses_the_library_and_credits_it(self):
        from production.library import write_catalogue
        from tests.test_stills import FakeFal, cfg, story
        from production import voice
        from production.stills import pipeline

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        lib = root / "lib"
        lib.mkdir()
        sounds = []
        for k, mood in enumerate(("unease", "dread", "terror")):
            tone(lib / f"m{k}.wav", 60, 110 + 40 * k)
            sounds.append(Sound(id=f"m{k}", kind="music", title=f"Track {mood}", file=f"m{k}.wav", moods=[mood],
                                seconds=60, license="cc-by-4.0", credit=f"\"Track {mood}\" Kevin MacLeod (incompetech.com)"))
        tone(lib / "st.wav", 2, 600)
        sounds.append(Sound(id="st", kind="sting", title="Orchestral horror stinger hit", file="st.wav", seconds=2))
        tone(lib / "bell.wav", 2, 900)
        sounds.append(Sound(id="bell", kind="sfx", title="Bus bell ring", file="bell.wav", tags=["bell"], seconds=2))
        write_catalogue(lib / "library.json", sounds)
        c = cfg(sound_library={"enabled": True, "local_dir": str(lib), "min_score": 0.5, "generate_missing": True,
                               "save_generated": True})
        made = FakeFal(root)
        summary = pipeline.produce(story(False), root / "out", voice.SilentVoice(), None, c, made)
        endpoints = [e for e, _ in made.calls]
        self.assertNotIn("fal-ai/ace-step-1.5", endpoints)  # every mood came from the library
        report = summary["agents"]["sound"]["library"]
        self.assertTrue(report["library"])
        self.assertEqual(report["from_library_by_kind"]["music"], 3)
        self.assertEqual(report["from_library_by_kind"]["sting"], 1)
        meta = json.loads((root / "out" / "metadata.json").read_text())
        self.assertIn("Kevin MacLeod (incompetech.com)", meta["description"])
        self.assertIn("m0", json.loads((root / "out" / "sounds_used.json").read_text()))
        if summary["agents"]["sound"]["generated"]:  # room tones the library lacked were paid for once
            self.assertTrue((root / "out" / "library_new" / "entries.json").exists())


if __name__ == "__main__":
    unittest.main()

"""Picture quality: the camera's moves and the film finish, the colorist, the set designer, the painter's rules."""

import base64
import io
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import numpy as np
    from PIL import Image
    HAVE = bool(shutil.which("ffmpeg"))
except ImportError:
    HAVE = False

from production.shots import Shot


def beat(index=0, **over):
    return Shot(index=index, scene=over.pop("scene", 0), t0=index, t1=index + 1, text=over.pop("text", "x"), **over)


def flat(path, colour=(90, 90, 90), size=(640, 360)):
    Image.new("RGB", size, colour).save(path)
    return path


def frame(video, at=0.0):
    """One frame of a video as an HD player shows it (BT.709), as an RGB array."""
    raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{at:.3f}", "-i", str(video),
                          "-frames:v", "1", "-vf", "scale=in_color_matrix=bt709:in_range=limited:out_range=full,"
                          "format=rgb24", "-f", "rawvideo", "-"], capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(1080, 1920, 3).astype(np.float32)


@unittest.skipUnless(HAVE, "video libraries or ffmpeg not installed")
class Camera(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_each_framing_gets_the_move_that_suits_it(self):
        from production.stills.camera import move_for

        self.assertEqual(move_for(beat(framing="wide"), 0), "pan_right")
        self.assertEqual(move_for(beat(framing="wide"), 1), "pan_left")  # the next wide goes the other way
        self.assertEqual(move_for(beat(framing="close"), 0), "push_in")
        self.assertEqual(move_for(beat(framing="close", scare=True), 0), "push_in_fast")
        self.assertEqual(move_for(beat(framing="wide", scare=True, view="cctv"), 0), "none")  # bolted to a wall
        self.assertIn(move_for(beat(framing="medium"), 1), ("push_in", "pull_out"))

    def test_a_longer_shot_travels_further_at_the_same_gentle_speed(self):
        from production.stills.camera import path

        short, long = path("push_in", 4.0), path("push_in", 7.0)
        self.assertEqual(short[0], 1.0)
        self.assertLess(short[1], long[1])
        self.assertLessEqual(path("push_in", 30.0)[1], 1.08)  # never a zoom you notice
        self.assertGreater(path("push_in_fast", 4.0)[1], path("push_in", 4.0)[1] + 0.05)  # a scare lunges
        self.assertEqual(path("push_in_fast", 4.0)[4], "settle")
        self.assertEqual(path("pan_left", 6.0)[2:4], (1.0, 0.0))
        self.assertEqual(path("none", 6.0)[:2], (1.0, 1.0))

    def test_the_canvas_is_the_frame_times_the_zoom_with_fine_grain(self):
        from production.stills.camera import canvas

        src = flat(self.root / "grey.png", size=(2560, 1440))
        self.assertEqual(canvas(src, self.root / "clean.png", 1.06, 0.0, 1), (2036, 1146))
        clean = np.asarray(Image.open(self.root / "clean.png").convert("L"), dtype=np.float32)
        self.assertLess(clean.std(), 0.01)
        self.assertEqual(canvas(src, self.root / "film.png", 1.0, 0.02, 1), (1920, 1080))
        film = np.asarray(Image.open(self.root / "film.png").convert("L"), dtype=np.float32)
        self.assertGreater(film.std(), 2.0)  # grain is there
        self.assertLess(film.std(), 6.0)  # and fine: texture, not noise
        self.assertAlmostEqual(film.mean(), 90, delta=1.0)  # without changing the picture's brightness

    def test_grain_leaves_pure_black_clean(self):
        from production.stills.camera import grain_on

        black = grain_on(np.zeros((90, 160, 3), dtype=np.float32), 0.03, 1)
        grey = grain_on(np.full((90, 160, 3), 0.3, dtype=np.float32), 0.03, 1)
        self.assertLess(black.std() * 3, grey.std())

    def test_a_shot_moves_steadily_along_its_path(self):
        from production.edit import FPS, Segment
        from production.stills.camera import path
        from production.stills.post import render

        # a dark picture with one soft bright line at 30 % of its width
        x = np.arange(2560) - 0.30 * 2560
        im = 30 + 200 * np.exp(-0.5 * (x / 3.0) ** 2)[None, :] * np.ones((1440, 1))
        Image.fromarray(np.clip(im, 0, 255).astype("uint8")).convert("RGB").save(self.root / "line.png")
        seg = Segment("seg_push.mp4", 0.0, 4.0, self.root / "line.png", "push_in")
        render(seg, self.root, {"grain": 0.0, "vignette": 0.0}, 1)
        video = self.root / "seg_push.mp4"
        info = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                               "stream=width,height,nb_read_frames,color_space", "-of", "csv=p=0", str(video)],
                              capture_output=True, text=True).stdout.strip()
        self.assertEqual(info, f"1920,1080,bt709,{4 * FPS}")
        raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video), "-f", "rawvideo",
                              "-pix_fmt", "gray", "-"], capture_output=True).stdout
        frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 1080, 1920)
        z0, z1 = path("push_in", 4.0)[:2]
        off = []
        for k, f in enumerate(frames):  # the line's place in each frame, to a fraction of a pixel
            ideal = (576 - 960) * (z0 + (z1 - z0) * k / (len(frames) - 1)) + 960
            prof = f[300:780, 520:600].astype(np.float64).mean(0)
            prof = np.clip(prof - np.percentile(prof, 20), 0, None)
            off.append(520 + (prof * np.arange(prof.size)).sum() / prof.sum() - ideal)
        off = np.array(off)
        self.assertLess(np.abs(off).max(), 0.5)  # on the planned path from the first frame to the last
        # and steady on it: a whole-pixel zoom (FFmpeg's zoompan) steps by 0.2-0.4 px from frame to frame
        self.assertLess(np.abs(np.diff(off)).max(), 0.1)

    def test_the_channels_red_stays_that_red(self):
        from production.edit import Segment
        from production.stills.post import render

        src = flat(self.root / "red.png", (150, 12, 12), (2560, 1440))
        render(Segment("seg_red.mp4", 0.0, 1.0, src, "none"), self.root, {"grain": 0.0, "vignette": 0.0}, 1)
        shown = frame(self.root / "seg_red.mp4", 0.5)[500:580, 900:1000].mean((0, 1))
        # with FFmpeg's default (standard-definition) matrix an HD player shows (160, 25, 8): orange-red
        self.assertLess(np.abs(shown - np.array([150, 12, 12])).max(), 5)

    def test_the_vignette_is_fixed_to_the_frame(self):
        from production.edit import Segment
        from production.stills.post import render

        src = flat(self.root / "grey.png", (120, 120, 120), (2560, 1440))
        for move in ("none", "pan_left"):
            seg = Segment(f"seg_{move}.mp4", 0.0, 1.0, src, move)
            render(seg, self.root, {"grain": 0.0, "vignette": 0.3}, 1)
            for at in (0.0, 0.9):
                v = frame(self.root / seg.name, at).mean(2)
                centre = v[500:580, 920:1000].mean()
                left, right = v[:60, :60].mean(), v[:60, -60:].mean()
                self.assertLess(left, centre * 0.8)
                self.assertAlmostEqual(left, right, delta=2.0)  # the same in both corners, wherever the pan is


@unittest.skipUnless(HAVE, "video libraries or ffmpeg not installed")
class Atmosphere(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_textures_wrap_so_they_can_scroll_for_ever(self):
        from production.stills.camera import fog_texture, rain_texture

        size = (320, 180)
        rain = np.asarray(Image.open(rain_texture(self.root / "rain.png", size)))
        self.assertEqual(rain.shape, (360, 640, 4))  # a 2x2 tile
        np.testing.assert_array_equal(rain[:180, :320], rain[180:, 320:])
        self.assertGreater(rain[..., 3].max(), 60)  # streaks
        self.assertLess(rain[..., 3].mean(), 30)  # mostly clear air
        fog = np.asarray(Image.open(fog_texture(self.root / "fog.png", size)), dtype=np.float32)
        self.assertEqual(fog.shape, (180, 640, 4))  # two tiles wide
        seam = np.abs(fog[:, 319, 3] - fog[:, 320, 3]).mean()  # where one tile meets the next
        inside = np.abs(fog[:, 159, 3] - fog[:, 160, 3]).mean()
        self.assertLess(seam, inside + 3)
        self.assertGreater(fog[-30:, :, 3].mean(), fog[:30, :, 3].mean())  # the mist lies low

    def test_a_shot_with_rain_fog_and_flicker_renders(self):
        from production.edit import FPS, Segment
        from production.stills.camera import fog_texture, rain_texture
        from production.stills.post import render

        size = (640, 360)
        textures = {"rain": rain_texture(self.root / "rain.png", size), "fog": fog_texture(self.root / "fog.png", size)}
        src = flat(self.root / "dark.png", (25, 25, 30), (704, 396))
        plain = Segment("plain.mp4", 0.0, 2.0, src, "push_in")
        stormy = Segment("stormy.mp4", 0.0, 2.0, src, "push_in", atmosphere=("rain", "fog", "flicker"))
        for seg in (plain, stormy):
            render(seg, self.root, {"grain": 0.0, "vignette": 0.0}, 1, size, textures)

        def frames(name):
            raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(self.root / name), "-f",
                                  "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True).stdout
            return np.frombuffer(raw, dtype=np.uint8).reshape(-1, 360, 640).astype(np.float32)

        a, b = frames("plain.mp4"), frames("stormy.mp4")
        self.assertEqual((len(a), len(b)), (2 * FPS, 2 * FPS))
        self.assertLess(np.abs(np.diff(a, axis=0)).mean(), 0.5)  # a still under a slow push barely changes
        self.assertGreater(np.abs(np.diff(b, axis=0)).mean(), 1.0)  # rain falls, the light stutters
        self.assertGreater(b[:, 300:].mean(), a[:, 300:].mean())  # mist lightens the bottom of the frame

    def test_rain_is_never_put_over_an_interior(self):
        from production.shots import atmosphere_in

        self.assertEqual(atmosphere_in("Rain lashes the empty road; fog rolls over the fields."), ["rain", "fog"])
        self.assertEqual(atmosphere_in("Inside the bus, rain streaks the windows."), [])
        self.assertEqual(atmosphere_in("A candle gutters on the table."), ["flicker"])
        self.assertEqual(atmosphere_in("Mara on the bedroom floor, fog at the window."), [])


@unittest.skipUnless(HAVE, "video libraries or ffmpeg not installed")
class Colorist(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.rng = np.random.default_rng(3)

    def picture(self, name, mean, cast=(0, 0, 0)):
        base = self.rng.random((180, 320, 1)).astype(np.float32) ** 2.2
        base = base / base.mean() * (mean / 255.0)
        rgb = np.clip(np.broadcast_to(base, (180, 320, 3)) + np.array(cast, dtype=np.float32) / 255.0, 0, 1)
        Image.fromarray((rgb * 255).astype("uint8")).save(self.root / name)
        return self.root / name

    def board(self, pictures, views=None):
        from production.stills.board import Board

        beats = [beat(i, view=(views or {}).get(i, "film")) for i in range(len(pictures))]
        b = Board(story={}, out=self.root, work=self.root, cfg={})
        b.data.update(beats=beats, pictures=dict(enumerate(pictures)))
        return b

    def test_only_the_picture_that_jumps_at_the_cut_is_graded(self):
        from production.stills.grade import ColoristAgent, measure

        pictures = [self.picture(f"p{k}.png", 38 + k) for k in range(4)] + [self.picture("bright.png", 125)]
        b = self.board(pictures)
        report = ColoristAgent().run(b)
        self.assertEqual(report["graded"], 1)
        self.assertEqual([b.data["pictures"][k] for k in range(4)], pictures[:4])  # untouched, not even re-saved
        after = measure(b.data["pictures"][4])[0]
        self.assertNotEqual(b.data["pictures"][4], pictures[4])
        self.assertLess(after, 85)
        self.assertGreater(after, report["film_brightness"])  # still the brighter shot, no longer a flash
        self.assertLess(report["changes"][4]["brightness"], -30)

    def test_gamma_keeps_black_black_and_white_white(self):
        from production.stills.grade import grade

        rgb = np.full((180, 320, 3), 0.6, dtype=np.float32)
        rgb[:20], rgb[-20:] = 0.0, 1.0
        out, changed = grade(rgb, (40.0, 30.0, 128.0, 128.0), 0.75)
        self.assertIn("brightness", changed)
        self.assertLess(out[90, 160, 0], 0.5)
        self.assertLess(out[:20].max(), 0.01)
        self.assertGreater(out[-20:].min(), 0.99)

    def test_a_colour_cast_is_pulled_back_but_a_camera_view_keeps_its_own(self):
        from production.stills.grade import ColoristAgent, measure

        pictures = [self.picture(f"p{k}.png", 40) for k in range(4)] + [self.picture("blue.png", 40, (0, 0, 40))]
        b = self.board(pictures)
        ColoristAgent().run(b)
        self.assertLess(measure(b.data["pictures"][4])[2], measure(pictures[4])[2] - 4)  # less blue
        b = self.board(pictures, views={4: "cctv"})
        report = ColoristAgent().run(b)
        self.assertNotIn("colour", report["changes"].get(4, {}))

    def test_a_film_lit_like_day_is_printed_down_as_a_whole(self):
        from production.stills.grade import ColoristAgent, measure

        pictures = [self.picture(f"p{k}.png", 55 + k) for k in range(5)]
        b = self.board(pictures)
        b.data["look"] = {"max_brightness": 40}
        report = ColoristAgent().run(b)
        self.assertEqual(report["graded"], 5)
        self.assertGreater(report["printed_down"], 0.2)
        for k in range(5):
            self.assertLess(measure(b.data["pictures"][k])[0], 48)
        b = self.board(pictures)  # without a ceiling a uniformly bright film is left as it is
        self.assertEqual(ColoristAgent().run(b)["graded"], 0)

    def test_a_flat_picture_gains_contrast_and_a_harsh_one_is_left_alone(self):
        from production.stills.grade import grade

        want = (40.0, 30.0, 128.0, 128.0)
        flat_picture = np.clip(0.16 + self.rng.normal(0, 0.02, (180, 320, 3)), 0, 1).astype(np.float32)
        harsh = (self.rng.random((180, 320, 3)) > 0.84).astype(np.float32)
        self.assertGreater(grade(flat_picture, want, 0.75)[1].get("contrast", 1.0), 1.1)
        self.assertNotIn("contrast", grade(harsh, want, 0.75)[1])


class FakePlanner:
    def __init__(self, answer):
        self.answer, self.prompts = answer, []

    def one(self, role):
        return self

    def generate_json(self, task, system, prompt, schema):
        self.prompts.append((task, prompt))
        return self.answer


class SetDesign(unittest.TestCase):
    def beats(self):
        return [beat(0, scene=0, picture="The Storyteller by his lantern."),
                beat(1, scene=1, picture="Tomas drives the night bus down an empty road."),
                beat(2, scene=1, picture="The bus's bell light glows red."),
                beat(3, scene=2, picture="The bus stands under a dead streetlight, doors open."),
                beat(4, scene=3, picture="A kitchen at dawn.")]

    def test_the_designer_groups_scenes_into_places(self):
        from production.stills.art import places

        design = ("Outside: a 1990s single-decker city bus in cream with a dark green band, route number blind "
                  "unlit. Inside: brown vinyl seats, yellow grab rails, two flickering strip lights.")
        router = FakePlanner({"places": [
            {"name": "the kitchen", "scenes": [3], "design": "A small kitchen with a round table and a humming fridge, "
             "lit by one grey window.", "views": ["the whole room from the door"]},
            {"name": "the night bus", "scenes": [1, 2, "x", 9], "design": design,
             "views": ["outside: the whole bus from the front left", "inside: down the aisle", "a third view"]},
            {"name": "", "scenes": [1], "design": design, "views": []},
            {"name": "too thin", "scenes": [3], "design": "a room", "views": []}]})
        scenes = [{"location": "the hollow"}, {"location": "Route 14"}, {"location": "the last stop"},
                  {"location": "home"}]
        got = places(router, scenes, self.beats(), {0}, "HOUSE")
        self.assertEqual([p["name"] for p in got], ["the night bus", "the kitchen"])  # most pictures first
        self.assertEqual(got[0]["scenes"], [1, 2])  # unknown scenes dropped
        self.assertEqual(len(got[0]["views"]), 2)
        task, prompt = router.prompts[0]
        self.assertEqual(task, "set_design")
        self.assertIn("SCENE 1 (2 pictures) - location: Route 14", prompt)
        self.assertNotIn("SCENE 0", prompt)  # the host's own place is not a set
        self.assertEqual(places(None, scenes, self.beats(), {0}, "HOUSE"), [])

    def test_a_beat_in_a_place_is_painted_from_its_sheet_and_design(self):
        from production.stills.art import IllustratorAgent

        b = beat(2, scene=1, framing="wide", picture="The bus stands under a dead streetlight.")
        place = {"name": "the night bus", "design": "Outside: a cream city bus with a green band.", "sheet": Path("x")}
        text = IllustratorAgent().prompt(b, "HOUSE.", {}, [], True, {}, "", place)
        self.assertIn("Reference image 1 is the location sheet of the night bus", text)
        self.assertIn("Place: the night bus: Outside: a cream city bus with a green band.", text)
        self.assertNotIn("look only", text)
        no_sheet = IllustratorAgent().prompt(b, "HOUSE.", {}, [], True, {}, "", dict(place, sheet=None))
        self.assertIn("look only", no_sheet)  # the style frame gives the look; the design still names the place
        self.assertIn("Place: the night bus", no_sheet)
        self.assertNotIn("Place:", IllustratorAgent().prompt(b, "HOUSE.", {}, [], True, {}, ""))

    def test_the_shot_leads_the_prompt_and_the_look_follows(self):
        from production.stills.art import IllustratorAgent

        b = beat(2, scene=1, framing="close", characters=["Tomas"], picture="Tomas stares into the mirror.")
        place = {"name": "the night bus", "design": "Outside: a cream city bus.", "sheet": Path("x")}
        text = IllustratorAgent().prompt(b, "A dark film frame.", {"Tomas": "58, grey stubble"}, ["Tomas"], True,
                                         {"noun": "film still", "avoid": "No plastic skin."}, "", place)
        order = [text.index(part) for part in ("One single 16:9 film still", "Tomas stares into the mirror.",
                                               "Place: the night bus", "Characters: Tomas: 58", "Reference image 1",
                                               "Reference image 2 is the character sheet of Tomas",
                                               "The look: A dark film frame.", "No plastic skin.")]
        self.assertEqual(order, sorted(order))
        self.assertLess(text.index("Tomas stares into the mirror."), 150)  # read first, followed best

    def test_a_view_through_another_camera_keeps_that_cameras_look(self):
        from production.stills.art import IllustratorAgent

        b = beat(6, scene=1, view="cctv", picture="The driver's seat from above.")
        place = {"name": "the night bus", "design": "Outside: a cream city bus.", "sheet": Path("x")}
        text = IllustratorAgent().prompt(b, "HOUSE.", {}, [], True, {}, "Seen on a security camera", place)
        self.assertIn("location sheet of the night bus", text)
        self.assertNotIn("has its look", text)  # the place, not the film's colours
        plain = IllustratorAgent().prompt(b, "HOUSE.", {}, [], True, {}, "Seen on a security camera")
        self.assertIn("keep that camera's look", plain)
        self.assertNotIn("look only", plain)


@unittest.skipUnless(HAVE, "video libraries or ffmpeg not installed")
class PainterRules(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def painter(self, verdicts, steps):
        from production.stills.images import Painter
        from tests.test_stills import painted

        asked, looks = [], []

        class Critic:
            def look(self, image, picture):
                looks.append(picture)
                return verdicts[min(len(looks), len(verdicts)) - 1]

        class P(Painter):
            def available(self, step):
                return step in steps

            def _one(self, step, prompt, refs, raw, key, retry=0):
                asked.append((step, prompt))
                retries.append(retry)
                return painted(raw, seed=len(asked))


        retries = self.retries = []
        return P(None, {}, self.root, Critic()), asked

    def test_the_next_painter_is_told_what_was_wrong(self):
        p, asked = self.painter([(False, "she is leaning, the line says standing"), (True, "fine")],
                                ("fal:shot", "fal:shot_alt"))
        out, step = p.paint("She stands behind him.", [], self.root / "beat.png", "k",
                            ["fal:shot", "fal:shot_alt", "pollinations"], check="standing")
        self.assertEqual(step, "fal:shot_alt")
        self.assertEqual([s for s, _ in asked], ["fal:shot", "fal:shot_alt"])
        self.assertNotIn("Correct this", asked[0][1])
        self.assertTrue(asked[1][1].startswith("Correct this from the previous attempt: she is leaning, the line "
                                               "says standing. She stands behind him."))  # the fault leads
        self.assertEqual(p.counts, {"fal:shot_alt (corrected)": 1})
        self.assertEqual(p.sizes, {"fal:shot": "640x360", "fal:shot_alt": "640x360"})

    def test_a_refused_paid_picture_is_never_replaced_by_a_free_painter(self):
        p, asked = self.painter([(False, "her face shows")], ("fal:shot", "fal:shot_alt", "pollinations"))
        out, step = p.paint("She stands behind him.", [], self.root / "beat.png", "k",
                            ["fal:shot", "fal:shot_alt", "pollinations"], check="standing")
        self.assertEqual([s for s, _ in asked], ["fal:shot", "fal:shot_alt"])  # pollinations was not asked
        self.assertEqual(step, "fal:shot")  # both faults were details: the strongest painter's picture stays
        self.assertEqual(p.counts, {"fal:shot (doubted)": 1})
        self.assertEqual(p.doubted, {"beat"})
        self.assertTrue(out.exists())

    def test_a_correction_that_broke_something_worse_does_not_win(self):
        # run 96: the reveal was refused for the way she looked; its correction came back as a split screen
        p, asked = self.painter([(False, "she looks left, not right"), (False, "a split screen with two panels")],
                                ("fal:hero", "fal:shot"))
        out, step = p.paint("The reveal.", [], self.root / "beat.png", "k", ["fal:hero", "fal:shot"], check="x")
        self.assertEqual(step, "fal:hero")
        self.assertTrue(out.exists())
        self.assertEqual(p.counts, {"fal:hero (doubted)": 1})
        # with serious faults only, the last (corrected) one is kept as before
        p, _ = self.painter([(False, "it shows written text"), (False, "a collage of panels")], ("fal:shot", "fal:shot_alt"))
        self.assertEqual(p.paint("x", [], self.root / "b2.png", "k2", ["fal:shot", "fal:shot_alt"], check="x")[1],
                         "fal:shot_alt")

    def test_free_painters_still_paint_when_no_paid_picture_could_be_made(self):
        p, asked = self.painter([(True, "fine")], ("pollinations",))
        out, step = p.paint("A bus.", [], self.root / "beat.png", "k", ["fal:shot", "pollinations"], check="a bus")
        self.assertEqual(step, "pollinations")

    def test_an_ordinary_beat_keeps_a_picture_refused_only_for_a_detail(self):
        from production.stills.images import MINOR

        p, asked = self.painter([(False, MINOR + "the monitor's screen should be dark")], ("fal:shot",))
        out, step = p.paint("A mother on the floor.", [], self.root / "beat.png", "k", ["fal:shot", "fal:shot"],
                            check="x", accept_minor=True)
        self.assertEqual(len(asked), 1)  # no money spent on the detail
        self.assertEqual(p.counts, {"fal:shot (minor fault kept)": 1})
        # a serious fault on an ordinary beat is still corrected
        p, asked = self.painter([(False, "a split screen of two panels"), (True, "fine")], ("fal:shot",))
        p.paint("x", [], self.root / "b2.png", "k2", ["fal:shot", "fal:shot"], check="x", accept_minor=True)
        self.assertEqual(len(asked), 2)
        self.assertIn("Correct this from the previous attempt: a split screen", asked[1][1])
        # a key beat corrects details too, and the prompt does not carry the severity label
        p, asked = self.painter([(False, MINOR + "she holds it in the wrong hand"), (True, "fine")],
                                ("fal:shot", "fal:shot_alt"))
        p.paint("x", [], self.root / "b3.png", "k3", ["fal:shot", "fal:shot_alt"], check="x")
        self.assertEqual(len(asked), 2)
        self.assertTrue(asked[1][1].startswith("Correct this from the previous attempt: she holds it"))

    def test_the_critic_says_whether_a_fault_is_serious(self):
        import json

        from production.paid import Budget
        from production.stills.images import MINOR, Critic, serious

        answers = [{"ok": False, "serious": False, "reason": "the duvet is blue"},
                   {"ok": False, "serious": True, "reason": "her face is calm"},
                   {"ok": False, "reason": "it is split into two panels"}]

        class Vision:
            def run(self, model, args):
                return {"output": json.dumps(answers.pop(0))}

        studio = SimpleNamespace(models={"vision": "v"}, args={}, fal=Vision(), budget=Budget(1.0, {"v": 0.002}))
        image = self.root / "p.png"
        Image.new("RGB", (64, 36), (20, 20, 20)).save(image)
        critic = Critic(studio, "dark")
        detail, face, panels = (critic.look(image, "x")[1] for _ in range(3))
        self.assertEqual(detail, MINOR + "the duvet is blue")
        self.assertFalse(serious(detail))
        self.assertTrue(serious(face) or not face.startswith(MINOR))  # the critic called it serious
        self.assertFalse(face.startswith(MINOR))
        self.assertTrue(serious(panels))  # no word from the critic: judged from the reason

    def test_a_sheet_that_fails_its_own_check_is_painted_again_by_the_same_painter(self):
        p, asked = self.painter([], ("fal:sheet",))
        seen = []

        def check(image):
            seen.append(image)
            return (len(seen) > 1, "her face shows; the hood must hide it")

        out, step = p.paint("A character sheet.", [], self.root / "sheet.png", "k", ["fal:sheet", "fal:sheet"],
                            check=check)
        self.assertEqual([s for s, _ in asked], ["fal:sheet", "fal:sheet"])
        self.assertIn("Correct this from the previous attempt: her face shows", asked[1][1])
        self.assertEqual(self.retries, [0, 1])  # the second try is its own paid picture, not the first one again
        self.assertEqual(p.counts, {"fal:sheet (corrected)": 1})

    def test_references_are_sent_as_small_jpegs(self):
        from production.stills.images import Critic, Painter

        big = self.root / "sheet.png"
        rng = np.random.default_rng(0)
        Image.fromarray(rng.integers(0, 60, (1440, 2560, 3), dtype=np.uint8)).save(big)
        p = Painter(None, {}, self.root / "art", Critic(None, "dark"))
        uri = p.reference(big)
        self.assertTrue(uri.startswith("data:image/jpeg;base64,"))
        data = base64.b64decode(uri.split(",", 1)[1])
        with Image.open(io.BytesIO(data)) as im:
            self.assertEqual(im.size, (2048, 1152))
        self.assertLess(len(data), big.stat().st_size / 3)
        self.assertIs(p.reference(big), uri)  # made once


class ReferenceCheck(unittest.TestCase):
    def test_a_sheet_is_held_against_its_design(self):
        import json

        from production.paid import Budget
        from production.stills.checks import sheet_ok

        class Vision:
            prompts = []

            def run(self, model, args):
                self.prompts.append(args["prompt"])
                return {"output": json.dumps({"ok": False, "reason": "her face shows"})}

        studio = SimpleNamespace(models={"vision": "v"}, args={}, fal=Vision(), budget=Budget(1.0, {"v": 0.002}))
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "sheet.png"
            Image.new("RGB", (320, 180), (20, 20, 20)).save(image)
            ok, why = sheet_ok(studio, image, "character", "The Woman in Yellow", "hood always up, face hidden")
            self.assertEqual((ok, why), (False, "her face shows"))
            self.assertIn("character reference sheet of The Woman in Yellow: hood always up", Vision.prompts[0])
            self.assertIn("a face that the design keeps hidden is shown", Vision.prompts[0])
            sheet_ok(studio, image, "location", "the night bus", "a cream city bus")
            self.assertIn("a different kind of vehicle", Vision.prompts[1])
            self.assertEqual(sheet_ok(None, image, "location", "the night bus", "x"), (True, "not checked"))


@unittest.skipUnless(HAVE, "video libraries or ffmpeg not installed")
class Clips(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def clip(self, name, source):
        path = self.root / name
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", source,
                        "-pix_fmt", "yuv420p", str(path)], check=True)
        return path

    def test_a_short_clip_is_slowed_then_held_to_fill_its_beat(self):
        from production.edit import FPS, Segment
        from production.stills.post import render

        clip = self.clip("c.mp4", "testsrc=size=640x360:rate=24:duration=4,eq=brightness=-0.3")
        seg = Segment("seg_clip.mp4", 0.0, 6.0, None, clip=clip)
        render(seg, self.root, {"grain": 0.018, "vignette": 0.22}, 1, (640, 360))
        info = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                               "stream=width,height,nb_read_frames,color_space", "-of", "csv=p=0",
                               str(self.root / "seg_clip.mp4")], capture_output=True, text=True).stdout.strip()
        self.assertEqual(info, f"640,360,bt709,{6 * FPS}")

    def test_a_frozen_or_bright_clip_is_refused(self):
        from production.stills.motion import frames, length, moves

        frozen = self.clip("f.mp4", "color=c=0x202020:size=320x180:rate=24:duration=3")
        alive = self.clip("a.mp4", "testsrc=size=320x180:rate=24:duration=3,eq=brightness=-0.3")
        self.assertFalse(moves(frames(frozen, [0.1, 0.9], self.root / "f")))
        self.assertTrue(moves(frames(alive, [0.1, 0.9], self.root / "a")))
        self.assertEqual(length(5.8, [5, 10]), 5)  # 5 s slowed 1.25x covers 5.8 s
        self.assertEqual(length(7.0, [5, 10]), 10)

    def test_letterbox_bars_frame_a_cinema_screen(self):
        from production.stills.camera import letterbox_filter

        self.assertEqual(letterbox_filter(0, (2560, 1440)), "")
        self.assertIn("h=184", letterbox_filter(2.39, (2560, 1440)))

    def test_halation_warms_the_dark_around_a_light_and_leaves_the_dark_alone(self):
        from production.stills.camera import halation

        rgb = np.zeros((90, 160, 3), dtype=np.float32)
        rgb[40:50, 75:85] = 1.0
        out = halation(rgb, 0.5)
        self.assertGreater(out[45, 86, 0], out[45, 86, 2])  # a warm haze just beside the light
        self.assertEqual(float(out[5, 5].max()), 0.0)  # far from it, black stays black


class FirstTry(unittest.TestCase):
    def test_a_person_with_a_sheet_needs_only_a_few_words(self):
        from production.stills.art import IllustratorAgent, short

        design = ("a woman in her early thirties, exhausted, dark brown hair tied back loosely; white skin, an oval "
                  "face with a small scar on the chin, a grey cardigan over dark pyjamas, never wears glasses")
        b = beat(4, scene=1, framing="close", characters=["Mara"], picture="Mara stares at the monitor, terrified.")
        place = {"name": "Mara's bedroom", "design": "A master bedroom with floral wallpaper and a dark wooden "
                 "bed. The only light is the monitor's red glow. Heavy curtains.", "sheet": Path("x")}
        with_sheet = IllustratorAgent().prompt(b, "LOOK.", {"Mara": design}, ["Mara"], True, {}, "", place,
                                               "one single frame, never split into two or more panels")
        without = IllustratorAgent().prompt(b, "LOOK.", {"Mara": design}, [], True, {}, "", dict(place, sheet=None))
        self.assertIn("Mara: a woman in her early thirties, exhausted, dark brown hair tied back loosely", with_sheet)
        self.assertNotIn("scar on the chin", with_sheet)  # the sheet shows it
        self.assertIn("scar on the chin", without)  # no sheet: the words are all there is
        self.assertNotIn("Heavy curtains", with_sheet)
        self.assertTrue(with_sheet.startswith("One single 16:9 frame, one frame, never split into panels."))
        self.assertIn("Avoid what went wrong before: one single frame", with_sheet)
        self.assertEqual(short("a tall man in a long coat", 3), "a tall man...")

    def test_refusals_are_counted_across_videos_and_warn_the_next(self):
        import sqlite3

        from production.stills import lessons

        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        self.assertEqual(lessons.load(conn), "")
        lessons.record(conn, ["The image is a diptych, split into two frames", "too bright for night",
                              "Mara is wearing blue jeans instead of pyjamas"])
        self.assertEqual(lessons.load(conn), "")  # once is chance
        counted = lessons.record(conn, ["a split screen with two panels", "It shows daylight through the window"])
        self.assertEqual(counted, {"split": 1, "daylight": 1})
        warned = lessons.load(conn)
        self.assertIn("never split", warned)
        self.assertIn("no daylight", warned)
        self.assertNotIn("clothes", warned)
        self.assertEqual(lessons.load(None), "")


@unittest.skipUnless(HAVE, "video libraries or ffmpeg not installed")
class RetryAllowance(unittest.TestCase):
    def test_retries_stop_when_their_share_of_the_cap_is_spent(self):
        from production.paid import Budget
        from production.stills.images import Painter
        from tests.test_stills import painted

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        asked = []

        class Critic:
            def look(self, image, picture):
                return False, "a split screen"  # serious, every time

        class P(Painter):
            def available(self, step):
                return True

            def _one(self, step, prompt, refs, raw, key, retry=0):
                asked.append(step)
                return painted(raw, seed=len(asked))

        studio = SimpleNamespace(budget=Budget(1.0, {"m": 0.04}), models={"shot": "m"})
        p = P(studio, {"retry_share": 0.1}, root, Critic())  # retries may spend $0.10: two of them
        for k in range(3):
            p.paint("x", [], root / f"b{k}.png", f"k{k}", ["fal:shot", "fal:shot"], check="x", capped=True)
        self.assertEqual(asked.count("fal:shot"), 3 + 2)  # three first tries, only two retries
        self.assertEqual(p.retries_stopped, 1)
        self.assertAlmostEqual(p.retry_spent, 0.08)
        self.assertTrue(all((root / f"b{k}.png").exists() for k in range(3)))  # each keeps its best try
        # an art-review redraw counts as a retry from its first call
        self.assertEqual(p.paint("x", [], root / "r.png", "kr", ["fal:shot"], check="x", capped=True,
                                 redraw=True), (None, "none"))


@unittest.skipUnless(HAVE, "video libraries or ffmpeg not installed")
class FinishedPicture(unittest.TestCase):
    def test_a_stretch_of_black_is_reported(self):
        from production.stills.post import picture_check

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        tags = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                        "color=c=gray:s=640x360:r=24:d=3,format=yuv420p", "-f", "lavfi", "-i",
                        "color=c=black:s=640x360:r=24:d=3,format=yuv420p", "-filter_complex",
                        "[0:v][1:v][0:v]concat=n=3:v=1[v]", "-map", "[v]", *tags, str(root / "gap.mp4")], check=True)
        issues = picture_check(root / "gap.mp4", 9.0)
        self.assertEqual(len(issues), 1)
        self.assertIn("3.0 s of black picture at 0:03", issues[0])
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                        "color=c=gray:s=640x360:r=24:d=2,format=yuv420p", str(root / "plain.mp4")], check=True)
        issues = picture_check(root / "plain.mp4", 2.0)  # no black, but nothing says its colours are HD
        self.assertEqual(len(issues), 1)
        self.assertIn("bt709", issues[0])


if __name__ == "__main__":
    unittest.main()

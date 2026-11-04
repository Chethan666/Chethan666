"""Pipeline 3: the drama writers' room with fake models."""

import json
import unittest

from drama_factory import pipeline
from tests.fakes import make_context
from tests.test_drama_production import BIBLE, episode

CRITERIA = ("hook", "conflict", "character", "dialogue", "cliffhanger", "continuity")


def review(**low):
    return {"scores": {c: low.get(c, 8) for c in CRITERIA}, "flags": {"gore": False},
            "notes": [{"shot": 3, "problem": "flat line", "fix": "Say it colder."}]}


def room(reviews=None, first=None, viewers_unlock=8):
    """A context whose fake models play every drama role."""
    ctx, brain = make_context()
    turn = {"review": 0}
    reviews = reviews or [review()]

    def next_review(prompt, spec):
        i = min(turn["review"] // 3, len(reviews) - 1)  # three judges per round
        turn["review"] += 1
        return reviews[i]

    counter = {"pitch": 0}

    def pitches(prompt, spec):
        counter["pitch"] += 1
        return {"pitches": [{"title": f"Series {spec.family} {counter['pitch']} {k}", "logline": "x", "hook": "y",
                             "central_secret": "z", "why_bingeable": "w"} for k in range(2)]}

    brain.overrides.update({
        "drama_pitches": pitches,
        "drama_score_pitch": lambda p, s: {"scores": {c: 8 for c in ("hook", "bingeability", "originality",
                                                                     "emotional_core", "clarity")}, "verdict": "ok"},
        "drama_compare_pitches": lambda p, s: {"reason": "r", "winner": "A"},
        "drama_bible": lambda p, s: dict(BIBLE, cast=[dict(c, look=c["look"] + " with a long scar across the left "
                                                                            "cheek and a silver ring")
                                                      for c in BIBLE["cast"]],
                                         episodes=[dict(e, number=n) for n, e in
                                                   enumerate(BIBLE["episodes"] * 4, 1)]),
        "drama_compare_bibles": lambda p, s: {"reason": "r", "winner": "B"},
        "drama_review_bible": lambda p, s: {"score": 8, "problems": [], "approved": True},
        "drama_revise_bible": lambda p, s: BIBLE,
        "drama_episode": lambda p, s: first or episode(),
        "drama_compare_episodes": lambda p, s: {"reason": "r", "winner": "A"},
        "drama_review": next_review,
        "drama_rewrite_shots": lambda p, s: {"shots": [{"number": 3, "shot": episode()["shots"][2]}]},
        "drama_revise": lambda p, s: episode(),
        "drama_replan": lambda p, s: {"number": 1, "title": "New", "hook": "h", "beats": ["b"], "cliffhanger": "c"},
        "drama_hook_options": lambda p, s: {"options": [{"shots": episode()["shots"][:2]}]},
        "drama_cliffhanger_options": lambda p, s: {"options": [{"shots": episode()["shots"][-2:]}]},
        "drama_compare_hook": lambda p, s: {"reason": "r", "winner": "A"},
        "drama_compare_cliffhanger": lambda p, s: {"reason": "r", "winner": "A"},
        "drama_viewer": lambda p, s: {"shots": [{"shot": 1, "interest": 8, "would_swipe": False, "note": "n"}],
                                      "unlock_next": viewers_unlock, "overall": "hooked"},
        "drama_final_check": lambda p, s: {"flags": {"gore": False}, "issues": []},
        "drama_packaging": lambda p, s: {"title": "The Still Room Ep 1", "caption": "Would you open it?",
                                         "hashtags": ["#drama"], "pinned_comment": "Who is she?"},
    })
    return ctx, brain


def log_of(ctx, episode_id):
    return json.loads(ctx.conn.execute("SELECT production_log FROM episodes WHERE id = ?", (episode_id,)).fetchone()[0])


class SeriesTests(unittest.TestCase):
    def test_pitch_contest_then_plan_contest_makes_an_active_series_and_shelves_runners_up(self):
        ctx, brain = room()
        series_id = pipeline.create_series(ctx)
        rows = {r["status"]: r for r in ctx.conn.execute("SELECT * FROM series")}
        self.assertEqual(rows["active"]["id"], series_id)
        self.assertIn("shelved", rows)
        log = json.loads(rows["active"]["production_log"])
        self.assertEqual([e["stage"] for e in log][:2], ["pitch_contest", "plan_contest"])
        self.assertEqual(len(brain.tasks("drama_bible")), 2)  # every writer planned
        # the next series starts from the shelf
        pipeline.create_series(ctx)
        self.assertEqual(len(brain.tasks("drama_pitches")), 3)  # no new pitch contest

    def test_bible_checks(self):
        self.assertTrue(any("precise look" in i for i in pipeline.bible_issues(BIBLE, 3)))
        self.assertTrue(any("outline all 12" in i for i in pipeline.bible_issues(BIBLE, 12)))


class EpisodeTests(unittest.TestCase):
    def setUp(self):
        self.ctx, self.brain = room()
        self.series_id = pipeline.create_series(self.ctx)

    def write(self):
        return pipeline.write_episode(self.ctx, self.series_id)

    def test_a_good_draft_goes_through_the_whole_room(self):
        episode_id, status = self.write()
        self.assertEqual(status, "approved")
        stages = [e["stage"] for e in log_of(self.ctx, episode_id)]
        for stage in ("draft_contest", "editors", "hook_lab", "cliffhanger_lab", "audience", "final_check"):
            self.assertIn(stage, stages)
        row = self.ctx.conn.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,)).fetchone()
        self.assertEqual(json.loads(row["packaging"])["title"], "The Still Room Ep 1")
        self.assertEqual(json.loads(row["audience_report"])["unlock_next"], 8)

    def test_noted_shots_are_rewritten_and_the_best_version_kept(self):
        self.ctx, self.brain = room(reviews=[review(dialogue=6), review(dialogue=5, hook=5), review()])
        self.series_id = pipeline.create_series(self.ctx)
        episode_id, status = self.write()
        self.assertEqual(status, "approved")
        log = log_of(self.ctx, episode_id)
        self.assertEqual([e["action"] for e in log if e["stage"] == "showrunner"][:2], ["shots", "revise"])
        self.assertIn("the earlier best", [e.get("kept") for e in log if e["stage"] == "editors"])

    def test_a_far_off_draft_is_written_again(self):
        self.ctx, self.brain = room(reviews=[review(conflict=4), review()])
        self.series_id = pipeline.create_series(self.ctx)
        episode_id, status = self.write()
        self.assertEqual(status, "approved")
        fresh = [e for e in log_of(self.ctx, episode_id) if e["stage"] == "draft_contest" and e["fresh"]]
        self.assertEqual(len(fresh), 1)

    def test_stuck_structure_rethinks_the_outline(self):
        self.ctx, self.brain = room(reviews=[review(cliffhanger=6, hook=6), review(cliffhanger=6, hook=6), review()])
        self.series_id = pipeline.create_series(self.ctx)
        episode_id, status = self.write()
        actions = [e["action"] for e in log_of(self.ctx, episode_id) if e["stage"] == "showrunner"]
        self.assertIn("replan", actions)
        self.assertEqual(status, "approved")

    def test_only_a_weak_hook_goes_to_the_hook_lab(self):
        self.ctx, self.brain = room(reviews=[review(hook=6), review()])
        self.series_id = pipeline.create_series(self.ctx)
        episode_id, _ = self.write()
        actions = [e["action"] for e in log_of(self.ctx, episode_id) if e["stage"] == "showrunner"]
        self.assertEqual(actions, ["hook"])

    def test_episodes_follow_on_and_a_rejected_one_is_written_again(self):
        self.ctx, self.brain = room(reviews=[review(conflict=6, character=6)])
        self.series_id = pipeline.create_series(self.ctx)
        _, status = self.write()
        self.assertEqual(status, "rejected")
        self.brain.overrides["drama_review"] = lambda p, s: review()
        _, status = self.write()
        self.assertEqual(status, "approved")
        _, status = self.write()
        numbers = [r["number"] for r in self.ctx.conn.execute("SELECT number FROM episodes ORDER BY number")]
        self.assertEqual(numbers, [1, 2])

    def test_free_checks(self):
        self.assertTrue(any("cast member" in p for p in pipeline.episode_issues(episode(bad_speaker=True), BIBLE)))
        silent = episode()
        for s in silent["shots"]:
            s["line"] = ""
        self.assertTrue(any("dialogue drama" in p for p in pipeline.episode_issues(silent, BIBLE)))
        self.assertEqual(pipeline.episode_issues(episode(), BIBLE), [])

    def test_a_safety_flag_that_survives_a_repair_rejects(self):
        self.brain.overrides["drama_final_check"] = lambda p, s: {"flags": {"gore": True}, "issues": ["blood"]}
        _, status = self.write()
        self.assertEqual(status, "rejected")
        self.assertEqual(len(self.brain.tasks("drama_final_check")), 4)  # two judges, two rounds


if __name__ == "__main__":
    unittest.main()

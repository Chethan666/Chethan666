"""The rewrite ladder: keep the best, diagnose the deepest problem, fix it at that level."""

import json
import unittest

from story_factory import db, pipeline, schemas
from tests.fakes import Brain, make_context


def scores(**low):
    return {c: low.get(c, 9) for c in schemas.REVIEW_CRITERIA}


def version(kind="long", coverage=1.0, unrecognized=(), structural=(), **low) -> pipeline.Version:
    v = pipeline.Version({"body": "x"}, "anthropic", list(structural))
    if not structural:
        v.agg = {"scores": scores(**low), "flags": [], "notes": [], "quoted_notes": []}
        v.panel = pipeline.review_issues(v.agg, 7)
        v.bible = pipeline.BibleVerdict(coverage=coverage)
        v.unrecognized = list(unrecognized)
    return v


def reviews_in_turn(brain: Brain, rounds: list[dict]):
    """Editors' scores change from round to round (the last entry repeats)."""
    seen = {"n": 0, "per_round": None}

    def review(prompt, spec):
        index = min(seen["n"] // 3, len(rounds) - 1)  # three judges per round
        seen["n"] += 1
        return {"scores": rounds[index], "flags": {"gore": False},
                "notes": [{"quote": "the door", "problem": "the middle sags", "fix": "raise the tension"}]}

    brain.overrides["review"] = review
    return seen


class DiagnoseTests(unittest.TestCase):
    q = dict({"min_score": 7, "kill_at_or_below_originality": 4, "redraft_below_coverage": 0.3,
                            "redraft_at_or_below_score": 5})

    def call(self, v, kind="long", fresh=1, stuck=frozenset(), last="", stalled=False):
        return pipeline.diagnose(v, kind, self.q, fresh, set(stuck), last, stalled)[0]

    def test_free_check_failures_are_repaired_first(self):
        self.assertEqual(self.call(version(structural=["too short"])), "structure")

    def test_a_familiar_premise_is_killed(self):
        self.assertEqual(self.call(version(originality=4)), "kill")
        self.assertNotEqual(self.call(version(originality=6)), "kill")

    def test_an_unrecognized_legend_goes_back_to_the_plan(self):
        self.assertEqual(self.call(version(unrecognized=["not the being"])), "replan")
        self.assertEqual(self.call(version(unrecognized=["not the being"]), fresh=0), "lines")

    def test_a_draft_far_from_its_plan_is_written_again(self):
        self.assertEqual(self.call(version(coverage=0.1)), "redraft")
        self.assertEqual(self.call(version(hook=4, tension=4)), "redraft")
        self.assertEqual(self.call(version(hook=4, tension=4), fresh=0), "sections")

    def test_structure_scores_stuck_after_a_revision_mean_a_new_plan(self):
        self.assertEqual(self.call(version(tension=6), stuck={"tension"}), "replan")
        self.assertEqual(self.call(version(tension=6)), "sections")  # first, try revising

    def test_hook_only_goes_to_the_hook_lab(self):
        self.assertEqual(self.call(version(hook=6)), "hook")
        self.assertEqual(self.call(version(hook=6), last="hook", stalled=True), "sections")

    def test_a_revision_that_made_things_worse_changes_the_kind_of_fix(self):
        self.assertEqual(self.call(version(prose_for_the_ear=6), last="sections", stalled=True), "lines")
        self.assertEqual(self.call(version(prose_for_the_ear=6), last="lines", stalled=True), "sections")
        self.assertEqual(self.call(version(prose_for_the_ear=6), kind="short", last="lines", stalled=True), "lines")
        self.assertEqual(self.call(version(prose_for_the_ear=5, tension=6), last="lines", stalled=True), "redraft")

    def test_shorts_revise_the_whole_script(self):
        self.assertEqual(self.call(version(prose_for_the_ear=6), kind="short"), "lines")

    def test_versions_order_by_approval_then_weakest_score(self):
        self.assertGreater(version().key(), version(tension=6).key())
        self.assertGreater(version(tension=6).key(), version(tension=5).key())
        self.assertGreater(version(tension=5).key(), version(structural=["x"]).key())


class SectionTests(unittest.TestCase):
    def test_sections_cover_the_body_and_locate_quoted_notes(self):
        body = "\n\n".join(f"Paragraph {i} has the words door{i} and more words here." for i in range(9))
        sections = pipeline.split_sections(body, 3)
        self.assertEqual(len(sections), 3)
        self.assertEqual("\n\n".join(sections), body)
        found = pipeline.locate_notes(sections, [{"quote": "Paragraph 7 has the words door7", "problem": "flat",
                                                   "fix": "sharper"}, {"quote": "not in the story at all"}])
        self.assertEqual(list(found), [3])
        self.assertIn("flat", found[3][0])

    def test_sentences_are_used_when_there_are_few_paragraphs(self):
        body = "One two three. Four five six. Seven eight nine. Ten eleven twelve."
        self.assertEqual(pipeline.split_sections(body, 2), ["One two three. Four five six.",
                                                             "Seven eight nine. Ten eleven twelve."])


class LadderFlowTests(unittest.TestCase):
    def test_a_worse_revision_never_replaces_the_best_version(self):
        ctx, brain = make_context()
        ctx.cfg["quality"]["max_fresh_starts"] = 0
        brain.pitch_scores = lambda title: 7  # below the parking bar
        good, worse = scores(tension=6), scores(tension=5, character=5, prose_for_the_ear=5)
        reviews_in_turn(brain, [good, worse, worse])
        outcome = pipeline.generate_story(ctx, "hollow_world")
        row = db.get_story(ctx.conn, outcome.story_id)
        self.assertEqual(row["status"], "rejected")  # a middling pitch isn't parked
        self.assertEqual(json.loads(row["review"])["scores"]["character"], 9)  # the best version's review
        log = json.loads(row["production_log"])
        self.assertIn("the earlier best", [e.get("kept") for e in log if e["stage"] == "editors"])
        self.assertTrue(json.loads(row["best_draft"])["body"])

    def test_stuck_tension_sends_the_story_back_to_the_plan(self):
        ctx, brain = make_context()
        reviews_in_turn(brain, [scores(tension=6), scores(tension=6), scores()])
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "approved", outcome.detail)
        log = json.loads(db.get_story(ctx.conn, outcome.story_id)["production_log"])
        actions = [e["action"] for e in log if e["stage"] == "showrunner"]
        self.assertEqual(actions, ["sections", "replan"])
        self.assertIn("replan", [e["stage"] for e in log])
        fresh = [e for e in log if e["stage"] == "draft_contest" and e["fresh"]]
        self.assertEqual(len(fresh), 1)

    def test_far_off_draft_is_redrafted_with_the_notes(self):
        ctx, brain = make_context()
        reviews_in_turn(brain, [scores(hook=4, tension=4), scores()])
        seen = []
        handler = brain._write_section
        brain.overrides["write_section"] = lambda prompt, spec: (seen.append(prompt), handler(prompt, spec))[1]
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "approved", outcome.detail)
        self.assertTrue(any("turned down by the editors" in p for p in seen))

    def test_section_rewrite_changes_only_the_quoted_section(self):
        ctx, brain = make_context()
        quoted = {}

        def review(prompt, spec):
            body = json.loads(prompt.split("Script:\n", 1)[1].split("\n\nScoring guide")[0])["body"] \
                if prompt.split("Script:\n", 1)[1].lstrip().startswith("{") else ""
            quoted.setdefault("q", " ".join(body.split()[-8:-2]) if body else "the door")
            first = brain.calls.count(("review", spec.label)) <= 1
            return {"scores": scores(prose_for_the_ear=6) if first else scores(), "flags": {"gore": False},
                    "notes": [{"quote": quoted["q"], "problem": "flat", "fix": "sharper"}]}

        brain.overrides["review"] = review
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "approved", outcome.detail)
        self.assertTrue(brain.tasks("rewrite_sections"))
        log = json.loads(db.get_story(ctx.conn, outcome.story_id)["production_log"])
        rewrite = next(e for e in log if e["stage"] == "section_rewrite")
        self.assertEqual(len(rewrite["rewritten"]), 1)  # only the section the note quotes

    def test_a_familiar_premise_is_rejected_not_parked(self):
        ctx, brain = make_context()
        brain.review_scores = 9
        reviews_in_turn(brain, [scores(originality=3)])
        outcome = pipeline.generate_story(ctx, "hollow_world")
        row = db.get_story(ctx.conn, outcome.story_id)
        self.assertEqual(row["status"], "rejected")
        self.assertEqual(row["diagnosis"], "kill")
        self.assertTrue(row["reject_reason"].startswith("premise:"))
        self.assertEqual(brain.tasks("revise"), [])  # nothing was wasted rewriting it

    def test_a_strong_premise_the_writers_could_not_land_is_parked(self):
        ctx, brain = make_context()
        ctx.cfg["quality"]["shelve_min_score"] = 7.5
        brain.pitch_scores = lambda title: 9
        brain.review_scores = 6
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "rejected")
        row = db.get_story(ctx.conn, outcome.story_id)
        self.assertEqual(row["status"], "parked")
        self.assertTrue(json.loads(row["best_draft"])["body"])
        shelf = db.best_shelved(ctx.conn, "hollow_world")
        self.assertNotEqual(shelf and shelf["id"], row["id"])  # not retried by the same models
        self.assertEqual(db.best_shelved(ctx.conn, "hollow_world", include_parked=True)["id"], row["id"])


class LegendRecognitionTests(unittest.TestCase):
    def add(self, ctx):
        return db.upsert_inspiration(ctx.conn, source="wikipedia", source_id="1", kind="folklore",
                                     title="Legend 1", url="", summary="A spirit of the roads.",
                                     region="Japan", license="CC BY-SA")

    def test_an_unrecognized_being_is_replanned_then_rejected(self):
        ctx, brain = make_context()
        self.add(ctx)
        brain.recognized = False
        outcome = pipeline.generate_story(ctx, "legends_retold")
        row = db.get_story(ctx.conn, outcome.story_id)
        self.assertEqual(row["status"], "rejected")  # legends are never parked
        self.assertIn("recognizably Legend 1", row["reject_reason"])
        log = json.loads(row["production_log"])
        self.assertIn("replan", [e.get("action") for e in log if e["stage"] == "showrunner"])
        self.assertEqual(len(db.unused_legends(ctx.conn)), 1)  # free to be retold properly

    def test_recognized_legend_is_approved(self):
        ctx, brain = make_context()
        self.add(ctx)
        outcome = pipeline.generate_story(ctx, "legends_retold")
        self.assertEqual(outcome.status, "approved", outcome.detail)
        self.assertTrue(brain.tasks("recognition"))

    def test_recheck_moves_an_unrecognizable_approved_legend_back(self):
        ctx, brain = make_context()
        self.add(ctx)
        ctx.cfg["quality"]["recognition_check"] = False  # approved before the test existed
        outcome = pipeline.generate_story(ctx, "legends_retold")
        self.assertEqual(outcome.status, "approved", outcome.detail)
        ctx.cfg["quality"]["recognition_check"] = True
        brain.recognized = False
        results = pipeline.recheck_legends(ctx)
        self.assertEqual([r["recognized"] for r in results], [False])
        row = db.get_story(ctx.conn, outcome.story_id)
        self.assertEqual(row["status"], "rejected")
        self.assertTrue(row["reject_reason"].startswith("recheck:"))
        self.assertTrue(row["body"])  # the script is kept
        self.assertEqual(pipeline.recheck_legends(ctx), [])  # nothing left to check

    def test_recheck_keeps_a_recognized_legend(self):
        ctx, brain = make_context()
        self.add(ctx)
        outcome = pipeline.generate_story(ctx, "legends_retold")
        row = db.get_story(ctx.conn, outcome.story_id)
        review = json.loads(row["review"])
        review.pop("recognition", None)
        db.update_story(ctx.conn, row["id"], review=review)
        self.assertEqual([r["recognized"] for r in pipeline.recheck_legends(ctx)], [True])
        self.assertEqual(db.get_story(ctx.conn, row["id"])["status"], "approved")


class ContestTests(unittest.TestCase):
    def test_shorts_compete_on_breadth(self):
        ctx, brain = make_context()
        outcome = pipeline.generate_story(ctx, "sixty_seconds")
        self.assertEqual(outcome.status, "approved", outcome.detail)
        self.assertEqual(len(brain.tasks("write_draft")), ctx.cfg["quality"]["short_drafts"])
        log = json.loads(db.get_story(ctx.conn, outcome.story_id)["production_log"])
        contest = next(e for e in log if e["stage"] == "draft_contest")
        self.assertEqual(contest["finalists"], ctx.cfg["quality"]["short_finalists"])
        self.assertEqual(brain.tasks("review_outline"), [])  # Shorts skip plan reviews
        self.assertEqual(brain.tasks("compare_plans"), [])

    def test_long_episodes_have_a_plan_contest(self):
        ctx, brain = make_context()
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "approved", outcome.detail)
        self.assertEqual(len(brain.tasks("story_bible")), 2)
        self.assertTrue(brain.tasks("compare_plans"))

    def test_plan_review_keeps_the_best_scoring_version_and_reviews_every_revision(self):
        ctx, brain = make_context()
        scores_given = iter([6, 6, 6, 5, 5, 5, 4, 4, 4])

        def review(prompt, spec):
            score = next(scores_given, 4)
            return {"score": score, "problems": ["the middle sags"], "approved": False}

        brain.overrides["review_outline"] = review
        story = pipeline.Story(series=pipeline.catalog.SERIES["hollow_world"], recipe=None, pitch={"title": "T"},
                               legend=None, dossier=None, sparks=[], world=[])
        first = brain._beat_sheet("", None)
        first["peak_moment"] = "the FIRST version"
        later = brain._beat_sheet("", None)
        brain.overrides["revise_outline"] = lambda prompt, spec: later
        chosen = pipeline.refine_plan(
            ctx, story, first, stage="outline", check=lambda o: pipeline.outline_issues(o, "long"),
            review_prompt=lambda o: "outline", revise=lambda o, p: later, author_family="anthropic", judges=3)
        self.assertEqual(chosen["peak_moment"], "the FIRST version")
        reviewed = [e for e in story.log if e["stage"] == "outline_review"]
        self.assertEqual(len(reviewed), ctx.cfg["quality"]["plan_revisions"] + 1)


class CraftCheckTests(unittest.TestCase):
    def test_outline_needs_an_escalation_ladder(self):
        outline = Brain()._beat_sheet("", None)
        self.assertEqual(pipeline.outline_issues(outline, "long"), [])
        outline["escalations"][0]["safety_lost"] = ""
        self.assertTrue(any("escalations" in i for i in pipeline.outline_issues(outline, "long")))

    def test_bible_needs_insider_details_and_promises_two(self):
        bible = Brain()._story_bible("", None)
        self.assertIn("insider1", dict(pipeline.bible_promises(bible, "long")))
        bible["insider_details"] = ["one"]
        self.assertTrue(any("insider_details" in i for i in pipeline.bible_issues(bible, "long")))

    def test_predictable_ending_gets_misdirection_kept_only_if_less_predictable(self):
        ctx, brain = make_context()
        brain.guessed = True
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "approved", outcome.detail)
        report = json.loads(db.get_story(ctx.conn, outcome.story_id)["audience_report"])["surprise_test"]
        self.assertEqual(report["guessed"], report["readers"])
        self.assertEqual(report["revision"], "discarded: no less predictable")

    def test_surprising_ending_is_left_alone(self):
        ctx, brain = make_context()
        outcome = pipeline.generate_story(ctx, "hollow_world")
        report = json.loads(db.get_story(ctx.conn, outcome.story_id)["audience_report"])["surprise_test"]
        self.assertEqual(report["guessed"], 0)
        self.assertNotIn("revision", report)

    def test_ending_lab_replaces_the_ending_only_when_judges_prefer_it(self):
        ctx, brain = make_context()
        brain.prefer_ending = "this time it rang for her"
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "approved", outcome.detail)
        row = db.get_story(ctx.conn, outcome.story_id)
        self.assertIn("this time it rang for her", row["body"])
        self.assertIn("\n\n", row["body"])  # paragraphs before the ending survive

    def test_narration_notes_keep_only_words_in_the_script(self):
        ctx, brain = make_context()
        story = pipeline.Story(series=pipeline.catalog.SERIES["hollow_world"], recipe=None, pitch={},
                               legend=None, dossier=None, sparks=[], world=[])
        script = {"intro": "", "body": "La Llorona walks. Some doors are better left closed tonight.",
                  "outro": ""}
        pronunciations, pauses = pipeline.annotate(ctx, story, script)
        self.assertEqual(pronunciations[0]["word"], "Llorona")
        self.assertEqual(pauses[0]["after"], "Some doors are better left closed")
        brain.overrides["annotate"] = lambda p, s: {"pronunciations": [{"word": "Krakonos", "say_as": "x"}],
                                                    "pauses": []}
        self.assertEqual(pipeline.annotate(ctx, story, script), ([], []))



class SceneCastTests(unittest.TestCase):
    def test_scene_characters_snap_to_cast_names(self):
        cast = [{"name": "Delano Marchetti", "look": "x"}, {"name": "The Storyteller", "look": "y"}]
        scenes = [{"characters": ["Delano (Memory)", "Del", "The Storyteller", "Nurse", "The Woman"]}]
        pipeline.snap_cast(scenes, cast)
        self.assertEqual(scenes[0]["characters"], ["Delano Marchetti", "The Storyteller", "Nurse", "The Woman"])
        pipeline.snap_cast(scenes, cast, drop_unknown=True)
        self.assertEqual(scenes[0]["characters"], ["Delano Marchetti", "The Storyteller"])


if __name__ == "__main__":
    unittest.main()

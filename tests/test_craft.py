"""The storyteller's craft: story bible, structure, writing in parts, style, pauses, benchmark."""

import json
import unittest

from story_factory import bench, db, pipeline
from story_factory.router import Router
from tests.fakes import Brain, FakeClient, make_context, test_config


def prompts_for(brain: Brain, ctx, task: str) -> list[str]:
    """Record the prompts a task receives, still answering with the default handler."""
    seen: list[str] = []
    handler = getattr(brain, f"_{task}")

    def record(prompt, spec):
        seen.append(prompt)
        return handler(prompt, spec)

    brain.overrides[task] = record
    return seen


class StoryBibleTests(unittest.TestCase):
    def test_bible_is_made_stored_and_steers_every_stage(self):
        ctx, brain = make_context()
        outline_prompts = prompts_for(brain, ctx, "beat_sheet")
        section_prompts = prompts_for(brain, ctx, "write_section")
        scene_prompts = prompts_for(brain, ctx, "scene_plan")
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "approved", outcome.detail)
        row = db.get_story(ctx.conn, outcome.story_id)
        bible = json.loads(row["bible"])
        self.assertEqual(bible["hero"]["name"], "Marisol")
        self.assertIn("Story bible", outline_prompts[0])
        self.assertTrue(all("she hums three notes" in p for p in section_prompts))
        self.assertIn("red accent: the stop button", scene_prompts[0])
        stages = [e["stage"] for e in json.loads(row["production_log"])]
        self.assertIn("story_bible", stages)
        self.assertIn("bible_check", stages)

    def test_incomplete_bible_is_revised(self):
        ctx, brain = make_context()
        first = brain._story_bible("", None)
        first["hero"]["human_detail"] = ""
        first["clues"] = []
        brain.overrides["story_bible"] = lambda prompt, spec: first
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "approved", outcome.detail)
        self.assertTrue(brain.tasks("revise_bible"))
        self.assertEqual(brain.tasks("review_bible"), [])  # editors only read complete bibles
        self.assertTrue(json.loads(db.get_story(ctx.conn, outcome.story_id)["bible"])["clues"])

    def test_bible_issues(self):
        bible = Brain()._story_bible("", None)
        self.assertEqual(pipeline.bible_issues(bible, "long"), [])
        bible["threat"]["rules"] = ["only one"]
        bible["refrain"] = "a refrain that is far too long to say again and again in one story"
        issues = pipeline.bible_issues(bible, "long")
        self.assertTrue(any("at least 2 rules" in i for i in issues))
        self.assertTrue(any("at most 12 words" in i for i in issues))
        self.assertEqual(pipeline.bible_issues(bible, "short"), ["the refrain must be at most 12 words"])
        short = Brain()._story_bible("", None)
        short.update(refrain="", refrain_turn="", breath_moment="", dramatic_irony="", clues=short["clues"][:1])
        short["threat"]["limits"] = []
        self.assertEqual(pipeline.bible_issues(short, "short"), [])  # Shorts need less

    def test_unmet_bible_promises_block_the_story(self):
        ctx, brain = make_context()
        brain.bible_present = lambda item_id: item_id.startswith("clue")
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "rejected")
        self.assertIn("deliver the story bible", outcome.detail)
        self.assertEqual(brain.tasks("review"), [])  # never reached the scored review

    def test_a_small_gap_only_advises(self):
        ctx, brain = make_context()
        brain.bible_present = lambda item_id: item_id != "breath"
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "approved", outcome.detail)

    def test_threat_breaking_its_rules_blocks(self):
        ctx, brain = make_context()
        brain.threat_breaks = ["it speaks, though it cannot speak"]
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "rejected")
        self.assertIn("breaks its own rules", outcome.detail)

    def test_quotes_must_really_be_on_the_page(self):
        text = "She hums the same three notes every night before the last stop."
        self.assertTrue(pipeline.quote_found("hums the same three notes", text))
        self.assertTrue(pipeline.quote_found("She hummed the same three notes every night", text))
        self.assertFalse(pipeline.quote_found("she whistles a song from her childhood", text))
        self.assertFalse(pipeline.quote_found("hums", text))


class StructureTests(unittest.TestCase):
    def test_outline_needs_rehooks_turns_midpoint_and_peak(self):
        outline = Brain()._beat_sheet("", None)
        self.assertEqual(pipeline.outline_issues(outline, "long"), [])
        outline["rehooks"] = outline["rehooks"][1:]
        outline["midpoint_reversal"] = ""
        outline["beats"][2]["turn"] = ""
        issues = pipeline.outline_issues(outline, "long")
        self.assertTrue(any("three rehooks" in i for i in issues))
        self.assertTrue(any("midpoint" in i for i in issues))
        self.assertTrue(any("first wrong detail" in i for i in issues))

    def test_outline_missing_structure_is_revised_before_review(self):
        ctx, brain = make_context()
        flat = brain._beat_sheet("", None)
        flat["beats"] = flat["beats"][:3]
        brain.overrides["beat_sheet"] = lambda prompt, spec: flat
        pipeline.generate_story(ctx, "hollow_world")
        self.assertTrue(brain.tasks("revise_outline"))

    def test_long_drafts_are_written_in_parts_and_framed(self):
        ctx, brain = make_context()
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "approved", outcome.detail)
        writers = len(ctx.cfg["roles"]["writers"])
        self.assertEqual(len(brain.tasks("write_section")), 3 * writers)
        self.assertEqual(len(brain.tasks("write_frame")), writers)
        self.assertEqual(brain.tasks("write_draft"), [])
        intro = db.get_story(ctx.conn, outcome.story_id)["intro"]
        self.assertTrue(intro.startswith(pipeline.RITUAL_OPENING))

    def test_later_parts_see_the_story_so_far(self):
        ctx, brain = make_context()
        seen = prompts_for(brain, ctx, "write_section")
        pipeline.generate_story(ctx, "hollow_world")
        self.assertNotIn("The story so far", seen[0])
        self.assertTrue(any("The story so far" in p and "part 3 of 3" in p for p in seen))

    def test_shorts_are_written_in_one_go(self):
        ctx, brain = make_context()
        outcome = pipeline.generate_story(ctx, "sixty_seconds")
        self.assertEqual(outcome.status, "approved", outcome.detail)
        self.assertEqual(brain.tasks("write_section"), [])
        self.assertTrue(brain.tasks("write_draft"))

    def test_split_beats_balances_running_time(self):
        beats = [{"target_seconds": s} for s in (60, 60, 60, 60, 60, 60, 60, 60, 60)]
        self.assertEqual([len(g) for g in pipeline.split_beats(beats, 3)], [3, 3, 3])
        uneven = [{"target_seconds": s} for s in (200, 20, 20, 20, 20)]
        groups = pipeline.split_beats(uneven, 3)
        self.assertEqual(len(groups), 3)
        self.assertEqual(sum(len(g) for g in groups), 5)
        self.assertEqual(len(pipeline.split_beats(beats[:2], 3)), 2)

    def test_ritual_opening_is_added_once_for_long_episodes(self):
        script = Brain().make_script(long=True)
        pipeline.tidy(script, "long")
        self.assertTrue(script["intro"].startswith("The lantern is lit. I found"))
        pipeline.tidy(script, "long")
        self.assertEqual(script["intro"].count("lantern is lit"), 1)
        short = Brain().make_script(long=False)
        self.assertEqual(pipeline.tidy(short, "short")["intro"], "")


class StyleTests(unittest.TestCase):
    def setUp(self):
        self.cfg = test_config()

    def issues(self, text, kind="long"):
        return pipeline.style_issues(text, self.cfg, kind)

    def test_clean_prose_passes(self):
        text = ("Marisol had driven the late bus for nine years. Nobody ever rang the bell past Calder Road. "
                "There is nothing out there. Tonight, at six minutes past midnight, the bell rang.")
        self.assertEqual(self.issues(text), [])

    def test_machine_phrasing_and_named_feelings(self):
        text = ("A sense of dread filled the room. The air grew thick. She felt terrified. He felt scared. "
                "They felt afraid.")
        issues = self.issues(text)
        self.assertTrue(any("a sense of dread" in i.lower() for i in issues))
        self.assertTrue(any("the air grew thick" in i.lower() for i in issues))
        self.assertTrue(any("instead of naming it" in i for i in issues))

    def test_long_sentences_repeated_openings_dashes_and_hard_words(self):
        long_sentence = " ".join(["word"] * 40) + "."
        self.assertTrue(any("average" in i for i in self.issues(long_sentence)))
        repeated = "She ran. She hid. She waited. She listened."
        self.assertTrue(any("in a row start with 'she'" in i for i in self.issues(repeated)))
        dashed = "He stopped — then turned — then ran — then fell. It was over."
        self.assertTrue(any("dashes" in i for i in self.issues(dashed)))
        hard = "Unquestionably, the extraordinary documentation necessitated reconsideration immediately."
        self.assertTrue(any("plainer words" in i for i in self.issues(hard)))

    def test_style_problems_get_a_dedicated_line_edit(self):
        ctx, brain = make_context()

        def framed(prompt, spec):
            frame = brain._write_frame(prompt, spec)
            frame["outro"] = "A sense of dread stays with you. " + frame["outro"]
            return frame

        def cleaned(prompt, spec):
            script = brain._line_edit(prompt, spec)
            script["outro"] = script["outro"].replace("A sense of dread stays with you.", "The bell still rings.")
            return script

        brain.overrides["write_frame"] = framed
        brain.overrides["line_edit"] = cleaned
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "approved", outcome.detail)
        self.assertTrue(brain.tasks("line_edit"))
        self.assertNotIn("sense of dread", db.get_story(ctx.conn, outcome.story_id)["narration"])


class EditorTests(unittest.TestCase):
    def test_structured_notes_are_quoted_with_fixes(self):
        reviews = [{"scores": {}, "flags": {}, "notes": [{"quote": "the door", "problem": "too soon",
                                                          "fix": "delay it"}]}]
        self.assertEqual(pipeline.aggregate_reviews(reviews)["notes"], ['At "the door": too soon Fix: delay it'])

    def test_judges_per_comparison_are_capped(self):
        ctx, brain = make_context()
        self.assertEqual(len(ctx.router.judges(["anthropic"], 2)), 2)
        self.assertEqual(len(ctx.router.judges(["anthropic"])), 3)
        pipeline.generate_story(ctx, "hollow_world")
        self.assertLessEqual(len(set(brain.tasks("compare_titles"))), ctx.cfg["quality"]["quick_judges"])
        self.assertLessEqual(len(set(brain.tasks("review"))), ctx.cfg["quality"]["judges_per_comparison"])

    def test_pauses_are_kept_only_when_on_the_page(self):
        narration = "The bell rang. Some doors are better left closed. Would you have opened it?"
        pauses = [{"after": "Some doors are better left closed", "ms": 9000},
                  {"after": "a phrase that is not there", "ms": 800},
                  {"after": "The bell rang", "ms": 100}]
        kept = pipeline.valid_pauses(pauses, narration, 5)
        self.assertEqual([p["after"] for p in kept], ["The bell rang", "Some doors are better left closed"])
        self.assertEqual([p["ms"] for p in kept], [300, 3000])

    def test_pauses_are_stored_with_the_story(self):
        ctx, brain = make_context()
        outcome = pipeline.generate_story(ctx, "hollow_world")
        pauses = json.loads(db.get_story(ctx.conn, outcome.story_id)["pauses"])
        self.assertEqual(pauses, [{"after": "Some doors are better left closed", "ms": 1200}])


class BenchmarkTests(unittest.TestCase):
    def test_benchmark_compares_two_setups_blind(self):
        brain = Brain()
        factory = lambda cfg, conn: (lambda spec: FakeClient(spec, brain))  # noqa: E731
        cfg_a, cfg_b = test_config(), test_config()
        cfg_b["roles"]["judges"] = cfg_b["roles"]["judges"] + ["openai:judge-e"]
        summary = bench.run(cfg_a, cfg_b, 2, factory)
        self.assertEqual(len(summary["results"]), 2)
        self.assertIn("openai:judge-e", summary["judges"])
        self.assertEqual(summary["a_wins"] + summary["b_wins"] + summary["ties"], 2)
        self.assertTrue(all(r["words_a"] and r["words_b"] for r in summary["results"]))
        self.assertIn("| Pitch |", bench.markdown(summary, "a.toml", "b.toml"))

    def test_a_setup_that_cannot_write_loses(self):
        brain = Brain()
        brain.failing = {"anthropic:writer-x"}
        factory = lambda cfg, conn: (lambda spec: FakeClient(spec, brain))  # noqa: E731
        cfg_a, cfg_b = test_config(), test_config()
        cfg_b["roles"]["writers"] = ["anthropic:writer-x"]
        summary = bench.run(cfg_a, cfg_b, 1, factory)
        self.assertEqual(summary["results"][0]["winner"], "A")

    def test_router_factory_is_used_for_every_setup(self):
        cfg = test_config()
        router = Router(cfg, db.connect(":memory:"), lambda spec: FakeClient(spec, Brain()))
        self.assertEqual(len(router.pool("judges")), 4)


if __name__ == "__main__":
    unittest.main()

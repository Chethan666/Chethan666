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
        self.assertIn("bible_review", stages)
        self.assertIn("plan_contest", stages)
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
        tasks = [task for task, _ in brain.calls]
        # Editors only read complete bibles: no review before the incomplete one was revised.
        self.assertLess(tasks.index("revise_bible"), tasks.index("review_bible"))
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

    def test_unmet_bible_promises_steer_revisions_then_the_panel_decides(self):
        ctx, brain = make_context()
        brain.bible_present = lambda item_id: item_id.startswith("clue")
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "approved", outcome.detail)
        log = json.loads(db.get_story(ctx.conn, outcome.story_id)["production_log"])
        # The panel approved it: one attempt to deliver the missing promises, then accepted.
        self.assertEqual(len([e for e in log if e["stage"] == "showrunner"]), 1)
        self.assertIn("bible_gaps_accepted", [e["stage"] for e in log])

    def test_unmet_promises_and_a_failing_panel_reject(self):
        ctx, brain = make_context()
        brain.bible_present = lambda item_id: item_id.startswith("clue")
        brain.review_scores = 5
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "rejected")

    def test_a_small_gap_only_advises(self):
        ctx, brain = make_context()
        brain.bible_present = lambda item_id: item_id != "breath"
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "approved", outcome.detail)
        self.assertEqual(brain.tasks("revise"), [])

    def test_confirmed_rule_breaks_block(self):
        ctx, brain = make_context()
        brain.threat_breaks = ["it speaks, though it cannot speak"]
        brain.real_breaks = {"it speaks, though it cannot speak"}
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "rejected")
        self.assertIn("breaks its own rules", outcome.detail)

    def test_an_unconfirmed_rule_break_claim_does_not_block(self):
        ctx, brain = make_context()
        brain.threat_breaks = ["the laundry rule is implied to be broken somehow"]
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "approved", outcome.detail)
        self.assertTrue(brain.tasks("confirm_breaks"))
        self.assertNotEqual(brain.tasks("bible_check")[0], brain.tasks("confirm_breaks")[0])  # another judge

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
        self.assertTrue(intro.startswith("The phone rings"))  # cold open first
        self.assertIn(pipeline.RITUAL_OPENING, intro)

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
        short["body"] = "The lantern is lit. At 3:47 her phone played a song. " + short["body"]
        self.assertTrue(pipeline.tidy(short, "short")["body"].startswith("At 3:47"))
        short["body"] = "You sign off on car 4. The lantern is lit. Pine smoke and wool. Is the lantern lit? No."
        self.assertEqual(pipeline.tidy(short, "short")["body"],
                         "You sign off on car 4. Pine smoke and wool. Is the lantern lit? No.")


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


class SoftStyleTests(unittest.TestCase):
    def test_judgment_calls_advise_but_never_block(self):
        ctx, brain = make_context()

        def framed(prompt, spec):
            frame = brain._write_frame(prompt, spec)
            frame["outro"] = "You wait. You listen. You count. You breathe. " + frame["outro"]
            return frame

        brain.overrides["write_frame"] = framed
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "approved", outcome.detail)
        self.assertTrue(brain.tasks("line_edit"))  # still offered to the line editor
        self.assertTrue(brain.tasks("review"))  # and the editors still read it


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


class ScenePlanRepairTests(unittest.TestCase):
    NARRATION = ("The bell rang. Nobody was there. She checked the mirror.\n\nThe seats were empty. "
                 "The bell rang again. She stopped the bus? Would you have opened the door?")

    def scene(self, text, key=False):
        return {"text": text, "visual": "v", "location": "l", "characters": ["Iris"], "mood": "m",
                "delivery": "slow", "sfx": ["bell"], "key_moment": key}

    def test_skipped_and_partial_text_is_snapped_to_full_coverage(self):
        scenes = [self.scene("The bell rang.", True), self.scene("She checked the"),  # partial sentence
                  self.scene("The bell rang again. She stopped"), self.scene("a line the planner invented")]
        fixed = pipeline.snap_scenes(self.NARRATION, scenes, "short")
        self.assertEqual([s["text"] for s in fixed], [
            "The bell rang. Nobody was there.", "She checked the mirror. The seats were empty.",
            "The bell rang again. She stopped the bus? Would you have opened the door?"])
        self.assertEqual(pipeline.scene_issues(self.NARRATION, fixed, "short"), ["3 scenes; need 4-10"])

    def test_too_many_scenes_and_key_moments_are_trimmed(self):
        sentences = [f"Sentence number {i} is here." for i in range(60)]
        narration = " ".join(sentences)
        scenes = [self.scene(t, key=i % 5 == 0) for i, t in enumerate(sentences)]
        fixed = pipeline.snap_scenes(narration, scenes, "long")
        self.assertEqual(len(fixed), 45)
        self.assertEqual(sum(s["key_moment"] for s in fixed), 6)
        self.assertEqual(pipeline.scene_issues(narration, fixed, "long"), [])


class ChannelGrowthTests(unittest.TestCase):
    def test_long_episodes_get_a_teaser_short_with_scenes(self):
        ctx, brain = make_context()
        outcome = pipeline.generate_story(ctx, "hollow_world")
        packaging = json.loads(db.get_story(ctx.conn, outcome.story_id)["packaging"])
        teaser = packaging["teaser_short"]
        self.assertTrue(teaser["narration"].endswith("Would you have answered?"))
        self.assertEqual(teaser["comment_question"], "Would you have answered?")
        self.assertTrue(teaser["scenes"])
        self.assertEqual(packaging["end_screen_line"], "There is another story waiting down the line.")

    def test_a_broken_teaser_is_left_out_without_blocking(self):
        ctx, brain = make_context()
        brain.overrides["teaser_short"] = lambda prompt, spec: {"body": "Too short.", "outro": "No question."}
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "approved", outcome.detail)
        self.assertIsNone(json.loads(db.get_story(ctx.conn, outcome.story_id)["packaging"])["teaser_short"])
        self.assertEqual(len(brain.tasks("teaser_short")), 2)

    def test_shorts_get_no_teaser(self):
        ctx, brain = make_context()
        pipeline.generate_story(ctx, "sixty_seconds")
        self.assertEqual(brain.tasks("teaser_short"), [])

    def test_busy_thumbnails_are_sent_back(self):
        ctx, brain = make_context()
        busy = brain._packaging("", None)
        busy.update(thumbnail_brief=" ".join(["figure"] * 60), thumbnail_text="MUTE IT NOW BEFORE IT HEARS")
        calls = []

        def packaging(prompt, spec):
            calls.append(prompt)
            return busy if len(calls) == 1 else brain._packaging(prompt, spec)

        brain.overrides["packaging"] = packaging
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertIn("FIX THESE PROBLEMS", calls[1])
        result = json.loads(db.get_story(ctx.conn, outcome.story_id)["packaging"])
        self.assertEqual(result["thumbnail_text"], "WHO?")

    def test_recent_heroes_reach_the_next_bible(self):
        ctx, brain = make_context()
        seen = prompts_for(brain, ctx, "story_bible")
        pipeline.generate_story(ctx, "hollow_world")
        pipeline.generate_story(ctx, "hollow_world")
        self.assertNotIn("Heroes of our recent episodes", seen[0])
        self.assertIn("- Marisol: a bus driver, 52", seen[-1])

    def test_strong_profanity_is_caught(self):
        script = Brain().make_script(long=False)
        script["body"] += " What the fuck was that."
        self.assertTrue(any("advertiser-friendly" in i
                            for i in pipeline.script_issues(script, "short", test_config())))

    def test_intro_is_rated_but_never_rewritten(self):
        ctx, brain = make_context()
        brain.audience_interest = lambda chunk: 3 if chunk == 1 else 9
        outcome = pipeline.generate_story(ctx, "hollow_world")
        report = json.loads(db.get_story(ctx.conn, outcome.story_id)["audience_report"])
        self.assertEqual(report["intro_interest"], 3)
        self.assertEqual(report["weak_chunks"], [])
        self.assertEqual(brain.tasks("rewrite_chunks"), [])


class FinalCheckTests(unittest.TestCase):
    def graded(self, *issues):
        return lambda prompt, spec: {"flags": {"gore": False}, "accuracy_issues": [],
                                     "cultural_issues": [{"issue": i, "severity": sev} for i, sev in issues]}

    def test_minor_suggestions_never_block(self):
        ctx, brain = make_context()
        brain.overrides["final_check"] = self.graded(("explain the Angelus bell", "minor"))
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "approved", outcome.detail)
        self.assertEqual(brain.tasks("revise"), [])

    def test_must_fix_issues_get_two_repair_rounds(self):
        ctx, brain = make_context()
        brain.overrides["final_check"] = self.graded(("a stereotype in part two", "must_fix"))
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "rejected")
        self.assertIn("a stereotype in part two", outcome.detail)
        self.assertEqual(len(brain.tasks("revise")), ctx.cfg["quality"]["final_check_repairs"])

    def test_one_judge_alone_cannot_veto_on_accuracy_or_culture(self):
        ctx, brain = make_context()
        first = []

        def one_strict_judge(prompt, spec):
            strict = spec.label == "openai:judge-b"
            first.append(spec.label)
            issues = [{"issue": "the bill should be a proboscis", "severity": "must_fix"}] if strict else []
            return {"flags": {"gore": False}, "accuracy_issues": [], "cultural_issues": issues}

        brain.overrides["final_check"] = one_strict_judge
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "approved", outcome.detail)
        self.assertEqual(brain.tasks("revise"), [])

    def test_a_safety_flag_from_one_judge_still_blocks(self):
        ctx, brain = make_context()

        def one_flag(prompt, spec):
            return {"flags": {"gore": spec.label == "openai:judge-b"}, "accuracy_issues": [], "cultural_issues": []}

        brain.overrides["final_check"] = one_flag
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "rejected")
        self.assertIn("rule violation: gore", outcome.detail)

    def test_legend_core_is_a_promise_checked_while_editing(self):
        bible = Brain()._story_bible("", None)
        dossier = {"core_elements": ["punishes those who work the fields at noon", "carries a scythe"]}
        promises = dict(pipeline.bible_promises(bible, "long", dossier))
        self.assertIn("punishes those who work the fields at noon", promises["legend1"])
        self.assertIn("legend2", promises)
        self.assertNotIn("legend1", dict(pipeline.bible_promises(bible, "long")))


class LegendFitTests(unittest.TestCase):
    def add(self, ctx, n):
        return db.upsert_inspiration(ctx.conn, source="wikipedia", source_id=str(n), kind="folklore",
                                     title=f"Legend {n}", url="", summary="A spirit of the roads.",
                                     region="Japan", license="CC BY-SA")

    def test_unfit_legends_are_skipped_for_good(self):
        ctx, brain = make_context()
        for n in (1, 2):
            self.add(ctx, n)
        brain.unfit_legends = {"Legend 1"}
        outcome = pipeline.generate_story(ctx, "legends_retold")
        self.assertEqual(outcome.status, "approved", outcome.detail)
        row = db.get_story(ctx.conn, outcome.story_id)
        self.assertEqual(row["legend_key"], "wikipedia:2")

    def test_no_fit_legend_skips_the_run(self):
        ctx, brain = make_context()
        self.add(ctx, 1)
        brain.unfit_legends = {"Legend 1"}
        outcome = pipeline.generate_story(ctx, "legends_retold")
        self.assertEqual(outcome.status, "skipped")
        self.assertIn("no suitable unused legend", outcome.detail)
        self.assertEqual(db.unused_legends(ctx.conn), [])  # never offered again

    def test_legend_fidelity_steers_pitches_and_the_bible(self):
        ctx, brain = make_context()
        self.add(ctx, 1)
        pitch_prompts = prompts_for(brain, ctx, "pitches")
        score_prompts = prompts_for(brain, ctx, "score_pitch")
        review_prompts = prompts_for(brain, ctx, "review_bible")
        pipeline.generate_story(ctx, "legends_retold")
        self.assertIn("Suggested ingredients", pitch_prompts[0])
        self.assertIn("score 1-3 if the pitch changes the being", score_prompts[0])
        self.assertIn("exactly as these research notes describe it", review_prompts[0])

    def test_ritual_line_cannot_be_the_refrain(self):
        bible = Brain()._story_bible("", None)
        bible["refrain"] = "The lantern is lit."
        self.assertTrue(any("not the Storyteller's ritual" in i for i in pipeline.bible_issues(bible, "long")))

    def test_bible_check_asks_about_invented_traits(self):
        ctx, brain = make_context()
        self.add(ctx, 1)
        seen = prompts_for(brain, ctx, "bible_check")
        pipeline.generate_story(ctx, "legends_retold")
        self.assertIn("research notes do not support", seen[0])
        self.assertIn("Interpreting its motives, feelings or inner life", seen[0])


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

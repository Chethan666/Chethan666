import json
import random
import unittest

from story_factory import db, pipeline
from story_factory.llm import BudgetExceeded, BudgetGuard, LLMError, parse_json
from tests.fakes import FakeLLM, premise, test_config


def make_context(seed: int = 7) -> tuple[pipeline.Context, FakeLLM]:
    cfg = test_config()
    llm = FakeLLM()
    return pipeline.Context(conn=db.connect(":memory:"), cfg=cfg, llm=llm, rng=random.Random(seed)), llm


def add_legend(conn, n: int, summary: str = "A spirit that walks the old roads at night.") -> int:
    return db.upsert_inspiration(conn, source="wikipedia", source_id=str(n), kind="folklore",
                                 title=f"Legend {n}", url="", summary=summary, region="Mexico",
                                 license="CC BY-SA")


class GenerateTests(unittest.TestCase):
    def test_approves_a_complete_story(self):
        ctx, _ = make_context()
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "approved", outcome.detail)
        story = db.get_story(ctx.conn, outcome.story_id)
        self.assertEqual(story["status"], "approved")
        self.assertEqual(story["kind"], "long")
        self.assertTrue(story["narration"].endswith("?"))
        scenes = json.loads(story["scenes"])
        self.assertTrue(18 <= len(scenes) <= 45)
        linked = ctx.conn.execute("SELECT COUNT(*) FROM story_entities WHERE story_id = ?",
                                  (story["id"],)).fetchone()[0]
        self.assertEqual(linked, 2)  # its new place plus the referenced red lantern

    def test_short_story(self):
        ctx, _ = make_context()
        outcome = pipeline.generate_story(ctx, "sixty_seconds")
        self.assertEqual(outcome.status, "approved", outcome.detail)
        story = db.get_story(ctx.conn, outcome.story_id)
        self.assertEqual(story["kind"], "short")
        self.assertEqual(story["intro"], "")

    def test_repeated_title_is_rejected(self):
        ctx, llm = make_context()
        first = premise(1)
        llm.premise_queue.append([first])
        self.assertEqual(pipeline.generate_story(ctx, "hollow_world").status, "approved")
        renamed = dict(premise(2), title=first["title"] + "!")
        fresh = premise(3)
        llm.premise_queue.append([renamed, fresh])
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.detail, fresh["title"])

    def test_same_story_with_new_title_is_rejected(self):
        ctx, llm = make_context()
        original = premise(1)
        llm.premise_queue.append([original])
        pipeline.generate_story(ctx, "hollow_world")
        # Same words in a different order: different wording, identical meaning.
        shuffled_words = original["synopsis"].rstrip(".").split()
        random.Random(0).shuffle(shuffled_words)
        disguised = {"title": "A Completely New Name", "logline": original["logline"],
                     "synopsis": " ".join(shuffled_words) + "."}
        llm.premise_queue.append([disguised])
        llm.premise_queue.append([premise(5)])
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.detail, premise(5)["title"])

    def test_borderline_similarity_goes_to_judge(self):
        ctx, llm = make_context()
        ctx.cfg["dedup"]["semantic_soft"] = 0.0  # every pair is borderline
        llm.premise_queue.append([premise(1)])
        pipeline.generate_story(ctx, "hollow_world")
        llm.judge_same = True
        llm.premise_queue.extend([[premise(2)], [premise(3)], [premise(4)]])
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "skipped")
        self.assertIn("judge_duplicate", llm.calls)

    def test_rejected_ideas_are_not_pitched_again(self):
        ctx, llm = make_context()
        llm.review_scores = 3
        idea = premise(1)
        llm.premise_queue.append([idea])
        self.assertEqual(pipeline.generate_story(ctx, "hollow_world").status, "rejected")
        llm.review_scores = 9
        llm.premise_queue.extend([[idea], [premise(2)]])
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.detail, premise(2)["title"])

    def test_low_review_is_revised_then_rejected(self):
        ctx, llm = make_context()
        llm.review_scores = 4
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "rejected")
        self.assertIn("improve hook", outcome.detail)
        self.assertEqual(llm.calls.count("revise"), ctx.cfg["quality"]["max_revisions"])

    def test_copied_legend_text_forces_rewrite(self):
        ctx, llm = make_context()
        legend_text = "the weeping woman wanders the riverbank at night searching for her lost children"
        add_legend(ctx.conn, 1, legend_text)
        copying = llm.make_script(long=True)
        copying["body"] = legend_text.capitalize() + ". " + copying["body"]
        llm.overrides["write_script"] = lambda prompt: copying
        outcome = pipeline.generate_story(ctx, "legends_retold")
        self.assertEqual(outcome.status, "approved", outcome.detail)
        self.assertIn("revise", llm.calls)
        self.assertNotIn(legend_text, db.get_story(ctx.conn, outcome.story_id)["narration"].lower())

    def test_each_legend_is_told_once(self):
        ctx, _ = make_context()
        add_legend(ctx.conn, 1)
        first = pipeline.generate_story(ctx, "legends_retold")
        self.assertEqual(first.status, "approved", first.detail)
        self.assertEqual(db.get_story(ctx.conn, first.story_id)["legend_key"], "wikipedia:1")
        second = pipeline.generate_story(ctx, "legends_retold")
        self.assertEqual(second.status, "skipped")
        self.assertIn("no unused legends", second.detail)

    def test_model_failure_mid_story_does_not_leave_it_stuck(self):
        ctx, llm = make_context()

        def failing(prompt):
            raise LLMError("HTTP 503")

        llm.overrides["review"] = failing
        with self.assertRaises(LLMError):
            pipeline.generate_story(ctx, "hollow_world")
        statuses = db.status_counts(ctx.conn)
        self.assertEqual(statuses, {"rejected": 1})
        del llm.overrides["review"]
        self.assertEqual(pipeline.generate_story(ctx, "hollow_world").status, "approved")

    def test_bad_scene_plan_is_retried(self):
        ctx, llm = make_context()
        attempts = []

        def scenes(prompt):
            attempts.append(prompt)
            if len(attempts) == 1:
                return {"scenes": [{"text": "made up words", "visual": "x", "location": "y", "key_moment": True}]}
            del llm.overrides["scene_plan"]
            return llm._scene_plan(prompt)

        llm.overrides["scene_plan"] = scenes
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "approved", outcome.detail)
        self.assertIn("FIX THESE PROBLEMS", attempts[1])


class BacklogTests(unittest.TestCase):
    def test_fill_reaches_targets_without_repeats(self):
        ctx, _ = make_context()
        ctx.cfg["backlog"] = {"hollow_world": 3, "legends_retold": 1, "sixty_seconds": 4}
        add_legend(ctx.conn, 1)
        outcomes = pipeline.fill_backlog(ctx, max_new=20)
        self.assertEqual(sum(o.status == "approved" for o in outcomes), 8)
        self.assertEqual(db.backlog_counts(ctx.conn),
                         {"hollow_world": 3, "legends_retold": 1, "sixty_seconds": 4})
        titles = [r["title_norm"] for r in db.live_stories(ctx.conn)]
        self.assertEqual(len(titles), len(set(titles)))
        sigs = [r["recipe_sig"] for r in db.live_stories(ctx.conn)]
        self.assertEqual(len(sigs), len(set(sigs)))

    def test_fill_stops_at_budget(self):
        ctx, llm = make_context()

        def broke(prompt):
            raise BudgetExceeded("cap reached")

        llm.overrides["premises"] = broke
        outcomes = pipeline.fill_backlog(ctx, max_new=5)
        self.assertEqual(len(outcomes), 1)
        self.assertIn("cap reached", outcomes[0].detail)

    def test_fill_survives_model_errors(self):
        ctx, llm = make_context()
        ctx.cfg["backlog"] = {"hollow_world": 1, "sixty_seconds": 1}

        def failing(prompt):
            raise LLMError("HTTP 500")

        llm.overrides["premises"] = failing
        outcomes = pipeline.fill_backlog(ctx, max_new=5)
        self.assertEqual({o.status for o in outcomes}, {"skipped"})


class StorageTests(unittest.TestCase):
    def test_claim_hands_out_each_story_once(self):
        ctx, _ = make_context()
        a = pipeline.generate_story(ctx, "hollow_world").story_id
        b = pipeline.generate_story(ctx, "hollow_world").story_id
        pipeline.generate_story(ctx, "sixty_seconds")
        self.assertEqual(db.claim_next(ctx.conn, "long", "w1")["id"], a)
        claimed = db.claim_next(ctx.conn, "long", "w2")
        self.assertEqual((claimed["id"], claimed["status"], claimed["claimed_by"]), (b, "in_production", "w2"))
        self.assertIsNone(db.claim_next(ctx.conn, "long", "w3"))

    def test_budget_guard(self):
        conn = db.connect(":memory:")
        guard = BudgetGuard(conn, 18.0)
        guard.check()
        db.record_cost(conn, pipeline="production", task="image", model="m",
                       input_units=0, output_units=0, usd=17.5)
        guard.check()
        with self.assertRaises(BudgetExceeded):
            guard.check(upcoming_usd=0.6)

    def test_recipe_unique_index(self):
        conn = db.connect(":memory:")
        fields = dict(kind="long", series="hollow_world", status="approved", title="A",
                      title_norm="a", recipe_sig="x|y|z")
        db.insert_story(conn, **fields)
        with self.assertRaises(Exception):
            db.insert_story(conn, **dict(fields, title="B", title_norm="b"))
        db.insert_story(conn, **dict(fields, title="C", title_norm="c", status="rejected"))

    def test_parse_json_accepts_fences(self):
        self.assertEqual(parse_json('```json\n{"a": 1}\n```'), {"a": 1})
        with self.assertRaises(LLMError):
            parse_json("[1, 2]")


class ScriptCheckTests(unittest.TestCase):
    def test_claims_true_is_caught(self):
        cfg = test_config()
        script = FakeLLM().make_script(long=False)
        script["body"] += " This is a true story."
        issues = pipeline.script_issues(script, "short", cfg)
        self.assertTrue(any("never claim" in i for i in issues))

    def test_scene_coverage(self):
        parts = ["One.", "Two three.", "Four five six.", "Seven eight nine ten?"]
        narration = " ".join(parts)
        good = [{"text": t, "visual": "v", "location": "l", "key_moment": i == 0}
                for i, t in enumerate(parts)]
        self.assertEqual(pipeline.scene_issues(narration, good, "short"), [])
        out_of_order = [good[0], good[2], good[1], good[3]]
        self.assertTrue(any("in-order" in i for i in pipeline.scene_issues(narration, out_of_order, "short")))
        repeated = [good[0], good[1], good[1], good[3]]
        self.assertTrue(pipeline.scene_issues(narration, repeated, "short"))
        skipped = [good[0], good[1], good[3]]
        issues = pipeline.scene_issues(narration, skipped, "short")
        self.assertTrue(any("cover" in i for i in issues))


if __name__ == "__main__":
    unittest.main()

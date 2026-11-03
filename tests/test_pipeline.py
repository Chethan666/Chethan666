import json
import random
import unittest

from story_factory import db, dedup, learning, pipeline, tournament
from story_factory.llm import BudgetExceeded, LLMError, ModelSpec, UsageRecorder, parse_json
from story_factory.router import Router
from tests.fakes import Brain, FakeClient, make_context, premise, test_config


def add_legend(conn, n: int, summary: str = "A spirit that walks the old roads at night.") -> int:
    return db.upsert_inspiration(conn, source="wikipedia", source_id=str(n), kind="folklore",
                                 title=f"Legend {n}", url="", summary=summary, region="Mexico",
                                 license="CC BY-SA")


class WritersRoomTests(unittest.TestCase):
    def test_full_flow_approves_a_long_story(self):
        ctx, brain = make_context()
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "approved", outcome.detail)
        story = db.get_story(ctx.conn, outcome.story_id)
        self.assertTrue(story["narration"].endswith("?"))
        self.assertTrue(18 <= len(json.loads(story["scenes"])) <= 45)
        self.assertEqual(json.loads(story["packaging"])["titles"][0], "The Extra Passenger")
        self.assertTrue(json.loads(story["beat_sheet"])["beats"])
        self.assertIn("chunk_interest", json.loads(story["audience_report"]))
        self.assertEqual(json.loads(story["pronunciations"])[0]["say_as"], "yoh-ROH-nah")
        authors = json.loads(story["author_models"])
        self.assertEqual(set(authors) >= {"pitch", "outline", "draft", "polisher"}, True)
        stages = [entry["stage"] for entry in json.loads(story["production_log"])]
        for stage in ("pitch", "outline_review", "draft_contest", "editors", "audience", "polish", "final_check"):
            self.assertIn(stage, stages)
        self.assertEqual(brain.violations, [])

    def test_judges_never_score_their_own_family(self):
        ctx, brain = make_context()
        pipeline.generate_story(ctx, "hollow_world")
        self.assertTrue(brain.tasks("score_pitch"))
        self.assertEqual(brain.violations, [])
        # The anthropic-written draft is reviewed only by other families.
        reviewers = {label.split(":")[0] for label in brain.tasks("review")}
        self.assertNotIn("openai", reviewers)  # the winning draft came from writer-b (openai)

    def test_best_draft_wins_the_contest(self):
        ctx, _ = make_context()
        outcome = pipeline.generate_story(ctx, "hollow_world")
        authors = json.loads(db.get_story(ctx.conn, outcome.story_id)["author_models"])
        self.assertEqual(authors["draft"], "openai:writer-b")

    def test_runner_up_pitches_are_shelved_then_developed_first(self):
        ctx, brain = make_context()
        first = pipeline.generate_story(ctx, "hollow_world")
        shelved = db.status_counts(ctx.conn).get("shelved", 0)
        self.assertGreater(shelved, 0)
        pitch_calls = len(brain.tasks("pitches"))
        second = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(second.status, "approved", second.detail)
        self.assertEqual(len(brain.tasks("pitches")), pitch_calls)  # no new pitching needed
        self.assertEqual(db.status_counts(ctx.conn).get("shelved", 0), shelved - 1)
        self.assertNotEqual(first.detail, second.detail)

    def test_weak_pitches_are_not_shelved(self):
        ctx, brain = make_context()
        brain.pitch_scores = lambda title: 5
        pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(db.status_counts(ctx.conn).get("shelved", 0), 0)

    def test_one_pitcher_failing_does_not_stop_the_room(self):
        ctx, brain = make_context()
        brain.failing.add("google:writer-c")
        self.assertEqual(pipeline.generate_story(ctx, "hollow_world").status, "approved")

    def test_repeated_pitches_are_rejected(self):
        ctx, brain = make_context()
        first = premise(1, "anthropic")
        brain.pitch_queue.extend([[first], [], []])
        self.assertEqual(pipeline.generate_story(ctx, "sixty_seconds").detail, first["title"])
        same_meaning = dict(premise(2, "openai"), synopsis=" ".join(reversed(first["synopsis"].rstrip(".").split())) + ".")
        renamed = dict(premise(3, "google"), title=first["title"] + "!")
        fresh = premise(4, "openai")
        brain.judge_same = True  # the reworded pitch is borderline similar, so the AI judge decides
        brain.pitch_queue.extend([[same_meaning], [renamed], [fresh]])
        outcome = pipeline.generate_story(ctx, "sixty_seconds")
        self.assertEqual(outcome.detail, fresh["title"])

    def test_twin_pitches_in_one_round_are_collapsed(self):
        ctx, brain = make_context()
        twin = premise(5, "openai")
        brain.pitch_queue.extend([[twin], [dict(twin, title=twin["title"] + " II")], []])
        pipeline.generate_story(ctx, "sixty_seconds")
        titles = [r["title"] for r in db.all_stories(ctx.conn)]
        self.assertEqual(len(titles), 1)

    def test_editors_loop_revises_then_rejects(self):
        ctx, brain = make_context()
        brain.review_scores = 5
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "rejected")
        self.assertIn("raise hook", outcome.detail)
        self.assertEqual(len(brain.tasks("revise")), ctx.cfg["quality"]["max_revisions"])

    def test_audience_weak_parts_are_rewritten(self):
        ctx, brain = make_context()
        brain.audience_interest = lambda chunk: 4 if chunk == 2 else 9
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "approved", outcome.detail)
        report = json.loads(db.get_story(ctx.conn, outcome.story_id)["audience_report"])
        self.assertEqual(report["weak_chunks"], [2])
        self.assertEqual(report["rewrite"], "kept")
        self.assertTrue(brain.tasks("rewrite_chunks"))

    def test_polish_that_changes_the_story_is_discarded(self):
        ctx, brain = make_context()
        brain.overrides["polish"] = lambda prompt, spec: dict(brain.make_script(True), pronunciations=[])
        outcome = pipeline.generate_story(ctx, "hollow_world")
        story = db.get_story(ctx.conn, outcome.story_id)
        log = json.loads(story["production_log"])
        self.assertIn({"stage": "polish", "kept": False}, log)

    def test_final_check_issue_is_fixed_or_rejected(self):
        ctx, brain = make_context()
        brain.final_issues.extend([["wrong region for the legend"]])
        self.assertEqual(pipeline.generate_story(ctx, "hollow_world").status, "approved")
        ctx, brain = make_context()
        brain.final_issues.extend([["problem"]] * 10)
        outcome = pipeline.generate_story(ctx, "hollow_world")
        self.assertEqual(outcome.status, "rejected")
        self.assertIn("final check", outcome.detail)

    def test_model_failure_mid_story_does_not_leave_it_stuck(self):
        ctx, brain = make_context()

        def down(prompt, spec):
            raise LLMError("HTTP 503")

        brain.overrides["beat_sheet"] = down
        with self.assertRaises(LLMError):
            pipeline.generate_story(ctx, "sixty_seconds")
        self.assertEqual(db.status_counts(ctx.conn).get("rejected"), 1)
        del brain.overrides["beat_sheet"]
        self.assertEqual(pipeline.generate_story(ctx, "sixty_seconds").status, "approved")


class EnhancementTests(unittest.TestCase):
    def test_hook_lab_replaces_the_opening_when_judges_prefer_an_alternative(self):
        ctx, brain = make_context()
        brain.prefer_opening = "The knocking started at 3:07"
        outcome = pipeline.generate_story(ctx, "hollow_world")
        story = db.get_story(ctx.conn, outcome.story_id)
        self.assertTrue(story["body"].startswith("The knocking started at 3:07"))
        self.assertEqual(story["hook_line"], "The knocking started at 3:07 every night.")

    def test_hook_lab_keeps_the_original_on_a_tie(self):
        ctx, brain = make_context()
        outcome = pipeline.generate_story(ctx, "hollow_world")
        log = json.loads(db.get_story(ctx.conn, outcome.story_id)["production_log"])
        self.assertIn({"stage": "hook_lab", "candidates": 3, "replaced": False}, log)

    def test_title_contest_orders_titles(self):
        ctx, brain = make_context()
        brain.prefer_title = "Seat Ten"
        outcome = pipeline.generate_story(ctx, "sixty_seconds")
        titles = json.loads(db.get_story(ctx.conn, outcome.story_id)["packaging"])["titles"]
        self.assertEqual(titles[0], "Seat Ten")
        self.assertEqual(len(titles), 4)

    def test_cast_is_stored_and_recurring_looks_stay_fixed(self):
        ctx, brain = make_context()
        first = pipeline.generate_story(ctx, "hollow_world")
        cast = json.loads(db.get_story(ctx.conn, first.story_id)["cast"])
        self.assertEqual({m["name"] for m in cast}, {"Protagonist", "The Storyteller"})
        canon = dict(db.known_looks(ctx.conn))["The Storyteller"]
        brain.cast_override = lambda names: [{"name": n, "look": "A completely different look"} for n in names]
        second = pipeline.generate_story(ctx, "hollow_world")
        cast = {m["name"]: m["look"] for m in json.loads(db.get_story(ctx.conn, second.story_id)["cast"])}
        self.assertEqual(cast["The Storyteller"], canon)
        self.assertEqual(cast["Protagonist"], "A completely different look")

    def test_scene_plan_with_unknown_character_is_retried(self):
        ctx, brain = make_context()
        attempts = []
        original = brain._scene_plan

        def plan(prompt, spec):
            attempts.append(prompt)
            result = original(prompt, spec)
            if len(attempts) == 1:
                result["scenes"][0]["characters"] = ["Nobody We Cast"]
            return result

        brain.overrides["scene_plan"] = plan
        self.assertEqual(pipeline.generate_story(ctx, "sixty_seconds").status, "approved")
        self.assertIn("missing from the cast", attempts[1])

    def test_costs_are_attributed_to_stories_and_reported(self):
        ctx, brain = make_context()
        outcome = pipeline.generate_story(ctx, "sixty_seconds")
        ctx.router.recorder.record("write_draft", ModelSpec("anthropic", "writer-a"), 10, 10)  # outside a story
        ctx.router.recorder.story_id = outcome.story_id
        ctx.router.recorder.record("write_draft", ModelSpec("anthropic", "writer-a"), 10, 10)
        ctx.router.recorder.story_id = None
        report = db.stats(ctx.conn)
        self.assertEqual(report["cost_per_approved_story"][0]["stories"], 1)
        self.assertEqual(report["draft_wins_by_model"], {"openai:writer-b": 1})
        self.assertTrue(report["spend_by_model"])


class AWSEmbedAndReportTests(unittest.TestCase):
    def test_titan_embeddings_request(self):
        import io
        import os
        from unittest import mock

        from story_factory import providers

        captured = {}

        def fake_urlopen(request, timeout):
            captured.update(url=request.full_url, auth=request.get_header("Authorization"),
                            body=json.loads(request.data))
            return io.BytesIO(json.dumps({"embedding": [0.1, 0.2], "inputTextTokenCount": 3}).encode())

        cfg = test_config()
        with mock.patch.dict(os.environ, {"AWS_BEARER_TOKEN_BEDROCK": "k", "AWS_REGION": "us-west-2"}), \
                mock.patch("urllib.request.urlopen", fake_urlopen):
            spec = ModelSpec.parse("aws:amazon.titan-embed-text-v2:0")
            client = providers.build(spec, cfg, UsageRecorder(db.connect(":memory:"), cfg))
            self.assertIsInstance(client, providers.AWSEmbedClient)
            self.assertEqual(client.embed("hello there"), [0.1, 0.2])
        self.assertEqual(spec.family, "amazon")
        self.assertEqual(captured["url"], "https://bedrock-runtime.us-west-2.amazonaws.com/model/"
                                          "amazon.titan-embed-text-v2%3A0/invoke")
        self.assertEqual(captured["auth"], "Bearer k")
        self.assertEqual(captured["body"]["inputText"], "hello there")

    def test_report_summarizes_the_latest_story(self):
        from story_factory.__main__ import report

        ctx, _ = make_context()
        outcome = pipeline.generate_story(ctx, "sixty_seconds")
        row = db.get_story(ctx.conn, outcome.story_id)
        text = report(ctx.conn, row)
        self.assertIn(f"Story #{outcome.story_id}", text)
        self.assertIn("**Status:** approved", text)
        self.assertIn("Titles (best first)", text)
        self.assertIn("Stages run:", text)

    def test_local_hash_embedder(self):
        from story_factory import providers

        cfg = test_config()
        embedder = providers.build(ModelSpec.parse("local:hash"), cfg, UsageRecorder(db.connect(":memory:"), cfg))
        a = embedder.embed("The night bus had one passenger too many.")
        b = embedder.embed("One passenger too many rode the night bus.")
        c = embedder.embed("A lighthouse keeper counts the waves until dawn.")
        self.assertGreater(dedup.cosine(a, b), dedup.cosine(a, c))
        self.assertAlmostEqual(dedup.cosine(a, a), 1.0)

    def test_aws_open_config_has_independent_judges(self):
        from story_factory.config import load_config

        cfg = load_config("config.aws-open.toml")
        families = {ModelSpec.parse(t).family for t in cfg["roles"]["judges"]}
        self.assertEqual(len(families), 4)
        writer_families = {ModelSpec.parse(t).family for t in cfg["roles"]["writers"]}
        self.assertTrue(families - writer_families)  # someone can always judge a writer

    def test_aws_only_config_loads(self):
        from story_factory.config import load_config

        cfg = load_config("config.aws-only.toml")
        for role, specs in cfg["roles"].items():
            for text in specs:
                self.assertEqual(ModelSpec.parse(text).provider, "aws", role)
        families = {ModelSpec.parse(t).family for t in cfg["roles"]["judges"]}
        self.assertEqual(families, {"anthropic", "openai", "xai"})


class LegendTests(unittest.TestCase):
    def test_legend_is_researched_once_and_told_once(self):
        ctx, brain = make_context()
        add_legend(ctx.conn, 1)
        first = pipeline.generate_story(ctx, "legends_retold")
        self.assertEqual(first.status, "approved", first.detail)
        self.assertEqual(len(brain.tasks("dossier_research")), 1)
        self.assertEqual(db.status_counts(ctx.conn).get("shelved", 0), 0)
        second = pipeline.generate_story(ctx, "legends_retold")
        self.assertEqual(second.status, "skipped")

    def test_legend_without_web_search_uses_wikipedia_notes(self):
        ctx, brain = make_context()
        add_legend(ctx.conn, 1)
        ctx.router.one("researcher").supports_web_search = False
        self.assertEqual(pipeline.generate_story(ctx, "legends_retold").status, "approved")
        self.assertEqual(brain.tasks("dossier_research"), [])
        self.assertEqual(len(brain.tasks("dossier")), 1)

    def test_copied_legend_text_is_rejected(self):
        ctx, brain = make_context()
        text = "the weeping woman wanders the riverbank at night searching for her lost children"
        add_legend(ctx.conn, 1, text)

        def copying(prompt, spec):
            script = brain.make_script(True)
            script["body"] = text.capitalize() + ". " + script["body"]
            return script

        brain.overrides["write_draft"] = copying
        brain.overrides["revise"] = copying
        outcome = pipeline.generate_story(ctx, "legends_retold")
        self.assertEqual(outcome.status, "rejected")
        self.assertIn("original words", outcome.detail)


class BacklogTests(unittest.TestCase):
    def test_fill_reaches_targets_without_repeats(self):
        ctx, _ = make_context()
        ctx.cfg["backlog"] = {"hollow_world": 3, "legends_retold": 1, "sixty_seconds": 3}
        add_legend(ctx.conn, 1)
        outcomes = pipeline.fill_backlog(ctx, max_new=20)
        self.assertEqual(sum(o.status == "approved" for o in outcomes), 7)
        self.assertEqual(db.backlog_counts(ctx.conn), {"hollow_world": 3, "legends_retold": 1, "sixty_seconds": 3})
        live = db.live_stories(ctx.conn)
        self.assertEqual(len({r["title_norm"] for r in live}), len(live))
        self.assertEqual(len({r["recipe_sig"] for r in live}), len(live))

    def test_fill_stops_at_budget(self):
        ctx, brain = make_context()

        def broke(prompt, spec):
            raise BudgetExceeded("cap reached")

        brain.overrides["pitches"] = broke
        outcomes = pipeline.fill_backlog(ctx, max_new=5)
        self.assertEqual(len(outcomes), 1)
        self.assertIn("cap reached", outcomes[0].detail)


class LearningTests(unittest.TestCase):
    def test_weights_favor_ingredients_that_retain_viewers(self):
        conn = db.connect(":memory:")
        for i in range(10):
            sid = db.insert_story(conn, kind="long", series="hollow_world", status="published",
                                  title=f"T{i}", title_norm=f"t{i}", setting="night bus" if i < 5 else "lighthouse",
                                  core_fear="dark", twist="ghost", format="classic", ending="twist",
                                  recipe_sig=f"s{i}")
            db.record_performance(conn, sid, views=1000, avg_view_percentage=70.0 if i < 5 else 30.0)
        weights = learning.ingredient_weights(conn)
        self.assertGreater(weights["setting"]["night bus"], 1.0)
        self.assertLess(weights["setting"]["lighthouse"], 1.0)

    def test_no_weights_without_enough_data(self):
        self.assertEqual(learning.ingredient_weights(db.connect(":memory:")), {})

    def test_learnings_reach_future_pitches(self):
        ctx, brain = make_context()
        for i in range(12):
            sid = db.insert_story(ctx.conn, kind="long", series="hollow_world", status="published",
                                  title=f"Old {i}", title_norm=f"old {i}", recipe_sig=f"old{i}")
            db.record_performance(ctx.conn, sid, views=100 * i, avg_view_percentage=float(i * 5))
        self.assertEqual(learning.learn(ctx.conn, ctx.router), ["Night-shift settings keep viewers longest."])
        seen = []
        original = brain._pitches
        brain.overrides["pitches"] = lambda prompt, spec: (seen.append(prompt), original(prompt, spec))[1]
        pipeline.generate_story(ctx, "hollow_world")
        self.assertIn("Night-shift settings keep viewers longest.", seen[0])


class ComponentTests(unittest.TestCase):
    def test_tournament_needs_agreement_in_both_orders(self):
        run = lambda fn, jobs: [fn(j) for j in jobs]
        always_first = lambda judge, a, b: "A"  # pure position bias
        ranked = tournament.rank(["x", "y"], lambda a, b: ["j"], always_first, run, prior={"x": 1, "y": 2}.get)
        self.assertEqual([r.points for r in ranked], [0.5, 0.5])
        self.assertEqual(ranked[0].item, "y")  # tie broken by prior
        prefers_x = lambda judge, a, b: "A" if a == "x" else "B"
        ranked = tournament.rank(["x", "y", "z"], lambda a, b: ["j"], prefers_x, run)
        self.assertEqual(ranked[0].item, "x")

    def test_router_judges_skip_author_families(self):
        brain = Brain()
        router = Router(test_config(), db.connect(":memory:"), factory=lambda spec: FakeClient(spec, brain))
        self.assertEqual({j.spec.family for j in router.judges(["anthropic"])}, {"openai", "google", "xai"})
        self.assertEqual({j.spec.family for j in router.judges(["anthropic", "openai"])}, {"google", "xai"})
        self.assertEqual(len(router.judges(["anthropic", "openai", "google", "xai"])), 4)

    def test_model_spec(self):
        spec = ModelSpec.parse("anthropic:claude-fable-5-1@xhigh")
        self.assertEqual((spec.family, spec.model, spec.effort), ("anthropic", "claude-fable-5-1", "xhigh"))
        with self.assertRaises(ValueError):
            ModelSpec.parse("acme:thing")

    def test_budget_cap(self):
        cfg = test_config()
        conn = db.connect(":memory:")
        cfg["budget"]["monthly_cap_usd"] = 18.0
        recorder = UsageRecorder(conn, cfg)
        recorder.check()
        cfg["prices"]["big"] = [1_000_000.0, 0.0]
        recorder.record("x", ModelSpec("anthropic", "big"), 20, 0)
        with self.assertRaises(BudgetExceeded):
            recorder.check()
        cfg["budget"]["monthly_cap_usd"] = 0
        UsageRecorder(conn, cfg).check()  # no cap configured

    def test_claim_hands_out_each_story_once(self):
        ctx, _ = make_context()
        a = pipeline.generate_story(ctx, "sixty_seconds").story_id
        b = pipeline.generate_story(ctx, "sixty_seconds").story_id
        self.assertEqual(db.claim_next(ctx.conn, "short", "w1")["id"], a)
        self.assertEqual(db.claim_next(ctx.conn, "short", "w2")["id"], b)
        self.assertIsNone(db.claim_next(ctx.conn, "short", "w3"))

    def test_recipe_unique_index(self):
        conn = db.connect(":memory:")
        fields = dict(kind="long", series="hollow_world", status="approved", title="A", title_norm="a",
                      recipe_sig="x|y|z")
        db.insert_story(conn, **fields)
        with self.assertRaises(Exception):
            db.insert_story(conn, **dict(fields, title="B", title_norm="b", status="shelved"))
        db.insert_story(conn, **dict(fields, title="C", title_norm="c", status="rejected"))

    def test_parse_json_accepts_fences(self):
        self.assertEqual(parse_json('```json\n{"a": 1}\n```'), {"a": 1})
        with self.assertRaises(LLMError):
            parse_json("[1, 2]")


class ScriptCheckTests(unittest.TestCase):
    def test_claims_true_and_cliches_are_caught(self):
        script = Brain().make_script(long=False)
        script["body"] += " This is a true story. Little did they know."
        issues = pipeline.script_issues(script, "short", test_config())
        self.assertTrue(any("never claim" in i for i in issues))
        self.assertTrue(any("cliche" in i for i in issues))

    def test_aggregate_reviews_uses_median_and_any_flag(self):
        base = {c: 9 for c in ("hook", "tension", "emotional_payoff", "originality", "prose_for_the_ear",
                               "drawability", "follows_recipe")}
        reviews = [{"scores": dict(base, hook=3), "flags": {"gore": True}, "notes": ["a"]},
                   {"scores": base, "flags": {}, "notes": ["a", "b"]},
                   {"scores": base, "flags": {}, "notes": []}]
        agg = pipeline.aggregate_reviews(reviews)
        self.assertEqual(agg["scores"]["hook"], 9)
        self.assertEqual(agg["flags"], ["gore"])
        self.assertEqual(agg["notes"], ["a", "b"])

    def test_chunk_text_keeps_sentences_whole(self):
        chunks = pipeline.chunk_text("One two three. Four five six. Seven eight.", 4)
        self.assertEqual(chunks, ["One two three. Four five six.", "Seven eight."])

    def test_scene_coverage(self):
        parts = ["One.", "Two three.", "Four five six.", "Seven eight nine ten?"]
        good = [{"text": t, "visual": "v", "location": "l", "key_moment": i == 0} for i, t in enumerate(parts)]
        narration = " ".join(parts)
        self.assertEqual(pipeline.scene_issues(narration, good, "short"), [])
        self.assertTrue(pipeline.scene_issues(narration, [good[0], good[2], good[1], good[3]], "short"))
        self.assertTrue(any("cover" in i for i in pipeline.scene_issues(narration, [good[0], good[1], good[3]], "short")))


if __name__ == "__main__":
    unittest.main()

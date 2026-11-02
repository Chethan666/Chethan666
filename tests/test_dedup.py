import random
import unittest

from story_factory import catalog, dedup
from tests.fakes import test_config


class SimilarityTests(unittest.TestCase):
    def test_normalize_and_titles(self):
        self.assertEqual(dedup.normalize("The Night-Bus, Again!"), "the night bus again")
        self.assertGreater(dedup.title_similarity("The Last Night Bus", "The last night-bus"), 0.95)
        self.assertLess(dedup.title_similarity("The Last Night Bus", "Rules for the Lighthouse"), 0.6)

    def test_lexical_similarity(self):
        a = "A nurse on the night shift hears her dead mother calling from ward nine."
        self.assertEqual(dedup.lexical_similarity(a, a), 1.0)
        self.assertLess(dedup.lexical_similarity(a, "A ferry captain counts one passenger too many."), 0.1)

    def test_cosine(self):
        self.assertAlmostEqual(dedup.cosine([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(dedup.cosine([1, 0], [0, 1]), 0.0)
        self.assertEqual(dedup.cosine([0, 0], [1, 0]), 0.0)

    def test_copied_passages(self):
        source = "The weeping woman wanders the riverbank at night searching for her lost children forever."
        copied = "Locals say the weeping woman wanders the riverbank at night searching for her lost children."
        own_words = "By the river, a woman cries for children she lost long ago."
        self.assertTrue(dedup.copied_passages(copied, [source], 8))
        self.assertFalse(dedup.copied_passages(own_words, [source], 8))


class RecipeTests(unittest.TestCase):
    def setUp(self):
        self.rules = test_config()["dedup"]
        self.series = catalog.SERIES["hollow_world"]

    def test_never_reuses_a_signature(self):
        rng = random.Random(1)
        used: set[str] = set()
        recent: list[dict] = []
        for _ in range(300):
            recipe = catalog.pick_recipe(rng, self.series, recent, used, self.rules)
            self.assertNotIn(recipe.signature, used)
            used.add(recipe.signature)
            recent.insert(0, {"setting": recipe.setting, "core_fear": recipe.core_fear, "twist": recipe.twist})

    def test_cooldowns(self):
        rng = random.Random(2)
        recent: list[dict] = []
        for _ in range(100):
            recipe = catalog.pick_recipe(rng, self.series, recent, set(), self.rules)
            window = recent[: self.rules["setting_cooldown"]]
            self.assertNotIn(recipe.setting, {r["setting"] for r in window})
            self.assertNotIn(recipe.twist, {r["twist"] for r in recent[: self.rules["twist_cooldown"]]})
            recent.insert(0, {"setting": recipe.setting, "core_fear": recipe.core_fear, "twist": recipe.twist})

    def test_series_constraints(self):
        short = catalog.SERIES["sixty_seconds"]
        recipe = catalog.pick_recipe(random.Random(3), short, [], set(), self.rules)
        self.assertIn(recipe.format, short.formats)
        self.assertIn(recipe.ending, short.endings)


if __name__ == "__main__":
    unittest.main()

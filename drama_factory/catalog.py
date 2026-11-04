"""Ingredients for drama pitches: each pitcher gets a different trope recipe, so the room
pitches genuinely different series instead of variations of one idea."""

from __future__ import annotations

import random
from dataclasses import dataclass

TROPES = (
    "a hidden identity revealed at the worst moment", "revenge on the family that cast them out",
    "a contract relationship that becomes real", "a second chance at a life they lost",
    "an impostor living someone else's life", "a war over an inheritance", "a curse that passes to whoever lies",
    "a time loop that resets at midnight", "a deal with something that isn't human", "a sibling swapped at birth",
    "a witness nobody believes", "a secret marriage", "a stranger who knows the future",
    "a returning presumed-dead spouse", "a debt that must be paid in years of life",
)
SETTINGS = (
    "a hospital on the night shift", "a luxury hotel in a storm", "a small island town", "a corporate tower",
    "a family estate", "a late-night radio station", "a cruise ship", "a boarding school", "a mountain village",
    "a big-city apartment block", "a vineyard", "a remote research station",
)
TWISTS = (
    "the ally is the one behind it", "the victim planned everything", "it has happened before",
    "the hero is not who they think they are", "the dead are not the ones who died",
    "love was the trap", "the secret protects someone innocent",
)


@dataclass(frozen=True)
class Recipe:
    trope: str
    setting: str
    twist: str

    @property
    def signature(self) -> str:
        return f"{self.trope}|{self.setting}|{self.twist}"

    def describe(self) -> str:
        return f"trope: {self.trope}; setting: {self.setting}; twist to build toward: {self.twist}"


def recipes(rng: random.Random, count: int, used: set[str]) -> list[Recipe]:
    """`count` different recipes not used by earlier series."""
    out: list[Recipe] = []
    for _ in range(count * 50):
        r = Recipe(rng.choice(TROPES), rng.choice(SETTINGS), rng.choice(TWISTS))
        if r.signature not in used and all(r.trope != o.trope for o in out):
            out.append(r)
        if len(out) == count:
            break
    return out

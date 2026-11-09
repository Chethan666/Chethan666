"""Story ingredients and series definitions.

Every story starts from a recipe drawn from these lists. The combinations run
into the millions, and cooldowns stop the same setting, fear or twist from
appearing in consecutive stories.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# Places and people from today's India (the channel tells Indian horror stories in English).
SETTINGS = [
    "a night bus on a ghat road",
    "a highway dhaba at 3 AM",
    "a government hospital ward on the night shift",
    "a college hostel emptied for the holidays",
    "the lift of an old apartment block in Mumbai",
    "a sleeper coach on an overnight train",
    "a newly rented paying-guest room in Bengaluru",
    "a lighthouse on the Konkan coast",
    "a river ferry crossing at dusk",
    "a petrol pump on an empty state highway",
    "a hill-station guest house in the off-season",
    "the last local train of the night",
    "a railway waiting room at 2 AM",
    "an old single-screen cinema hall",
    "a call centre's empty night floor in Gurugram",
    "a palace museum during the night guard's rounds",
    "an auto-rickshaw on a late-night ride",
    "the stairwell of a high-rise under construction",
    "a mountain road in Himachal after a landslide",
    "a houseboat on the Kerala backwaters",
    "a small pilgrim town after the festival crowds leave",
    "a British-era bungalow in a hill cantonment",
    "a joint-family house during a power cut",
    "a crumbling haveli in Rajasthan",
    "an old stepwell outside a village",
    "a sugarcane field at harvest time",
    "a fishing village during the monsoon",
    "a tea estate bungalow in Assam",
    "a hospital's record room in the basement",
    "a ropeway car stopped mid-air",
    "a village school after dark",
    "a toll plaza on an empty expressway",
    "an old-age home's night corridor",
    "a wedding hall after the guests have left",
    "a warehouse by the docks in Kolkata",
    "a farmhouse at the edge of a forest in the Western Ghats",
]

CORE_FEARS = [
    "being watched by something patient",
    "total isolation",
    "losing someone you love",
    "being replaced by a copy",
    "not being believed",
    "the wrong person waiting at home",
    "a night that keeps repeating",
    "a familiar place turning strange",
    "being followed",
    "what waits in the dark",
    "being forgotten",
    "guilt that comes back",
    "a rule from the elders that nobody explains",
    "a trusted voice that lies",
    "a door that should stay closed",
    "a voice that calls your name at night",
    "a promise to the dead that was broken",
    "what the family has kept hidden",
]

PROTAGONISTS = [
    "a night-shift nurse at a government hospital",
    "a long-haul truck driver",
    "a newly hired hotel receptionist",
    "a grieving widower",
    "a first-year engineering student",
    "a museum night guard",
    "a food-delivery rider",
    "a retired schoolteacher",
    "a single father",
    "an auto-rickshaw driver",
    "a translator travelling alone",
    "an office cleaner on the night shift",
    "a late-night radio jockey",
    "a lighthouse keeper",
    "a train ticket examiner",
    "a young doctor posted to a village health centre",
    "a software engineer new to the city",
    "a farmer's eldest daughter",
    "an elderly boatman",
    "a call centre worker",
    "a new bride in her in-laws' house",
    "a security guard at a construction site",
    "a medical intern on her first night duty",
    "a postman in a hill village",
]

FORMATS = {
    "classic": "a classic third-person narrative",
    "rules": "a list of rules the protagonist must follow, revealed as the story goes",
    "second_person": "told in the second person, putting the viewer in the story ('you')",
    "two_timelines": "two timelines that meet at the end",
    "logbook": "narrated through the protagonist's own logbook or voice notes",
    "heartbreak_ghost": "a ghost story whose fear turns into heartbreak",
}

TWISTS = [
    "the protagonist was the ghost all along",
    "the helpful stranger is the real threat",
    "the threat was trying to protect them",
    "it already happened years ago",
    "the protagonist caused it",
    "the loop has run many times before",
    "a case of mistaken identity",
    "it was never a person",
    "the rules were written by the thing itself",
    "the warning came from their future self",
    "the place remembers everyone who stayed",
    "the one they were mourning never left",
]

ENDINGS = {
    "terror": "ends in pure dread, the danger still out there",
    "twist": "ends on a twist that recolors everything before it",
    "justice": "ends with the cruel getting what they earned",
    "heartbreak": "ends in heartbreak and love, more sad than scary",
    "bittersweet": "the protagonist escapes, but pays a price",
    "ambiguous": "ends on an open question the audience will argue about",
}


@dataclass(frozen=True)
class Series:
    name: str
    kind: str  # "long" or "short"
    title: str
    description: str
    formats: tuple[str, ...]
    endings: tuple[str, ...]
    uses_world: bool = False
    uses_legend: bool = False


SERIES: dict[str, Series] = {
    "hollow_world": Series(
        name="hollow_world",
        kind="long",
        title="Tales from the Hollow",
        description=(
            "Original stories set in the channel's shared world: wherever a line "
            "runs between places (roads, rails, rivers, corridors, the threshold of a "
            "door) there is a hollow, and something waits in it. Stories happen across "
            "India, from its biggest cities to its smallest villages."
        ),
        formats=tuple(FORMATS),
        endings=tuple(ENDINGS),
        uses_world=True,
    ),
    "legends_retold": Series(
        name="legends_retold",
        kind="long",
        title="Legends Retold",
        description=(
            "A real traditional legend from India, retold faithfully and respectfully, "
            "framed by an original modern story of someone who meets it."
        ),
        formats=("classic", "two_timelines", "logbook", "second_person"),
        endings=tuple(ENDINGS),
        uses_legend=True,
    ),
    "sixty_seconds": Series(
        name="sixty_seconds",
        kind="short",
        title="60-Second Horror",
        description="A complete micro horror story in under a minute, ending on a sting.",
        formats=("classic", "second_person", "rules", "heartbreak_ghost"),
        endings=("terror", "twist", "heartbreak", "ambiguous"),
    ),
}


@dataclass(frozen=True)
class Recipe:
    setting: str
    core_fear: str
    protagonist: str
    format: str
    twist: str
    ending: str

    @property
    def signature(self) -> str:
        # The combination that must never repeat.
        return f"{self.setting}|{self.core_fear}|{self.twist}"


def _choose(rng: random.Random, values: list[str] | tuple[str, ...], weights: dict[str, float] | None) -> str:
    if not weights:
        return rng.choice(values)
    return rng.choices(values, weights=[weights.get(v, 1.0) for v in values])[0]


def pick_recipe(
    rng: random.Random,
    series: Series,
    recent: list[dict],
    used_signatures: set[str],
    dedup_cfg: dict,
    weights: dict[str, dict[str, float]] | None = None,
    max_draws: int = 2000,
) -> Recipe:
    """Draw a recipe whose signature is new and that respects the cooldowns.

    `recent` holds the latest non-rejected stories, newest first, as dicts with
    `setting`, `core_fear` and `twist` keys. `weights` (from real audience
    retention) makes better-performing ingredients more likely.
    """
    weights = weights or {}
    blocked_settings = {s["setting"] for s in recent[: dedup_cfg["setting_cooldown"]]}
    blocked_twists = {s["twist"] for s in recent[: dedup_cfg["twist_cooldown"]]}
    blocked_fears = {s["core_fear"] for s in recent[: dedup_cfg["fear_cooldown"]]}
    for _ in range(max_draws):
        recipe = Recipe(
            setting=_choose(rng, SETTINGS, weights.get("setting")),
            core_fear=_choose(rng, CORE_FEARS, weights.get("core_fear")),
            protagonist=rng.choice(PROTAGONISTS),
            format=_choose(rng, series.formats, weights.get("format")),
            twist=_choose(rng, TWISTS, weights.get("twist")),
            ending=_choose(rng, series.endings, weights.get("ending")),
        )
        if (
            recipe.signature not in used_signatures
            and recipe.setting not in blocked_settings
            and recipe.twist not in blocked_twists
            and recipe.core_fear not in blocked_fears
        ):
            return recipe
    raise RuntimeError("No unused recipe found; extend the catalog lists")

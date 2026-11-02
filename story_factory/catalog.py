"""Story ingredients and series definitions.

Every story starts from a recipe drawn from these lists. The combinations run
into the millions, and cooldowns stop the same setting, fear or twist from
appearing in consecutive stories.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

SETTINGS = [
    "a night bus on an empty highway",
    "a roadside motel with one working sign",
    "a hospital ward on the night shift",
    "a backpackers' hostel dormitory",
    "the elevator of an old apartment block",
    "a sleeper car on an overnight train",
    "a newly rented apartment",
    "a lighthouse on a remote coast",
    "a car ferry crossing in fog",
    "a gas station at 3 AM",
    "a ski lodge during a snowstorm",
    "the last subway train of the night",
    "a lakeside campsite",
    "an airport terminal after closing",
    "an old single-screen cinema",
    "a 24-hour laundromat",
    "a museum during the night guard's rounds",
    "a rideshare car on a late pickup",
    "the stairwell of a high-rise",
    "a mountain pass road",
    "a cruise ship's lower decks",
    "a desert diner off the interstate",
    "a polar research station",
    "a call center's empty night floor",
    "a toll booth on a quiet road",
    "a retirement home's night corridor",
    "an empty hotel ballroom",
    "a multistorey parking garage",
    "a storage unit facility",
    "an abandoned shopping mall",
    "a rural farmhouse during harvest",
    "a radio station's overnight broadcast booth",
    "a fishing village during the off-season",
    "a hospital's basement archive",
    "a mountain cable car stopped mid-air",
    "a vineyard cellar",
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
    "rules nobody explains",
    "a trusted voice that lies",
    "a door that should stay closed",
]

PROTAGONISTS = [
    "a night-shift nurse",
    "a long-haul truck driver",
    "a newly hired hotel clerk",
    "a grieving widower",
    "a first-year university student",
    "a museum security guard",
    "a food-delivery rider",
    "a retired schoolteacher",
    "a single father",
    "a rideshare driver",
    "a translator travelling alone",
    "an office cleaner",
    "a radio sound engineer",
    "a lighthouse keeper",
    "a train conductor",
    "a nurse returning to her home town",
    "a backpacker on her first trip abroad",
    "a farmer's eldest daughter",
    "an elderly ferry captain",
    "a call center worker",
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
            "runs between places (roads, rails, corridors, sea lanes) there is a "
            "hollow, and something waits in it. Stories happen anywhere on Earth."
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
            "A real traditional legend, retold faithfully and respectfully, framed "
            "by an original modern story of someone who meets it."
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


def pick_recipe(
    rng: random.Random,
    series: Series,
    recent: list[dict],
    used_signatures: set[str],
    dedup_cfg: dict,
    max_draws: int = 2000,
) -> Recipe:
    """Draw a recipe whose signature is new and that respects the cooldowns.

    `recent` holds the latest non-rejected stories, newest first, as dicts with
    `setting`, `core_fear` and `twist` keys.
    """
    blocked_settings = {s["setting"] for s in recent[: dedup_cfg["setting_cooldown"]]}
    blocked_twists = {s["twist"] for s in recent[: dedup_cfg["twist_cooldown"]]}
    blocked_fears = {s["core_fear"] for s in recent[: dedup_cfg["fear_cooldown"]]}
    for _ in range(max_draws):
        recipe = Recipe(
            setting=rng.choice(SETTINGS),
            core_fear=rng.choice(CORE_FEARS),
            protagonist=rng.choice(PROTAGONISTS),
            format=rng.choice(series.formats),
            twist=rng.choice(TWISTS),
            ending=rng.choice(series.endings),
        )
        if (
            recipe.signature not in used_signatures
            and recipe.setting not in blocked_settings
            and recipe.twist not in blocked_twists
            and recipe.core_fear not in blocked_fears
        ):
            return recipe
    raise RuntimeError("No unused recipe found; extend the catalog lists")

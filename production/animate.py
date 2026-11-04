"""Moving chalk: chosen scenes become real video clips.

A drawn chalk picture goes to an image-to-video model (fal.ai, the same queue API as the
drama pipeline) with a prompt that keeps the chalk-on-black look and asks only for the
movement the scene needs. Key moments come first, then the tensest scenes, within a
budget per video. A clip that fails is simply left as the drawn scene with its camera move.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

TENSE = {"terr": 3, "panic": 3, "shock": 3, "horror": 3, "tense": 2, "fear": 2, "dread": 2, "menac": 2,
         "eerie": 1, "uneas": 1, "ominous": 1}
STYLE = ("Animate this hand-drawn chalk illustration. Keep the exact style: white chalk lines on black "
         "paper, simple stick figures, rough hand-drawn strokes, the only colour is the existing red detail. "
         "Do not add colour, shading, faces, text or new objects. Gentle chalk-line boil.")


def prompt(scene: dict[str, Any], spec: dict[str, Any]) -> str:
    who = ", ".join(f"{f.get('name', 'a figure')} ({f.get('pose', 'stand')})" for f in spec.get("figures", [])[:3])
    camera = {"push_in": "slow push in", "pull_out": "slow pull back", "pan_left": "slow pan left",
              "pan_right": "slow pan right", "shake": "slight handheld shake"}.get(str(spec.get("camera")), "locked camera")
    light = " The light flickers." if spec.get("light") == "flicker" else ""
    return (f"{STYLE} Action: {scene.get('visual', '')}. Figures: {who or 'none'}; they move simply and "
            f"slowly, like a chalk animation. {camera}.{light} No cuts.")


def choose(scenes: list[dict[str, Any]], limit: int) -> list[int]:
    """Key moments first, then the tensest scenes, in story order."""
    if limit <= 0:
        return []
    def tension(scene: dict[str, Any]) -> int:
        mood = str(scene.get("mood", "")).lower()
        return max((v for k, v in TENSE.items() if k in mood), default=0)

    ranked = sorted(range(len(scenes)), key=lambda i: (not scenes[i].get("key_moment"), -tension(scenes[i]), i))
    return sorted(ranked[:limit])


class Animator:
    def __init__(self, fal: Any, model: str, args: dict[str, Any], price_per_second: float,
                 max_usd: float, max_scenes: int, workers: int = 3):
        self.fal, self.model, self.args = fal, model, dict(args)
        self.price, self.max_usd, self.max_scenes, self.workers = price_per_second, max_usd, max_scenes, workers
        self.spent = 0.0
        self.lock = threading.Lock()

    @property
    def seconds(self) -> float:
        return float(self.args.get("duration", 5))

    def _reserve(self) -> bool:
        cost = self.price * self.seconds
        with self.lock:
            if self.max_usd and self.spent + cost > self.max_usd:
                return False
            self.spent += cost
            return True

    def animate(self, picture: Path, scene: dict[str, Any], spec: dict[str, Any], out: Path) -> Path | None:
        from drama_production.media import MediaError, data_uri, download, first_url

        if not self._reserve():
            return None
        try:
            result = self.fal.run(self.model, dict(self.args, image_url=data_uri(picture), prompt=prompt(scene, spec)))
            return download(first_url(result, "video"), out)
        except MediaError as exc:
            log.warning("Scene stays drawn (no moving clip): %s", exc)
            with self.lock:
                self.spent -= self.price * self.seconds  # failed jobs are not billed
            return None

    def run(self, work: Path, scenes: list[dict[str, Any]], direction: list[dict[str, Any]]) -> dict[int, Path]:
        chosen = choose(scenes, self.max_scenes)

        def one(i: int) -> tuple[int, Path | None]:
            return i, self.animate(work / f"scene_{i:03d}_0.png", scenes[i], direction[i], work / f"motion_{i:03d}.mp4")

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            made = dict(pool.map(one, chosen))
        moving = {i: p for i, p in made.items() if p is not None}
        log.info("Moving clips: %d of %d chosen scenes (about $%.2f)", len(moving), len(chosen), self.spent)
        return moving

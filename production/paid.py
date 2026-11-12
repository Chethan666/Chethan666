"""Paid media for the studio: one spending cap, a cache, and the vision checker.

Every paid call goes through `Budget`: it reserves the estimated price first (so parallel
workers never overspend the cap), releases it if the job fails, and records what was spent
in the shared costs table. Results are cached by the exact request, so a rerun of a failed
video pays only for what it has not made yet.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from pathlib import Path
from typing import Any

from production.media import MediaError, data_uri, download, first_url

log = logging.getLogger(__name__)


class OverBudget(MediaError):
    pass


class BalanceEmpty(RuntimeError):
    """The fal.ai account has run out of credit: stop the video rather than finish it in fallbacks."""


def _empty_balance(exc: Exception) -> bool:
    text = str(exc).lower()
    return "exhausted balance" in text or "user is locked" in text


class Budget:
    def __init__(self, cap: float, prices: dict[str, float], conn: Any = None):
        self.cap, self.prices, self.conn = cap, prices, conn
        self.spent = 0.0
        self.held = 0.0
        self.by_task: dict[str, float] = {}
        self.lock = threading.Lock()

    def price(self, model: str, units: float = 1.0) -> float:
        return float(self.prices.get(model, 0.0)) * units

    def left(self) -> float:
        with self.lock:
            return self.cap - self.spent - self.held

    def reserve(self, model: str, units: float = 1.0) -> float:
        usd = self.price(model, units)
        with self.lock:
            if self.cap and self.spent + self.held + usd > self.cap + 1e-9:
                raise OverBudget(f"over the ${self.cap:.2f} cap (spent ${self.spent:.2f})")
            self.held += usd
        return usd

    def settle(self, usd: float, task: str, model: str, ok: bool, units: float = 1.0) -> None:
        with self.lock:
            self.held -= usd
            if ok:
                self.spent += usd
                self.by_task[task] = self.by_task.get(task, 0.0) + usd
        if ok and self.conn is not None and usd:
            from story_factory import db
            try:
                db.record_cost(self.conn, pipeline="video", task=task, model=model, input_units=0,
                               output_units=int(round(units)), usd=usd)
            except Exception as exc:  # noqa: BLE001 - bookkeeping never stops a video
                log.warning("Could not record a cost: %s", exc)


def digest(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


class Studio:
    """fal calls with the cap, the cache and retries."""

    def __init__(self, fal: Any, budget: Budget, cache: Path, cfg: dict[str, Any]):
        self.fal, self.budget, self.cache, self.cfg = fal, budget, cache, cfg
        self.models: dict[str, str] = cfg.get("models", {})
        self.args: dict[str, dict] = cfg.get("model_args", {})
        self.stop_on_empty = True  # an empty balance stops the video while pictures and clips are made
        self.balance_ran_out = False
        cache.mkdir(parents=True, exist_ok=True)

    def model(self, role: str) -> str:
        return self.models[role]

    def call(self, role: str, arguments: dict[str, Any], kind: str, out: Path, task: str,
             units: float = 1.0, key: Any = None, tries: int = 2) -> Path:
        """Run a model (role from the config) and download its `kind` result (image, video,
        audio) to `out`. Cached by the request; paid once."""
        model = self.models[role]
        args = dict(self.args.get(role, {}), **arguments)
        suffix = out.suffix or {"image": ".png", "video": ".mp4", "audio": ".mp3"}[kind]
        cached = self.cache / f"{task}-{digest(model, key if key is not None else args)}{suffix}"
        if cached.exists() and cached.stat().st_size > 0:
            _link(cached, out)
            return out
        last: Exception | None = None
        for attempt in range(tries):
            usd = self.budget.reserve(model, units)
            try:
                result = self.fal.run(model, args)
                url = result_url(result, kind)
                download(url, cached)
            except MediaError as exc:
                self.budget.settle(usd, task, model, ok=False)
                if _empty_balance(exc):
                    self.balance_ran_out = True
                    if not self.stop_on_empty:  # the expensive work is done: finish with free fallbacks
                        raise MediaError(f"fal.ai balance is empty ({task})") from exc
                    raise BalanceEmpty("fal.ai balance is empty: top it up at fal.ai/dashboard/billing, "
                                       "then run the video again (paid so far: $%.2f)" % self.budget.spent) from exc
                last = exc
                log.warning("%s (%s) attempt %d failed: %s", task, model, attempt + 1, str(exc)[:300])
                if "422" in str(exc) or "content" in str(exc).lower():
                    break  # a refused request will not pass on a retry
                continue
            self.budget.settle(usd, task, model, ok=True, units=units)
            _link(cached, out)
            return out
        raise last or MediaError(f"{task}: no result")


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.write_bytes(src.read_bytes())


def result_url(result: dict[str, Any], kind: str) -> str:
    if kind in ("image", "video"):
        return first_url(result, kind)
    for key in ("audio", "audio_file", "audio_url", "output"):
        value = result.get(key)
        if isinstance(value, dict) and value.get("url"):
            return value["url"]
        if isinstance(value, str) and value.startswith("http"):
            return value
    if isinstance(result.get("video"), dict) and result["video"].get("url"):  # video-to-audio models
        return result["video"]["url"]
    raise MediaError(f"no {kind} in result: {str(result)[:300]}")


VISION_SYSTEM = ("You are the picture checker of a chalk-animation horror channel. Every frame must look like "
                 "white chalk drawn on a black chalkboard: simple stick figures, rough chalk strokes, black "
                 "background, and no colour at all except a single blood-red accent. Answer with JSON only.")


def vision_check(studio: Studio, image: Path, picture: str, task: str = "vision") -> tuple[bool, str]:
    """A vision model looks at a picture: is it in the channel's style, and does it show the shot?
    Returns (ok, reason). If the checker itself fails, the picture passes (the pixel check still ran)."""
    if "vision" not in studio.models:
        return True, "no vision model"
    model = studio.models["vision"]
    try:
        usd = studio.budget.reserve(model)
    except OverBudget:
        return True, "no budget left for checks"
    prompt = (f"The shot should show: {picture}\n\nLook at the image. Judge only the drawing style, not small "
              "details of the content. Fail it ONLY if one of these is clearly true: it is a photograph, a 3D "
              "render or a painted, fully coloured illustration instead of a chalk drawing on a dark board; large "
              "areas use a colour other than white chalk and red (green, blue, yellow...); it contains readable "
              "words or a watermark (a short number is fine); it is split into several panels; or it shows "
              "something entirely unrelated to the shot. Small extra details, a second red detail, simple "
              "facial expressions, slightly different poses or framing are all fine. Reply exactly as JSON: "
              '{"ok": true or false, "reason": "short reason"}')
    try:
        result = studio.fal.run(model, dict(studio.args.get("vision", {}), prompt=prompt, system_prompt=VISION_SYSTEM,
                                            image_urls=[data_uri(_small(image))]))
    except MediaError as exc:
        studio.budget.settle(usd, task, model, ok=False)
        log.warning("Vision check unavailable: %s", str(exc)[:200])
        return True, "checker failed"
    studio.budget.settle(usd, task, model, ok=True)
    text = str(result.get("output") or result.get("text") or result)
    match = re.search(r"\{.*\}", text, re.S)
    try:
        verdict = json.loads(match.group(0)) if match else {}
    except ValueError:
        verdict = {}
    if not isinstance(verdict, dict) or "ok" not in verdict:
        return True, "unreadable verdict"
    return bool(verdict.get("ok")), str(verdict.get("reason", ""))[:200]


def _small(image: Path, width: int = 768) -> Path:
    """A smaller JPEG copy for the checker (cheaper and faster to send)."""
    from PIL import Image

    out = image.with_name(image.stem + "_check.jpg")
    with Image.open(image) as im:
        im = im.convert("RGB")
        if im.width > width:
            im = im.resize((width, int(im.height * width / im.width)))
        im.save(out, quality=85)
    return out

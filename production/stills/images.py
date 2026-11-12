"""Painters and the critic.

A picture is asked of each painter in the chain until one passes the critic:

    fal:<role>   a paid model on fal.ai, with the style frame and character sheets as
                 references (Seedream 4.5 edit for shots, Nano Banana 2 as the redraw)
    cloudflare   FLUX.1 schnell on Cloudflare Workers AI: free allowance of 10,000
                 neurons a day (CF_ACCOUNT_ID, CF_API_TOKEN). Text only, square output
                 (cropped to 16:9), so composition is asked to stay central
    pollinations keyless and free, slow (one picture per ~45 s), no guarantee

The critic checks every picture in code (not too bright for the channel, not blank, not
almost black) and then, when a vision model is configured, by eye: the house style, the
right moment, no text or watermark, no panels, no broken anatomy.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from production.media import MediaError, _request, data_uri

from ..look import fit
from ..paid import OverBudget, Studio, _small, digest

log = logging.getLogger(__name__)

CF_URL = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}"
POLLINATIONS = "https://image.pollinations.ai/prompt/{prompt}?width=1280&height=720&nologo=true&seed={seed}"


def still_check(path: Path) -> tuple[bool, str]:
    """Code check of a picture: dark enough for the channel, and something actually painted."""
    with Image.open(path) as im:
        v = np.asarray(im.convert("L").resize((320, 180)), dtype=np.float32) / 255.0
    mean, spread = float(v.mean()), float(v.std())
    if mean > 0.62:
        return False, f"too bright for the channel (mean {mean:.2f})"
    if spread < 0.035:
        return False, "flat or blank"
    if float(np.mean(v < 0.04)) > 0.97:
        return False, "almost black"
    return True, "ok"


CRITIC_SYSTEM = ("You are the picture editor of a horror storytelling channel told in still illustrations. "
                 "Every picture must match the channel's house style and show the narrated moment clearly. "
                 "Answer with JSON only.")


class Critic:
    """Checks a picture in code, then by eye with a vision model when one is configured."""

    def __init__(self, studio: Studio | None, style: str):
        self.studio, self.style = studio, style
        self.lock = threading.Lock()
        self.log: list[str] = []

    def check(self, image: Path, picture: str) -> tuple[bool, str]:
        ok, why = still_check(image)
        if not ok:
            return ok, why
        return self.look(image, picture)

    def look(self, image: Path, picture: str) -> tuple[bool, str]:
        studio = self.studio
        if studio is None or "vision" not in studio.models:
            return True, "no vision model"
        model = studio.models["vision"]
        try:
            usd = studio.budget.reserve(model)
        except OverBudget:
            return True, "no budget left for checks"
        prompt = (f"House style: {self.style}\nThe picture should show: {picture}\n\nLook at the image. Fail it ONLY "
                  "if one of these is clearly true: it is a photograph or looks photographic instead of an "
                  "illustration; it is bright and cheerful rather than dark; it contains readable words, letters "
                  "or a watermark; it is split into panels or is a collage or a character sheet; a person has "
                  "clearly broken anatomy (extra limbs, melted face); or it shows something unrelated to the "
                  'moment. Small differences in detail are fine. Reply exactly as JSON: {"ok": true or false, '
                  '"reason": "short reason"}')
        try:
            result = studio.fal.run(model, dict(studio.args.get("vision", {}), prompt=prompt,
                                                system_prompt=CRITIC_SYSTEM, image_urls=[data_uri(_small(image))]))
        except MediaError as exc:
            studio.budget.settle(usd, "vision", model, ok=False)
            log.warning("Vision check unavailable: %s", str(exc)[:200])
            return True, "checker failed"
        studio.budget.settle(usd, "vision", model, ok=True)
        text = str(result.get("output") or result.get("text") or result)
        match = re.search(r"\{.*\}", text, re.S)
        try:
            verdict = json.loads(match.group(0)) if match else {}
        except ValueError:
            verdict = {}
        if not isinstance(verdict, dict) or "ok" not in verdict:
            return True, "unreadable verdict"
        return bool(verdict.get("ok")), str(verdict.get("reason", ""))[:200]


class Painter:
    """Asks each painter in the chain for a picture until the critic passes one."""

    def __init__(self, studio: Studio | None, cfg: dict[str, Any], work: Path, critic: Critic,
                 http: Any = _request):
        self.studio, self.cfg, self.work, self.critic, self.http = studio, cfg, work, critic, http
        self.cache = work / "free"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.pollinations_lock = threading.Lock()
        self.pollinations_last = 0.0
        self.counts: dict[str, int] = {}
        self.rejections: list[str] = []
        self.lock = threading.Lock()

    def _count(self, key: str) -> None:
        with self.lock:
            self.counts[key] = self.counts.get(key, 0) + 1

    def available(self, step: str) -> bool:
        if step.startswith("fal:"):
            return self.studio is not None and step[4:] in self.studio.models
        if step == "cloudflare":
            return bool(os.environ.get("CF_ACCOUNT_ID") and os.environ.get("CF_API_TOKEN"))
        if step == "pollinations":
            return bool(self.cfg.get("pollinations", True))
        return False

    def paint(self, prompt: str, refs: list[Path], out: Path, key: Any, chain: list[str],
              check: str | None = None) -> tuple[Path | None, str]:
        """The first picture in the chain that the critic passes (or, failing that, the first one that
        passed the code check). `check` is what the picture should show; None skips the critic."""
        doubted: tuple[Path, str] | None = None
        for k, step in enumerate(s for s in chain if self.available(s)):
            raw = self.work / f"raw_{out.stem}_{k}.png"
            try:
                made = self._one(step, prompt, refs, raw, key)
            except OverBudget:
                continue
            except (MediaError, OSError, ValueError) as exc:
                log.warning("%s: %s failed: %s", out.stem, step, str(exc)[:200])
                continue
            if made is None:
                continue
            candidate = self.work / f"fit_{out.stem}_{k}.png"
            try:
                fit(made, candidate)
            except OSError as exc:
                log.warning("%s: %s gave an unreadable picture: %s", out.stem, step, exc)
                continue
            if check is None:
                candidate.replace(out)
                self._count(step)
                return out, step
            ok, why = still_check(candidate)
            if ok:
                doubted = doubted or (candidate, step)
                ok, why = self.critic.look(candidate, check)
            if ok:
                candidate.replace(out)
                self._count(step)
                return out, step
            with self.lock:
                self.rejections.append(f"{out.stem} ({step}): {why}")
            log.info("%s from %s rejected (%s); next painter", out.stem, step, why)
        if doubted is not None:
            doubted[0].replace(out)
            self._count(doubted[1] + " (doubted)")
            return out, doubted[1]
        return None, "none"

    def _one(self, step: str, prompt: str, refs: list[Path], raw: Path, key: Any) -> Path | None:
        if step.startswith("fal:"):
            role = step[4:]
            if refs:
                args = {"prompt": prompt, "image_urls": [data_uri(p) for p in refs if p.exists()]}
            else:  # an edit model needs a reference: use the role's text-to-image twin
                role = self.cfg.get("text_roles", {}).get(role, role)
                if role not in self.studio.models:
                    return None
                args = {"prompt": prompt}
            return self.studio.call(role, args, "image", raw, f"picture_{role}", key=[key, role, len(refs)])
        cached = self.cache / f"{step}-{digest(prompt)}.png"
        if cached.exists() and cached.stat().st_size:
            return cached
        data = self._cloudflare(prompt) if step == "cloudflare" else self._pollinations(prompt, key)
        cached.write_bytes(data)
        return cached

    def _cloudflare(self, prompt: str) -> bytes:
        model = self.cfg.get("cloudflare_model", "@cf/black-forest-labs/flux-1-schnell")
        url = CF_URL.format(account=os.environ["CF_ACCOUNT_ID"], model=model)
        text = ("Wide cinematic composition with the subject in the middle band of the frame. " + prompt)[:2000]
        reply = self.http(url, method="POST", headers={"Authorization": f"Bearer {os.environ['CF_API_TOKEN']}"},
                          body={"prompt": text, "steps": int(self.cfg.get("cloudflare_steps", 4))}, timeout=180)
        image = (reply.get("result") or {}).get("image") if isinstance(reply, dict) else None
        if not image:
            raise MediaError(f"cloudflare: no image in {str(reply)[:200]}")
        return base64.b64decode(image)

    def _pollinations(self, prompt: str, key: Any) -> bytes:
        with self.pollinations_lock:  # one request at a time, spaced out (the free tier's limit)
            wait = float(self.cfg.get("pollinations_gap_seconds", 16)) - (time.time() - self.pollinations_last)
            if wait > 0:
                time.sleep(wait)
            url = POLLINATIONS.format(prompt=urllib.parse.quote(prompt[:900]), seed=int(digest(key), 16) % 100000)
            try:
                return self.http(url, timeout=240, raw=True)
            finally:
                self.pollinations_last = time.time()

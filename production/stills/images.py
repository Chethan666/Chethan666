"""Painters and the critic.

A picture is asked of each painter in the chain until one passes the critic:

    fal:<role>   a paid model on fal.ai, with the place's sheet (or the style frame) and the
                 character sheets as references (Seedream 4.5 edit for shots, Nano Banana 2
                 as the redraw)
    cloudflare   FLUX.1 schnell on Cloudflare Workers AI: free allowance of 10,000
                 neurons a day (CF_ACCOUNT_ID, CF_API_TOKEN). Text only, square output
                 (cropped to 16:9), so composition is asked to stay central
    pollinations keyless and free, slow (one picture per ~45 s), no guarantee

The critic checks every picture in code (not too bright for the channel, not blank, not
almost black) and then, when a vision model is configured, by eye: the house style, the
right moment, no text or watermark, no panels, no broken anatomy.

When the critic refuses a picture, the next painter is told what was wrong with it, so the
second try is a correction and not a second guess. A refused paid picture is never replaced
by a free painter's: free painters take no references, so the people and the place would
change (run 94 tried exactly that and got an unrelated room).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from production.media import MediaError, _request, data_uri

from ..look import H, W, fit
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


CRITIC_SYSTEM = ("You are the picture editor of a horror storytelling channel told in still pictures. "
                 "Every picture must match the channel's house style and show the narrated moment clearly. "
                 "Answer with JSON only.")


def house_look(cfg: dict[str, Any]) -> dict[str, Any]:
    """The channel's look (`stills.look`) with its style text; `stills.style` overrides the text."""
    looks = cfg.get("looks") or {}
    name = str(cfg.get("look") or "cinematic")
    look = dict(looks.get(name) or looks.get("cinematic") or {})
    look.setdefault("style", "A dark, moody horror picture with one motivated light source.")
    look.setdefault("noun", "picture")
    for key in ("people", "avoid", "wrong", "lettering"):
        look.setdefault(key, "")
    look.setdefault("grain", 0.0)
    look.setdefault("vignette", 0.0)
    look.setdefault("synthetic_media", False)
    if str(cfg.get("style") or "").strip():
        look["style"] = str(cfg["style"]).strip()
    look["name"] = name
    return look


SERIOUS = ("panel", "split", "collage", "two different", "side by side", "text", "lettering", "written", "watermark",
           "timestamp", "anatomy", "extra finger", "extra limb", "melted", "doubled", "warped", "cartoon",
           "drawing", "painting", "3d render", "plastic", "too bright", "no person", "nobody", "not visible",
           "unrelated", "does not show", "doesn't show",
           # the wrong action, or the opposite of what the line says
           "instead of", "rather than", "not sleeping", "not lying", "not standing", "not sitting",
           "contradict", "opposite",
           # another person, or a face unlike the portrait; a calm face where the line is terror
           "different person", "another person", "different face", "not match the portrait", "calm",
           "animated", "doll", "cgi", "render")


MINOR = "minor: "
SEVERE = "serious: "


def serious(reason: str) -> bool:
    """A refusal for a fault that ruins a picture, rather than a detail of the description. The
    critic's "detail" stands; its "serious" stands only for a fault of a serious kind (SERIOUS):
    in run 106 it called nearly everything serious (a close-up framed a little wider, hair worn
    down), spent the whole retry allowance and still kept 8 of 12 pictures unpassed."""
    if reason.startswith(MINOR):
        return False
    text = plain(reason).lower()
    return any(s in text for s in SERIOUS)


def plain(reason: str) -> str:
    """The critic's reason without the serious/minor mark."""
    for mark in (SEVERE, MINOR):
        if reason.startswith(mark):
            return reason[len(mark):]
    return reason


class Critic:
    """Checks a picture in code, then by eye with a vision model when one is configured."""

    def __init__(self, studio: Studio | None, style: str, wrong: str = ""):
        self.studio, self.style = studio, style
        self.wrong = wrong or "it does not match the house style at all"
        self.lock = threading.Lock()
        self.log: list[str] = []
        self.faces: dict[Path, str] = {}

    def _face(self, portrait: Path) -> str:
        """A portrait as the checker is sent it, made once (pictures are checked in parallel)."""
        with self.lock:
            if portrait not in self.faces:
                self.faces[portrait] = data_uri(_small(portrait, 384))
            return self.faces[portrait]

    def check(self, image: Path, picture: str) -> tuple[bool, str]:
        ok, why = still_check(image)
        if not ok:
            return ok, why
        return self.look(image, picture)

    def look(self, image: Path, picture: str, faces: list[tuple[str, Path]] | None = None) -> tuple[bool, str]:
        """`faces`: (name, close portrait) of the people who should be in it, sent after the picture
        so the checker can compare each face with its portrait."""
        studio = self.studio
        if studio is None or "vision" not in studio.models:
            return True, "no vision model"
        model = studio.models["vision"]
        try:
            usd = studio.budget.reserve(model)
        except OverBudget:
            return True, "no budget left for checks"
        prompt = (f"House style: {self.style}\nThe picture should show: {picture}\n\nLook at the image. Fail it ONLY "
                  f"if one of these is clearly true: {self.wrong}; it is bright and cheerful rather than dark; "
                  "it contains readable words, letters, a timestamp or a watermark; it is split into panels or is "
                  "a collage or a character sheet; a person has clearly broken anatomy (extra limbs or fingers, "
                  "a melted or doubled face); a named person clearly contradicts their description (wrong hair "
                  "colour, glasses missing or added, different clothes, a different age); or it shows something "
                  "unrelated to the moment" + (
                      "; or a named person has clearly a different face from their portrait (images 2 on are "
                      "close portraits of " + ", ".join(n for n, _ in faces) + ": a different person, not "
                      "just another angle, expression or light) - that is serious" if faces else "") +
                  ". Do NOT fail it for small differences: a slightly different shade of "
                  "a colour, a person a little nearer or further than described, more or less shadow on them, the "
                  "direction someone looks or faces, a slightly different pose, a missing small prop or background "
                  "detail. When you fail it, say in one sentence what is wrong "
                  "and what the picture should show instead, and whether it is serious. Serious is ONLY: a "
                  "different person, or a face unlike their portrait; the picture contradicts what the line says "
                  "happens (someone asleep shown awake, a shut door shown open, a light the line says is off "
                  "shown on - then begin the reason with 'contradicts the line:'); a calm face where the line is "
                  "terror; split frames or text; broken anatomy; a cartoon, 3D-render or doll-like look; a "
                  "bright, cheerful picture. Everything else is NOT serious: the exact framing (a close-up "
                  "framed wider), a pose, where someone looks, hair worn up or down, a prop's state or "
                  "position, which hand holds what, a costume detail, something extra in the background. "
                  "Reply exactly as JSON: "
                  '{"ok": true or false, "serious": true or false, "reason": "short reason"}')
        try:
            result = studio.fal.run(model, dict(studio.args.get("vision", {}), prompt=prompt,
                                                system_prompt=CRITIC_SYSTEM, image_urls=[data_uri(_small(image))] +
                                                [self._face(p) for _, p in faces or [] if p.exists()]))
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
        reason = str(verdict.get("reason", ""))[:200]
        if not verdict.get("ok") and verdict.get("serious") is False:
            reason = MINOR + reason  # the critic's own word that this is a detail
        elif not verdict.get("ok") and verdict.get("serious") is True:
            reason = SEVERE + reason  # ... or that a viewer would notice it
        return bool(verdict.get("ok")), reason


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
        self.sizes: dict[str, str] = {}  # the size each painter's pictures really come back at
        self.doubted: set[str] = set()  # pictures used although no check passed them
        # Retries and redraws may spend at most `retry_share` of the video's cap: the first try is
        # meant to be right, a retry is the safety net. Past it, a picture keeps its best try.
        cap = float(getattr(getattr(studio, "budget", None), "cap", 0.0) or 0.0)
        self.retry_cap = float(cfg.get("retry_share", 0.15)) * cap if cap else float("inf")
        self.retry_spent = 0.0
        self.retries_stopped = 0
        self.rejections: list[str] = []
        self.lock = threading.Lock()
        self._refs: dict[tuple[str, int, int, int], str] = {}

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

    def reference(self, path: Path, px: int = 2048) -> str:
        """A reference picture as the models are sent it: a JPEG at most `px` wide. The
        full-size PNG is ten times the upload (four of them: 30 MB a request) for no gain."""
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size, px)
        with self.lock:
            if key in self._refs:
                return self._refs[key]
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((px, px), Image.LANCZOS)
            data = io.BytesIO()
            im.save(data, "JPEG", quality=92)
        uri = "data:image/jpeg;base64," + base64.b64encode(data.getvalue()).decode()
        with self.lock:
            self._refs[key] = uri
        return uri

    def paint(self, prompt: str, refs: list[Path], out: Path, key: Any, chain: list[str],
              check: str | Callable[[Path], tuple[bool, str]] | None = None,
              accept_minor: bool = False, capped: bool = False, redraw: bool = False,
              faces: list[tuple[str, Path]] | None = None) -> tuple[Path | None, str]:
        """The first picture in the chain that passes its check. `check` is what the picture
        should show (the critic looks at it), or a check of its own for a reference sheet
        (picture -> passed, reason); None takes the first picture made.

        Each painter after a refusal is told what was wrong; a chain may name the same painter
        twice for that. With `accept_minor` (an ordinary beat), a picture refused only for a detail
        is kept at once: no money is spent correcting what a viewer would not notice. With `capped`,
        retries (and a `redraw` from the start) stop once the retry allowance is spent. If none
        passes, the picture kept (and counted as doubted) is the first one
        whose fault was not serious, so the strongest painter's picture wins over a correction that
        broke something worse (run 96: a reveal refused for the way she looked lost to a split
        screen); with only serious faults, the last one."""
        doubted: tuple[Path, str] | None = None
        mild: tuple[Path, str] | None = None
        why = ""
        paid_picture = False
        for k, step in enumerate(s for s in chain if self.available(s)):
            paid = step.startswith("fal:")
            if paid_picture and not paid:
                continue  # a refused paid picture still has the right people and place; a free one would not
            # the correction leads the prompt: at the end of 400 words it would be the least-read part
            fault = plain(why)
            text = f"Correct this from the previous attempt: {fault[:300].rstrip('. ')}. {prompt}" if why else prompt
            raw = self.work / f"raw_{out.stem}_{k}.png"
            extra = capped and paid and (bool(why) or redraw)  # money spent beyond a first try
            price = self._price(step)
            if extra:
                with self.lock:
                    if self.retry_spent + price > self.retry_cap + 1e-9:
                        self.retries_stopped += 1
                        log.info("%s: retry allowance spent; keeping the best try so far", out.stem)
                        break
                    self.retry_spent += price  # held now, so parallel workers cannot overspend
            try:
                made = self._one(step, text, refs, raw, key, retry=k if why else 0)
            except OverBudget:
                self._refund(extra, price)
                continue
            except (MediaError, OSError, ValueError) as exc:
                self._refund(extra, price)
                log.warning("%s: %s failed: %s", out.stem, step, str(exc)[:200])
                continue
            if made is None:
                self._refund(extra, price)
                continue
            candidate = self.work / f"fit_{out.stem}_{k}.png"
            try:
                with Image.open(made) as im:
                    size = f"{im.width}x{im.height}"
                fit(made, candidate, tuple(self.cfg.get("picture_size") or (W, H)))
            except OSError as exc:
                log.warning("%s: %s gave an unreadable picture: %s", out.stem, step, exc)
                continue
            paid_picture = paid_picture or paid
            with self.lock:
                self.sizes.setdefault(step, size)
            if check is None:
                candidate.replace(out)
                self._count(step)
                return out, step
            if callable(check):
                doubted = (candidate, step)
                ok, reason = check(candidate)
            else:
                ok, reason = still_check(candidate)
                if ok:
                    doubted = (candidate, step)
                    ok, reason = (self.critic.look(candidate, check, faces) if faces
                                  else self.critic.look(candidate, check))
            if ok:
                candidate.replace(out)
                self._count(step + " (corrected)" if why else step)
                return out, step
            if accept_minor and doubted is not None and doubted[0] == candidate and not serious(reason):
                candidate.replace(out)
                self._count(step + " (minor fault kept)")
                with self.lock:
                    self.rejections.append(f"{out.stem} ({step}, kept): {reason}")
                return out, step
            why = reason
            if mild is None and doubted is not None and doubted[0] == candidate and not serious(reason):
                kept = candidate.with_name(f"kept_{candidate.name}")
                candidate.replace(kept)
                mild = doubted = (kept, step)
            with self.lock:
                self.rejections.append(f"{out.stem} ({step}): {reason}")
            log.info("%s from %s rejected (%s); next painter", out.stem, step, reason)
        keep = mild or doubted
        if keep is not None:
            keep[0].replace(out)
            self._count(keep[1] + " (doubted)")
            self.doubted.add(out.stem)
            return out, keep[1]
        return None, "none"

    def _price(self, step: str) -> float:
        if not step.startswith("fal:") or self.studio is None or step[4:] not in self.studio.models:
            return 0.0
        role = step[4:]
        units = float((self.cfg.get("model_units") or {}).get(role, 1.0))
        return float(self.studio.budget.price(self.studio.models[role], units))

    def _refund(self, held: bool, price: float) -> None:
        if held:
            with self.lock:
                self.retry_spent -= price

    def _one(self, step: str, prompt: str, refs: list[Path], raw: Path, key: Any, retry: int = 0) -> Path | None:
        """One picture from one painter. `retry`: the corrected try's place in the chain, so the
        same painter asked twice is paid twice rather than handed its first picture again."""
        if step.startswith("fal:"):
            role = step[4:]
            if role in (self.cfg.get("text_only_roles") or ()):  # it takes no pictures: the words alone
                refs = []
            if refs:
                px = int((self.cfg.get("reference_px") or {}).get(role, 2048))
                args = {"prompt": prompt, "image_urls": [self.reference(p, px) for p in refs if p.exists()]}
            else:  # an edit model needs a reference: use the role's text-to-image twin
                role = self.cfg.get("text_roles", {}).get(role, role)
                if role not in self.studio.models:
                    return None
                args = {"prompt": prompt}
            units = float((self.cfg.get("model_units") or {}).get(role, 1.0))  # e.g. a 2K picture at 1.5x the price
            return self.studio.call(role, args, "image", raw, f"picture_{role}", units=units,
                                    key=[key, role, len(refs)] + (["retry", retry] if retry else []))
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

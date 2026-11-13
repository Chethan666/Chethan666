"""The channel's sound library: music, room tones, effects and stings the agents pick from.

The library is a catalogue (`library.json`) plus one audio file per sound, published as a
GitHub Release (tag `sound-library`) by `.github/workflows/sound-library.yml`, or kept in a
local folder. Each entry says what the sound is, its moods and intensity, whether it loops,
its license and source, and the credit line it needs (Kevin MacLeod's CC BY tracks need
one; CC0 and Sonniss sounds don't).

At production time the sound director searches the catalogue for every need (a music cue
per mood, a room tone per place, an effect per beat, a sting), shows the planner model a
short list of candidates and lets it pick. Whatever the library cannot cover is generated
as before, and saved back into the library after the run (`library sync`), so each sound
is paid for at most once.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

KINDS = ("music", "ambience", "sfx", "sting")
MOODS = ("unease", "dread", "terror")
# Licenses a monetized channel may use. Non-commercial, no-derivatives and share-alike are never added.
ALLOWED = ("cc0", "cc-by-3.0", "cc-by-4.0", "cc-by", "sonniss", "generated", "public-domain")
STOP = set("a an the of and or in on at to with from into for by is are its it this that slowly quietly "
           "softly loud sound sounds effect noise very".split())
SYNONYMS = {
    "phone": ["telephone", "ring", "ringing"], "telephone": ["phone"], "ring": ["phone", "ringing"],
    "creak": ["creaking", "squeak"], "door": ["doors"], "footstep": ["footsteps", "step", "steps", "walk"],
    "step": ["footstep", "footsteps"], "knock": ["knocking", "knocks"], "wind": ["windy", "gust"],
    "rain": ["raining", "storm"], "storm": ["thunder", "rain"], "thunder": ["storm"],
    "whisper": ["whispers", "whispering", "voice"], "scream": ["screams", "shriek"],
    "heartbeat": ["heart", "pulse"], "hum": ["buzz", "drone", "electrical"], "buzz": ["hum", "buzzing"],
    "office": ["room", "indoor", "hum"], "forest": ["woods", "night", "crickets"], "woods": ["forest"],
    "car": ["engine", "vehicle"], "engine": ["car", "motor"], "glass": ["shatter", "break"],
    "water": ["drip", "dripping"], "drip": ["water", "dripping"], "clock": ["ticking", "tick"],
    "stinger": ["sting", "hit", "impact"], "sting": ["stinger", "hit"], "hit": ["impact", "sting"],
}


def tokens(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text)).lower()):
        if w in STOP or len(w) < 2:
            continue
        for suffix in ("ing", "es", "s", "ed"):
            if len(w) > 4 and w.endswith(suffix):
                w = w[: -len(suffix)]
                break
        out.append(w)
    return out


def license_of(text: str) -> str:
    """A license name or URL as one of ALLOWED, or "" when it may not be used commercially."""
    t = str(text).lower()
    if not t.strip():
        return ""
    if "publicdomain/zero" in t or "cc0" in t or "zero 1.0" in t or "cc-zero" in t:
        return "cc0"
    if "public domain" in t or "publicdomain/mark" in t:
        return "public-domain"
    if any(x in t for x in ("noncommercial", "non-commercial", "-nc", "nc-", " nc ", "by-nc", "noderiv", "-nd",
                            "by-nd", "sharealike", "share-alike", "-sa", "by-sa", "gpl")):
        return ""
    if "attribution" in t or "licenses/by/" in t or t.startswith("cc-by") or t.startswith("cc by"):
        if "3.0" in t:
            return "cc-by-3.0"
        if "4.0" in t:
            return "cc-by-4.0"
        return "cc-by"
    if t in ALLOWED:
        return t
    return ""


@dataclass
class Sound:
    id: str
    kind: str  # music, ambience, sfx, sting
    title: str
    file: str  # the file name in the library (the release asset name)
    description: str = ""
    tags: list[str] = field(default_factory=list)
    moods: list[str] = field(default_factory=list)  # music: unease, dread, terror
    intensity: int = 2  # 1 (barely there) to 5 (full terror)
    seconds: float = 0.0
    loop: bool = False
    license: str = "cc0"
    source: str = ""  # the page it came from, kept for any copyright dispute
    author: str = ""
    credit: str = ""  # the line the video description must carry, "" when none is needed
    uses: int = 0  # how many videos used it: the least used is preferred, so videos sound different
    origin: str = ""  # incompetech, opengameart, fma, zip, acestep, generated

    def text(self) -> str:
        return " ".join([self.title, self.description, " ".join(self.tags)])

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Sound":
        known = {f for f in cls.__dataclass_fields__}  # noqa: SLF001
        return cls(**{k: v for k, v in d.items() if k in known})


class Library:
    def __init__(self, sounds: list[Sound], base: str | Path, cache: Path, http: Callable | None = None):
        self.sounds = [s for s in sounds if s.kind in KINDS and license_of(s.license) and s.file]
        self.by_id = {s.id: s for s in self.sounds}
        self.base, self.cache, self.http = base, cache, http
        self._index = {s.id: set(_expand(tokens(s.text()))) for s in self.sounds}

    # --- loading ------------------------------------------------------------------
    @classmethod
    def load(cls, cfg: dict[str, Any], cache: Path, http: Callable | None = None) -> "Library | None":
        """From `local_dir` when it has a library.json, else from the GitHub Release; None if neither."""
        cache.mkdir(parents=True, exist_ok=True)
        local = str(cfg.get("local_dir") or "").strip()
        if local and (Path(local) / "library.json").is_file():
            data = json.loads((Path(local) / "library.json").read_text())
            return cls([Sound.from_dict(x) for x in data.get("sounds", [])], Path(local), cache, http)
        base = release_base(cfg)
        if not base:
            return None
        try:
            got = cache / "library.json"
            (http or _download)(f"{base}/library.json", got)
            data = json.loads(got.read_text())
        except Exception as exc:  # noqa: BLE001 - without a library every sound is generated as before
            log.warning("Sound library not available (%s): %s", base, str(exc)[:200])
            return None
        return cls([Sound.from_dict(x) for x in data.get("sounds", [])], base, cache, http)

    def fetch(self, sound: Sound) -> Path | None:
        out = self.cache / sound.file
        if out.exists() and out.stat().st_size > 0:
            return out
        try:
            if isinstance(self.base, Path):
                shutil.copyfile(self.base / sound.file, out)
            else:
                (self.http or _download)(f"{self.base}/{sound.file}", out)
        except Exception as exc:  # noqa: BLE001
            log.warning("Library sound %s not fetched: %s", sound.id, str(exc)[:200])
            return None
        return out

    # --- search -------------------------------------------------------------------
    def search(self, kind: str, text: str = "", mood: str = "", k: int = 5,
               intensity: int | None = None) -> list[tuple[Sound, float]]:
        """The best `k` sounds of one kind for a need, each with a score from 0 to about 1."""
        want = set(_expand(tokens(text)))
        q = set(tokens(text))
        out = []
        for s in self.sounds:
            if s.kind != kind:
                continue
            if kind == "music":
                score = (1.0 if mood and mood in s.moods else 0.25 if not s.moods else 0.0)
                if q:
                    score += 0.3 * len(q & self._index[s.id]) / len(q)
                if s.seconds and s.seconds < 45:
                    score -= 0.2  # a short track loops too often under a long scene
            else:
                if not q:
                    score = 0.3
                else:
                    hits = len(q & self._index[s.id]) + 0.5 * len((want - q) & self._index[s.id])
                    score = hits / len(q)
                if mood and mood in s.moods:
                    score += 0.1
            if intensity is not None:
                score -= 0.05 * abs(int(s.intensity) - intensity)
            score -= 0.03 * math.log1p(max(s.uses, 0))  # prefer the less used
            out.append((s, round(score, 3)))
        out.sort(key=lambda x: (-x[1], x[0].uses, x[0].id))
        return out[:k]

    def credits(self, ids: list[str]) -> list[str]:
        seen, out = set(), []
        for i in ids:
            s = self.by_id.get(i)
            if s and s.credit and s.credit not in seen:
                seen.add(s.credit)
                out.append(s.credit)
        return out

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self.sounds:
            counts[s.kind] = counts.get(s.kind, 0) + 1
        return counts


def _expand(words: list[str]) -> list[str]:
    out = list(words)
    for w in words:
        out.extend(tokens(" ".join(SYNONYMS.get(w, []))))
    return out


def release_base(cfg: dict[str, Any]) -> str:
    repo = str(cfg.get("repo") or os.environ.get("GITHUB_REPOSITORY") or "").strip()
    tag = str(cfg.get("release_tag") or "sound-library")
    return f"https://github.com/{repo}/releases/download/{tag}" if repo else ""


def _download(url: str, out: Path) -> Path:
    from production.media import download
    out.parent.mkdir(parents=True, exist_ok=True)
    download(url, out)
    return out


# --- the catalogue file ---------------------------------------------------------------
def read_catalogue(path: Path) -> list[Sound]:
    if not path.is_file():
        return []
    return [Sound.from_dict(x) for x in json.loads(path.read_text()).get("sounds", [])]


def write_catalogue(path: Path, sounds: list[Sound]) -> None:
    sounds = sorted(sounds, key=lambda s: (KINDS.index(s.kind) if s.kind in KINDS else 9, s.id))
    data = {"version": 1, "counts": {k: sum(1 for s in sounds if s.kind == k) for k in KINDS},
            "sounds": [asdict(s) for s in sounds]}
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False))


def credits_page(sounds: list[Sound]) -> str:
    lines = ["# Sound library credits and licenses", "",
             "Every sound here may be used in monetized videos. CC0, Sonniss and generated sounds need no",
             "credit; CC BY sounds need the credit line below in the video description (added automatically).",
             ""]
    for s in sorted(sounds, key=lambda s: (s.origin, s.title)):
        lines.append(f"- **{s.title}** ({s.kind}, {s.license}) {s.source} {('- ' + s.credit) if s.credit else ''}")
    return "\n".join(lines) + "\n"


def merge(existing: list[Sound], new: list[Sound], used: list[str] | None = None) -> list[Sound]:
    """New entries added (an id already present keeps its use count); `used` ids counted once more."""
    by = {s.id: s for s in existing}
    for s in new:
        if s.id in by:
            s.uses = max(s.uses, by[s.id].uses)
        by[s.id] = s
    for i in used or []:
        if i in by:
            by[i].uses += 1
    return list(by.values())

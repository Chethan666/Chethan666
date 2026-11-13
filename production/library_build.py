"""Builds the sound library from free, commercially usable sources (`sounds/sources.json`).

Run by `.github/workflows/sound-library.yml` on GitHub's servers (open internet), never
during a video. Every source is optional: one that fails is logged and skipped. Sources:

    incompetech   Kevin MacLeod's music (CC BY 4.0: free for monetized videos, credit line needed)
    opengameart   OpenGameArt pages, collections or searches; only CC0 and CC BY are kept
    fma           Free Music Archive tracks from a Hugging Face dataset, filtered to CC0 / CC BY
                  instrumental tracks in dark genres
    zip           any pack of sounds by URL, with its license given in the source entry
                  (e.g. a Sonniss GDC bundle part, or a CC0 pack)
    acestep       music made with ACE-Step on fal.ai, for moods the free sources miss (paid, about
                  $0.0002 a second; off unless the workflow is started with generate_music)

Every file is converted to 48 kHz Ogg Vorbis, trimmed of silence (effects) and levelled, its
length measured, and its catalogue entry written: kind, moods, intensity, tags, license,
source page and credit line. Non-commercial, no-derivatives and share-alike files are never
added.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import logging
import re
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable

from .library import KINDS, Sound, credits_page, license_of, merge, read_catalogue, tokens, write_catalogue

log = logging.getLogger(__name__)

AUDIO = (".mp3", ".ogg", ".wav", ".flac", ".m4a", ".aif", ".aiff", ".opus")
AGENT = "Mozilla/5.0 (compatible; sound-library-builder/1.0)"
KMAC_CREDIT = ('"{title}" Kevin MacLeod (incompetech.com) Licensed under Creative Commons: By Attribution 4.0 '
               "License http://creativecommons.org/licenses/by/4.0/")
# Words in a track's feel, genre or description that say which score mood it can carry.
MOOD_WORDS = {
    "terror": ["intense", "aggressive", "action", "frantic", "chaotic", "chase", "violent", "panic", "terror",
               "climax", "driving", "pounding", "horrific", "brutal"],
    "dread": ["suspense", "suspenseful", "tense", "tension", "dark", "unnerving", "ominous", "menacing", "dread",
              "creepy", "sinister", "horror", "scary", "threat", "pulse", "heartbeat"],
    "unease": ["mysterious", "eerie", "somber", "sombre", "haunting", "haunted", "ambient", "drone", "atmosphere",
               "atmospheric", "unease", "uneasy", "melancholy", "lonely", "bleak", "spooky", "ghost", "calm",
               "sad", "quiet", "night"],
}
SFX_KIND_WORDS = {"ambience": ["ambience", "ambient", "atmosphere", "amb", "room", "tone", "roomtone", "loop",
                               "wind", "rain", "forest", "crowd", "traffic", "night", "hum", "background"],
                  "sting": ["sting", "stinger", "hit", "impact", "riser", "jumpscare", "scare", "boom", "whoosh"]}


def fetch(url: str, timeout: int = 120) -> bytes:
    if url.startswith("file://") or Path(url).exists():
        return Path(url.removeprefix("file://")).read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception:  # noqa: BLE001
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def moods_of(text: str) -> list[str]:
    words = set(tokens(text)) | set(re.findall(r"[a-z]+", text.lower()))
    found = [m for m in ("unease", "dread", "terror") if any(w in words for w in MOOD_WORDS[m])]
    return found or ["unease"]


def intensity_of(moods: list[str]) -> int:
    return 5 if "terror" in moods else 3 if "dread" in moods else 2


def slug(text: str, n: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n] or "sound"


def make_id(kind: str, title: str, source: str) -> str:
    return f"{kind}-{slug(title, 32)}-{hashlib.sha1(source.encode()).hexdigest()[:6]}"


def words_from_filename(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"([a-z])([A-Z])", r"\1 \2", stem)
    return re.sub(r"[_\-.]+|\d{2,}", " ", stem).strip()


class Builder:
    def __init__(self, out: Path, http: Callable[[str], bytes] = fetch, fal: Any = None, max_usd: float = 1.0,
                 ffmpeg: str = "ffmpeg"):
        self.out, self.http, self.fal, self.max_usd, self.ffmpeg = out, http, fal, max_usd, ffmpeg
        out.mkdir(parents=True, exist_ok=True)
        self.sounds: list[Sound] = []
        self.report: dict[str, Any] = {}
        self.spent = 0.0
        self.seen: set[str] = set()

    # --- audio processing ---------------------------------------------------------------
    def process(self, raw: Path, kind: str, max_seconds: float) -> tuple[str, float] | None:
        """Convert to the library's format; returns (file name, seconds) or None when unusable."""
        name = f"{kind}-{hashlib.sha1(raw.read_bytes()).hexdigest()[:16]}.ogg"
        dst = self.out / name
        level = "loudnorm=I=-20:TP=-1.5:LRA=11" if kind in ("music", "ambience") else "loudnorm=I=-16:TP=-1"
        trim = ("silenceremove=start_periods=1:start_threshold=-55dB,areverse,"
                "silenceremove=start_periods=1:start_threshold=-55dB,areverse," if kind in ("sfx", "sting") else "")
        cmd = [self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(raw), "-t", str(max_seconds),
               "-af", f"{trim}{level}", "-ar", "48000", "-ac", "2", "-c:a", "libvorbis", "-q:a", "5", str(dst)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not dst.exists():
            log.warning("Could not convert %s: %s", raw.name, r.stderr[-200:])
            return None
        seconds = duration(dst)
        if seconds < 0.3:
            dst.unlink(missing_ok=True)
            return None
        return name, seconds

    def add(self, raw: bytes | Path, suffix: str, *, kind: str, title: str, source: str, license: str,
            description: str = "", tags: list[str] | None = None, moods: list[str] | None = None,
            author: str = "", credit: str = "", origin: str = "", loop: bool | None = None,
            intensity: int | None = None, max_seconds: float | None = None) -> Sound | None:
        lic = license_of(license)
        if not lic:
            self.count(origin, "skipped_license")
            return None
        key = f"{source}|{title}"
        if key in self.seen:
            return None
        self.seen.add(key)
        limit = max_seconds or {"music": 420, "ambience": 240, "sfx": 20, "sting": 8}[kind]
        with tempfile.TemporaryDirectory() as tmp:
            path = raw if isinstance(raw, Path) else Path(tmp) / f"in{suffix or '.bin'}"
            if not isinstance(raw, Path):
                path.write_bytes(raw)
            got = self.process(path, kind, limit)
        if got is None:
            self.count(origin, "unusable")
            return None
        file, seconds = got
        if kind == "music" and seconds < 25:
            kind = "ambience" if seconds >= 8 else "sting"
        moods = moods or (moods_of(f"{title} {description} {' '.join(tags or [])}") if kind == "music" else [])
        sound = Sound(id=make_id(kind, title, source), kind=kind, title=title[:120], file=file,
                      description=description[:400], tags=[t[:40] for t in (tags or [])][:20], moods=moods,
                      intensity=intensity or intensity_of(moods), seconds=round(seconds, 2),
                      loop=bool(loop) if loop is not None else kind == "ambience", license=lic, source=source,
                      author=author[:80], credit=credit if lic.startswith("cc-by") else "", origin=origin)
        self.sounds.append(sound)
        self.count(origin, "added")
        return sound

    def count(self, origin: str, what: str) -> None:
        r = self.report.setdefault(origin or "other", {})
        r[what] = r.get(what, 0) + 1

    # --- sources --------------------------------------------------------------------
    def run(self, sources: list[dict[str, Any]], only: set[str] | None = None) -> list[Sound]:
        for src in sources:
            kind = str(src.get("type"))
            if src.get("enabled") is False or (only and kind not in only):
                continue
            try:
                getattr(self, f"src_{kind}")(src)
            except Exception as exc:  # noqa: BLE001 - one source failing never stops the others
                log.warning("Source %s failed: %s", src.get("name") or kind, str(exc)[:300])
                self.count(kind, "failed")
        return self.sounds

    def src_incompetech(self, src: dict[str, Any]) -> None:
        base = str(src.get("files", "https://incompetech.com/music/royalty-free/mp3-royaltyfree/"))
        feels = {f.lower() for f in src.get("feels", [])}
        exclude = {f.lower() for f in src.get("exclude_feels", [])}
        picks: list[dict[str, Any]] = []
        try:
            data = json.loads(self.http(str(src.get("catalogue", "https://incompetech.com/music/royalty-free/"
                                                                  "pieces.json"))))
            items = data if isinstance(data, list) else next((v for v in data.values() if isinstance(v, list)), [])
            for p in items:
                p = {str(k).lower(): v for k, v in p.items()} if isinstance(p, dict) else {}
                feel = {f.strip().lower() for f in re.split(r"[,;/|]", str(p.get("feel", ""))) if f.strip()}
                genre = str(p.get("genre", "")).lower()
                if (feel & feels or any(g in genre for g in src.get("genres", []))) and not feel & exclude:
                    picks.append(p)
        except Exception as exc:  # noqa: BLE001
            log.warning("incompetech catalogue not read (%s); using the listed titles", str(exc)[:120])
        if not picks:
            picks = [{"title": t} for t in src.get("titles", [])]
        picks = picks[: int(src.get("max", 60))]
        for p in picks:
            title = str(p.get("title") or "").strip()
            if not title:
                continue
            filename = str(p.get("filename") or f"{title}.mp3")
            url = base + urllib.parse.quote(filename)
            try:
                raw = self.http(url)
            except Exception as exc:  # noqa: BLE001
                log.warning("incompetech %s: %s", title, str(exc)[:120])
                self.count("incompetech", "missing")
                continue
            desc = " ".join(str(p.get(k, "")) for k in ("feel", "genre", "instruments", "description")).strip()
            self.add(raw, ".mp3", kind="music", title=title, description=desc,
                     tags=[t.strip() for t in re.split(r"[,;]", str(p.get("feel", ""))) if t.strip()],
                     source=f"https://incompetech.com/music/royalty-free/index.html?isrc={p.get('isrc', '')}"
                     if p.get("isrc") else url, license="cc-by-4.0", author="Kevin MacLeod",
                     credit=KMAC_CREDIT.format(title=title), origin="incompetech",
                     moods=src.get("moods", {}).get(title))

    def src_opengameart(self, src: dict[str, Any]) -> None:
        pages = list(src.get("pages", []))
        for coll in src.get("collections", []) + src.get("searches", []):
            try:
                text = self.http(coll).decode("utf-8", "replace")
            except Exception as exc:  # noqa: BLE001
                log.warning("OpenGameArt list %s: %s", coll, str(exc)[:120])
                continue
            for slug_ in re.findall(r'href="(/content/[a-z0-9\-]+)"', text):
                url = "https://opengameart.org" + slug_
                if url not in pages:
                    pages.append(url)
        for page in pages[: int(src.get("max_pages", 60))]:
            try:
                self.oga_page(page, src)
            except Exception as exc:  # noqa: BLE001
                log.warning("OpenGameArt %s: %s", page, str(exc)[:120])
                self.count("opengameart", "failed_page")

    def oga_page(self, page: str, src: dict[str, Any]) -> None:
        text = self.http(page).decode("utf-8", "replace")
        lic = oga_license(text)
        if not lic:
            self.count("opengameart", "skipped_license")
            return
        title = html.unescape(re.search(r"<title>(.*?)(\s*\|.*)?</title>", text, re.S).group(1)).strip() \
            if re.search(r"<title>", text) else page.rsplit("/", 1)[-1]
        author = ""
        m = re.search(r'href="/users/([^"]+)"', text)
        if m:
            author = urllib.parse.unquote(m.group(1))
        m = re.search(r'<meta name="description" content="([^"]*)"', text) or \
            re.search(r'field-name-body.*?<p>(.*?)</p>', text, re.S)
        desc = html.unescape(re.sub(r"<[^>]+>", " ", m.group(1))).strip() if m else ""
        tags = [html.unescape(t) for t in re.findall(r'href="/art-search-advanced\?keys=([^"&]+)', text)][:12]
        files = []
        for f in re.findall(r'(?:https?://opengameart\.org)?(/sites/default/files/[^"\'<>\s]+)', text):
            url = "https://opengameart.org" + urllib.parse.quote(urllib.parse.unquote(f), safe="/")
            if url not in files and (url.lower().endswith(AUDIO) or url.lower().endswith(".zip")):
                files.append(url)
        kind_hint = src.get("kind", "music")
        page_moods = src.get("moods", {}).get(page)
        for url in files[: int(src.get("max_files_per_page", 8))]:
            raw = self.http(url)
            members = [(Path(url).name, raw)] if not url.lower().endswith(".zip") else unzip_audio(raw)
            prefer = [m for m in members if m[0].lower().endswith((".ogg", ".flac", ".wav"))]
            for name, data in (prefer or members)[: int(src.get("max_files_per_page", 8))]:
                kind = classify(name, desc, kind_hint)
                credit = (f'"{title}" by {author or "unknown"} (OpenGameArt.org, {page}), licensed {lic.upper()}'
                          if lic.startswith("cc-by") else "")
                self.add(data, Path(name).suffix, kind=kind, title=f"{title} - {words_from_filename(name)}"
                         if len(members) > 1 else title, description=desc, tags=tags + [words_from_filename(name)],
                         source=page, license=lic, author=author, credit=credit, origin="opengameart",
                         moods=page_moods)

    def src_fma(self, src: dict[str, Any]) -> None:
        from datasets import Audio, load_dataset  # installed by the library workflow only

        ds = load_dataset(src.get("dataset", "benjamin-paine/free-music-archive-full"), src.get("config"),
                          split="train", streaming=True)
        try:
            ds = ds.cast_column("audio", Audio(decode=False))
        except Exception:  # noqa: BLE001
            pass
        genres = [g.lower() for g in src.get("genres", [])]
        wanted, scanned, deadline = int(src.get("max", 30)), 0, time.time() + float(src.get("minutes", 25)) * 60
        added = 0
        for row in ds:
            scanned += 1
            if scanned > int(src.get("scan", 30000)) or time.time() > deadline or added >= wanted:
                break
            r = {str(k).lower(): v for k, v in row.items()}
            lic = license_of(str(r.get("license") or r.get("license_title") or r.get("license_url") or ""))
            if not lic:
                continue
            instrumental = r.get("instrumental")
            if instrumental not in (None, 1, True, "1", "yes", "Yes"):
                continue
            gv = r.get("genres") or r.get("genre") or r.get("genre_top") or ""
            g = gv if isinstance(gv, str) else " ".join(str(x) for x in gv)
            if genres and not any(x in g.lower() for x in genres):
                continue
            audio = r.get("audio") or {}
            data = audio.get("bytes") if isinstance(audio, dict) else None
            if not data:
                continue
            title = str(r.get("title") or r.get("track_title") or "untitled")
            artist = str(r.get("artist") or r.get("artist_name") or "unknown")
            page = str(r.get("url") or r.get("track_url") or "https://freemusicarchive.org")
            credit = f'"{title}" by {artist} (Free Music Archive, {page}), licensed {lic.upper()}'
            got = self.add(data, Path(str(audio.get("path") or "x.mp3")).suffix, kind="music", title=f"{title} - {artist}",
                           description=f"{g} instrumental", tags=[x for x in re.split(r"[,;|]", g) if x][:8],
                           source=page, license=lic, author=artist, credit=credit, origin="fma")
            added += 1 if got else 0
        self.report.setdefault("fma", {})["scanned"] = scanned

    def src_zip(self, src: dict[str, Any]) -> None:
        url = str(src["url"])
        include = re.compile(src.get("include", ".*"), re.I)
        exclude = re.compile(src["exclude"], re.I) if src.get("exclude") else None
        raw = self.http(url)
        n = 0
        for name, data in unzip_audio(raw):
            if not include.search(name) or (exclude and exclude.search(name)):
                continue
            if n >= int(src.get("max", 150)):
                break
            words = words_from_filename(name)
            kind = classify(name, words, src.get("kind", "sfx"))
            got = self.add(data, Path(name).suffix, kind=kind, title=words, description=f"{words} ({Path(name).parent})",
                           tags=tokens(str(Path(name).parent))[:8], source=f"{src.get('page', url)}#{name}",
                           license=src.get("license", ""), author=src.get("author", ""),
                           credit=src.get("credit", "").format(title=words) if src.get("credit") else "",
                           origin=src.get("name", "zip"))
            n += 1 if got else 0

    def src_acestep(self, src: dict[str, Any]) -> None:
        if self.fal is None:
            log.info("ACE-Step skipped: no fal key, or generate_music is off")
            return
        from production.media import download
        model = str(src.get("model", "fal-ai/ace-step/prompt-to-audio"))
        per_second = float(src.get("usd_per_second", 0.0002))
        for k, t in enumerate(src.get("tracks", [])):
            seconds = float(t.get("seconds", 120))
            cost = seconds * per_second
            if self.spent + cost > self.max_usd:
                log.warning("ACE-Step stopped at the $%.2f cap", self.max_usd)
                break
            args = dict(src.get("args", {}), prompt=t["prompt"], duration=seconds)
            try:
                result = self.fal.run(model, args)
            except Exception as exc:  # noqa: BLE001
                log.warning("ACE-Step track %d: %s", k, str(exc)[:200])
                continue
            self.spent += cost
            audio = result.get("audio") or result.get("audio_file") or {}
            url = audio.get("url") if isinstance(audio, dict) else str(audio)
            with tempfile.TemporaryDirectory() as tmp:
                path = download(url, Path(tmp) / "ace.wav")
                self.add(path, ".wav", kind="music", title=t.get("title", f"ACE-Step {t['mood']} {k + 1}"),
                         description=t["prompt"], tags=[t["mood"], "generated"], moods=[t["mood"]],
                         intensity=t.get("intensity"), source=f"fal:{model}", license="generated",
                         origin="acestep", loop=True)
        self.report.setdefault("acestep", {})["usd"] = round(self.spent, 3)


def oga_license(page_html: str) -> str:
    """The most permissive usable license an OpenGameArt page offers, or ""."""
    found = []
    for m in re.findall(r'(creativecommons\.org/(?:publicdomain/zero|licenses/[a-z\-]+)/[0-9.]+)', page_html):
        found.append(license_of(m))
    for m in re.findall(r'class="license-name"[^>]*>([^<]+)<', page_html):
        found.append(license_of(m))
    for lic in ("cc0", "public-domain", "cc-by-4.0", "cc-by-3.0", "cc-by"):
        if lic in found:
            return lic
    return ""


def classify(name: str, text: str, hint: str) -> str:
    words = set(tokens(f"{name} {text}"))
    if hint in ("music", "sting"):
        return hint
    for kind in ("sting", "ambience"):
        if words & set(SFX_KIND_WORDS[kind]):
            return kind
    return hint if hint in KINDS else "sfx"


def unzip_audio(raw: bytes) -> list[tuple[str, bytes]]:
    out = []
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for info in z.infolist():
            name = info.filename
            if info.is_dir() or "__MACOSX" in name or not name.lower().endswith(AUDIO):
                continue
            if info.file_size > 200_000_000:
                continue
            out.append((name, z.read(info)))
    return out


def duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def build(sources_path: Path, out: Path, existing: Path | None = None, only: set[str] | None = None,
          fal: Any = None, max_usd: float = 1.0, http: Callable[[str], bytes] = fetch) -> dict[str, Any]:
    """Run every source into `out`, merged with the existing catalogue; writes library.json and CREDITS.md."""
    sources = json.loads(sources_path.read_text())["sources"]
    builder = Builder(out, http=http, fal=fal, max_usd=max_usd)
    old = read_catalogue(existing) if existing else []
    builder.seen = {f"{s.source}|{s.title}" for s in old}
    new = builder.run(sources, only)
    sounds = merge(old, new)
    if len(sounds) > 950:
        log.warning("%d sounds: a GitHub release holds about 1,000 files; lower the sources' max values", len(sounds))
    write_catalogue(out / "library.json", sounds)
    (out / "CREDITS.md").write_text(credits_page(sounds))
    summary = {"added": len(new), "total": len(sounds),
               "by_kind": {k: sum(1 for s in sounds if s.kind == k) for k in KINDS},
               "music_by_mood": {m: sum(1 for s in sounds if s.kind == "music" and m in s.moods)
                                 for m in ("unease", "dread", "terror")},
               "sources": builder.report, "usd": round(builder.spent, 3)}
    (out / "build-report.json").write_text(json.dumps(summary, indent=2))
    return summary


def sync(runs: list[Path], existing: Path | None, out: Path) -> dict[str, Any]:
    """After videos: add each run's paid-for sounds (out/library_new) to the catalogue and count
    the library sounds each run used. Writes the new files and library.json into `out`, ready to
    upload to the release."""
    builder = Builder(out)
    old = read_catalogue(existing) if existing else []
    used: list[str] = []
    for run in runs:
        used += json.loads((run / "sounds_used.json").read_text()) if (run / "sounds_used.json").is_file() else []
        entries_file = run / "library_new" / "entries.json"
        if not entries_file.is_file():
            continue
        for e in json.loads(entries_file.read_text()):
            path = run / "library_new" / e["file"]
            if not path.is_file():
                continue
            moods = [e["mood"]] if e.get("mood") else None
            builder.add(path, path.suffix, kind=e["kind"], title=e["text"][:80] or e["file"],
                        description=e["text"], tags=["generated"], moods=moods, source=f"generated:{e['file']}",
                        license="generated", origin="generated", loop=e.get("loop") or None)
    ids_before = {s.id for s in old}
    sounds = merge(old, [s for s in builder.sounds if s.id not in ids_before], used)
    write_catalogue(out / "library.json", sounds)
    (out / "CREDITS.md").write_text(credits_page(sounds))
    return {"added": len(builder.sounds), "used": len(used), "total": len(sounds)}

"""Research: folklore from Wikipedia and audience demand from YouTube.

Wikipedia gives traditional legends to retell and motifs to spark original
stories (CC BY-SA text, used as research notes only; scripts are checked for
copied passages). The official YouTube Data API gives titles of popular horror
videos as demand signals; their plots are never used.

Not used, on purpose: Reddit and creepypasta sites. Their stories belong to
their authors.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from . import db

log = logging.getLogger(__name__)

Fetch = Callable[[str], dict[str, Any]]

USER_AGENT = "HollowLinesStoryFactory/0.1 (research bot; https://github.com/Chethan666/Chethan666)"
WIKI_API = "https://en.wikipedia.org/w/api.php"
YOUTUBE_API = "https://www.googleapis.com/youtube/v3"
WIKI_LICENSE = "CC BY-SA 4.0 (Wikipedia); research notes only, retold in original words"

# Pages about copyrighted works or lists, not legends.
SKIP_TITLE = re.compile(
    r"^(List of|Lists of|Category:)|\((film|novel|TV series|television series|video game|band|"
    r"song|album|comics|manga|anime|character|play|book)\b",
    re.IGNORECASE,
)


def http_fetch(url: str, pause: float = 0.5, attempts: int = 5) -> dict[str, Any]:
    """GET JSON politely: a short pause before each request, and on "429 Too Many
    Requests" a wait (Retry-After, or doubling from 5 seconds) before trying again."""
    for attempt in range(attempts):
        time.sleep(pause)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == attempts - 1:
                raise
            retry_after = exc.headers.get("Retry-After", "") if exc.headers else ""
            wait = float(retry_after) if retry_after.isdigit() else 5.0 * 2 ** attempt
            log.info("Rate limited by %s; waiting %.0f s", urllib.parse.urlsplit(url).netloc, wait)
            time.sleep(min(wait, 120.0))
    raise AssertionError("unreachable")


def _url(base: str, **params: Any) -> str:
    return f"{base}?{urllib.parse.urlencode(params)}"


def _members(fetch: Fetch, category: str, kind: str, limit: int) -> list[dict[str, Any]]:
    data = fetch(_url(WIKI_API, action="query", list="categorymembers", cmtitle=category,
                      cmtype=kind, cmlimit=min(limit, 500), format="json"))
    return data.get("query", {}).get("categorymembers", [])


def wikipedia_category_pages(fetch: Fetch, category: str, limit: int,
                             subcategories: bool = False) -> list[tuple[int, str]]:
    """Pages in a category (and optionally its direct subcategories), up to `limit`."""
    found: dict[int, str] = {}
    categories = [category]
    if subcategories:
        categories += [c["title"] for c in _members(fetch, category, "subcat", 50)]
    for name in categories:
        for m in _members(fetch, name, "page", limit):
            if len(found) >= limit:
                break
            if not SKIP_TITLE.search(m["title"]):
                found.setdefault(m["pageid"], m["title"])
    return list(found.items())


def wikipedia_summaries(fetch: Fetch, pageids: list[int]) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for start in range(0, len(pageids), 20):  # the extracts API returns at most 20 intros
        batch = pageids[start:start + 20]
        data = fetch(_url(WIKI_API, action="query", prop="extracts|info", exintro=1,
                          explaintext=1, exlimit=20, inprop="url", format="json",
                          pageids="|".join(map(str, batch))))
        for page in data.get("query", {}).get("pages", {}).values():
            extract = (page.get("extract") or "").strip()
            if extract:
                out[int(page["pageid"])] = {"title": page["title"], "url": page.get("fullurl", ""),
                                            "summary": extract[:1500]}
    return out


def research_wikipedia(conn: sqlite3.Connection, cfg: dict, fetch: Fetch = http_fetch) -> int:
    rules = cfg["research"]
    saved = 0
    for entry in rules["wikipedia_categories"]:
        try:
            pages = wikipedia_category_pages(fetch, entry["category"], rules["wikipedia_page_limit"],
                                             rules.get("wikipedia_subcategories", False))
            summaries = wikipedia_summaries(fetch, [pid for pid, _ in pages])
        except Exception as exc:  # one bad category must not stop the rest
            log.warning("Skipping %s: %s", entry["category"], exc)
            continue
        for pageid, info in summaries.items():
            db.upsert_inspiration(conn, source="wikipedia", source_id=str(pageid), kind="folklore",
                                  title=info["title"], url=info["url"], summary=info["summary"],
                                  region=entry["region"], license=WIKI_LICENSE)
            saved += 1
    return saved


def legend_regions(cfg: dict) -> list[str]:
    """The regions of the configured Wikipedia categories. Only their legends are retold or used
    as sparks, so folklore an earlier configuration brought in from elsewhere stays unused."""
    return sorted({str(entry["region"]) for entry in cfg["research"]["wikipedia_categories"]})


def ensure_legend_supply(conn: sqlite3.Connection, cfg: dict, fetch: Fetch = http_fetch) -> int:
    """Research Wikipedia automatically when unused legends from the configured regions run low,
    or when a configured region has never been researched. Returns legends added."""
    regions = legend_regions(cfg)
    unused = len(db.unused_legends(conn, regions))
    if unused >= cfg["research"]["min_unused_legends"] and set(regions) <= db.folklore_regions(conn):
        return 0
    research_wikipedia(conn, cfg, fetch)
    return len(db.unused_legends(conn, regions)) - unused


def research_youtube(conn: sqlite3.Connection, cfg: dict, api_key: str | None = None,
                     fetch: Fetch = http_fetch) -> int:
    """Popular recent horror titles. Costs about 101 quota units per query, no money."""
    api_key = api_key or os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        log.info("YOUTUBE_API_KEY not set; skipping YouTube demand research")
        return 0
    rules = cfg["research"]
    after = (datetime.now(timezone.utc) - timedelta(days=rules["youtube_published_within_days"]))
    saved = 0
    for query in rules["youtube_queries"]:
        try:
            search = fetch(_url(f"{YOUTUBE_API}/search", part="snippet", type="video",
                                order="viewCount", q=query, maxResults=rules["youtube_max_results"],
                                publishedAfter=after.strftime("%Y-%m-%dT%H:%M:%SZ"), key=api_key))
            items = [i for i in search.get("items", []) if i.get("id", {}).get("videoId")]
            if not items:
                continue
            stats = fetch(_url(f"{YOUTUBE_API}/videos", part="statistics",
                               id=",".join(i["id"]["videoId"] for i in items), key=api_key))
        except Exception as exc:
            log.warning("Skipping YouTube query %r: %s", query, exc)
            continue
        views = {v["id"]: int(v.get("statistics", {}).get("viewCount", 0)) for v in stats.get("items", [])}
        for item in items:
            video_id = item["id"]["videoId"]
            snippet = item["snippet"]
            db.upsert_inspiration(conn, source="youtube", source_id=video_id, kind="trend",
                                  title=snippet["title"],
                                  url=f"https://www.youtube.com/watch?v={video_id}",
                                  summary=f"{snippet.get('channelTitle', '')} | query: {query}",
                                  region="Global", license="Title used as a demand signal only",
                                  popularity=views.get(video_id, 0))
            saved += 1
    return saved

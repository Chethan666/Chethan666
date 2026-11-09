"""SQLite storage shared by the story factory and the production pipeline.

Story lifecycle (`stories.status`):

    idea -> scripted -> approved -> in_production -> produced -> published
      |  \\________\\___________________________________-> rejected / failed
      +-> shelved (a strong runner-up pitch, developed in a later run)
      +-> parked (editors never approved it, but the premise wasn't the problem: kept
                  with its best draft and diagnosis for a stronger model)

Rejected and shelved rows are kept so the same idea is never generated twice.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

STATUSES = ("idea", "shelved", "parked", "scripted", "approved", "in_production", "produced", "published",
            "rejected", "failed")

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspirations (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('folklore', 'trend')),
    title TEXT NOT NULL,
    url TEXT,
    summary TEXT,
    region TEXT,
    license TEXT,
    popularity INTEGER,
    used_count INTEGER NOT NULL DEFAULT 0,
    fetched_at TEXT NOT NULL,
    UNIQUE (source, source_id)
);

CREATE TABLE IF NOT EXISTS stories (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('long', 'short')),
    series TEXT NOT NULL,
    status TEXT NOT NULL,
    title TEXT NOT NULL,
    title_norm TEXT NOT NULL,
    logline TEXT,
    synopsis TEXT,
    setting TEXT,
    core_fear TEXT,
    protagonist TEXT,
    format TEXT,
    twist TEXT,
    ending TEXT,
    recipe_sig TEXT,
    legend_key TEXT,
    inspiration_ids TEXT NOT NULL DEFAULT '[]',
    synopsis_embedding TEXT,
    intro TEXT,
    body TEXT,
    outro TEXT,
    narration TEXT,
    word_count INTEGER,
    comment_question TEXT,
    hook_line TEXT,
    thumbnail_idea TEXT,
    description_credit TEXT,
    scenes TEXT,
    review TEXT,
    reject_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    claimed_by TEXT,
    claimed_at TEXT,
    video_id TEXT,
    published_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_stories_status ON stories (status, kind, series);
-- A legend is retold only once (rejected attempts don't count).
CREATE UNIQUE INDEX IF NOT EXISTS ux_stories_legend
    ON stories (legend_key) WHERE legend_key IS NOT NULL AND status != 'rejected';
-- The same setting + fear + twist combination never returns.
CREATE UNIQUE INDEX IF NOT EXISTS ux_stories_recipe
    ON stories (recipe_sig) WHERE recipe_sig IS NOT NULL AND status != 'rejected';

CREATE TABLE IF NOT EXISTS world_entities (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    name_norm TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    first_story_id INTEGER REFERENCES stories (id),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS story_entities (
    story_id INTEGER NOT NULL REFERENCES stories (id),
    entity_id INTEGER NOT NULL REFERENCES world_entities (id),
    PRIMARY KEY (story_id, entity_id)
);

CREATE TABLE IF NOT EXISTS performance (
    id INTEGER PRIMARY KEY,
    story_id INTEGER NOT NULL REFERENCES stories (id),
    measured_at TEXT NOT NULL,
    views INTEGER,
    avg_view_percentage REAL,
    ctr REAL,
    likes INTEGER,
    comments INTEGER
);

CREATE TABLE IF NOT EXISTS learnings (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    insights TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS costs (
    id INTEGER PRIMARY KEY,
    at TEXT NOT NULL,
    pipeline TEXT NOT NULL,
    task TEXT NOT NULL,
    model TEXT,
    input_units INTEGER NOT NULL DEFAULT 0,
    output_units INTEGER NOT NULL DEFAULT 0,
    usd REAL NOT NULL DEFAULT 0
);
"""

WORLD_SEED = [
    ("character", "The Storyteller",
     "The channel's host. A tall stick figure in a long coat who carries a lantern with a "
     "red flame. Calm, kind, a little ominous. He collects stories from the hollows and "
     "never says how he found them."),
    ("rule", "The Hollow",
     "Wherever a line runs between places (a road, a railway, a corridor, a sea lane) "
     "there is a hollow beside it. Most people pass through without noticing. Some don't."),
    ("object", "The red lantern",
     "The Storyteller's lantern. Its flame turns red when something from a hollow is near. "
     "It occasionally appears, unexplained, inside other stories."),
]


# Columns added after the first release; `connect` adds any that are missing.
MIGRATIONS = {
    "stories": {
        "pitch": "TEXT", "pitch_score": "REAL", "author_models": "TEXT", "beat_sheet": "TEXT",
        "audience_report": "TEXT", "pronunciations": "TEXT", "packaging": "TEXT",
        "production_log": "TEXT", "cast": "TEXT", "bible": "TEXT", "pauses": "TEXT",
        "best_draft": "TEXT", "diagnosis": "TEXT", "production": "TEXT",
    },
    "inspirations": {"dossier": "TEXT", "unsuitable": "TEXT"},
    "world_entities": {"look": "TEXT"},
    "costs": {"story_id": "INTEGER"},
}
JSON_COLUMNS = ("scenes", "review", "pitch", "author_models", "beat_sheet", "audience_report",
                "pronunciations", "packaging", "production_log", "cast", "bible", "pauses", "best_draft", "production")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_name(name: str) -> str:
    return " ".join(name.lower().split())


def connect(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    # Model calls run on worker threads; UsageRecorder serializes their writes.
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    if str(path) != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    migrate(conn)
    seed_world(conn)
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    for table, columns in MIGRATIONS.items():
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, kind in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")


def seed_world(conn: sqlite3.Connection) -> None:
    for kind, name, description in WORLD_SEED:
        conn.execute(
            "INSERT OR IGNORE INTO world_entities (kind, name, name_norm, description, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (kind, name, normalize_name(name), description, now()),
        )


# --- Inspirations -----------------------------------------------------------

def upsert_inspiration(conn: sqlite3.Connection, **fields: Any) -> int:
    fields.setdefault("fetched_at", now())
    cols = ", ".join(fields)
    marks = ", ".join("?" for _ in fields)
    updates = ", ".join(f"{c} = excluded.{c}" for c in fields if c not in ("source", "source_id"))
    conn.execute(
        f"INSERT INTO inspirations ({cols}) VALUES ({marks}) "
        f"ON CONFLICT (source, source_id) DO UPDATE SET {updates}",
        tuple(fields.values()),
    )
    row = conn.execute(
        "SELECT id FROM inspirations WHERE source = ? AND source_id = ?",
        (fields["source"], fields["source_id"]),
    ).fetchone()
    return row["id"]


def _in_regions(regions: Sequence[str] | None) -> tuple[str, list[str]]:
    """An SQL condition (and its parameters) limiting inspirations to `regions`; none means all."""
    if not regions:
        return "", []
    return f" AND region IN ({', '.join('?' for _ in regions)})", list(regions)


def unused_legends(conn: sqlite3.Connection, regions: Sequence[str] | None = None) -> list[sqlite3.Row]:
    """Legends not yet retold and not judged unfit, from `regions` only when given."""
    where, params = _in_regions(regions)
    return conn.execute(
        "SELECT * FROM inspirations i WHERE kind = 'folklore' AND unsuitable IS NULL AND NOT EXISTS ("
        "  SELECT 1 FROM stories s WHERE s.legend_key = i.source || ':' || i.source_id"
        f"  AND s.status != 'rejected'){where}", params
    ).fetchall()


def folklore_regions(conn: sqlite3.Connection) -> set[str]:
    """Every region that research has brought in folklore from."""
    return {r["region"] for r in conn.execute("SELECT DISTINCT region FROM inspirations WHERE kind = 'folklore'")}


def mark_unsuitable(conn: sqlite3.Connection, inspiration_id: int, reason: str) -> None:
    """A legend judged unable to carry a faithful horror retelling is never offered again."""
    conn.execute("UPDATE inspirations SET unsuitable = ? WHERE id = ?", (reason[:500] or "unsuitable", inspiration_id))


def folklore_sparks(conn: sqlite3.Connection, limit: int,
                    regions: Sequence[str] | None = None) -> list[sqlite3.Row]:
    where, params = _in_regions(regions)
    return conn.execute(
        f"SELECT * FROM inspirations WHERE kind = 'folklore'{where} ORDER BY used_count, RANDOM() LIMIT ?",
        (*params, limit),
    ).fetchall()


def top_trends(conn: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM inspirations WHERE kind = 'trend' "
        "ORDER BY COALESCE(popularity, 0) DESC LIMIT ?",
        (limit,),
    ).fetchall()


def mark_inspirations_used(conn: sqlite3.Connection, ids: Iterable[int]) -> None:
    conn.executemany("UPDATE inspirations SET used_count = used_count + 1 WHERE id = ?", [(i,) for i in ids])


# --- Stories ----------------------------------------------------------------

def insert_story(conn: sqlite3.Connection, **fields: Any) -> int:
    fields.setdefault("created_at", now())
    fields.setdefault("updated_at", fields["created_at"])
    for key in ("inspiration_ids", *JSON_COLUMNS):
        if key in fields and not isinstance(fields[key], (str, type(None))):
            fields[key] = json.dumps(fields[key], ensure_ascii=False)
    cols = ", ".join(fields)
    marks = ", ".join("?" for _ in fields)
    cur = conn.execute(f"INSERT INTO stories ({cols}) VALUES ({marks})", tuple(fields.values()))
    return cur.lastrowid


def update_story(conn: sqlite3.Connection, story_id: int, **fields: Any) -> None:
    if "status" in fields and fields["status"] not in STATUSES:
        raise ValueError(f"Unknown status {fields['status']!r}")
    for key in JSON_COLUMNS:
        if key in fields and not isinstance(fields[key], (str, type(None))):
            fields[key] = json.dumps(fields[key], ensure_ascii=False)
    fields["updated_at"] = now()
    sets = ", ".join(f"{c} = ?" for c in fields)
    conn.execute(f"UPDATE stories SET {sets} WHERE id = ?", (*fields.values(), story_id))


def get_story(conn: sqlite3.Connection, story_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM stories WHERE id = ?", (story_id,)).fetchone()


def all_stories(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Every story ever pitched, newest first. Rejected ideas still block repeats."""
    return conn.execute("SELECT * FROM stories ORDER BY id DESC").fetchall()


def live_stories(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Stories that are not rejected, newest first."""
    return conn.execute(
        "SELECT * FROM stories WHERE status != 'rejected' ORDER BY id DESC"
    ).fetchall()


def recent_heroes(conn: sqlite3.Connection, limit: int = 12) -> list[str]:
    """'Name: who they are' for the heroes of recent stories that weren't rejected."""
    heroes = []
    for row in conn.execute("SELECT bible FROM stories WHERE bible IS NOT NULL AND status != 'rejected' "
                            "ORDER BY id DESC LIMIT ?", (limit,)):
        hero = (json.loads(row["bible"]) or {}).get("hero") or {}
        if hero.get("name"):
            heroes.append(f"{hero['name']}: {hero.get('who', '')}")
    return heroes


def recent_recipes(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    rows = conn.execute(
        "SELECT setting, core_fear, twist FROM stories WHERE status != 'rejected' "
        "AND recipe_sig IS NOT NULL ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def used_signatures(conn: sqlite3.Connection) -> set[str]:
    return {
        r["recipe_sig"]
        for r in conn.execute(
            "SELECT recipe_sig FROM stories WHERE status != 'rejected' AND recipe_sig IS NOT NULL"
        )
    }


def best_shelved(conn: sqlite3.Connection, series: str, include_parked: bool = False) -> sqlite3.Row | None:
    statuses = "('shelved', 'parked')" if include_parked else "('shelved')"
    return conn.execute(
        f"SELECT * FROM stories WHERE status IN {statuses} AND series = ? "
        "ORDER BY pitch_score DESC, id LIMIT 1", (series,)
    ).fetchone()


def set_dossier(conn: sqlite3.Connection, inspiration_id: int, dossier: dict) -> None:
    conn.execute("UPDATE inspirations SET dossier = ? WHERE id = ?",
                 (json.dumps(dossier, ensure_ascii=False), inspiration_id))


def backlog_counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        "SELECT series, COUNT(*) AS n FROM stories WHERE status = 'approved' GROUP BY series"
    ).fetchall()
    return {r["series"]: r["n"] for r in rows}


def status_counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute("SELECT status, COUNT(*) AS n FROM stories GROUP BY status").fetchall()
    return {r["status"]: r["n"] for r in rows}


def claim_next(conn: sqlite3.Connection, kind: str, worker: str, series: str | None = None) -> sqlite3.Row | None:
    """Atomically hand the oldest approved story to the production pipeline."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        query = "SELECT id FROM stories WHERE status = 'approved' AND kind = ?"
        params: list[Any] = [kind]
        if series:
            query += " AND series = ?"
            params.append(series)
        row = conn.execute(query + " ORDER BY id LIMIT 1", params).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None
        stamp = now()
        conn.execute(
            "UPDATE stories SET status = 'in_production', claimed_by = ?, claimed_at = ?, "
            "updated_at = ? WHERE id = ?",
            (worker, stamp, stamp, row["id"]),
        )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return get_story(conn, row["id"])


# --- Story world ------------------------------------------------------------

def world_entities(conn: sqlite3.Connection, limit: int = 30) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM world_entities ORDER BY (first_story_id IS NULL) DESC, id DESC LIMIT ?",
        (limit,),
    ).fetchall()


def link_entities(conn: sqlite3.Connection, story_id: int, new: list[dict], referenced: list[str]) -> None:
    for entity in new:
        name = str(entity.get("name", "")).strip()
        description = str(entity.get("description", "")).strip()
        if not name or not description:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO world_entities (kind, name, name_norm, description, first_story_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (str(entity.get("kind", "character")), name, normalize_name(name), description, story_id, now()),
        )
        referenced = [*referenced, name]
    for name in referenced:
        row = conn.execute(
            "SELECT id FROM world_entities WHERE name_norm = ?", (normalize_name(name),)
        ).fetchone()
        if row:
            conn.execute(
                "INSERT OR IGNORE INTO story_entities (story_id, entity_id) VALUES (?, ?)",
                (story_id, row["id"]),
            )


def known_looks(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Recurring characters whose look is fixed in the world canon."""
    return [(r["name"], r["look"]) for r in conn.execute(
        "SELECT name, look FROM world_entities WHERE kind = 'character' AND look IS NOT NULL ORDER BY id")]


def fix_looks(conn: sqlite3.Connection, cast: list[dict]) -> None:
    """Store the first look given to each world character; later stories reuse it."""
    for member in cast:
        conn.execute(
            "UPDATE world_entities SET look = ? WHERE name_norm = ? AND kind = 'character' AND look IS NULL",
            (str(member.get("look", "")).strip() or None, normalize_name(str(member.get("name", "")))),
        )


# --- Costs ------------------------------------------------------------------

def record_cost(conn: sqlite3.Connection, *, pipeline: str, task: str, model: str | None,
                input_units: int, output_units: int, usd: float, story_id: int | None = None) -> None:
    conn.execute(
        "INSERT INTO costs (at, pipeline, task, model, input_units, output_units, usd, story_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (now(), pipeline, task, model, input_units, output_units, usd, story_id),
    )


# --- Performance feedback ---------------------------------------------------

def record_performance(conn: sqlite3.Connection, story_id: int, **metrics: Any) -> None:
    conn.execute(
        "INSERT INTO performance (story_id, measured_at, views, avg_view_percentage, ctr, likes, comments) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (story_id, now(), metrics.get("views"), metrics.get("avg_view_percentage"),
         metrics.get("ctr"), metrics.get("likes"), metrics.get("comments")),
    )


def latest_performance(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Each published story with its most recent metrics."""
    return conn.execute(
        "SELECT s.*, p.views, p.avg_view_percentage, p.ctr, p.likes, p.comments FROM stories s "
        "JOIN performance p ON p.story_id = s.id WHERE p.id = ("
        "  SELECT MAX(id) FROM performance WHERE story_id = s.id)"
    ).fetchall()


def save_learnings(conn: sqlite3.Connection, insights: list[str]) -> None:
    conn.execute("INSERT INTO learnings (created_at, insights) VALUES (?, ?)",
                 (now(), json.dumps(insights, ensure_ascii=False)))


def latest_learnings(conn: sqlite3.Connection) -> list[str]:
    row = conn.execute("SELECT insights FROM learnings ORDER BY id DESC LIMIT 1").fetchone()
    return json.loads(row["insights"]) if row else []


def month_spend(conn: sqlite3.Connection) -> float:
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    row = conn.execute(
        "SELECT COALESCE(SUM(usd), 0) AS total FROM costs WHERE substr(at, 1, 7) = ?", (month,)
    ).fetchone()
    return float(row["total"])


# --- Reports ------------------------------------------------------------------

def stats(conn: sqlite3.Connection) -> dict[str, Any]:
    by_series = [dict(r) for r in conn.execute(
        "SELECT series, status, COUNT(*) AS n FROM stories GROUP BY series, status ORDER BY series, status")]
    cost_per_story = [dict(r) for r in conn.execute(
        "SELECT s.kind, COUNT(DISTINCT s.id) AS stories, ROUND(SUM(c.usd), 2) AS usd, "
        "ROUND(SUM(c.usd) / COUNT(DISTINCT s.id), 2) AS usd_per_story "
        "FROM costs c JOIN stories s ON s.id = c.story_id "
        "WHERE s.status NOT IN ('rejected', 'shelved', 'parked', 'idea') GROUP BY s.kind")]
    unattributed = conn.execute("SELECT ROUND(COALESCE(SUM(usd), 0), 2) FROM costs WHERE story_id IS NULL").fetchone()[0]
    by_model = [dict(r) for r in conn.execute(
        "SELECT model, COUNT(*) AS calls, ROUND(SUM(usd), 2) AS usd FROM costs GROUP BY model ORDER BY usd DESC")]
    winners: dict[str, int] = {}
    for row in conn.execute("SELECT author_models FROM stories WHERE author_models IS NOT NULL "
                            "AND status NOT IN ('rejected', 'shelved', 'parked', 'idea')"):
        draft = json.loads(row["author_models"]).get("draft")
        if draft:
            winners[draft] = winners.get(draft, 0) + 1
    reasons = [dict(r) for r in conn.execute(
        "SELECT substr(reject_reason, 1, 70) AS reason, COUNT(*) AS n FROM stories "
        "WHERE status IN ('rejected', 'parked') GROUP BY 1 ORDER BY n DESC LIMIT 8")]
    return {"stories": by_series, "cost_per_approved_story": cost_per_story,
            "pitching_and_research_usd": unattributed, "spend_by_model": by_model,
            "draft_wins_by_model": winners, "top_reject_reasons": reasons}

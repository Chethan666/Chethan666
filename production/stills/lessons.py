"""What went wrong before: the critic's refusals, counted across videos, warn the next prompts.

Every refusal is sorted into one of a few faults painters make again and again (a split frame,
daylight in a night scene, a calm face on a terrified line...). The counts are kept in the story
database, which carries over between runs, and the faults seen most often are named in every
new picture prompt, so the first try avoids them and fewer pictures are paid for twice.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# fault: (words in a refusal that mean it, the warning a prompt carries)
FAULTS: dict[str, tuple[tuple[str, ...], str]] = {
    "split": (("split", "panel", "diptych", "collage", "two different", "side by side", "two frames"),
              "one single frame, never split into two or more panels"),
    "daylight": (("daylight", "too bright", "bright and cheerful", "evenly lit", "well-lit", "sunlight"),
                 "night: no daylight, lit only by the scene's own small lights"),
    "face": (("neutral expression", "calm", "expression", "not afraid", "not terrified", "no fear"),
             "faces show the line's feeling clearly"),
    "clothes": (("wearing", "clothes", "jeans", "pyjamas", "jacket", "costume", "duvet"),
                "each person wears exactly the clothes on their sheet"),
    "extra_people": (("two human figures", "extra person", "another person", "second person", "people in it",
                      "no people", "should be no"),
                     "only the people named, nobody else"),
    "text": (("text", "letters", "written", "watermark", "logo", "timestamp", "label"),
             "no writing of any kind"),
    "action": (("not standing", "not sitting", "instead of", "rather than", "not holding", "not staring",
                "not looking", "should show"),
               "the exact action the line describes"),
}
TABLE = "CREATE TABLE IF NOT EXISTS picture_lessons (fault TEXT PRIMARY KEY, count INTEGER NOT NULL)"


def fault_of(reason: str) -> str | None:
    text = reason.lower()
    for name, (words, _) in FAULTS.items():
        if any(w in text for w in words):
            return name
    return None


def load(conn: Any, top: int = 3, at_least: int = 2) -> str:
    """The warnings for the faults seen most often (at least `at_least` times), as one phrase."""
    if conn is None:
        return ""
    try:
        conn.execute(TABLE)
        rows = conn.execute("SELECT fault, count FROM picture_lessons WHERE count >= ? ORDER BY count DESC, fault "
                            "LIMIT ?", (at_least, top)).fetchall()
    except Exception as exc:  # noqa: BLE001 - lessons help, they never stop a video
        log.warning("Picture lessons unavailable: %s", exc)
        return ""
    return "; ".join(FAULTS[f][1] for f, _ in rows if f in FAULTS)


def record(conn: Any, reasons: list[str]) -> dict[str, int]:
    """Count this video's refusals by fault; returns what was counted."""
    counted: dict[str, int] = {}
    for reason in reasons:
        fault = fault_of(reason)
        if fault:
            counted[fault] = counted.get(fault, 0) + 1
    if conn is None or not counted:
        return counted
    try:
        conn.execute(TABLE)
        for fault, n in counted.items():
            conn.execute("INSERT INTO picture_lessons (fault, count) VALUES (?, ?) "
                         "ON CONFLICT(fault) DO UPDATE SET count = count + excluded.count", (fault, n))
        conn.commit()
    except Exception as exc:  # noqa: BLE001
        log.warning("Picture lessons not saved: %s", exc)
    return counted

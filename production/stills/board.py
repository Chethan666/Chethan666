"""The shared board the agents work on, and the showrunner that runs them."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)


def _json(value: Any, default: Any) -> Any:
    """A story field that may be stored as JSON text (the database) or as data (a script file)."""
    if value is None or value == "":
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


@dataclass
class Board:
    """Everything one video's agents share: the story, the tools, and what each agent made."""

    story: dict[str, Any]
    out: Path
    work: Path
    cfg: dict[str, Any]
    router: Any = None  # the story factory's model router (the planner role directs)
    studio: Any = None  # paid media calls with the cap and the cache (None: free tools only)
    budget: Any = None
    kokoro: Any = None  # the free voice, behind every paid voice
    conn: Any = None
    data: dict[str, Any] = field(default_factory=dict)
    reports: dict[str, Any] = field(default_factory=dict)

    def save(self) -> None:
        (self.work / "board.json").write_text(json.dumps(self.reports, indent=2, default=str))


class Agent(Protocol):
    name: str
    critical: bool  # a failure stops the video (otherwise the agent's fallback stands)

    def run(self, board: Board) -> dict[str, Any]: ...


class Showrunner:
    """Runs the agents in order, times them, and keeps each one's report."""

    def __init__(self, agents: list[Any]):
        self.agents = agents

    def run(self, board: Board) -> dict[str, Any]:
        for agent in self.agents:
            started = time.time()
            log.info("[%s] starting", agent.name)
            try:
                report = agent.run(board) or {}
            except Exception as exc:
                if getattr(agent, "critical", True):
                    board.reports[agent.name] = {"error": str(exc)[:500]}
                    board.save()
                    raise
                log.exception("[%s] failed; carrying on with its fallback", agent.name)
                report = {"error": str(exc)[:500]}
            report["seconds"] = round(time.time() - started, 1)
            board.reports[agent.name] = report
            board.save()
            log.info("[%s] done: %s", agent.name, {k: v for k, v in report.items() if not isinstance(v, (list, dict))})
        return board.reports

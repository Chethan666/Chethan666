"""One approved story in, a finished stills video out, made by the agent team."""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

from ..paid import Budget, Studio
from ..run import _json
from .art import ArtDirectorAgent, IllustratorAgent
from .board import Board, Showrunner
from .checks import ArtReviewAgent, ScriptCheckAgent
from .grade import ColoristAgent
from .post import EditAgent, EditPlanAgent, PackagerAgent, QAAgent, SoundAgent
from .storyboard import StoryboardAgent
from .voices import CastingAgent, VoiceAgent

log = logging.getLogger(__name__)


def team() -> list[Any]:
    """The agents, in the order the showrunner runs them."""
    return [ScriptCheckAgent(), CastingAgent(), VoiceAgent(), StoryboardAgent(), ArtDirectorAgent(),
            IllustratorAgent(), ArtReviewAgent(), ColoristAgent(), EditPlanAgent(), SoundAgent(), EditAgent(), PackagerAgent(),
            QAAgent()]


def produce(story: dict[str, Any], out_dir: Path, kokoro: Any, router: Any, cfg: dict[str, Any],
            fal: Any = None, conn: Any = None, keep_work: bool = False) -> dict[str, Any]:
    started = time.time()
    out_dir = out_dir.resolve()  # FFmpeg runs inside the work folder: every path must be absolute
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    budget = Budget(float(cfg.get("max_usd", 0.0)), cfg.get("prices", {}), conn)
    studio = Studio(fal, budget, work / "cache", cfg) if fal is not None else None
    if studio is not None:
        studio.stop_on_empty = False  # an empty fal balance means free fallbacks, never a lost video
    board = Board(story=story, out=out_dir, work=work, cfg=cfg, router=router, studio=studio, budget=budget,
                  kokoro=kokoro, conn=conn)
    board.data.update(bible=_json(story.get("bible"), {}), packaging=_json(story.get("packaging"), {}))
    reports = Showrunner(team()).run(board)
    d = board.data
    summary = {
        "mode": "stills", "video": str(d.get("video", "")), "seconds": round(d.get("total", 0.0), 1),
        "story_seconds": round(d.get("story_seconds", 0.0), 1), "beats": len(d.get("beats", [])),
        "point_of_view": d.get("pov"), "voices": d.get("casting"),
        "usd": round(budget.spent, 2), "usd_by_task": {k: round(v, 2) for k, v in sorted(budget.by_task.items())},
        "fal_balance_ran_out": bool(studio and studio.balance_ran_out),
        "qa": reports.get("qa", {}), "agents": reports,
        "review": "pending", "render_minutes": round((time.time() - started) / 60, 1),
    }
    (out_dir / "report.json").write_text(json.dumps(summary, indent=2, default=str))
    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)
    log.info("Made stills video: %s", {k: v for k, v in summary.items() if k != "agents"})
    return summary

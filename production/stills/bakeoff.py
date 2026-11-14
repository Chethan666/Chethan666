"""The picture bake-off: the same story's pictures painted by several setups, side by side.

One set of references (style frame, location sheets, character sheets and face portraits) is
made once; then every setup paints every beat from those same references with its own chain
of painters, and an optional detail pass (an upscaler) on top. Nothing is voiced by a paid
voice and no video is made: the output is for choosing a setup.

    out/compare_beat_NN.jpg   the beat painted by each setup, one above the other, full frame
    out/detail_beat_NN.jpg    the middle of each picture at the size a viewer sees it (1440p), 1:1
    out/faces.jpg             the portraits every setup painted from
    out/<setup>/beat_NN.jpg   every picture at full size
    out/report.json           cost, refusals and the judge's scores per setup

The judge (the vision model, given the portraits and the three versions of a beat, unlabelled
and in a shuffled order) scores each version for the same faces, the moment, and how much it
looks like a frame of a real film.
"""

from __future__ import annotations

import copy
import json
import logging
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from production.media import MediaError, data_uri

from .. import draw
from ..paid import Budget, OverBudget, Studio, _small
from .art import ArtDirectorAgent, IllustratorAgent
from .board import Board, Showrunner, _json
from .checks import ScriptCheckAgent
from .images import Painter
from .storyboard import StoryboardAgent
from .voices import CastingAgent, VoiceAgent

log = logging.getLogger(__name__)

PANEL = (1280, 720)
VIEW = (2560, 1440)  # the picture as the viewer sees it, full screen at 1440p


def detail(picture: Path, out: Path, task: str, studio: Studio) -> Path | None:
    """The detail pass: the picture upscaled to 4K by the `upscale` model, or None."""
    if studio is None or "upscale" not in studio.models:
        return None
    small = picture.with_name(picture.stem + "_up.jpg")
    with Image.open(picture) as im:
        im.convert("RGB").save(small, quality=95)
    try:
        return studio.call("upscale", {"image_url": data_uri(small)}, "image", out, task, key=[str(picture.name), task])
    except (MediaError, OverBudget) as exc:
        log.warning("Detail pass on %s failed: %s", picture.name, str(exc)[:200])
        return None


def _label(im: Image.Image, text: str) -> Image.Image:
    bar = 44
    canvas = Image.new("RGB", (im.width, im.height + bar), (16, 16, 16))
    canvas.paste(im, (0, bar))
    ImageDraw.Draw(canvas).text((12, 8), text, font=draw.font(26), fill=(230, 230, 230))
    return canvas


def _stack(panels: list[Image.Image], out: Path) -> Path:
    sheet = Image.new("RGB", (max(p.width for p in panels), sum(p.height for p in panels)), (16, 16, 16))
    y = 0
    for p in panels:
        sheet.paste(p, (0, y))
        y += p.height
    sheet.save(out, quality=90)
    return out


def _view(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB").resize(VIEW, Image.LANCZOS)


def compare(names: list[str], labels: dict[str, str], pictures: dict[str, dict[int, Path]], beats: list[Any],
            out: Path) -> list[Path]:
    made = []
    for b in beats:
        whole, middle = [], []
        for n in names:
            path = pictures[n].get(b.index)
            if path is None or not path.exists():
                continue
            view = _view(path)
            whole.append(_label(view.resize(PANEL, Image.LANCZOS), f"{labels[n]}  |  beat {b.index}"))
            x, y = (VIEW[0] - PANEL[0]) // 2, (VIEW[1] - PANEL[1]) // 2
            middle.append(_label(view.crop((x, y, x + PANEL[0], y + PANEL[1])),
                                 f"{labels[n]}  |  beat {b.index}, middle at 1:1"))
        if whole:
            made.append(_stack(whole, out / f"compare_beat_{b.index:02d}.jpg"))
            made.append(_stack(middle, out / f"detail_beat_{b.index:02d}.jpg"))
    return made


JUDGE = ("You judge pictures for a horror channel told in film stills. The first images are close "
         "portraits of the people who should appear. The last {n} images are versions of the same "
         "picture, labelled by their order: {letters}. The picture should show: {picture} (narrated: "
         "\"{text}\"). Score every version from 1 to 10 on: same_faces (each named person has the face "
         "of their portrait; 10 if nobody is named and nobody appears), moment (it shows exactly this "
         "moment), cinematic (it looks like a frame of a real film: real people, real light, real "
         "texture and detail, nothing plastic or waxy). Then name the best version. Reply exactly as "
         "JSON: {{\"scores\": {{\"A\": {{\"same_faces\": 0, \"moment\": 0, \"cinematic\": 0}}, ...}}, "
         "\"best\": \"A\", \"why\": \"one sentence\"}}")


def judge(studio: Studio, beat: Any, faces: list[Path], versions: dict[str, Path], seed: int) -> dict[str, Any]:
    """The judge's scores for each setup on one beat; versions are shown unlabelled, shuffled."""
    if studio is None or "vision" not in studio.models or len(versions) < 2:
        return {}
    order = list(versions)
    random.Random(seed).shuffle(order)
    letters = [chr(65 + k) for k in range(len(order))]
    prompt = JUDGE.format(n=len(order), letters=", ".join(letters), picture=str(beat.picture)[:500],
                          text=str(beat.text)[:200])
    images = [data_uri(_small(p, 384)) for p in faces] + [data_uri(_small(versions[n], 1024)) for n in order]
    args = dict(studio.args.get("judge", {}) or studio.args.get("vision", {}), prompt=prompt, image_urls=images)
    model = studio.models["vision"]
    try:
        usd = studio.budget.reserve(model)
    except OverBudget:
        return {}
    try:
        result = studio.fal.run(model, args)
    except MediaError as exc:
        studio.budget.settle(usd, "judge", model, ok=False)
        log.warning("Judge failed on beat %s: %s", beat.index, str(exc)[:200])
        return {}
    studio.budget.settle(usd, "judge", model, ok=True)
    text = str(result.get("output") or result.get("text") or result)
    match = re.search(r"\{.*\}", text, re.S)
    try:
        verdict = json.loads(match.group(0)) if match else {}
    except ValueError:
        return {}
    scores = verdict.get("scores") if isinstance(verdict, dict) else None
    if not isinstance(scores, dict):
        return {}
    by_setup = {order[k]: scores.get(letter) for k, letter in enumerate(letters) if isinstance(scores.get(letter), dict)}
    best = str(verdict.get("best", "")).strip().upper()[:1]
    return {"scores": by_setup, "best": order[letters.index(best)] if best in letters else None,
            "why": str(verdict.get("why", ""))[:300]}


def run(story: dict[str, Any], out_dir: Path, kokoro: Any, router: Any, cfg: dict[str, Any], fal: Any,
        conn: Any = None, names: list[str] | None = None) -> dict[str, Any]:
    started = time.time()
    out_dir = out_dir.resolve()
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(cfg)
    cfg["models"] = {k: v for k, v in (cfg.get("models") or {}).items() if v and k not in ("voice", "sfx", "music")}
    setups: dict[str, dict[str, Any]] = cfg.get("bakeoff", {}).get("setups", {})
    names = [n for n in (names or list(setups)) if n in setups]
    if not names:
        raise ValueError(f"no bake-off setups to run (known: {', '.join(setups)})")
    budget = Budget(float(cfg.get("max_usd", 0.0)), cfg.get("prices", {}), conn)
    studio = Studio(fal, budget, work / "cache", cfg) if fal is not None else None
    if studio is not None:
        studio.stop_on_empty = False
    board = Board(story=story, out=out_dir, work=work, cfg=cfg, router=router, studio=studio, budget=budget,
                  kokoro=kokoro, conn=conn)
    board.data.update(bible=_json(story.get("bible"), {}), packaging=_json(story.get("packaging"), {}))
    # the free voice: the narration's timing places the beats; the pictures do not depend on whose voice
    reports = Showrunner([ScriptCheckAgent(), CastingAgent(), VoiceAgent(), StoryboardAgent(),
                          ArtDirectorAgent()]).run(board)
    d = board.data
    beats = d["beats"]
    references = round(budget.spent, 3)
    labels = {n: str(setups[n].get("label", n)) for n in names}
    pictures: dict[str, dict[int, Path]] = {}
    results: dict[str, Any] = {}
    for n in names:
        setup = setups[n]
        spent = budget.spent
        t0 = time.time()
        cfg_n = dict(cfg, chains=dict(cfg["chains"], **(setup.get("chains") or {})),
                     retry_share=float(cfg.get("bakeoff", {}).get("retry_share", cfg.get("retry_share", 0.15))))
        board.cfg = cfg_n
        painter = Painter(studio, cfg_n, work / f"art_{n}", d["critic"])
        d["painter"] = painter
        report = IllustratorAgent().run(board)
        made: dict[int, Path] = dict(d["pictures"])
        folder = out_dir / n
        folder.mkdir(parents=True, exist_ok=True)
        upscaled = 0
        for index, path in sorted(made.items()):
            final = path
            if setup.get("detail"):
                up = detail(path, work / f"up_{n}_{index:03d}.png", f"picture_detail_{n}", studio)
                if up is not None:
                    final, upscaled = up, upscaled + 1
            keep = folder / f"beat_{index:02d}.jpg"
            with Image.open(final) as im:
                im.convert("RGB").save(keep, quality=93)
            made[index] = keep
        pictures[n] = made
        results[n] = {"label": labels[n], "usd": round(budget.spent - spent, 3), "seconds": round(time.time() - t0),
                      "upscaled": upscaled, "by_painter": report.get("by_painter"),
                      "refusals": len(painter.rejections), "doubted_beats": report.get("doubted_beats"),
                      "retry_usd": report.get("retry_usd"), "painter_sizes": report.get("painter_sizes"),
                      "rejected": painter.rejections[:20]}
    board.cfg = cfg
    faces = [p for p in (d.get("portraits") or {}).values() if p and p.exists()]
    if faces:
        tiles = []
        for p in faces:
            with Image.open(p) as im:
                tiles.append(im.convert("RGB").resize((640, 360), Image.LANCZOS))
        _stack(tiles, out_dir / "faces.jpg")
    for name, p in (d.get("sheets") or {}).items():
        if p and p.exists():
            with Image.open(p) as im:
                im.convert("RGB").save(out_dir / f"sheet_{re.sub(r'[^a-z0-9]+', '_', name.lower())}.jpg", quality=90)
    compare(names, labels, pictures, beats, out_dir)
    verdicts: dict[int, Any] = {}
    portraits = d.get("portraits") or {}
    for b in beats:
        versions = {n: pictures[n][b.index] for n in names if b.index in pictures[n]}
        who = [portraits[c] for c in b.characters if c in portraits][:3]
        verdicts[b.index] = judge(studio, b, who, versions, seed=b.index)
    for n in names:  # the judge's small copies are not part of the output
        for extra in (out_dir / n).glob("*_check.jpg"):
            extra.unlink()
    for n in names:
        rows = [v["scores"][n] for v in verdicts.values() if v and isinstance(v.get("scores", {}).get(n), dict)]

        def mean(key: str) -> float | None:
            vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
            return round(sum(vals) / len(vals), 2) if vals else None
        results[n]["judge"] = {"same_faces": mean("same_faces"), "moment": mean("moment"),
                               "cinematic": mean("cinematic"),
                               "best_on": sum(1 for v in verdicts.values() if v and v.get("best") == n)}
    summary = {"mode": "bakeoff", "beats": len(beats), "setups": results, "references_usd": references,
               "usd": round(budget.spent, 2), "usd_by_task": {k: round(v, 3) for k, v in sorted(budget.by_task.items())},
               "fal_balance_ran_out": bool(studio and studio.balance_ran_out),
               "judge": {str(k): v for k, v in verdicts.items()},
               "agents": {k: reports.get(k) for k in ("art", "storyboard")},
               "minutes": round((time.time() - started) / 60, 1)}
    (out_dir / "report.json").write_text(json.dumps(summary, indent=2, default=str))
    (out_dir / "beats.json").write_text(json.dumps([{"index": b.index, "text": b.text, "picture": b.picture,
                                                     "characters": b.characters, "scare": b.scare}
                                                    for b in beats], indent=2))
    shutil.rmtree(work, ignore_errors=True)
    log.info("Bake-off: %s", {n: {k: results[n][k] for k in ("usd", "refusals", "judge")} for n in names})
    return summary

"""The Storyteller's fixed references, painted once.

The host is the same man in every episode, so his character sheet and close portrait are made
once by the strongest painter, looked at by a person, and kept in the repository
(`stills.host_references`); every video then paints him from the same two pictures for free.

    python -m production --config config.aws-open.toml host --out out/host_refs   # ~$0.60 for 2 candidates

Each candidate is a sheet painted from words alone and a portrait painted from that sheet, both
checked by the vision model (face in shadow, one man, a plain lantern). The pictures are saved
at full size as JPEG with host_report.json; copy the chosen pair to production/assets/host/.
"""

from __future__ import annotations

import io
import base64
import json
import logging
from pathlib import Path
from typing import Any

from PIL import Image

from production.media import MediaError

from ..paid import Budget, OverBudget, Studio
from .art import HOST_DESIGN, HOST_FACE
from .checks import sheet_ok
from .images import house_look
from .storyboard import HOST

log = logging.getLogger(__name__)

LANTERN = ("The lantern is a plain old iron railway lantern with glass panes and a blood-red flame inside: no "
           "faces, masks, skulls or carvings on it.")


def _jpeg_uri(path: Path) -> str:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((2048, 2048), Image.LANCZOS)
        data = io.BytesIO()
        im.save(data, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(data.getvalue()).decode()


def _keep(src: Path, out: Path) -> Path:
    with Image.open(src) as im:
        im.convert("RGB").save(out, quality=95)
    return out


def make(cfg: dict[str, Any], fal: Any, out: Path, candidates: int = 2, max_usd: float = 1.5) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    look = house_look(cfg)
    house, avoid = look["style"], look.get("avoid", "")
    budget = Budget(max_usd, cfg.get("prices", {}))
    studio = Studio(fal, budget, work / "cache", cfg)
    sheet_text = (f"{house} A character reference sheet of one man, {HOST}, the host of a horror channel: "
                  f"{HOST_DESIGN}. {HOST_FACE} The same man three times side by side on a plain dark background, "
                  "lit only by the red glow of his lantern from below: full body from the front, full body in "
                  "three-quarter view, and a head-and-shoulders close-up. Identical hat, coat and lantern in all "
                  f"three. {LANTERN} No text, no labels. {avoid}").strip()
    face_text = (f"{house} A close head-and-shoulders portrait of the man in the reference image, {HOST}: "
                 f"{HOST_DESIGN}. Exactly the same man, hat and coat. {HOST_FACE} Lit only by his lantern's red "
                 f"glow from below, against a plain dark background. One single frame. {LANTERN} No text, no "
                 f"labels. {avoid}").strip()
    made: list[dict[str, Any]] = []
    for k in range(1, candidates + 1):
        item: dict[str, Any] = {"candidate": k}
        try:
            raw = studio.call("hero_text", {"prompt": sheet_text, "seed": 7000 + k}, "image",
                              work / f"sheet_{k}.png", "host_sheet", key=["host_sheet", k, sheet_text])
            sheet = _keep(raw, out / f"host_sheet_{k}.jpg")
            item["sheet"], item["sheet_check"] = sheet.name, sheet_ok(studio, sheet, "character", HOST, HOST_DESIGN)
            raw = studio.call("hero", {"prompt": face_text, "image_urls": [_jpeg_uri(raw)], "seed": 7100 + k},
                              "image", work / f"portrait_{k}.png", "host_portrait", key=["host_portrait", k, face_text])
            face = _keep(raw, out / f"host_portrait_{k}.jpg")
            item["portrait"], item["portrait_check"] = face.name, sheet_ok(studio, face, "portrait", HOST, HOST_DESIGN)
        except (MediaError, OverBudget) as exc:
            item["error"] = str(exc)[:300]
            log.warning("Host candidate %d: %s", k, exc)
        made.append(item)
    report = {"candidates": made, "usd": round(budget.spent, 2),
              "next": "copy the chosen host_sheet_N.jpg and host_portrait_N.jpg to production/assets/host/"
                      "sheet.jpg and portrait.jpg"}
    (out / "host_report.json").write_text(json.dumps(report, indent=2, default=str))
    import shutil
    shutil.rmtree(work, ignore_errors=True)
    return report

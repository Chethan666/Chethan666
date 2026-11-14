"""Packaging: the review kit (the thumbnails are the stills packager's, in
production/stills/post.py).

- A contact sheet of every shot and a 60-second preview, so a person can approve the video in
  two minutes before anything is published.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from . import draw
from .shots import Shot

log = logging.getLogger(__name__)

def contact_sheet(shots: list[Shot], pictures: dict[int, Path], out: Path,
                  columns: int = 8, thumb: tuple[int, int] = (240, 135), moving: set[int] | None = None) -> Path:
    rows = -(-len(shots) // columns) or 1
    sheet = Image.new("RGB", (columns * (thumb[0] + 8) + 8, rows * (thumb[1] + 34) + 8), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    f = draw.font(16, bold=False)
    for k, s in enumerate(shots):
        x, y = 8 + (k % columns) * (thumb[0] + 8), 8 + (k // columns) * (thumb[1] + 34)
        path = pictures.get(s.index)
        if path and path.exists():
            with Image.open(path) as im:
                sheet.paste(im.convert("RGB").resize(thumb), (x, y))
        clip = " CLIP" if moving and s.index in moving else ""
        d.text((x, y + thumb[1] + 4), f"#{s.index} {s.framing}{clip}{' SCARE' if s.scare else ''}", font=f,
               fill=(220, 60, 60) if s.scare else (200, 200, 200))
    sheet.save(out, quality=85)
    return out


def preview(video: Path, out: Path, seconds: float = 60.0) -> Path | None:
    result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-t",
                             f"{seconds:.0f}", "-c", "copy", str(out)], capture_output=True, text=True)
    return out if result.returncode == 0 else None

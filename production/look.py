"""The video frame and fitting a picture to a size."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFilter

W, H = 1920, 1080


def fit(src: Path, dst: Path, size: tuple[int, int] = (W, H)) -> Path:
    """Cover-crop any picture to `size`."""
    with Image.open(src) as im:
        im = im.convert("RGB")
        scale = max(size[0] / im.width, size[1] / im.height)
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        if scale > 1.05:  # an enlarged picture goes soft: give back some edge
            im = im.filter(ImageFilter.UnsharpMask(radius=1.6, percent=55, threshold=2))
        x, y = (im.width - size[0]) // 2, (im.height - size[1]) // 2
        im.crop((x, y, x + size[0], y + size[1])).save(dst)
    return dst

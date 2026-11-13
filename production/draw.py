"""Lettering for cards, Shorts and the fallback thumbnail: the channel's two colours and its font."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

CHALK = (236, 234, 226)  # the channel's bone white
RED = (200, 24, 30)  # and its blood red


def font(size: int, bold: bool = True) -> ImageFont.ImageFont:
    path = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def thumbnail(picture: Image.Image, text: str, size: tuple[int, int] = (1280, 720)) -> Image.Image:
    """The key moment's picture with up to three big words, the last one in red."""
    image = picture.resize(size, Image.LANCZOS)
    words = text.upper().split()[:3]
    if not words:
        return image
    draw = ImageDraw.Draw(image)
    f = font(int(size[1] * 0.16))
    x, y = int(size[0] * 0.05), int(size[1] * 0.07)
    for i, word in enumerate(words):
        color = RED if i == len(words) - 1 and len(words) > 1 else CHALK
        draw.text((x, y), word, font=f, fill=color, stroke_width=int(size[1] * 0.012), stroke_fill=(0, 0, 0))
        y += int(size[1] * 0.18)
    return image

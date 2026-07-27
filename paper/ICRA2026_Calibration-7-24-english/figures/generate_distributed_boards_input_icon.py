#!/usr/bin/env python3
"""Render a text-free hand-drawn input glyph for the method overview."""

from __future__ import annotations

from pathlib import Path
from random import Random

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "pic" / "generated" / "distributed_boards_input_sketch.png"
BLUE = (27, 102, 160)
PALE_BLUE = (27, 102, 160, 28)
RNG = Random(14)


def jitter(value: float, amount: float = 1.8) -> float:
    return value + RNG.uniform(-amount, amount)


def hand_line(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], *, width: int = 7, dashed: bool = False) -> None:
    """Draw two close imperfect marker strokes for a restrained sketch effect."""
    for stroke, alpha in ((0, 220), (1, 100)):
        shifted = [(jitter(x, 1.1 + stroke), jitter(y, 1.1 + stroke)) for x, y in points]
        if dashed:
            for start in range(0, len(shifted) - 1, 2):
                draw.line(shifted[start : start + 2], fill=(*BLUE, alpha), width=max(2, width - 2))
        else:
            draw.line(shifted, fill=(*BLUE, alpha), width=width, joint="curve")


def hand_circle(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float], *, width: int = 7) -> None:
    for offset, alpha in ((0.0, 220), (1.2, 96)):
        x0, y0, x1, y1 = box
        draw.ellipse(
            (jitter(x0, offset + 0.8), jitter(y0, offset + 0.8), jitter(x1, offset + 0.8), jitter(y1, offset + 0.8)),
            outline=(*BLUE, alpha),
            width=width,
        )


def make_tag(size: int, angle: float) -> Image.Image:
    tile = Image.new("RGBA", (size * 2, size * 2), (255, 255, 255, 0))
    draw = ImageDraw.Draw(tile, "RGBA")
    m = size // 2
    border = int(size * 0.11)
    hand_line(draw, [(m - size * 0.42, m - size * 0.42), (m + size * 0.42, m - size * 0.42),
                     (m + size * 0.42, m + size * 0.42), (m - size * 0.42, m + size * 0.42),
                     (m - size * 0.42, m - size * 0.42)], width=max(5, border))

    # A minimal, non-branded AprilTag-like binary structure, all in the same ink.
    cell = size * 0.16
    for ix, iy in ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (0, 1)):
        x = int(m + ix * cell - cell * 0.34)
        y = int(m + iy * cell - cell * 0.34)
        draw.rectangle((x, y, int(x + cell * 0.68), int(y + cell * 0.68)), fill=(*BLUE, 210))

    return tile.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)


def main() -> None:
    width, height = 1600, 1080
    background = Image.new("RGBA", (width, height), "white")
    ink = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    glow = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(ink, "RGBA")
    glow_draw = ImageDraw.Draw(glow, "RGBA")

    cx, cy, radius = 800, 525, 355
    hand_circle(draw, (cx - radius, cy - radius, cx + radius, cy + radius), width=8)
    glow_draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=PALE_BLUE, width=18)

    # Subtle lens guide arcs make the circular camera field explicit without labels.
    draw.arc((cx - 0.72 * radius, cy - 0.72 * radius, cx + 0.72 * radius, cy + 0.72 * radius),
             start=206, end=334, fill=(*BLUE, 108), width=4)
    draw.arc((cx - 0.40 * radius, cy - 0.40 * radius, cx + 0.40 * radius, cy + 0.40 * radius),
             start=30, end=152, fill=(*BLUE, 82), width=3)

    # Tags intentionally span center, middle, and peripheral areas of the image circle.
    tag_specs = [
        (800, 286, 116, -3),
        (548, 372, 104, 17),
        (1056, 374, 104, -15),
        (510, 663, 98, -17),
        (1090, 671, 96, 18),
        (807, 736, 92, 3),
        (790, 515, 88, -8),
    ]
    for x, y, size, angle in tag_specs:
        # Dashed geometric rays express coverage while keeping the drawing abstract.
        hand_line(draw, [(cx, cy), (x, y)], width=4, dashed=True)
        tag = make_tag(size, angle)
        background.alpha_composite(tag, (int(x - tag.width / 2), int(y - tag.height / 2)))

    # The optical center is deliberately small and neutral.
    hand_circle(draw, (cx - 13, cy - 13, cx + 13, cy + 13), width=5)
    background = Image.alpha_composite(background, glow.filter(ImageFilter.GaussianBlur(7)))
    background = Image.alpha_composite(background, ink)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    background.convert("RGB").save(OUTPUT, quality=96, dpi=(420, 420))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()

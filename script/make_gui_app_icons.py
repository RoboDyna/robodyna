#!/usr/bin/env python3
"""Render the per-suite RoboDyna app icons.

Each GUI gets its own dock/taskbar icon with three stacked lines at one shared
font size: ``Robo`` / ``Dyna`` in stamp ink and the suite word in the suite
accent color (Base teal, HH orange, Exp coral). The shared hub launcher
(``interactive/robodyna_gui.py``) instead gets a two-line ``Robo`` / ``Dyna``
mark whose letters cycle through the four brand colors. The cream paper and its
grain come from the original ``robodyna_app_icon.png`` stamp.

Run after changing colors, wording, or layout::

    python script/make_gui_app_icons.py
"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
TEXTURE_DIR = ROOT / "assets" / "dyna_textures"
STAMP_PATH = TEXTURE_DIR / "robodyna_app_icon.png"

# Condensed heavy face, closest stock Ubuntu match to the stamp's compressed
# letterforms; stroke_width thickens it further. DejaVu is the fallback if the
# variable font or its named instance is unavailable.
FONT_PATH = Path("/usr/share/fonts/truetype/ubuntu/UbuntuSans[wdth,wght].ttf")
FONT_VARIATION = "Condensed ExtraBold"
FONT_FALLBACK = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
STROKE = 3

CANVAS = 512
CREAM = (254, 248, 235)
INK = (52, 56, 61)
SIDE_MARGIN = 30
EDGE_MARGIN = 14
# Gap between cap bands, as a fraction of cap height.
GAP_RATIO = 0.16
# Grain amplitude for the glyph fill, matching the stamp's speckled ink.
GRAIN = 20

# suite key (as passed to setup_gui_app_icon) -> (word, accent color, filename)
SUITE_ICONS = {
    "interactive": ("Base", (2, 156, 163), "robodyna_app_icon_base.png"),
    "household": ("HH", (253, 139, 23), "robodyna_app_icon_hh.png"),
    "experiment": ("Exp", (239, 84, 56), "robodyna_app_icon_exp.png"),
}
MARK_LINES = ("Robo", "Dyna")

# Hub launcher icon: no suite word, ``Robo`` in logo navy, and the ``Dyna``
# letters cycling through the four brand colors position by position.
HUB_MARK_INK = (4, 43, 76)
HUB_LETTER_COLORS = (
    (1, 157, 162),
    (253, 142, 25),
    (237, 82, 57),
    (52, 139, 215),
)
HUB_ICON_FILENAME = "robodyna_app_icon_gui.png"


def _font(size: int) -> ImageFont.FreeTypeFont:
    if FONT_PATH.is_file():
        font = ImageFont.truetype(str(FONT_PATH), size)
        try:
            font.set_variation_by_name(FONT_VARIATION)
            return font
        except (OSError, ValueError):
            pass
    return ImageFont.truetype(str(FONT_FALLBACK), size)


def _cap_height(font: ImageFont.FreeTypeFont) -> int:
    cap = font.getbbox("H", stroke_width=STROKE)
    return cap[3] - cap[1]


def _descent(font: ImageFont.FreeTypeFont, word: str) -> int:
    """Depth of ``word`` below the baseline (the y in Dyna, the p in Exp)."""
    # getbbox measures from the ascender top; the baseline sits at the cap
    # bottom, so anything past it is descender depth.
    baseline = font.getbbox("H", stroke_width=STROKE)[3]
    return max(0, font.getbbox(word, stroke_width=STROKE)[3] - baseline)


def _ascent(font: ImageFont.FreeTypeFont, word: str) -> int:
    """Ink height of ``word`` above the baseline (the b in Robo tops the caps)."""
    baseline = font.getbbox("H", stroke_width=STROKE)[3]
    return baseline - font.getbbox(word, stroke_width=STROKE)[1]


def _layout(font: ImageFont.FreeTypeFont) -> tuple[list[int], int, int]:
    """Baselines (relative to the block's ink top), block height, widest line.

    Each line's own descender is added to the gap below it, so a tail like the
    ``y`` in Dyna clears the next line's caps instead of colliding with them.
    The suite words share one worst-case layout, keeping the three icons aligned.
    """
    words = [spec[0] for spec in SUITE_ICONS.values()]
    cap_h = _cap_height(font)
    gap = round(cap_h * GAP_RATIO)
    baselines = [_ascent(font, MARK_LINES[0])]
    for text in MARK_LINES:
        baselines.append(baselines[-1] + cap_h + gap + _descent(font, text))
    block = baselines[-1] + max(_descent(font, word) for word in words)
    widest = max(
        font.getbbox(text, stroke_width=STROKE)[2]
        - font.getbbox(text, stroke_width=STROKE)[0]
        for text in list(MARK_LINES) + words
    )
    return baselines, block, widest


def _fit_font() -> ImageFont.FreeTypeFont:
    """Largest shared font size where the three stacked lines fit the canvas."""
    max_w = CANVAS - 2 * SIDE_MARGIN
    max_h = CANVAS - 2 * EDGE_MARGIN
    fitted = None
    for size in range(24, 400):
        font = _font(size)
        _, block, widest = _layout(font)
        if widest > max_w or block > max_h:
            break
        fitted = font
    if fitted is None:
        raise RuntimeError(f"cannot fit the icon words into {max_w}x{max_h}")
    return fitted


def _paper(stamp: Image.Image) -> Image.Image:
    """Cream canvas tiled from the stamp's blank margin so the grain matches."""
    patch = stamp.crop((0, 0, 48, 19)).resize((64, 64), Image.Resampling.LANCZOS)
    paper = Image.new("RGB", (CANVAS, CANVAS), CREAM)
    for y in range(0, CANVAS, patch.height):
        for x in range(0, CANVAS, patch.width):
            paper.paste(patch, (x, y))
    return paper


def _speckle(color: tuple[int, int, int], rng: random.Random) -> Image.Image:
    """Full-canvas noisy fill so glyphs get the stamp's speckled ink."""
    fill = Image.new("RGB", (CANVAS, CANVAS))
    fill.putdata(
        [
            tuple(min(255, max(0, c + rng.randint(-GRAIN, GRAIN))) for c in color)
            for _ in range(CANVAS * CANVAS)
        ]
    )
    return fill


def render_icon(
    stamp: Image.Image,
    word: str,
    color: tuple[int, int, int],
    font: ImageFont.FreeTypeFont,
) -> Image.Image:
    icon = _paper(stamp)
    lines = [(text, INK) for text in MARK_LINES] + [(word, color)]

    baselines, block, _ = _layout(font)
    block_top = (CANVAS - block) // 2

    rng = random.Random(0xB0B0)
    for (text, line_color), offset in zip(lines, baselines):
        baseline = block_top + offset
        stencil = Image.new("L", (CANVAS, CANVAS), 0)
        ImageDraw.Draw(stencil).text(
            (CANVAS // 2, baseline),
            text,
            font=font,
            anchor="ms",
            fill=255,
            stroke_width=STROKE,
            stroke_fill=255,
        )
        icon.paste(_speckle(line_color, rng), (0, 0), stencil)
    return icon


def _hub_layout(font: ImageFont.FreeTypeFont) -> tuple[list[int], int, float]:
    """Baselines, block height, and widest line for the two-line hub mark."""
    cap_h = _cap_height(font)
    gap = round(cap_h * GAP_RATIO)
    baselines = [_ascent(font, MARK_LINES[0])]
    baselines.append(baselines[0] + cap_h + gap + _descent(font, MARK_LINES[0]))
    block = baselines[-1] + max(_descent(font, text) for text in MARK_LINES)
    widest = max(font.getlength(text) for text in MARK_LINES) + 2 * STROKE
    return baselines, block, widest


def _fit_hub_font() -> ImageFont.FreeTypeFont:
    max_w = CANVAS - 2 * SIDE_MARGIN
    max_h = CANVAS - 2 * EDGE_MARGIN
    fitted = None
    for size in range(24, 400):
        font = _font(size)
        _, block, widest = _hub_layout(font)
        if widest > max_w or block > max_h:
            break
        fitted = font
    if fitted is None:
        raise RuntimeError(f"cannot fit the hub words into {max_w}x{max_h}")
    return fitted


def render_hub_icon(stamp: Image.Image, font: ImageFont.FreeTypeFont) -> Image.Image:
    """``Robo`` in logo navy over ``Dyna`` with one brand color per letter."""
    icon = _paper(stamp)
    baselines, block, _ = _hub_layout(font)
    block_top = (CANVAS - block) // 2
    robo, dyna = MARK_LINES

    rng = random.Random(0xB0B0)

    def stamp_text(xy: tuple[float, float], text: str, color: tuple[int, int, int], anchor: str):
        stencil = Image.new("L", (CANVAS, CANVAS), 0)
        ImageDraw.Draw(stencil).text(
            xy,
            text,
            font=font,
            anchor=anchor,
            fill=255,
            stroke_width=STROKE,
            stroke_fill=255,
        )
        icon.paste(_speckle(color, rng), (0, 0), stencil)

    stamp_text((CANVAS // 2, block_top + baselines[0]), robo, HUB_MARK_INK, "ms")
    # Dyna letters are drawn one at a time at their prefix advance so each can
    # take its own color while keeping the whole-word spacing.
    baseline = block_top + baselines[1]
    left = (CANVAS - font.getlength(dyna)) / 2
    for index, letter in enumerate(dyna):
        color = HUB_LETTER_COLORS[index % len(HUB_LETTER_COLORS)]
        stamp_text((left + font.getlength(dyna[:index]), baseline), letter, color, "ls")
    return icon


def main() -> int:
    stamp = Image.open(STAMP_PATH).convert("RGB")
    font = _fit_font()
    baselines, block, _ = _layout(font)
    print(
        f"shared font size {font.size} (cap height {_cap_height(font)}px, "
        f"baselines {baselines}, block {block}px)"
    )
    for word, color, filename in SUITE_ICONS.values():
        dest = TEXTURE_DIR / filename
        render_icon(stamp, word, color, font).save(dest, optimize=True)
        print(f"wrote {dest.relative_to(ROOT)}")
    hub_dest = TEXTURE_DIR / HUB_ICON_FILENAME
    render_hub_icon(stamp, _fit_hub_font()).save(hub_dest, optimize=True)
    print(f"wrote {hub_dest.relative_to(ROOT)}")
    print("Run script/install_gui_dock_icons.py to refresh the Ubuntu dock entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

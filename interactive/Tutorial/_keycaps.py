"""Keycap PNGs for tutorial HUD overlays (same look as the arrow-key demo)."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

KEY_SIZE = 180
RADIUS = 28
PAD = 16
GAP = 14
CANVAS_PAD = 12
SUBLABEL_H = 38
INSTRUCTION_H = 78
TITLE_H = 40

ARM_SUBLABELS = ("left", "right", "both")
ARM_INSTRUCTION = "Select gripper"
VIEW_INSTRUCTION = "Switch view,\nhead camera and gripper"
ARROW_INSTRUCTION = "Move the arm left, right, forward, and back"
HEIGHT_SUBLABELS = ("up", "down")
HEIGHT_INSTRUCTION = "Raise and lower the arm.\nZ min and max is capped."
YAW_SUBLABELS = ("left", "right")
YAW_INSTRUCTION = "Rotate the gripper left or right"
TILT_SUBLABELS = ("left", "right")
TILT_INSTRUCTION = "Tilt the gripper left or right"
SPACE_INSTRUCTION = "Open and close the gripper"
SPACE_BAR_W = KEY_SIZE * 2 + GAP
GRASP_TITLE = "Pick the cube"
GRASP_INSTRUCTION = "Arrows move. E/Q raise and lower. Space closes."
HOLD_TITLE = "Press the button"
HOLD_INSTRUCTION = "Arrows to the button. Space closes. Q to hold; E to release."
SWITCH_TITLE = "Turn key on/off"
SWITCH_INSTRUCTION = "Arrows to aim. Space closes. Q turns it ON, Q again OFF."
PUSH_TITLE = "Push the box over the line"
PUSH_INSTRUCTION = "Close gripper, lower arm and push"
BALL_TITLE = "Pick up the ball"
BALL_INSTRUCTION = "Ball rolls toward you. Space closes; E lifts. Falls off → respawns."
STOVE_TITLE = "Turn on/off the stove"
STOVE_INSTRUCTION = "Arrows + Q/E to the knob. Space grasp. R/T yaw: left lights fire."
MALLET_TITLE = "Pick up the mallet"
MALLET_INSTRUCTION = "Space closes on the handle; E lifts the mallet."
FORCE_TITLE = "Press multistep button"
FORCE_INSTRUCTION = (
    "Press Q until the bar enters the yellow band — that counts. "
    "Release, then press again for the next yellow."
)
FORCE_BAR_H = 56
FORCE_STEP_LABELS = ("1 light", "2 medium", "3 firm", "4 full")

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
)

_FILL = (36, 38, 44, 255)
_FILL_INNER = (48, 51, 58, 255)
_OUTLINE = (210, 214, 220, 255)
_INNER_OUTLINE = (88, 92, 102, 255)
_GLYPH = (245, 247, 250, 255)
_FILL_DONE = (28, 72, 46, 255)
_FILL_INNER_DONE = (36, 92, 58, 255)
_OUTLINE_DONE = (120, 220, 150, 255)
_INNER_OUTLINE_DONE = (70, 160, 100, 255)
_GLYPH_DONE = (230, 255, 236, 255)
_CANVAS = (18, 18, 22, 230)
_SUBLABEL = (200, 204, 212, 255)
_INSTRUCTION = (232, 234, 238, 255)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _rounded_rect(draw, xy, radius, fill, outline=None, width=3):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_keycap(label: str, *, pressed: bool = False, size: int = KEY_SIZE) -> Image.Image:
    scale = size / float(KEY_SIZE)
    pad = max(6, int(round(PAD * scale)))
    radius = max(8, int(round(RADIUS * scale)))
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    fill = _FILL_DONE if pressed else _FILL
    inner = _FILL_INNER_DONE if pressed else _FILL_INNER
    outline = _OUTLINE_DONE if pressed else _OUTLINE
    inner_outline = _INNER_OUTLINE_DONE if pressed else _INNER_OUTLINE
    glyph = _GLYPH_DONE if pressed else _GLYPH
    _rounded_rect(
        d,
        [pad + 3, pad + 6, size - pad + 3, size - pad + 6],
        radius,
        fill=(0, 0, 0, 90),
    )
    _rounded_rect(
        d,
        [pad, pad, size - pad, size - pad],
        radius,
        fill=fill,
        outline=outline,
        width=max(2, int(round(4 * scale))),
    )
    inset = max(5, int(round(8 * scale)))
    _rounded_rect(
        d,
        [pad + inset, pad + inset, size - pad - inset, size - pad - inset],
        max(4, radius - inset),
        fill=inner,
        outline=inner_outline,
        width=max(1, int(round(2 * scale))),
    )
    text = str(label)
    if len(text) == 1:
        font_px = int(round(92 * scale))
    elif len(text) == 2:
        font_px = int(round(64 * scale))
    else:
        font_px = int(round(48 * scale))
    font = _font(max(16, font_px))
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - tw) / 2 - bbox[0]
    y = (size - th) / 2 - bbox[1] - 2
    d.text((x, y), text, font=font, fill=glyph)
    return img


def cluster_size() -> tuple[int, int]:
    return (
        KEY_SIZE * 3 + GAP * 2 + CANVAS_PAD * 2,
        KEY_SIZE + CANVAS_PAD * 2 + SUBLABEL_H + INSTRUCTION_H,
    )


def _wrap_text(text: str, font, draw, max_width: int) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        current = ""
        for word in para.split():
            trial = word if not current else f"{current} {word}"
            bbox = draw.textbbox((0, 0), trial, font=font)
            if bbox[2] - bbox[0] <= max_width or not current:
                current = trial
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines or [text]


def _draw_centered_lines(draw, lines: list[str], font, y: int, canvas_w: int, fill) -> None:
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (canvas_w - tw) / 2 - bbox[0]
        draw.text((x, y - bbox[1]), line, font=font, fill=fill)
        y += th + 4


def _measure_lines_h(lines: list[str], font, draw) -> int:
    total = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        total += (bbox[3] - bbox[1]) + 4
    return total


def _overlay_canvas(
    width: int,
    body_h: int,
    instruction: str = "",
    title: str = "",
    *,
    font_size: int = 26,
    title_size: int = 28,
):
    """Canvas with optional title above the keys and instruction below."""
    probe = Image.new("RGBA", (max(8, width), 8), _CANVAS)
    draw = ImageDraw.Draw(probe)
    title_font = _font(title_size)
    title_lines = _wrap_text(title, title_font, draw, width - CANVAS_PAD * 2) if title else []
    title_h = _measure_lines_h(title_lines, title_font, draw) if title_lines else 0
    inst_font = _font(font_size)
    inst_lines = (
        _wrap_text(instruction, inst_font, draw, width - CANVAS_PAD * 2) if instruction else []
    )
    inst_h = _measure_lines_h(inst_lines, inst_font, draw) if inst_lines else 0
    title_band = (CANVAS_PAD + title_h + 10) if title else 0
    bottom = (16 + inst_h + 12) if instruction else CANVAS_PAD
    height = title_band + int(body_h) + bottom
    canvas = Image.new("RGBA", (width, int(height)), _CANVAS)
    title_y = CANVAS_PAD
    body_y = title_band
    inst_y = title_band + int(body_h) + 8
    return (
        canvas,
        ImageDraw.Draw(canvas),
        title_lines,
        title_font,
        title_y,
        inst_lines,
        inst_font,
        inst_y,
        body_y,
    )


def _paint_overlay_text(
    d,
    width: int,
    title_lines,
    title_font,
    title_y: int,
    inst_lines,
    inst_font,
    inst_y: int,
) -> None:
    if title_lines:
        _draw_centered_lines(d, title_lines, title_font, title_y, width, _INSTRUCTION)
    if inst_lines:
        _draw_centered_lines(d, inst_lines, inst_font, inst_y, width, _INSTRUCTION)


def draw_key_row(
    labels: tuple[str, ...],
    pressed: set[str] | None = None,
    *,
    sublabels: tuple[str, ...] | None = None,
    instruction: str = "",
    title: str = "",
) -> Image.Image:
    pressed = pressed or set()
    width = cluster_size()[0]
    body_h = KEY_SIZE + (SUBLABEL_H if sublabels else 10)
    canvas, d, title_lines, title_font, title_y, lines, inst_font, inst_y, y0 = _overlay_canvas(
        width, body_h, instruction, title=title
    )
    n = len(labels)
    row_w = n * KEY_SIZE + max(0, n - 1) * GAP
    x0 = (width - row_w) // 2
    sub_font = _font(28)
    for i, label in enumerate(labels):
        cap = draw_keycap(label, pressed=label in pressed)
        x = x0 + i * (KEY_SIZE + GAP)
        canvas.alpha_composite(cap, (x, y0))
        if sublabels is not None and i < len(sublabels):
            hint = sublabels[i]
            bbox = d.textbbox((0, 0), hint, font=sub_font)
            tw = bbox[2] - bbox[0]
            tx = x + (KEY_SIZE - tw) / 2 - bbox[0]
            ty = y0 + KEY_SIZE + 4 - bbox[1]
            d.text((tx, ty), hint, font=sub_font, fill=_SUBLABEL)
    _paint_overlay_text(d, width, title_lines, title_font, title_y, lines, inst_font, inst_y)
    return canvas


def draw_arm_keys(pressed: set[str] | None = None) -> Image.Image:
    return draw_key_row(
        ("1", "2", "3"),
        pressed,
        sublabels=ARM_SUBLABELS,
        instruction=ARM_INSTRUCTION,
    )


def draw_view_key(pressed: bool = False) -> Image.Image:
    return draw_key_row(
        ("V",),
        {"V"} if pressed else set(),
        instruction=VIEW_INSTRUCTION,
    )


PLAY_KEY_SIZE = 112
PLAY_GAP = 10
PLAY_PAD = 10
PLAY_SUB_H = 34
PLAY_LABELS = ("1", "2", "3", "V", "Esc")
PLAY_SUBLABELS = ("left", "right", "both", "view", "quit")


def play_cluster_size() -> tuple[int, int]:
    n = len(PLAY_LABELS)
    return (
        n * PLAY_KEY_SIZE + (n - 1) * PLAY_GAP + PLAY_PAD * 2,
        PLAY_KEY_SIZE + PLAY_PAD * 2 + PLAY_SUB_H,
    )


def draw_play_keys(pressed: set[str] | None = None) -> Image.Image:
    """Compact 1 / 2 / 3 / V / Esc strip used after the Part 1 lesson."""
    pressed = pressed or set()
    width, height = play_cluster_size()
    canvas = Image.new("RGBA", (width, height), _CANVAS)
    d = ImageDraw.Draw(canvas)
    sub_font = _font(20)
    y0 = PLAY_PAD
    x0 = PLAY_PAD
    for i, label in enumerate(PLAY_LABELS):
        cap = draw_keycap(label, pressed=label in pressed, size=PLAY_KEY_SIZE)
        x = x0 + i * (PLAY_KEY_SIZE + PLAY_GAP)
        canvas.alpha_composite(cap, (x, y0))
        hint = PLAY_SUBLABELS[i]
        bbox = d.textbbox((0, 0), hint, font=sub_font)
        tw = bbox[2] - bbox[0]
        tx = x + (PLAY_KEY_SIZE - tw) / 2 - bbox[0]
        ty = y0 + PLAY_KEY_SIZE + 2 - bbox[1]
        d.text((tx, ty), hint, font=sub_font, fill=_SUBLABEL)
    return canvas


PLAY2_KEY = 72
PLAY2_GAP = 8
PLAY2_GROUP = 16
PLAY2_PAD = 10
PLAY2_CAPTION = "Practice — Esc quits."


def _play2_arrow_block() -> tuple[int, int]:
    k, g = PLAY2_KEY, PLAY2_GAP
    return (3 * k + 2 * g, 2 * k + g)


def part2_play_cluster_size() -> tuple[int, int]:
    aw, ah = _play2_arrow_block()
    letters_w = 6 * PLAY2_KEY + 3 * PLAY2_GAP + 2 * PLAY2_GROUP
    width = PLAY2_PAD + aw + PLAY2_GROUP + letters_w + PLAY2_PAD
    height = PLAY2_PAD + ah + PLAY2_GAP + PLAY2_KEY + PLAY2_PAD + PLAY_SUB_H
    return width, height


def draw_part2_play_keys(pressed: set[str] | None = None) -> Image.Image:
    """Compact arrows + E/Q + R/T + F/G + Space + Esc for Part 2 practice."""
    pressed = pressed or set()
    width, height = part2_play_cluster_size()
    canvas = Image.new("RGBA", (width, height), _CANVAS)
    d = ImageDraw.Draw(canvas)
    k, g = PLAY2_KEY, PLAY2_GAP
    x0 = y0 = PLAY2_PAD
    aw, ah = _play2_arrow_block()
    canvas.alpha_composite(
        draw_arrow_keycap("up", pressed="up" in pressed, size=k),
        (x0 + k + g, y0),
    )
    by = y0 + k + g
    canvas.alpha_composite(
        draw_arrow_keycap("left", pressed="left" in pressed, size=k),
        (x0, by),
    )
    canvas.alpha_composite(
        draw_arrow_keycap("down", pressed="down" in pressed, size=k),
        (x0 + k + g, by),
    )
    canvas.alpha_composite(
        draw_arrow_keycap("right", pressed="right" in pressed, size=k),
        (x0 + 2 * (k + g), by),
    )
    lx = x0 + aw + PLAY2_GROUP
    groups = (("E", "Q"), ("R", "T"), ("F", "G"))
    for gi, pair in enumerate(groups):
        gx = lx + gi * (2 * k + PLAY2_GAP + PLAY2_GROUP)
        for j, label in enumerate(pair):
            cap = draw_keycap(label, pressed=label in pressed, size=k)
            canvas.alpha_composite(cap, (gx + j * (k + g), by))
    sy = y0 + ah + PLAY2_GAP
    canvas.alpha_composite(
        draw_wide_keycap("Space", pressed="Space" in pressed, width=aw, height=k),
        (x0, sy),
    )
    canvas.alpha_composite(
        draw_keycap("Esc", pressed="Esc" in pressed, size=k),
        (lx, sy),
    )
    inst_font = _font(22)
    lines = _wrap_text(PLAY2_CAPTION, inst_font, d, width - PLAY2_PAD * 2)
    _draw_centered_lines(d, lines, inst_font, sy + k + 6, width, _INSTRUCTION)
    return canvas


def _arrow_pts(cx, cy, direction, length=46, width=38):
    if direction == "up":
        return [
            (cx, cy - length),
            (cx - width, cy + length * 0.45),
            (cx - width * 0.28, cy + length * 0.45),
            (cx - width * 0.28, cy + length),
            (cx + width * 0.28, cy + length),
            (cx + width * 0.28, cy + length * 0.45),
            (cx + width, cy + length * 0.45),
        ]
    if direction == "down":
        return [
            (cx, cy + length),
            (cx - width, cy - length * 0.45),
            (cx - width * 0.28, cy - length * 0.45),
            (cx - width * 0.28, cy - length),
            (cx + width * 0.28, cy - length),
            (cx + width * 0.28, cy - length * 0.45),
            (cx + width, cy - length * 0.45),
        ]
    if direction == "left":
        return [
            (cx - length, cy),
            (cx + length * 0.45, cy - width),
            (cx + length * 0.45, cy - width * 0.28),
            (cx + length, cy - width * 0.28),
            (cx + length, cy + width * 0.28),
            (cx + length * 0.45, cy + width * 0.28),
            (cx + length * 0.45, cy + width),
        ]
    return [
        (cx + length, cy),
        (cx - length * 0.45, cy - width),
        (cx - length * 0.45, cy - width * 0.28),
        (cx - length, cy - width * 0.28),
        (cx - length, cy + width * 0.28),
        (cx - length * 0.45, cy + width * 0.28),
        (cx - length * 0.45, cy + width),
    ]


def draw_arrow_keycap(direction: str, *, pressed: bool = False, size: int = KEY_SIZE) -> Image.Image:
    scale = size / float(KEY_SIZE)
    pad = max(6, int(round(PAD * scale)))
    radius = max(8, int(round(RADIUS * scale)))
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    fill = _FILL_DONE if pressed else _FILL
    inner = _FILL_INNER_DONE if pressed else _FILL_INNER
    outline = _OUTLINE_DONE if pressed else _OUTLINE
    inner_outline = _INNER_OUTLINE_DONE if pressed else _INNER_OUTLINE
    glyph = _GLYPH_DONE if pressed else _GLYPH
    _rounded_rect(
        d,
        [pad + 3, pad + 6, size - pad + 3, size - pad + 6],
        radius,
        fill=(0, 0, 0, 90),
    )
    _rounded_rect(
        d,
        [pad, pad, size - pad, size - pad],
        radius,
        fill=fill,
        outline=outline,
        width=max(2, int(round(4 * scale))),
    )
    inset = max(5, int(round(8 * scale)))
    _rounded_rect(
        d,
        [pad + inset, pad + inset, size - pad - inset, size - pad - inset],
        max(4, radius - inset),
        fill=inner,
        outline=inner_outline,
        width=max(1, int(round(2 * scale))),
    )
    cx = cy = size / 2
    d.polygon(
        _arrow_pts(cx, cy - 2 * scale, direction, length=46 * scale, width=38 * scale),
        fill=glyph,
    )
    return img


def arrow_cluster_size() -> tuple[int, int]:
    return (
        KEY_SIZE * 3 + GAP * 2 + CANVAS_PAD * 2,
        KEY_SIZE * 2 + GAP + CANVAS_PAD * 2 + INSTRUCTION_H,
    )


def draw_arrow_cluster(pressed: set[str] | None = None) -> Image.Image:
    pressed = pressed or set()
    width, height = arrow_cluster_size()
    canvas = Image.new("RGBA", (width, height), _CANVAS)
    d = ImageDraw.Draw(canvas)
    x0, y0 = CANVAS_PAD, CANVAS_PAD
    canvas.alpha_composite(
        draw_arrow_keycap("up", pressed="up" in pressed),
        (x0 + KEY_SIZE + GAP, y0),
    )
    canvas.alpha_composite(
        draw_arrow_keycap("left", pressed="left" in pressed),
        (x0, y0 + KEY_SIZE + GAP),
    )
    canvas.alpha_composite(
        draw_arrow_keycap("down", pressed="down" in pressed),
        (x0 + KEY_SIZE + GAP, y0 + KEY_SIZE + GAP),
    )
    canvas.alpha_composite(
        draw_arrow_keycap("right", pressed="right" in pressed),
        (x0 + 2 * (KEY_SIZE + GAP), y0 + KEY_SIZE + GAP),
    )
    inst_font = _font(26)
    max_w = width - CANVAS_PAD * 2
    lines = _wrap_text(ARROW_INSTRUCTION, inst_font, d, max_w)
    inst_y = y0 + KEY_SIZE * 2 + GAP + 8
    _draw_centered_lines(d, lines, inst_font, inst_y, width, _INSTRUCTION)
    return canvas


def draw_height_keys(pressed: set[str] | None = None) -> Image.Image:
    return draw_key_row(
        ("E", "Q"),
        pressed,
        sublabels=HEIGHT_SUBLABELS,
        instruction=HEIGHT_INSTRUCTION,
    )


def draw_yaw_keys(pressed: set[str] | None = None) -> Image.Image:
    return draw_key_row(
        ("R", "T"),
        pressed,
        sublabels=YAW_SUBLABELS,
        instruction=YAW_INSTRUCTION,
    )


def draw_tilt_keys(pressed: set[str] | None = None) -> Image.Image:
    return draw_key_row(
        ("F", "G"),
        pressed,
        sublabels=TILT_SUBLABELS,
        instruction=TILT_INSTRUCTION,
    )


def draw_wide_keycap(
    label: str,
    *,
    pressed: bool = False,
    width: int = SPACE_BAR_W,
    height: int = KEY_SIZE,
) -> Image.Image:
    scale = height / float(KEY_SIZE)
    pad = max(6, int(round(PAD * scale)))
    radius = max(8, int(round(RADIUS * scale)))
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    fill = _FILL_DONE if pressed else _FILL
    inner = _FILL_INNER_DONE if pressed else _FILL_INNER
    outline = _OUTLINE_DONE if pressed else _OUTLINE
    inner_outline = _INNER_OUTLINE_DONE if pressed else _INNER_OUTLINE
    glyph = _GLYPH_DONE if pressed else _GLYPH
    _rounded_rect(
        d,
        [pad + 3, pad + 6, width - pad + 3, height - pad + 6],
        radius,
        fill=(0, 0, 0, 90),
    )
    _rounded_rect(
        d,
        [pad, pad, width - pad, height - pad],
        radius,
        fill=fill,
        outline=outline,
        width=max(2, int(round(4 * scale))),
    )
    inset = max(5, int(round(8 * scale)))
    _rounded_rect(
        d,
        [pad + inset, pad + inset, width - pad - inset, height - pad - inset],
        max(4, radius - inset),
        fill=inner,
        outline=inner_outline,
        width=max(1, int(round(2 * scale))),
    )
    font = _font(max(16, int(round(56 * scale))))
    bbox = d.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (width - tw) / 2 - bbox[0]
    y = (height - th) / 2 - bbox[1] - 2
    d.text((x, y), label, font=font, fill=glyph)
    return img


def draw_space_key(pressed: bool = False) -> Image.Image:
    width, height = cluster_size()
    canvas = Image.new("RGBA", (width, height), _CANVAS)
    d = ImageDraw.Draw(canvas)
    cap = draw_wide_keycap("Space", pressed=pressed)
    x0 = (width - cap.width) // 2
    y0 = CANVAS_PAD
    canvas.alpha_composite(cap, (x0, y0))
    inst_font = _font(26)
    max_w = width - CANVAS_PAD * 2
    lines = _wrap_text(SPACE_INSTRUCTION, inst_font, d, max_w)
    inst_y = y0 + KEY_SIZE + 14
    _draw_centered_lines(d, lines, inst_font, inst_y, width, _INSTRUCTION)
    return canvas


def draw_control_stage(stage: str, pressed: set[str] | None = None) -> Image.Image:
    pressed = pressed or set()
    if stage == "arrows":
        return draw_arrow_cluster(pressed)
    if stage == "height":
        return draw_height_keys(pressed)
    if stage == "yaw":
        return draw_yaw_keys(pressed)
    if stage == "tilt":
        return draw_tilt_keys(pressed)
    if stage == "space":
        return draw_space_key(pressed="Space" in pressed)
    raise ValueError(f"Unknown control stage: {stage}")


def draw_space_plus_key(
    letter: str,
    *,
    space_pressed: bool = False,
    letter_pressed: bool = False,
    space_sub: str = "",
    letter_sub: str = "",
    instruction: str = "",
) -> Image.Image:
    """Wide Space key beside a letter key, with optional sublabels."""
    width, height = cluster_size()
    canvas = Image.new("RGBA", (width, height), _CANVAS)
    d = ImageDraw.Draw(canvas)
    space = draw_wide_keycap("Space", pressed=space_pressed)
    key = draw_keycap(letter, pressed=letter_pressed)
    row_w = space.width + GAP + KEY_SIZE
    x0 = (width - row_w) // 2
    y0 = CANVAS_PAD
    canvas.alpha_composite(space, (x0, y0))
    canvas.alpha_composite(key, (x0 + space.width + GAP, y0))
    sub_font = _font(28)
    if space_sub:
        bbox = d.textbbox((0, 0), space_sub, font=sub_font)
        tw = bbox[2] - bbox[0]
        tx = x0 + (space.width - tw) / 2 - bbox[0]
        ty = y0 + KEY_SIZE + 4 - bbox[1]
        d.text((tx, ty), space_sub, font=sub_font, fill=_SUBLABEL)
    if letter_sub:
        bbox = d.textbbox((0, 0), letter_sub, font=sub_font)
        tw = bbox[2] - bbox[0]
        kx = x0 + space.width + GAP
        tx = kx + (KEY_SIZE - tw) / 2 - bbox[0]
        ty = y0 + KEY_SIZE + 4 - bbox[1]
        d.text((tx, ty), letter_sub, font=sub_font, fill=_SUBLABEL)
    if instruction:
        inst_font = _font(26)
        max_w = width - CANVAS_PAD * 2
        lines = _wrap_text(instruction, inst_font, d, max_w)
        inst_y = y0 + KEY_SIZE + SUBLABEL_H + 4
        _draw_centered_lines(d, lines, inst_font, inst_y, width, _INSTRUCTION)
    return canvas


def _arrows_eq_size(*, extra_h: int = 0) -> tuple[int, int, int, int, int]:
    k, g, pad = PLAY2_KEY, PLAY2_GAP, PLAY2_PAD
    aw, ah = _play2_arrow_block()
    width = pad + aw + PLAY2_GROUP + k + pad
    height = pad + ah + pad + extra_h
    return width, height, k, g, pad


def _composite_arrows_eq(canvas: Image.Image, pressed: set[str], *, x0: int, y0: int, k: int, g: int) -> int:
    """Draw arrow cluster + E (up) / Q (down). Returns the block height."""
    aw, ah = _play2_arrow_block()
    canvas.alpha_composite(
        draw_arrow_keycap("up", pressed="up" in pressed, size=k),
        (x0 + k + g, y0),
    )
    by = y0 + k + g
    canvas.alpha_composite(
        draw_arrow_keycap("left", pressed="left" in pressed, size=k),
        (x0, by),
    )
    canvas.alpha_composite(
        draw_arrow_keycap("down", pressed="down" in pressed, size=k),
        (x0 + k + g, by),
    )
    canvas.alpha_composite(
        draw_arrow_keycap("right", pressed="right" in pressed, size=k),
        (x0 + 2 * (k + g), by),
    )
    lx = x0 + aw + PLAY2_GROUP
    canvas.alpha_composite(
        draw_keycap("E", pressed="E" in pressed, size=k),
        (lx, y0),
    )
    canvas.alpha_composite(
        draw_keycap("Q", pressed="Q" in pressed, size=k),
        (lx, by),
    )
    return ah


def draw_arrows_eq_keys(instruction: str, pressed: set[str] | None = None) -> Image.Image:
    """Arrows + E/Q with an instruction line (hold button / on-off switch)."""
    pressed = pressed or set()
    width, height, k, g, pad = _arrows_eq_size(extra_h=INSTRUCTION_H)
    canvas = Image.new("RGBA", (width, height), _CANVAS)
    d = ImageDraw.Draw(canvas)
    x0 = y0 = pad
    ah = _composite_arrows_eq(canvas, pressed, x0=x0, y0=y0, k=k, g=g)
    inst_font = _font(22)
    lines = _wrap_text(instruction, inst_font, d, width - pad * 2)
    _draw_centered_lines(d, lines, inst_font, y0 + ah + 8, width, _INSTRUCTION)
    return canvas


def draw_grasp_keys(
    pressed: set[str] | None = None,
    instruction: str = GRASP_INSTRUCTION,
    title: str = GRASP_TITLE,
) -> Image.Image:
    """Title, then arrows + E/Q + Space, then the instruction line."""
    pressed = pressed or set()
    k, g, pad = PLAY2_KEY, PLAY2_GAP, PLAY2_PAD
    aw, ah = _play2_arrow_block()
    title_h = TITLE_H if title else 0
    inst_h = INSTRUCTION_H if instruction else 8
    width = pad + aw + PLAY2_GROUP + k + pad
    height = pad + title_h + ah + g + k + pad + inst_h
    canvas = Image.new("RGBA", (width, height), _CANVAS)
    d = ImageDraw.Draw(canvas)
    x0 = pad
    y0 = pad + title_h
    if title:
        title_font = _font(24)
        lines = _wrap_text(title, title_font, d, width - pad * 2)
        _draw_centered_lines(d, lines, title_font, pad + 4, width, _INSTRUCTION)
    _composite_arrows_eq(canvas, pressed, x0=x0, y0=y0, k=k, g=g)
    sy = y0 + ah + g
    space_w = aw + PLAY2_GROUP + k
    canvas.alpha_composite(
        draw_wide_keycap("Space", pressed="Space" in pressed, width=space_w, height=k),
        (x0, sy),
    )
    if instruction:
        inst_font = _font(22)
        lines = _wrap_text(instruction, inst_font, d, width - pad * 2)
        _draw_centered_lines(d, lines, inst_font, sy + k + 8, width, _INSTRUCTION)
    return canvas


def draw_hold_keys(pressed: set[str] | None = None) -> Image.Image:
    return draw_grasp_keys(pressed, instruction=HOLD_INSTRUCTION, title=HOLD_TITLE)


def draw_switch_keys(pressed: set[str] | None = None) -> Image.Image:
    return draw_grasp_keys(pressed, instruction=SWITCH_INSTRUCTION, title=SWITCH_TITLE)


def draw_push_keys(pressed: set[str] | None = None) -> Image.Image:
    """Same keys as pick (including Q/E) with the push-box caption."""
    return draw_grasp_keys(pressed, instruction=PUSH_INSTRUCTION, title=PUSH_TITLE)


def draw_action_stage(stage: str, pressed: set[str] | None = None) -> Image.Image:
    pressed = pressed or set()
    if stage == "grasp":
        return draw_grasp_keys(pressed)
    if stage == "hold":
        return draw_hold_keys(pressed)
    if stage == "switch":
        return draw_switch_keys(pressed)
    if stage == "push":
        return draw_push_keys(pressed)
    raise ValueError(f"Unknown action stage: {stage}")


def draw_ball_keys(pressed: set[str] | None = None) -> Image.Image:
    return draw_grasp_keys(pressed, instruction=BALL_INSTRUCTION, title=BALL_TITLE)


def draw_stove_keys(pressed: set[str] | None = None) -> Image.Image:
    """Arrows + E/Q + R/T + Space — approach, height, grasp, then yaw."""
    pressed = pressed or set()
    k, g, pad = PLAY2_KEY, PLAY2_GAP, PLAY2_PAD
    aw, ah = _play2_arrow_block()
    # arrows | E Q | R T  (letter pairs sit on the arrow bottom row)
    letters_w = 2 * (2 * k + g) + PLAY2_GROUP
    title_h = TITLE_H
    width = pad + aw + PLAY2_GROUP + letters_w + pad
    height = pad + title_h + ah + g + k + pad + INSTRUCTION_H
    canvas = Image.new("RGBA", (width, height), _CANVAS)
    d = ImageDraw.Draw(canvas)
    x0 = pad
    y0 = pad + title_h
    title_font = _font(24)
    lines = _wrap_text(STOVE_TITLE, title_font, d, width - pad * 2)
    _draw_centered_lines(d, lines, title_font, pad + 4, width, _INSTRUCTION)
    canvas.alpha_composite(
        draw_arrow_keycap("up", pressed="up" in pressed, size=k),
        (x0 + k + g, y0),
    )
    by = y0 + k + g
    canvas.alpha_composite(
        draw_arrow_keycap("left", pressed="left" in pressed, size=k),
        (x0, by),
    )
    canvas.alpha_composite(
        draw_arrow_keycap("down", pressed="down" in pressed, size=k),
        (x0 + k + g, by),
    )
    canvas.alpha_composite(
        draw_arrow_keycap("right", pressed="right" in pressed, size=k),
        (x0 + 2 * (k + g), by),
    )
    lx = x0 + aw + PLAY2_GROUP
    for j, label in enumerate(("E", "Q")):
        canvas.alpha_composite(
            draw_keycap(label, pressed=label in pressed, size=k),
            (lx + j * (k + g), by),
        )
    rx = lx + 2 * k + g + PLAY2_GROUP
    for j, label in enumerate(("R", "T")):
        canvas.alpha_composite(
            draw_keycap(label, pressed=label in pressed, size=k),
            (rx + j * (k + g), by),
        )
    sub_font = _font(18)
    for hx, hw, hint in (
        (lx, 2 * k + g, "height"),
        (rx, 2 * k + g, "yaw"),
    ):
        bbox = d.textbbox((0, 0), hint, font=sub_font)
        tw = bbox[2] - bbox[0]
        tx = hx + (hw - tw) / 2 - bbox[0]
        ty = by - PLAY_SUB_H + 2 - bbox[1]
        d.text((tx, ty), hint, font=sub_font, fill=_SUBLABEL)
    sy = y0 + ah + g
    space_w = aw + PLAY2_GROUP + letters_w
    canvas.alpha_composite(
        draw_wide_keycap("Space", pressed="Space" in pressed, width=space_w, height=k),
        (x0, sy),
    )
    inst_font = _font(22)
    lines = _wrap_text(STOVE_INSTRUCTION, inst_font, d, width - pad * 2)
    _draw_centered_lines(d, lines, inst_font, sy + k + 8, width, _INSTRUCTION)
    return canvas


def draw_mallet_keys(pressed: set[str] | None = None) -> Image.Image:
    return draw_grasp_keys(pressed, instruction=MALLET_INSTRUCTION, title=MALLET_TITLE)


def _force_bar_colors():
    return (
        (40, 90, 210, 255),
        (40, 180, 80, 255),
        (230, 190, 30, 255),
        (230, 120, 30, 255),
        (210, 40, 40, 255),
    )


def draw_force_key_stage(
    pressed: set[str] | None = None,
    *,
    force_n: float = 0.0,
    peak_n: float = 0.0,
    target_level: int = 1,
    cleared: int = 0,
    thresholds: tuple[float, ...] = (3.0, 6.0, 10.0, 14.0),
    feedback: str = "",
) -> Image.Image:
    """Same footprint as grasp/ball — arrows + Q/E + a readable force bar."""
    pressed = pressed or set()
    k, g, pad = PLAY2_KEY, PLAY2_GAP, PLAY2_PAD
    aw, ah = _play2_arrow_block()
    title_h = TITLE_H
    width = pad + aw + PLAY2_GROUP + k + pad
    height = pad + title_h + ah + g + FORCE_BAR_H + pad + INSTRUCTION_H
    canvas = Image.new("RGBA", (width, height), _CANVAS)
    d = ImageDraw.Draw(canvas)
    x0 = pad
    y0 = pad + title_h
    title_font = _font(24)
    lines = _wrap_text(FORCE_TITLE, title_font, d, width - pad * 2)
    _draw_centered_lines(d, lines, title_font, pad + 4, width, _INSTRUCTION)
    _composite_arrows_eq(canvas, pressed, x0=x0, y0=y0, k=k, g=g)
    y_bar = y0 + ah + g
    bar_w = aw + PLAY2_GROUP + k
    bar = [x0, y_bar, x0 + bar_w, y_bar + 28]
    d.rounded_rectangle(
        bar, radius=6, fill=(32, 34, 40, 255), outline=(90, 94, 104, 255), width=2
    )
    max_f = float(thresholds[-1]) * 1.15 if thresholds else 16.0
    thr = tuple(float(t) for t in thresholds)
    n_levels = len(thr)
    bw = bar[2] - bar[0]
    for i in range(max(0, int(cleared))):
        lo = 0.0 if i == 0 else thr[i - 1]
        hi = thr[i] if i < n_levels else max_f
        x_lo = bar[0] + bw * (lo / max_f)
        x_hi = bar[0] + bw * (min(hi, max_f) / max_f)
        d.rectangle(
            [x_lo, bar[1] + 4, x_hi, bar[3] - 4],
            fill=(36, 110, 60, 120),
        )
    shown = max(float(force_n), float(peak_n) * 0.2)
    frac = float(min(1.0, max(0.0, shown / max(max_f, 1e-6))))
    fill_x = bar[0] + 3 + (bw - 6) * frac
    level = 0
    for i, t in enumerate(thr):
        if shown >= t:
            level = i + 1
    colors = _force_bar_colors()
    fill = colors[min(level, len(colors) - 1)]
    if fill_x > bar[0] + 4:
        d.rounded_rectangle(
            [bar[0] + 3, bar[1] + 3, fill_x, bar[3] - 3],
            radius=4,
            fill=fill,
        )
    target = max(1, min(int(target_level), n_levels))
    lo = float(thr[target - 1])
    hi = float(thr[target]) if target < n_levels else max_f
    x_lo = bar[0] + bw * (lo / max_f)
    x_hi = bar[0] + bw * (min(hi, max_f) / max_f)
    d.rectangle(
        [x_lo, bar[1] - 4, x_hi, bar[3] + 4],
        outline=(255, 230, 80, 255),
        width=3,
    )
    for t in thr:
        tx = bar[0] + bw * (float(t) / max_f)
        d.line([(tx, bar[1]), (tx, bar[3])], fill=(140, 145, 155, 200), width=1)
    step_label = (
        FORCE_STEP_LABELS[target - 1]
        if 0 < target <= len(FORCE_STEP_LABELS)
        else f"step {target}"
    )
    label_font = _font(18)
    caption = f"{force_n:.1f} N · yellow = {step_label} ({lo:.0f}–{hi:.0f} N)"
    if feedback:
        caption = f"{caption} · {feedback}"
    bbox = d.textbbox((0, 0), caption, font=label_font)
    tw = bbox[2] - bbox[0]
    if tw > bar_w - 4:
        label_font = _font(15)
        bbox = d.textbbox((0, 0), caption, font=label_font)
        tw = bbox[2] - bbox[0]
    d.text(
        (x0 + (bar_w - tw) / 2 - bbox[0], y_bar + 32 - bbox[1]),
        caption,
        font=label_font,
        fill=_INSTRUCTION,
    )
    inst_font = _font(20)
    lines = _wrap_text(FORCE_INSTRUCTION, inst_font, d, width - pad * 2)
    _draw_centered_lines(
        d, lines, inst_font, y_bar + FORCE_BAR_H + 2, width, _INSTRUCTION
    )
    return canvas


def draw_left_right_arrows(
    instruction: str,
    pressed: set[str] | None = None,
    *,
    title: str = "",
) -> Image.Image:
    pressed = pressed or set()
    width = cluster_size()[0]
    body_h = KEY_SIZE
    canvas, d, title_lines, title_font, title_y, lines, inst_font, inst_y, y0 = _overlay_canvas(
        width, body_h, instruction, title=title
    )
    gap = GAP * 2
    row_w = KEY_SIZE * 2 + gap
    x0 = (width - row_w) // 2
    canvas.alpha_composite(
        draw_arrow_keycap("left", pressed="left" in pressed),
        (x0, y0),
    )
    canvas.alpha_composite(
        draw_arrow_keycap("right", pressed="right" in pressed),
        (x0 + KEY_SIZE + gap, y0),
    )
    _paint_overlay_text(d, width, title_lines, title_font, title_y, lines, inst_font, inst_y)
    return canvas


def draw_space_instruction(
    instruction: str,
    pressed: bool = False,
    *,
    title: str = "",
) -> Image.Image:
    width = cluster_size()[0]
    body_h = KEY_SIZE
    canvas, d, title_lines, title_font, title_y, lines, inst_font, inst_y, y0 = _overlay_canvas(
        width, body_h, instruction, title=title
    )
    cap = draw_wide_keycap("Space", pressed=pressed)
    x0 = (width - cap.width) // 2
    canvas.alpha_composite(cap, (x0, y0))
    _paint_overlay_text(d, width, title_lines, title_font, title_y, lines, inst_font, inst_y)
    return canvas


def draw_arrows_and_space(
    instruction: str,
    pressed: set[str] | None = None,
    *,
    title: str = "",
) -> Image.Image:
    pressed = pressed or set()
    width = cluster_size()[0]
    gap = GAP * 2
    space_cap = draw_wide_keycap("Space", pressed="Space" in pressed)
    body_h = KEY_SIZE + 16 + space_cap.height
    canvas, d, title_lines, title_font, title_y, lines, inst_font, inst_y, y0 = _overlay_canvas(
        width, body_h, instruction, title=title
    )
    row_w = KEY_SIZE * 2 + gap
    x0 = (width - row_w) // 2
    canvas.alpha_composite(draw_arrow_keycap("left", pressed="left" in pressed), (x0, y0))
    canvas.alpha_composite(
        draw_arrow_keycap("right", pressed="right" in pressed),
        (x0 + KEY_SIZE + gap, y0),
    )
    sx = (width - space_cap.width) // 2
    canvas.alpha_composite(space_cap, (sx, y0 + KEY_SIZE + 16))
    _paint_overlay_text(d, width, title_lines, title_font, title_y, lines, inst_font, inst_y)
    return canvas


def draw_mouse_click(
    instruction: str = "",
    *,
    highlight_left: bool = True,
    title: str = "",
) -> Image.Image:
    """Mouse with the left button highlighted, optional title above."""
    width = cluster_size()[0]
    body_w, mouse_h = 168, 220
    body_h = mouse_h + 8
    canvas, d, title_lines, title_font, title_y, lines, inst_font, inst_y, y0 = _overlay_canvas(
        width, body_h, instruction, title=title
    )
    x0 = (width - body_w) // 2
    d.rounded_rectangle(
        [x0 + 6, y0 + 8, x0 + body_w + 6, y0 + mouse_h + 8],
        radius=70,
        fill=(0, 0, 0, 80),
    )
    d.rounded_rectangle(
        [x0, y0, x0 + body_w, y0 + mouse_h],
        radius=70,
        fill=_FILL,
        outline=_OUTLINE,
        width=4,
    )
    mid = x0 + body_w // 2
    split_y = y0 + 78
    left_fill = _FILL_INNER_DONE if highlight_left else _FILL_INNER
    left_outline = _OUTLINE_DONE if highlight_left else _INNER_OUTLINE
    d.rounded_rectangle(
        [x0 + 10, y0 + 10, mid - 4, split_y],
        radius=22,
        fill=left_fill,
        outline=left_outline,
        width=3,
    )
    d.rounded_rectangle(
        [mid + 4, y0 + 10, x0 + body_w - 10, split_y],
        radius=22,
        fill=_FILL_INNER,
        outline=_INNER_OUTLINE,
        width=3,
    )
    d.ellipse(
        [mid - 10, y0 + 28, mid + 10, y0 + 62],
        fill=_INNER_OUTLINE,
        outline=_OUTLINE,
        width=2,
    )
    _paint_overlay_text(d, width, title_lines, title_font, title_y, lines, inst_font, inst_y)
    return canvas


def draw_mouse_and_arrows(
    instruction: str,
    pressed: set[str] | None = None,
    *,
    title: str = "",
) -> Image.Image:
    pressed = pressed or set()
    mouse = draw_mouse_click("", highlight_left=True)
    arrows = draw_left_right_arrows("", pressed)
    width = cluster_size()[0]
    arrow_h = KEY_SIZE
    mouse_h = 228
    body_h = arrow_h + 12 + mouse_h
    canvas, d, title_lines, title_font, title_y, lines, inst_font, inst_y, y0 = _overlay_canvas(
        width, body_h, instruction, title=title
    )
    arrow_body = arrows.crop((0, 0, arrows.width, KEY_SIZE + 4))
    canvas.alpha_composite(arrow_body, ((width - arrow_body.width) // 2, y0))
    canvas.alpha_composite(mouse, ((width - mouse.width) // 2, y0 + arrow_h + 12))
    _paint_overlay_text(d, width, title_lines, title_font, title_y, lines, inst_font, inst_y)
    return canvas


def draw_kb_stage(stage: str, pressed: set[str] | None = None) -> Image.Image:
    pressed = pressed or set()
    if stage == "num_keys":
        return draw_key_row(
            ("1", "2", "3"),
            pressed,
            title="Turn on the lights",
            instruction="Press 1, 2, and 3.",
        )
    if stage == "num_mouse":
        return draw_mouse_click(
            "Click the buttons.",
            title="Turn off the lights",
        )
    if stage == "bowl_keys":
        return draw_left_right_arrows(
            "Press the arrow keys.",
            pressed,
            title="Move the bowl to the red areas",
        )
    if stage == "bowl_mouse":
        return draw_mouse_click(
            "Click the arrow buttons.",
            title="Move the bowl to the red areas",
        )
    if stage == "switch_keys":
        return draw_space_instruction(
            "Press Space.",
            pressed="Space" in pressed,
            title="Turn on and off the key",
        )
    if stage == "switch_mouse":
        return draw_mouse_click(
            "Click the key.",
            title="Turn on and off the key",
        )
    if stage == "push_keys":
        return draw_space_instruction(
            "Hold Space.",
            pressed="Space" in pressed,
            title="Press and hold the key to turn on the lamp",
        )
    if stage == "push_mouse":
        return draw_mouse_click(
            "Hold the left mouse button on the key.",
            title="Press and hold the key to turn on the lamp",
        )
    if stage in ("place_click", "place_drop"):
        return draw_mouse_click(
            "Click the red circles.",
            title="Move the cube onto the circles",
        )
    if stage == "cup_place":
        return draw_mouse_click(
            "Click the green region.",
            title="Place the cup on the green area",
        )
    if stage == "apple_box":
        return draw_mouse_click(
            "Click the apple, then the box.",
            title="Move the apple inside the box",
        )
    if stage == "gummy_keys":
        return draw_arrows_and_space(
            "Arrows move the bowl. Space triggers.",
            pressed,
            title="Move the bowl to the red area, then press the trigger",
        )
    if stage == "gummy_mouse":
        return draw_mouse_click(
            "Click the arrows, then the trigger.",
            title="Move the bowl to the red area, then press the trigger",
        )
    if stage == "dual_gate":
        return draw_left_right_arrows(
            "Press both arrow keys at the same time.",
            pressed,
            title="Open the gate",
        )
    if stage == "cue_aim":
        return draw_mouse_and_arrows(
            "Click the circle, then use the arrows.",
            pressed,
            title="Place the stick tip and rotate",
        )
    if stage == "stop_ball":
        return draw_mouse_click(
            "Click the ball.",
            title="Click the moving ball to stop it",
        )
    if stage == "knob_lamp":
        return draw_mouse_click(
            "Click the stove knob.",
            title="Turn the knob on, then off",
        )
    if stage == "board_tilt":
        return draw_mouse_and_arrows(
            "Click the circle, then use the arrows.",
            pressed,
            title="Place the board and tilt it",
        )
    raise ValueError(f"Unknown keyboard tutorial stage: {stage}")


def draw_advanced_stage(
    stage: str,
    pressed: set[str] | None = None,
    **kwargs,
) -> Image.Image:
    pressed = pressed or set()
    if stage == "ball":
        return draw_ball_keys(pressed)
    if stage == "stove":
        return draw_stove_keys(pressed)
    if stage == "mallet":
        return draw_mallet_keys(pressed)
    if stage == "force_key":
        return draw_force_key_stage(pressed, **kwargs)
    raise ValueError(f"Unknown advanced stage: {stage}")

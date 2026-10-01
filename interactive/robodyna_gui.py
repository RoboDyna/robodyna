#!/usr/bin/env python3
"""Shared RoboDyna launcher: one window that opens the base, household, and experiment GUIs.

Each suite box is painted with that suite's app-icon accent (see
``script/make_gui_app_icons.SUITE_ICONS``) and previews the suite through a
4-column collage of published scene snapshots.
"""
from __future__ import annotations

import math
import os
import random
import signal
import subprocess
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import font as tkfont, messagebox, ttk

from PIL import Image, ImageTk

_INTERACTIVE_DIR = Path(__file__).resolve().parent
if str(_INTERACTIVE_DIR) not in sys.path:
    sys.path.insert(0, str(_INTERACTIVE_DIR))

from _task_briefing import (  # noqa: E402
    GUI_PAGE_BG,
    GUI_WM_CLASS,
    apply_gui_logo,
    setup_gui_app_icon,
)
from base_task_gui import RoundedButton, TASKS as BASE_TASKS  # noqa: E402
from experiment_logs import (  # noqa: E402
    EXPERIMENT_ENV,
    EXPERIMENT_LOG_ENV,
    EXPERIMENT_USER_ENV,
)
from household_task_gui import TASKS as HOUSEHOLD_TASKS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "docs/final_task_demos"
PAGE_BG = GUI_PAGE_BG

# Brand hues (Base teal, HH orange, Exp coral). Boxes use a softened tint of these.
BRAND_BASE = (1, 157, 162)
BRAND_HOUSEHOLD = (253, 142, 25)
BRAND_EXPERIMENT = (237, 82, 57)
# Blend toward white so the cards read as pastel, not saturated.
SOFTEN = 0.48
HEADING_INK = (4, 43, 76)
BLURB_INK = (58, 86, 110)

ABOUT_ACCENT = (119, 174, 222)
# (text, tag) chunks; "" is the plain body style.
ABOUT_SEGMENTS = (
    ("RoboDyna", "bold"),
    (
        " is a robotic manipulation benchmark suite designed for evaluating robotic "
        "policies in highly dynamic environments. This GUI provides an ",
        "",
    ),
    ("interactive means for performing the dynamic tasks", "bold"),
    (" using ", ""),
    ("keyboard and mouse", "bold"),
    (", either by directly controlling the ", ""),
    ("robot arms", "italic"),
    (" or ", ""),
    ("interacting with objects", "italic"),
    (". Using the GUI, the user can ", ""),
    ("collect data", "bold"),
    (", ", ""),
    ("demos", "bold"),
    (", or perform ", ""),
    ("experiments", "bold"),
    (" for human evaluation. The collected data can be directly used for ", ""),
    ("policy training", "bold"),
    (".", ""),
)

def _about_tokens() -> list[list[tuple[str, str]]]:
    """Split the About prose into whitespace-separated tokens of styled pieces.

    A token can mix styles (``collect data`` in bold followed by a plain comma),
    so line measurement treats it as one unbreakable unit.
    """
    tokens: list[list[tuple[str, str]]] = []
    pending: list[tuple[str, str]] = []
    for chunk, tag in ABOUT_SEGMENTS:
        for index, part in enumerate(chunk.split(" ")):
            if index and pending:
                tokens.append(pending)
                pending = []
            if part:
                pending.append((part, tag))
    if pending:
        tokens.append(pending)
    return tokens


ABOUT_TOKENS = _about_tokens()


def _wrapped_line_count(
    width: int,
    fonts: dict[str, tkfont.Font],
) -> int:
    """Greedy word-wrap line count for the About prose at ``width`` pixels."""
    space = fonts[""].measure(" ")
    lines = 1
    used = 0
    for token in ABOUT_TOKENS:
        token_width = sum(fonts[tag].measure(text) for text, tag in token)
        if used and used + space + token_width > width:
            lines += 1
            used = token_width
        else:
            used += token_width if not used else space + token_width
    return lines


COLLAGE_COLUMNS = 4
COLLAGE_GAP = 6
# Head stills are 4:3; every collage cell keeps that shape.
CELL_ASPECT = 3 / 4
EXPERIMENT_COLLAGE_MAX = 12
# Per-group quotas for the experiment mix, biggest suite first.
EXPERIMENT_MIX_QUOTAS = (5, 4, 3)
THUMB_WIDTH = 360
TUTORIAL_KB_STEMS = ("buttons", "placement", "base", "household")


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


def _shade(rgb: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(round(channel * factor)))) for channel in rgb)


def _soften(rgb: tuple[int, int, int], amount: float = SOFTEN) -> tuple[int, int, int]:
    """Pull ``rgb`` toward white by ``amount`` (0 = original, 1 = white)."""
    return tuple(int(round(c + (255 - c) * amount)) for c in rgb)


BASE_ACCENT = _soften(BRAND_BASE)
HOUSEHOLD_ACCENT = _soften(BRAND_HOUSEHOLD)
EXPERIMENT_ACCENT = _soften(BRAND_EXPERIMENT)


def _base_snapshots() -> list[Path]:
    """Opt 1+2 still for every base task, in GUI order."""
    paths = []
    for _label, task in BASE_TASKS:
        path = DEMO_DIR / task / "scene_snapshot_opt1+2.png"
        if path.is_file():
            paths.append(path)
    return paths


def _household_snapshots() -> list[Path]:
    paths = []
    for _label, task, _script in HOUSEHOLD_TASKS:
        path = DEMO_DIR / task / "scene_snapshot.png"
        if path.is_file():
            paths.append(path)
    return paths


def _tutorial_snapshots() -> list[Path]:
    candidates = [DEMO_DIR / "tutorial_empty" / f"scene_snapshot_part{part}.png" for part in (1, 2, 3, 4)]
    candidates += [DEMO_DIR / "tutorial_keyboard" / f"scene_snapshot_kb_{stem}.png" for stem in TUTORIAL_KB_STEMS]
    return [path for path in candidates if path.is_file()]


def _experiment_snapshots(rng: random.Random) -> list[Path]:
    """Random base / household / tutorial mix, capped at ``EXPERIMENT_COLLAGE_MAX``."""
    groups = (_base_snapshots(), _household_snapshots(), _tutorial_snapshots())
    picked: list[Path] = []
    for pool, quota in zip(groups, EXPERIMENT_MIX_QUOTAS):
        picked.extend(rng.sample(pool, min(quota, len(pool))))
    if len(picked) < EXPERIMENT_COLLAGE_MAX:
        chosen = set(picked)
        spare = [path for pool in groups for path in pool if path not in chosen]
        rng.shuffle(spare)
        picked.extend(spare[: EXPERIMENT_COLLAGE_MAX - len(picked)])
    rng.shuffle(picked)
    return picked[:EXPERIMENT_COLLAGE_MAX]


def _fit_cell(source: Image.Image, width: int, height: int) -> Image.Image:
    """Center-crop ``source`` to the cell shape, then resize to it."""
    src_w, src_h = source.size
    target = width / height
    if src_w / src_h > target:
        crop_w = max(1, int(round(src_h * target)))
        left = (src_w - crop_w) // 2
        box = (left, 0, left + crop_w, src_h)
    else:
        crop_h = max(1, int(round(src_w / target)))
        top = (src_h - crop_h) // 2
        box = (0, top, src_w, top + crop_h)
    return source.crop(box).resize((width, height), Image.Resampling.LANCZOS)


def build_collage(
    tiles: list[Image.Image],
    *,
    cell_width: int,
    background: tuple[int, int, int],
    columns: int = COLLAGE_COLUMNS,
    gap: int = COLLAGE_GAP,
) -> Image.Image | None:
    if not tiles:
        return None
    cell_width = max(16, int(cell_width))
    cell_height = max(12, int(round(cell_width * CELL_ASPECT)))
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new(
        "RGB",
        (columns * cell_width + (columns - 1) * gap, rows * cell_height + (rows - 1) * gap),
        background,
    )
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        sheet.paste(
            _fit_cell(tile, cell_width, cell_height),
            (column * (cell_width + gap), row * (cell_height + gap)),
        )
    return sheet


@dataclass(frozen=True)
class SuiteSpec:
    key: str
    title: str
    blurb: str
    script: str
    accent: tuple[int, int, int]
    brand: tuple[int, int, int]


SUITES = (
    SuiteSpec(
        key="base",
        title="Base Tasks",
        blurb=(
            "23 dynamic table-top tasks that examine core understanding of physics. "
            "Each task comes in four variety with different levels of difficulty."
        ),
        script="base_task_gui.py",
        accent=BASE_ACCENT,
        brand=BRAND_BASE,
    ),
    SuiteSpec(
        key="household",
        title="Household Tasks",
        blurb=(
            "12 household tasks in office and kitchen environments examining physical "
            "understanding as part of daily activities."
        ),
        script="household_task_gui.py",
        accent=HOUSEHOLD_ACCENT,
        brand=BRAND_HOUSEHOLD,
    ),
    SuiteSpec(
        key="experiment",
        title="Experiments",
        blurb=(
            "A guided procedure for human evaluation, involving random selection of "
            "tasks and questionnaires."
        ),
        script="experiment_gui.py",
        accent=EXPERIMENT_ACCENT,
        brand=BRAND_EXPERIMENT,
    ),
)


class RoboDynaLauncher(tk.Tk):
    DESIGN_SIZE = (1680, 1040)
    # Small screens can go below the design size; the page then scrolls.
    MIN_SIZE = (900, 560)
    # Room left for window decorations / desktop panels when sizing to the screen.
    SCREEN_MARGIN = (80, 120)
    # Overflow this small is layout rounding, not cut-off content, so it must not
    # pop a scrollbar at the design size.
    SCROLL_SLACK = 8
    UI_SCALE_MIN = 0.6
    UI_SCALE_MAX = 1.15
    # Base logo bar is 160px in the suite GUIs; the hub shows it 30% larger.
    LOGO_HEIGHT = 208
    OUTER_PAD = 24
    CARD_GAP = 18
    CARD_PAD = 18
    CARD_BORDER = 2
    # About prose is auto-sized within this range to fill its box.
    ABOUT_FONT_MIN = 12
    ABOUT_FONT_MAX = 24
    # Bold/italic must stay the same size as the body, so the family is chosen at
    # runtime: some environments map "Sans" to a bitmap font whose bold is a
    # different (larger) face. Preference order, first consistent one wins.
    ABOUT_FONT_PREFS = ("DejaVu Sans", "Noto Sans", "Nimbus Sans", "nimbus sans l", "Helvetica")

    def __init__(self):
        super().__init__(className=GUI_WM_CLASS["gui"])
        self.title("RoboDyna")
        start_width, start_height = self._startup_size()
        self.geometry(f"{start_width}x{start_height}")
        self.minsize(*self.MIN_SIZE)
        self.configure(bg=PAGE_BG)
        setup_gui_app_icon(self, suite="gui", script_path=Path(__file__))
        self.protocol("WM_DELETE_WINDOW", self.exit_app)

        self.child: subprocess.Popen | None = None
        self._ui_scale = 0.0
        self._ui_scale_job: str | None = None
        self._thumbs: dict[Path, Image.Image] = {}
        self._cards: dict[str, tk.Frame] = {}
        self._about_fit_job: str | None = None
        self._about_fit_key: tuple[int, int] | None = None
        self._about_family = self._resolve_about_family()
        self._page_size: tuple[int, int] | None = None

        self._sources = {
            "base": _base_snapshots(),
            "household": _household_snapshots(),
            "experiment": _experiment_snapshots(random.Random()),
        }

        self._build_scroll_host()

        top = tk.Frame(self.page, bg=PAGE_BG)
        top.pack(fill="x", padx=self.OUTER_PAD, pady=(16, 0))
        self.exit_button = RoundedButton(
            top,
            text="Exit",
            command=self.exit_app,
            bg="#e34a33",
            activebackground="#eb6854",
            font=("Sans", 15, "bold"),
            width=120,
            height=54,
            radius=22,
        )
        self.exit_button.pack(side="right", anchor="n")
        self.logo_label = tk.Label(top, bg=PAGE_BG, bd=0, highlightthickness=0)
        self.logo_label.pack(anchor="center", pady=(0, 4))
        apply_gui_logo(self.logo_label, height=self.LOGO_HEIGHT)

        self.body = tk.Frame(self.page, bg=PAGE_BG)
        self.body.pack(fill="both", expand=True, padx=self.OUTER_PAD, pady=(8, self.OUTER_PAD))
        for column in range(len(SUITES)):
            self.body.columnconfigure(column, weight=1, uniform="suite")
        # Rows 0 and 1 stay at their requested heights and row 2 soaks up the
        # leftover window space, so the About box bottom lands exactly on the
        # bottom of the Base box that spans both rows.
        self.body.rowconfigure(0, weight=0)
        self.body.rowconfigure(1, weight=0)
        self.body.rowconfigure(2, weight=1)
        for column, spec in enumerate(SUITES):
            card = self._suite_card(self.body, spec)
            # Uniform columns split the body evenly, so every card takes the same
            # half-gap on both sides to keep the three boxes equally wide.
            card.grid(
                row=0,
                column=column,
                # Base is the tallest box, so it spans the About row beside it.
                rowspan=2 if spec.key == "base" else 1,
                # Household / Experiments fill row 0 so the two boxes always match
                # in height, however their blurbs happen to wrap.
                sticky="nsew",
                padx=self.CARD_GAP // 2,
            )
            self._cards[spec.key] = card
        self.about_card = self._about_card(self.body)
        self.about_card.grid(
            row=1,
            column=1,
            columnspan=len(SUITES) - 1,
            sticky="nsew",
            padx=self.CARD_GAP // 2,
            pady=(self.CARD_GAP, 0),
        )

        self.bind("<Configure>", self._on_root_configure)
        self.after(0, self._apply_ui_scale)
        self.after(250, self._poll_child)
        self.after(250, self._poll_layout)

    def _startup_size(self) -> tuple[int, int]:
        """Design size, capped to what the current screen can actually show."""
        available = (
            self.winfo_screenwidth() - self.SCREEN_MARGIN[0],
            self.winfo_screenheight() - self.SCREEN_MARGIN[1],
        )
        return tuple(
            max(smallest, min(design, room))
            for design, room, smallest in zip(self.DESIGN_SIZE, available, self.MIN_SIZE)
        )

    def _build_scroll_host(self):
        """Host the whole page in a canvas so a cut-off layout can be scrolled."""
        host = tk.Frame(self, bg=PAGE_BG)
        host.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(host, bg=PAGE_BG, highlightthickness=0, bd=0)
        self.scrollbar = ttk.Scrollbar(host, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self._on_scroll_set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.page = tk.Frame(self.canvas, bg=PAGE_BG)
        self.page_window = self.canvas.create_window((0, 0), window=self.page, anchor="nw")
        self.page.bind("<Configure>", lambda _event: self._sync_page_window())
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        self.canvas.bind_all("<Button-4>", lambda _event: self.canvas.yview_scroll(-3, "units"))
        self.canvas.bind_all("<Button-5>", lambda _event: self.canvas.yview_scroll(3, "units"))
        self.bind("<Up>", lambda _event: self.canvas.yview_scroll(-3, "units"))
        self.bind("<Down>", lambda _event: self.canvas.yview_scroll(3, "units"))
        self.bind("<Prior>", lambda _event: self.canvas.yview_scroll(-1, "pages"))
        self.bind("<Next>", lambda _event: self.canvas.yview_scroll(1, "pages"))
        self.bind("<Home>", lambda _event: self.canvas.yview_moveto(0.0))
        self.bind("<End>", lambda _event: self.canvas.yview_moveto(1.0))

    def _on_scroll_set(self, first: str, last: str):
        """Keep the scrollbar hidden unless the page really is taller than the view."""
        self.scrollbar.set(first, last)
        overflowing = float(first) > 0.0 or float(last) < 1.0
        mapped = bool(self.scrollbar.winfo_ismapped())
        if overflowing and not mapped:
            self.scrollbar.pack(side="right", fill="y")
        elif not overflowing and mapped:
            self.scrollbar.pack_forget()

    def _on_canvas_configure(self, event):
        # A new viewport means a new scroll extent; start from the top so the logo
        # and card headings are never left scrolled away by a plain resize.
        self._sync_page_window(event.width, event.height, reset_view=True)

    def _sync_page_window(
        self,
        width: int | None = None,
        height: int | None = None,
        *,
        reset_view: bool = False,
    ):
        """Match the page to the viewport, but never below its natural height.

        Stretching to the viewport keeps the ``expand`` based card / About
        alignment identical to an unscrolled window; the natural height floor is
        what makes the leftover content reachable on a small screen.
        """
        width = self.canvas.winfo_width() if width is None else width
        height = self.canvas.winfo_height() if height is None else height
        height = max(1, int(height))
        natural = self.page.winfo_reqheight()
        page_height = natural if natural > height + self.SCROLL_SLACK else height
        size = (max(1, int(width)), page_height)
        if size != self._page_size:
            self._page_size = size
            self.canvas.itemconfigure(self.page_window, width=size[0], height=size[1])
        self.canvas.configure(scrollregion=(0, 0, size[0], page_height))
        if reset_view or page_height <= height:
            self.canvas.yview_moveto(0.0)

    def _on_wheel(self, event):
        if event.delta:
            self.canvas.yview_scroll(-int(event.delta / 120), "units")

    def _poll_layout(self):
        """Track the natural height, which grows without resizing the page frame.

        Rescaled fonts, rebuilt collages, and About refits all change what the
        page asks for, so the scroll extent is re-derived on a heartbeat instead
        of trusting a single post-resize pass.
        """
        self._sync_page_window()
        self.after(250, self._poll_layout)

    def _resolve_about_family(self) -> str:
        """First family whose regular, bold, and italic share the same size.

        Falls back to ``Sans`` if none of the preferences render consistently.
        """
        available = {name.lower() for name in tkfont.families(self)}
        for family in self.ABOUT_FONT_PREFS:
            if family.lower() not in available:
                continue
            reg = tkfont.Font(root=self, family=family, size=18)
            bold = tkfont.Font(root=self, family=family, size=18, weight="bold")
            ital = tkfont.Font(root=self, family=family, size=18, slant="italic")
            reg_ls = reg.metrics("linespace")
            if (
                abs(bold.metrics("linespace") - reg_ls) <= 1
                and abs(ital.metrics("linespace") - reg_ls) <= 1
            ):
                return family
        return "Sans"

    def _about_font_specs(self, px_size: int) -> dict[str, tuple]:
        family = self._about_family
        return {
            "": (family, px_size),
            "bold": (family, px_size, "bold"),
            "italic": (family, px_size, "italic"),
        }

    def _about_card(self, parent) -> tk.Frame:
        """Read-only prose card that fills the gap beside the taller Base box."""
        background = _hex(ABOUT_ACCENT)
        ink = _hex(HEADING_INK)
        card = tk.Frame(
            parent,
            bg=background,
            highlightbackground=_hex(_shade(ABOUT_ACCENT, 0.82)),
            highlightthickness=self.CARD_BORDER,
        )
        pad = tk.Frame(card, bg=background, padx=self.CARD_PAD, pady=self.CARD_PAD)
        pad.pack(fill="both", expand=True)
        self.about_title = tk.Label(
            pad,
            text="About",
            bg=background,
            fg=ink,
            anchor="w",
            font=("Sans", 24, "bold"),
        )
        self.about_title.pack(anchor="w")
        # A Text widget (not a Label) so bold / italic runs can sit inline.
        self.about_body = tk.Text(
            pad,
            bg=background,
            fg=ink,
            wrap="word",
            relief="flat",
            bd=0,
            highlightthickness=0,
            width=1,
            height=4,
            cursor="arrow",
            font=("Sans", 13),
            spacing3=4,
        )
        self.about_body.pack(fill="x", pady=(6, 0))
        specs = self._about_font_specs(13)
        self.about_body.configure(font=specs[""])
        self.about_body.tag_configure("bold", font=specs["bold"])
        self.about_body.tag_configure("italic", font=specs["italic"])
        for chunk, tag in ABOUT_SEGMENTS:
            self.about_body.insert("end", chunk, tag or ())
        self.about_body.configure(state="disabled")
        # Its own resize is the reliable signal that the wrap width changed.
        self.about_body.bind("<Configure>", self._on_about_configure)
        card._pad = pad
        return card

    def _on_about_configure(self, _event=None):
        if self._about_fit_job is not None:
            self.after_cancel(self._about_fit_job)
        self._about_fit_job = self.after(60, self._fit_about_text)

    def _fit_about_text(self):
        """Pick the largest About font whose wrapped text fills the box beside Base."""
        self._about_fit_job = None
        text = getattr(self, "about_body", None)
        if text is None or not self._cards:
            return
        width = text.winfo_width() - 8
        if width < 120:
            self.after(60, self._fit_about_text)
            return
        scale = self._ui_scale or 1.0
        base = self._cards["base"]
        pad = max(1, int(round(self.CARD_PAD * scale)))
        # Space from the About top down to where the Base box ends. Both terms are
        # independent of this text, so growing the font cannot feed back into them.
        target = base.winfo_y() + base.winfo_reqheight() - self.about_card.winfo_y()
        chrome = 2 * self.CARD_BORDER + 2 * pad + self.about_title.winfo_reqheight() + 6
        available = target - chrome - int(text.cget("spacing3") or 0) - 4
        if available < 40:
            return
        if (width, available) == self._about_fit_key:
            return
        self._about_fit_key = (width, available)
        smallest = max(8, int(round(self.ABOUT_FONT_MIN * scale)))
        for size in range(self.ABOUT_FONT_MAX, self.ABOUT_FONT_MIN - 1, -1):
            px_size = max(8, int(round(size * scale)))
            specs = self._about_font_specs(px_size)
            fonts = {tag: tkfont.Font(root=self, font=spec) for tag, spec in specs.items()}
            lines = _wrapped_line_count(width, fonts)
            if lines * fonts[""].metrics("linespace") <= available or px_size <= smallest:
                text.configure(font=specs[""], height=lines)
                text.tag_configure("bold", font=specs["bold"])
                text.tag_configure("italic", font=specs["italic"])
                self.after_idle(self._sync_page_window)
                return

    def _suite_card(self, parent, spec: SuiteSpec) -> tk.Frame:
        background = _hex(spec.accent)
        heading = _hex(HEADING_INK)
        blurb_fg = _hex(BLURB_INK)
        card = tk.Frame(
            parent,
            bg=background,
            highlightbackground=_hex(_shade(spec.accent, 0.82)),
            highlightthickness=self.CARD_BORDER,
        )
        # Inner padding lives on the frame itself so the scaler has one knob for it.
        pad = tk.Frame(card, bg=background, padx=self.CARD_PAD, pady=self.CARD_PAD)
        pad.pack(fill="both", expand=True)
        title = tk.Label(
            pad,
            text=spec.title,
            bg=background,
            fg=heading,
            anchor="w",
            font=("Sans", 24, "bold"),
        )
        title.pack(anchor="w")
        blurb = tk.Label(
            pad,
            text=spec.blurb,
            bg=background,
            fg=blurb_fg,
            anchor="w",
            justify="left",
            wraplength=420,
            font=("Sans", 13),
        )
        blurb.pack(anchor="w", pady=(4, 12))
        collage = tk.Label(pad, bg=background, bd=0, highlightthickness=0)
        collage.pack(anchor="center")
        # Buttons keep the stronger brand hue so they stay readable on the pastel card.
        button_bg = spec.brand
        button = RoundedButton(
            pad,
            text="Open",
            command=lambda key=spec.key: self._launch(key),
            bg=_hex(button_bg),
            activebackground=_hex(_shade(button_bg, 1.12)),
            font=("Sans", 16, "bold"),
            width=180,
            height=60,
            radius=26,
        )
        button.pack(anchor="w", pady=(14, 0))
        card._spec = spec
        card._pad = pad
        card._title = title
        card._blurb = blurb
        card._collage = collage
        card._collage_photo = None
        card._button = button
        card._cell_width = 0
        return card

    def _tile(self, path: Path) -> Image.Image | None:
        """Load (and cache) a snapshot at thumbnail size for fast collage rebuilds."""
        cached = self._thumbs.get(path)
        if cached is not None:
            return cached
        try:
            with Image.open(path) as source:
                source.seek(0)
                image = source.convert("RGB")
        except (OSError, EOFError, ValueError):
            return None
        if image.width > THUMB_WIDTH:
            height = max(1, int(round(image.height * THUMB_WIDTH / image.width)))
            image = image.resize((THUMB_WIDTH, height), Image.Resampling.LANCZOS)
        self._thumbs[path] = image
        return image

    def _cell_width_for(self, count: int, inner_width: int, max_height: int) -> int:
        rows = max(1, math.ceil(count / COLLAGE_COLUMNS))
        by_width = (inner_width - COLLAGE_GAP * (COLLAGE_COLUMNS - 1)) / COLLAGE_COLUMNS
        by_height = ((max_height - COLLAGE_GAP * (rows - 1)) / rows) / CELL_ASPECT
        return max(24, int(min(by_width, by_height)))

    def _refresh_collage(self, card: tk.Frame, inner_width: int, max_height: int):
        paths = self._sources.get(card._spec.key, [])
        cell_width = self._cell_width_for(len(paths), inner_width, max_height)
        if cell_width == card._cell_width:
            return
        tiles = [tile for tile in (self._tile(path) for path in paths) if tile is not None]
        sheet = build_collage(tiles, cell_width=cell_width, background=card._spec.accent)
        if sheet is None:
            card._collage.configure(
                image="",
                text="No snapshots yet — run script/bench_script/publish_gui_snapshots.py.",
                fg=_hex(BLURB_INK),
                font=("Sans", 12),
                wraplength=inner_width,
            )
            card._collage_photo = None
            card._cell_width = cell_width
            return
        photo = ImageTk.PhotoImage(sheet, master=card._collage)
        card._collage.configure(image=photo, text="")
        card._collage_photo = photo
        card._cell_width = cell_width

    def _on_root_configure(self, event):
        if event.widget is not self:
            return
        if self._ui_scale_job is not None:
            self.after_cancel(self._ui_scale_job)
        self._ui_scale_job = self.after(80, self._apply_ui_scale)

    def _apply_ui_scale(self):
        self._ui_scale_job = None
        width = max(self.winfo_width(), 1)
        height = max(self.winfo_height(), 1)
        scale = min(width / self.DESIGN_SIZE[0], height / self.DESIGN_SIZE[1])
        scale = max(self.UI_SCALE_MIN, min(self.UI_SCALE_MAX, scale))
        rescaled = abs(scale - self._ui_scale) >= 0.02
        if rescaled:
            self._ui_scale = scale

        def font(size, weight=""):
            px_size = max(8, int(round(size * scale)))
            return ("Sans", px_size, weight) if weight else ("Sans", px_size)

        def px(value):
            return max(1, int(round(value * scale)))

        if rescaled:
            apply_gui_logo(self.logo_label, height=px(self.LOGO_HEIGHT))
            self.exit_button.configure(
                font=font(15, "bold"), width=px(120), height=px(54), radius=px(22)
            )
            self.about_title.configure(font=font(24, "bold"))
            self.about_card._pad.configure(padx=px(self.CARD_PAD), pady=px(self.CARD_PAD))

        # Card math follows the page, which is narrower than the window whenever
        # the scrollbar is showing.
        page_width = self.page.winfo_width()
        usable = page_width if page_width > 1 else width
        card_width = (usable - 2 * px(self.OUTER_PAD)) / len(SUITES) - px(self.CARD_GAP)
        inner_width = max(160, int(card_width) - 2 * (px(self.CARD_PAD) + self.CARD_BORDER) - 4)
        # Logo bar, card text, button, and outer padding all sit outside the collage.
        reserve = px(self.LOGO_HEIGHT) + px(240) + 2 * px(self.OUTER_PAD)
        max_height = max(140, height - reserve)
        for card in self._cards.values():
            if rescaled:
                card._title.configure(font=font(24, "bold"))
                card._blurb.configure(font=font(13), wraplength=inner_width)
                card._button.configure(
                    font=font(16, "bold"), width=px(180), height=px(60), radius=px(26)
                )
                card._pad.configure(padx=px(self.CARD_PAD), pady=px(self.CARD_PAD))
                card._cell_width = 0
            else:
                card._blurb.configure(wraplength=inner_width)
            self._refresh_collage(card, inner_width, max_height)
        self._about_fit_key = None
        self.after_idle(self._fit_about_text)
        # Font / collage changes move the natural page height, so the scrollable
        # extent has to be recomputed even when the page frame itself did not resize.
        self.after_idle(self._sync_page_window)

    def _launch(self, key: str):
        if self.child is not None:
            messagebox.showinfo("Already open", "Close the open launcher before starting another one.")
            return
        spec = next(item for item in SUITES if item.key == key)
        script = _INTERACTIVE_DIR / spec.script
        if not script.is_file():
            messagebox.showerror("Unavailable", f"Missing launcher:\n{script}")
            return
        child_env = os.environ.copy()
        if key != "experiment":
            # Free play from the hub: never inherit a participant session.
            for variable in (EXPERIMENT_ENV, EXPERIMENT_USER_ENV, EXPERIMENT_LOG_ENV):
                child_env.pop(variable, None)
        try:
            self.child = subprocess.Popen(
                [sys.executable, str(script)],
                cwd=ROOT,
                start_new_session=True,
                env=child_env,
            )
        except OSError as exc:
            messagebox.showerror("Could not start launcher", str(exc))
            self.child = None
            return
        self.withdraw()

    def _poll_child(self):
        if self.child is not None and self.child.poll() is not None:
            self.child = None
            try:
                self.deiconify()
                self.lift()
                self.focus_force()
            except tk.TclError:
                pass
        self.after(250, self._poll_child)

    def _stop_child(self):
        child = self.child
        if child is None:
            return
        try:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=2)
        except ProcessLookupError:
            pass
        finally:
            self.child = None

    def exit_app(self):
        self._stop_child()
        self.destroy()


def main():
    RoboDynaLauncher().mainloop()


if __name__ == "__main__":
    main()

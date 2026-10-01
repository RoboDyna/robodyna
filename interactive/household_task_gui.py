#!/usr/bin/env python3
"""Scrollable graphical launcher for the household interactive tasks."""
from __future__ import annotations

import json
import os
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent / "household"
TUTORIAL_DIR = Path(__file__).resolve().parent / "Tutorial"
DEMO_DIR = ROOT / "docs/final_task_demos"
README_PATH = ROOT / "README.md"
# Must match interactive._interactive_common.TASK_RESULT_ENV
TASK_RESULT_ENV = "ROBODYNA_TASK_RESULT_FILE"


def record_status_note(payload) -> str:
    """Append-only note when the child wrote collect_data-format files."""
    if not isinstance(payload, dict):
        return ""
    bits = []
    hdf5 = str(payload.get("record_hdf5") or "").strip()
    video = str(payload.get("record_video") or "").strip()
    viewer = str(payload.get("record_viewer") or "").strip()
    if hdf5:
        bits.append(hdf5)
    if video:
        bits.append(video)
    if viewer:
        bits.append(viewer)
    if bits:
        return " Recorded " + "; ".join(bits) + "."
    path = str(payload.get("record_path") or "").strip()
    ep = payload.get("record_episode")
    if path and ep is not None:
        return f" Recorded {path}/data/episode{ep}.hdf5 + video/episode{ep}.mp4."
    return ""


def ps_status_note(payload) -> str:
    """Append `` PS 0.75`` when the child reported a partial score."""
    if not isinstance(payload, dict):
        return ""
    raw = payload.get("partial_score")
    if raw is None and isinstance(payload.get("metrics"), dict):
        raw = payload["metrics"].get("partial_score")
    if raw is None:
        return ""
    try:
        return f" PS {float(raw):.2f}"
    except (TypeError, ValueError):
        return ""

_INTERACTIVE_DIR = Path(__file__).resolve().parent
if str(_INTERACTIVE_DIR) not in sys.path:
    sys.path.insert(0, str(_INTERACTIVE_DIR))
from _task_briefing import (  # noqa: E402
    GUI_INK,
    GUI_MUTED,
    GUI_PAGE_BG,
    GUI_WM_CLASS,
    apply_gui_logo,
    build_briefing_text,
    setup_gui_app_icon,
    show_task_briefing,
)
from _trial_metrics import show_trial_metrics  # noqa: E402
from experiment_config import load_experiment_config  # noqa: E402
from experiment_logs import (  # noqa: E402
    append_play,
    experiment_mode,
    is_slot_locked,
    load_user_assignment,
    log_controller_tag,
    payload_fail_reason,
    session_control_mode,
    slot_success_label,
    stamp_child_env,
    terminal_play_count,
)

TASKS = (
    ("Trap Bug", "trap_bug", "interactive_trap_bug.py"),
    ("Boil Milk", "boil_milk", "interactive_boil_milk.py"),
    ("Fill Coffee Jar", "fill_coffee_jar", "interactive_fill_coffee_jar.py"),
    ("Pour Beer", "pour_beer", "interactive_pour_beer.py"),
    ("Cook Food", "cook_food", "interactive_cook_food.py"),
    ("Cook Food Timer", "cook_food_timer", "interactive_cook_food_timer.py"),
    ("Measure Ingredient", "measure_ingredient", "interactive_measure_ingredient.py"),
    ("Make Soup", "make_soup", "interactive_make_soup.py"),
    ("Catch Cup", "catch_cup", "interactive_catch_cup.py"),
    ("Catch Mouse Object Drop", "catch_mouse_object_drop", "interactive_catch_mouse_object_drop.py"),
    ("Stop Ball", "stop_ball", "interactive_stop_ball.py"),
    ("Clean Table", "clean_table", "interactive_clean_table.py"),
)

TUTORIAL_PARTS = (
    ("Selection", "tutorial_part1", "Select arms (1, 2, 3) then switch camera (V)."),
    ("Basic control", "tutorial_part2", "Move with arrows, E/Q, R/T, F/G, then Space."),
    ("Basic actions", "tutorial_part3", "Grasp, hold-button, switch, then push a box."),
    ("Advance actions", "tutorial_part4", "Stove knob on/off, then multi-stage force key."),
)

KEYBOARD_TUTORIAL_PARTS = (
    ("Buttons", "tutorial_kb_buttons", "Number keys, arrows+bowl, on/off, then push — keyboard then mouse."),
    ("Placement", "tutorial_kb_placement", "Click to place a cup on the green pad, then pack an apple in the box."),
    ("Base", "tutorial_kb_base", "Base keyboard tutorial — open the base GUI."),
    ("Household", "tutorial_kb_household", "Stop a ball, toggle a lamp, then tilt the board."),
)

PLAY_BLUE = "#3182bd"
PLAY_BLUE_ACTIVE = "#4295d0"
LOCKED_GRAY = "#59616b"
RECORDING_GREEN = "#2a7a52"
PAGE_BG = GUI_PAGE_BG
HEADER_BG = GUI_PAGE_BG
HEADER_FG = GUI_INK
HEADER_MUTED = GUI_MUTED
CARD_BG = "#202c38"
CARD_BORDER = "#405367"
TEXT_PRIMARY = "#f4f8fb"
TEXT_SECONDARY = "#aebdca"
HINT_FG = "#7fb6dc"
RANDOM_SEED_MAX = 100000

def load_task_descriptions(readme_path: Path = README_PATH) -> dict[str, str]:
    """Parse README household-task ``<sub>…</sub>`` blurbs into task → description."""
    if not readme_path.exists():
        return {}
    text = readme_path.read_text(encoding="utf-8")
    # Prefer the Household Tasks section when present.
    section = text
    marker = "## Household Tasks"
    if marker in text:
        section = text.split(marker, 1)[1]
    row_re = re.compile(r"^\|\s*\*\*`([a-z0-9_]+)`\*\*.*$", re.MULTILINE)
    sub_re = re.compile(r"<sub>(.*?)</sub>", re.DOTALL)
    out: dict[str, str] = {}
    for match in row_re.finditer(section):
        task = match.group(1)
        line_end = section.find("\n", match.start())
        line = section[match.start() : line_end if line_end >= 0 else None]
        subs = [re.sub(r"\s+", " ", s).strip() for s in sub_re.findall(line)]
        if not subs:
            continue
        out[task] = subs[0]
    return out


TASK_DESCRIPTIONS = load_task_descriptions()


def task_description(task: str) -> str:
    """README blurb for ``task``, or empty if unavailable."""
    return str(TASK_DESCRIPTIONS.get(task, "") or "")


def resolve_seed(value: str) -> int:
    """Return the entered seed, or generate a fresh one when left blank."""
    value = value.strip()
    if not value:
        return secrets.randbelow(RANDOM_SEED_MAX + 1)
    try:
        seed = int(value)
    except ValueError as exc:
        raise ValueError("Seed must be a whole number or left blank for a random seed.") from exc
    if not 0 <= seed <= RANDOM_SEED_MAX:
        raise ValueError(f"Seed must be between 0 and {RANDOM_SEED_MAX}.")
    return seed


class RoundedButton(tk.Canvas):
    """Canvas-backed button with genuinely rounded corners."""

    def __init__(
        self,
        parent,
        *,
        text,
        command,
        bg,
        activebackground,
        font,
        width,
        height,
        radius=28,
        on_enter=None,
        on_leave=None,
    ):
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=parent.cget("bg"),
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        self.command = command
        self.normal_color = bg
        self.active_color = activebackground
        self.disabled_color = "#59616b"
        self.disabled_text = "#b4bac2"
        self.button_state = "normal"
        self.radius = radius
        self._shape = None
        self._label = None
        self._text = text
        self._font = font
        self._on_enter_cb = on_enter
        self._on_leave_cb = on_leave
        self.bind("<Configure>", self._redraw)
        self.bind("<Enter>", self._enter)
        self.bind("<Leave>", self._leave)
        self.bind("<ButtonRelease-1>", self._click)

    def _rounded_polygon(self, width, height):
        r = min(self.radius, width // 2, height // 2)
        return (
            r, 1, width - r, 1, width - 1, 1, width - 1, r,
            width - 1, height - r, width - 1, height - 1,
            width - r, height - 1, r, height - 1, 1, height - 1,
            1, height - r, 1, r, 1, 1,
        )

    def _redraw(self, _event=None):
        self.delete("all")
        width = max(2, self.winfo_width())
        height = max(2, self.winfo_height())
        if self.button_state == "normal":
            color = self.normal_color
            text_fill = "white"
        else:
            color = self.disabled_color
            text_fill = self.disabled_text
        self._shape = self.create_polygon(
            self._rounded_polygon(width, height),
            smooth=True,
            splinesteps=24,
            fill=color,
            outline="",
        )
        self._label = self.create_text(
            width / 2,
            height / 2,
            text=self._text,
            fill=text_fill,
            font=self._font,
            width=max(8, width - 16),
            justify="center",
        )

    def _enter(self, _event):
        if self.button_state == "normal" and self._shape is not None:
            self.itemconfigure(self._shape, fill=self.active_color)
        if self._on_enter_cb is not None:
            self._on_enter_cb()

    def _leave(self, _event):
        if self.button_state == "normal" and self._shape is not None:
            self.itemconfigure(self._shape, fill=self.normal_color)
        if self._on_leave_cb is not None:
            self._on_leave_cb()

    def _click(self, _event):
        if self.button_state == "normal":
            self.command()

    def configure(self, cnf=None, **kwargs):
        options = dict(cnf or {})
        options.update(kwargs)
        if "text" in options:
            self._text = options.pop("text")
        if "bg" in options:
            self.normal_color = options.pop("bg")
        if "activebackground" in options:
            self.active_color = options.pop("activebackground")
        if "disabledbackground" in options:
            self.disabled_color = options.pop("disabledbackground")
        if "disabledforeground" in options:
            self.disabled_text = options.pop("disabledforeground")
        if "state" in options:
            self.button_state = options.pop("state")
            options.setdefault(
                "cursor", "arrow" if self.button_state != "normal" else "hand2"
            )
        if "font" in options:
            self._font = options.pop("font")
        if "radius" in options:
            self.radius = int(options.pop("radius"))
        if options:
            super().configure(**options)
        self._redraw()

    config = configure


class HouseholdTaskLauncher(tk.Tk):
    # Head-camera stills are ~4:3. Width seeds the first layout; height follows
    # each image's native aspect so previews are not letterboxed.
    IMAGE_SIZE = (1400, 1050)
    COLUMNS = 4
    CARD_PAD = 8
    PREVIEW_SIDE_PAD = 28
    DESIGN_SIZE = (1560, 980)
    UI_SCALE_MIN = 0.55
    UI_SCALE_MAX = 1.15

    def __init__(self):
        # className becomes StartupWMClass for the Ubuntu dock .desktop entry.
        super().__init__(className=GUI_WM_CLASS["household"])
        self.title("Household Tasks")
        self.geometry("1560x980")
        self.minsize(900, 640)
        self.configure(bg=PAGE_BG)
        setup_gui_app_icon(
            self,
            suite="household",
            script_path=Path(__file__),
        )
        self.protocol("WM_DELETE_WINDOW", self.exit_app)

        self.child: subprocess.Popen | None = None
        self.active_index: int | None = None
        self.active_tutorial: int | None = None
        self.result_file: Path | None = None
        self._run_meta: dict | None = None
        self._recording_run = False
        self._recording_phase = False
        self.preview_sources: list[Image.Image | None] = []
        self.preview_photos: list[ImageTk.PhotoImage | None] = []
        self.preview_labels: list[tk.Label] = []
        self.task_buttons: list[RoundedButton] = []
        self.tutorial_sources: list[Image.Image | None] = []
        self.tutorial_photos: list[ImageTk.PhotoImage | None] = []
        self.tutorial_preview_labels: list[tk.Label] = []
        self.tutorial_buttons: list[RoundedButton] = []
        self.card_index_labels: list[tk.Label] = []
        self.card_title_labels: list[tk.Label] = []
        self.category_header_labels: list[tk.Label] = []
        self._preview_resize_job: str | None = None
        self._ui_scale_job: str | None = None
        self._preview_width = self.IMAGE_SIZE[0]
        self._ui_scale = 1.0
        self._header_layout_key: tuple | None = None
        self._visible_task_indices: list[int] = list(range(len(TASKS)))
        self._experiment_cfg = load_experiment_config() if experiment_mode() else None
        self._assignment = load_user_assignment() if self._experiment_cfg is not None else None
        if self._experiment_cfg is not None:
            assignment = self._assignment or {}
            grouped = self._experiment_cfg.grouped_visible_indices(
                "household",
                len(TASKS),
                assigned=assignment.get("household_tasks"),
                picks=assignment.get("household_picks"),
            )
            self._visible_task_indices = [
                index for _label, idxs in grouped for index in idxs
            ]
        self._idle_status = (
            f"Tutorial (4 parts)  ·  {len(self._visible_task_indices)} tasks  |  "
            "Hover a task for its README description."
        )
        self._idle_status_fg = HEADER_MUTED

        self._build_ui()
        if experiment_mode():
            user = os.environ.get("ROBODYNA_EXPERIMENT_USER", "").strip()
            title = f"Household Tasks  ·  {user}" if user else "Household Tasks  ·  Experiment"
            self.title(title)
            self.title_label.configure(text=title)
            self.subtitle_label.configure(
                text=(
                    "Play each task once. Finished tasks stay gray and cannot be replayed. "
                    "Tutorial is always available."
                )
            )
            self.exit_button.configure(text="Back")
            n_vis = len(self._visible_task_indices)
            self._idle_status = (
                f"Experiment  ·  {user or 'participant'}  ·  {n_vis} tasks  |  "
                "Completed tasks are gray. Tutorial is always available."
            )
            self._idle_status_fg = HEADER_MUTED
            self.status.configure(text=self._idle_status, fg=self._idle_status_fg)
            self._apply_experiment_protocol()
            self._apply_completed_locks()
            self._refresh_tutorial_mode()
        self.bind("<Configure>", self._on_root_configure)
        self.after(0, self._apply_ui_scale)
        self.after(250, self._poll_child)

    def _build_ui(self):
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass
        self.style.configure(
            "Task.TCombobox",
            padding=(8, 6),
            arrowsize=16,
            fieldbackground="#f7fafc",
            foreground="#182633",
            selectbackground="#f7fafc",
            selectforeground="#182633",
        )

        self.logo_bar = tk.Frame(self, bg=PAGE_BG)
        self.logo_bar.pack(fill="x", padx=24, pady=(16, 0))
        self.logo_label = tk.Label(self.logo_bar, bg=PAGE_BG, bd=0, highlightthickness=0)
        self.logo_label.pack(anchor="center")
        apply_gui_logo(self.logo_label, height=160)

        self.header = tk.Frame(
            self,
            bg=HEADER_BG,
            highlightbackground="#d4d5db",
            highlightthickness=1,
        )
        self.header.pack(fill="x", padx=24, pady=(12, 12))

        # Title + Exit live in the top row so Exit is packed first and never clipped.
        self.header_top = tk.Frame(self.header, bg=HEADER_BG)
        self.header_top.pack(fill="x")
        self.exit_button = RoundedButton(
            self.header_top,
            text="Exit",
            command=self.exit_app,
            bg="#e34a33",
            activebackground="#eb6854",
            font=("Sans", 17, "bold"),
            width=150,
            height=76,
            radius=28,
        )
        self.exit_button.pack(side="right", padx=(8, 22), pady=16)

        self.heading = tk.Frame(self.header_top, bg=HEADER_BG)
        self.heading.pack(side="left", padx=24, pady=18)
        self.title_label = tk.Label(
            self.heading,
            text="Household Tasks",
            bg=HEADER_BG,
            fg=HEADER_FG,
            anchor="w",
            justify="left",
            font=("Sans", 34, "bold"),
        )
        self.title_label.pack(anchor="w")
        self.subtitle_label = tk.Label(
            self.heading,
            text="Tutorial first, then select a household task",
            bg=HEADER_BG,
            fg=HEADER_MUTED,
            anchor="w",
            justify="left",
            wraplength=720,
            font=("Sans", 14),
        )
        self.subtitle_label.pack(anchor="w", pady=(3, 0))

        # Master is header so this can sit on the title row or wrap onto a full-width row.
        self.controls = tk.Frame(self.header, bg=HEADER_BG)
        self.controls.pack(in_=self.header_top, side="right", padx=22, pady=16)

        self.option_group = tk.Frame(self.controls, bg=HEADER_BG)
        self.option_group.pack(side="left", padx=(0, 16))
        self.show_briefing = tk.BooleanVar(value=True)
        self.record_data = tk.BooleanVar(value=False)
        self.save_video = tk.BooleanVar(value=False)
        self.show_metrics = tk.BooleanVar(value=True)
        self.briefing_check = self._header_tick(
            self.option_group, "Instructions", self.show_briefing
        )
        self.metrics_check = self._header_tick(
            self.option_group, "Trial metrics", self.show_metrics
        )
        self.record_check = self._header_tick(
            self.option_group, "Record data", self.record_data
        )
        self.video_check = self._header_tick(
            self.option_group, "Save video", self.save_video
        )

        self.seed_group = tk.Frame(self.controls, bg=HEADER_BG)
        self.seed_group.pack(side="left", padx=(0, 18))
        self.seed_caption = tk.Label(
            self.seed_group,
            text="Seed (blank = random)",
            bg=HEADER_BG,
            fg=HEADER_MUTED,
            font=("Sans", 13, "bold"),
        )
        self.seed_caption.pack(anchor="w")
        self.seed_entry = tk.Entry(
            self.seed_group,
            width=16,
            font=("Sans", 13, "bold"),
            bg="#f7fafc",
            fg="#182633",
            insertbackground="#182633",
            relief="flat",
        )
        self.seed_entry.pack(ipady=5, pady=(4, 0))

        self.control_group = tk.Frame(self.controls, bg=HEADER_BG)
        self.control_group.pack(side="left", padx=(0, 14))
        self.control_caption = tk.Label(
            self.control_group,
            text="Control",
            bg=HEADER_BG,
            fg=HEADER_MUTED,
            font=("Sans", 13, "bold"),
        )
        self.control_caption.pack(anchor="w")
        self.control = ttk.Combobox(
            self.control_group,
            values=("keyboard+mouse", "robot"),
            state="readonly",
            width=14,
            font=("Sans", 13, "bold"),
            style="Task.TCombobox",
        )
        self.control.set("robot")
        self.control.pack(pady=(4, 0))
        self.control.bind("<<ComboboxSelected>>", self._on_control_change)
        self._style_control_menu(("Sans", 13, "bold"))
        self._control_groups = (self.option_group, self.seed_group, self.control_group)

        self.status = tk.Label(
            self,
            text=self._idle_status,
            bg=PAGE_BG,
            fg=self._idle_status_fg,
            anchor="w",
            justify="left",
            wraplength=1460,
            font=("Sans", 19),
        )
        self.status.pack(fill="x", padx=34, pady=(2, 12))

        # Scrollable 4-column task grid, matching the base suite layout.
        self.outer = tk.Frame(self, bg=PAGE_BG)
        self.outer.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.canvas = tk.Canvas(self.outer, bg=PAGE_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.outer, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.page = tk.Frame(self.canvas, bg=PAGE_BG)
        self.page_window = self.canvas.create_window((0, 0), window=self.page, anchor="nw")
        self.page.bind("<Configure>", self._update_scroll_region)
        self.canvas.bind("<Configure>", self._resize_page)
        self.canvas.bind_all("<MouseWheel>", self._mousewheel)
        self.canvas.bind_all("<Button-4>", lambda event: self.canvas.yview_scroll(-3, "units"))
        self.canvas.bind_all("<Button-5>", lambda event: self.canvas.yview_scroll(3, "units"))

        self._add_tutorial_section()
        # Flat 4-column grid (same in free-play and experiment). Experiment still
        # filters to the assigned tasks via ``_visible_task_indices``; tiling them
        # side-by-side avoids one-per-line when only Easy + Hard are assigned.
        vis = self._visible_task_indices
        for row_start in range(0, len(vis), self.COLUMNS):
            self._add_task_row(vis[row_start : row_start + self.COLUMNS])

    @staticmethod
    def _scaled_font(size: float, weight: str = "", scale: float = 1.0) -> tuple:
        px = max(8, int(round(size * scale)))
        return ("Sans", px, weight) if weight else ("Sans", px)

    def _style_control_menu(self, font) -> None:
        """Enlarge the Control combobox dropdown list independently of the field."""
        try:
            popdown = self.tk.call("ttk::combobox::PopdownWindow", self.control)
            self.tk.call(f"{popdown}.f.l", "configure", "-font", font)
        except tk.TclError:
            self.option_add("*TCombobox*Listbox.font", font)

    @staticmethod
    def _px(value: float, scale: float) -> int:
        return max(1, int(round(value * scale)))

    def _compute_ui_scale(self, width: int | None = None, height: int | None = None) -> float:
        design_w, design_h = self.DESIGN_SIZE
        width = int(width if width is not None else max(self.winfo_width(), 1))
        height = int(height if height is not None else max(self.winfo_height(), 1))
        scale = min(width / design_w, height / design_h)
        return max(self.UI_SCALE_MIN, min(self.UI_SCALE_MAX, scale))

    def _on_root_configure(self, event):
        if event.widget is not self:
            return
        if self._ui_scale_job is not None:
            self.after_cancel(self._ui_scale_job)
        self._ui_scale_job = self.after(80, self._apply_ui_scale)

    def _header_tick(self, parent, text, variable):
        """Square checkbox used for Instructions / Trial metrics / Record data / Save video."""
        btn = tk.Checkbutton(
            parent,
            text=text,
            variable=variable,
            onvalue=True,
            offvalue=False,
            bg=HEADER_BG,
            fg=HEADER_FG,
            activebackground=HEADER_BG,
            activeforeground=HEADER_FG,
            selectcolor="#ffffff",
            highlightthickness=0,
            bd=0,
            indicatoron=True,
            font=("Sans", 14, "bold"),
            cursor="hand2",
            anchor="w",
        )
        btn.pack(anchor="w")
        return btn

    def _set_option_checks(self, state: str) -> None:
        for widget in (
            self.briefing_check,
            self.metrics_check,
            self.record_check,
            self.video_check,
        ):
            widget.configure(state=state)

    def _capture_run_note(self) -> str:
        meta = getattr(self, "_run_meta", None)
        if (
            experiment_mode()
            and isinstance(meta, dict)
            and meta.get("suite") in ("base", "household")
        ):
            cfg = self._experiment_cfg or load_experiment_config()
            self._experiment_cfg = cfg
            suite = str(meta.get("suite") or "")
            task = str(meta.get("task") or "")
            scenario = meta.get("scenario")
            data = cfg.should_record_data(suite, task, scenario)
            video = cfg.should_save_video(suite, task, scenario)
        else:
            data = bool(self.record_data.get())
            video = bool(self.save_video.get())
        if data and video:
            return " Recording data and head-camera video after the viewer closes."
        if data:
            return " Recording data after the viewer closes."
        if video:
            return " Saving head-camera video after the viewer closes."
        return ""

    def _apply_record_launch(
        self,
        child_env: dict,
        command: list[str],
        *,
        suite: str | None = None,
        task: str | None = None,
        scenario: str | None = None,
    ) -> None:
        """Pass Record data / Save video through to the interactive child."""
        if experiment_mode() and suite and task:
            cfg = self._experiment_cfg or load_experiment_config()
            self._experiment_cfg = cfg
            want_data = cfg.should_record_data(suite, task, scenario)
            want_video = cfg.should_save_video(suite, task, scenario)
            want_lerobot = want_data and cfg.should_export_lerobot(suite, task, scenario)
        else:
            want_data = bool(self.record_data.get())
            want_video = bool(self.save_video.get())
            want_lerobot = False
        child_env["ROBODYNA_SAVE_VIDEO"] = "1" if want_video else "0"
        child_env["ROBODYNA_EXPORT_LEROBOT"] = "1" if want_lerobot else "0"
        if want_data or want_video:
            child_env.setdefault("ROBODYNA_DISABLE_RAY_TRACING", "1")
        if want_data:
            child_env["ROBODYNA_RECORD_DATA"] = "1"
            child_env["ROBODYNA_RECORD_CONFIG"] = "demo_dynamic"
            command.append("--record-data")
        elif want_video:
            child_env["ROBODYNA_RECORD_CONFIG"] = "demo_dynamic"
        self._recording_run = bool(experiment_mode() and want_data)

    def _apply_ui_scale(self):
        self._ui_scale_job = None
        if getattr(self, "title_label", None) is None:
            return
        scale = self._compute_ui_scale()
        if abs(scale - self._ui_scale) < 0.02:
            self.status.configure(
                wraplength=max(360, self.winfo_width() - self._px(68, scale))
            )
            self._relayout_header(scale)
            return
        self._ui_scale = scale
        s = scale

        self.logo_bar.pack_configure(padx=self._px(24, s), pady=(self._px(16, s), 0))
        apply_gui_logo(self.logo_label, height=self._px(160, s))
        self.header.pack_configure(padx=self._px(24, s), pady=(self._px(12, s), self._px(12, s)))
        self.title_label.configure(font=self._scaled_font(34, "bold", s))
        self.subtitle_label.configure(font=self._scaled_font(14, scale=s))
        self.briefing_check.configure(font=self._scaled_font(14, "bold", s))
        self.metrics_check.configure(font=self._scaled_font(14, "bold", s))
        self.record_check.configure(font=self._scaled_font(14, "bold", s))
        self.video_check.configure(font=self._scaled_font(14, "bold", s))
        self.seed_caption.configure(font=self._scaled_font(13, "bold", s))
        self.control_caption.configure(font=self._scaled_font(13, "bold", s))
        self.seed_entry.configure(font=self._scaled_font(13, "bold", s))
        self.seed_entry.pack_configure(ipady=self._px(5, s), pady=(self._px(4, s), 0))
        self.control.configure(font=self._scaled_font(13, "bold", s))
        self._style_control_menu(self._scaled_font(13, "bold", s))
        self.control.pack_configure(pady=(self._px(4, s), 0))
        self.style.configure(
            "Task.TCombobox",
            padding=(self._px(8, s), self._px(6, s)),
            arrowsize=max(10, self._px(16, s)),
            fieldbackground="#f7fafc",
            foreground="#182633",
            selectbackground="#f7fafc",
            selectforeground="#182633",
        )
        self.exit_button.configure(
            font=self._scaled_font(17, "bold", s),
            width=self._px(150, s),
            height=self._px(76, s),
            radius=self._px(28, s),
        )
        self.status.configure(
            font=self._scaled_font(19, scale=s),
            wraplength=max(360, self.winfo_width() - self._px(68, s)),
        )
        self.status.pack_configure(padx=self._px(34, s), pady=(self._px(2, s), self._px(12, s)))
        self.outer.pack_configure(padx=self._px(12, s), pady=(0, self._px(12, s)))

        idx_padx = self._px(13, s)
        idx_pady = self._px(6, s)
        for label in self.card_index_labels:
            label.configure(
                font=self._scaled_font(17, "bold", s),
                padx=idx_padx,
                pady=idx_pady,
            )
        for label in self.card_title_labels:
            label.configure(font=self._scaled_font(15, "bold", s))
        for label in getattr(self, "category_header_labels", []):
            label.configure(font=self._scaled_font(16, "bold", s))
        if getattr(self, "tutorial_index_label", None) is not None:
            self.tutorial_index_label.configure(
                font=self._scaled_font(17, "bold", s),
                padx=idx_padx,
                pady=idx_pady,
            )
        if getattr(self, "tutorial_title_label", None) is not None:
            self.tutorial_title_label.configure(font=self._scaled_font(27, "bold", s))
        if getattr(self, "tutorial_badge_label", None) is not None:
            self.tutorial_badge_label.configure(font=self._scaled_font(12, "bold", s))

        btn_h = self._px(70, s)
        btn_radius = self._px(26, s)
        btn_font = self._scaled_font(16, "bold", s)
        for button in self.task_buttons:
            button.configure(font=btn_font, height=btn_h, radius=btn_radius)
        for button in self.tutorial_buttons:
            button.configure(font=btn_font, height=btn_h, radius=btn_radius)

        self._relayout_header(s)

    def _relayout_header(self, scale: float):
        """Keep Exit visible and reflow title/controls as the window width changes."""
        if getattr(self, "_relayouting", False):
            return
        self._relayouting = True
        try:
            self._relayout_header_body(scale)
        finally:
            self._relayouting = False

    def _relayout_header_body(self, scale: float):
        header_w = int(self.header.winfo_width())
        if header_w <= 1:
            header_w = max(1, int(self.winfo_width()) - 2 * self._px(24, scale))
        pad_exit = self._px(16, scale)
        pad_head = self._px(24, scale)
        pad_ctrl = self._px(22, scale)
        gap = self._px(16, scale)
        self.update_idletasks()
        exit_w = self.exit_button.winfo_reqwidth() + pad_exit + self._px(8, scale)
        avail = max(self._px(160, scale), header_w - exit_w)
        controls_w = self._controls_natural_width(scale)
        min_heading = self._px(220, scale)
        stacked = controls_w + gap + min_heading > avail
        wrap_groups = stacked and controls_w > avail - self._px(12, scale)
        if stacked:
            wrap = max(self._px(160, scale), avail - 2 * pad_head)
        else:
            wrap = max(self._px(160, scale), avail - controls_w - gap - 2 * pad_head)
        wrap -= wrap % 8
        key = (stacked, wrap_groups, wrap, round(scale, 2))
        if key == self._header_layout_key:
            return
        self._header_layout_key = key

        self.title_label.configure(wraplength=wrap)
        self.subtitle_label.configure(wraplength=wrap)
        self.heading.pack_forget()
        self.controls.pack_forget()
        self.exit_button.pack_forget()

        self.exit_button.pack(
            side="right",
            padx=(self._px(8, scale), pad_exit),
            pady=self._px(16, scale),
        )
        self._pack_control_groups(wrap=wrap_groups, scale=scale)
        if stacked:
            self.heading.pack(
                side="left",
                fill="x",
                expand=True,
                padx=pad_head,
                pady=(self._px(14, scale), self._px(10, scale)),
            )
            self.controls.pack(
                in_=self.header,
                side="top",
                anchor="w",
                fill="x",
                padx=pad_ctrl,
                pady=(0, self._px(14, scale)),
            )
        else:
            self.heading.pack(side="left", padx=pad_head, pady=self._px(18, scale))
            self.controls.pack(
                in_=self.header_top,
                side="right",
                padx=pad_ctrl,
                pady=self._px(16, scale),
            )

    def _controls_natural_width(self, scale: float) -> int:
        pads = (self._px(16, scale), self._px(18, scale), self._px(14, scale))
        return sum(
            group.winfo_reqwidth() + pad
            for group, pad in zip(self._control_groups, pads)
        )

    def _pack_control_groups(self, *, wrap: bool, scale: float) -> None:
        pads = (self._px(16, scale), self._px(18, scale), self._px(14, scale))
        for group in self._control_groups:
            group.pack_forget()
        if wrap:
            for group in self._control_groups:
                group.pack(side="top", anchor="w", pady=(0, self._px(6, scale)))
        else:
            for group, pad in zip(self._control_groups, pads):
                group.pack(side="left", padx=(0, pad))

    def _update_scroll_region(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _resize_page(self, event):
        page_width = max(event.width, 720)
        self.canvas.itemconfigure(self.page_window, width=page_width)
        preview_width = max(
            480, page_width - 2 * (self.CARD_PAD + self.PREVIEW_SIDE_PAD)
        )
        if abs(preview_width - self._preview_width) < 8:
            return
        if self._preview_resize_job is not None:
            self.after_cancel(self._preview_resize_job)
        self._preview_resize_job = self.after(
            80, lambda width=preview_width: self._apply_preview_width(width)
        )

    def _cell_preview_width(self, total_width: int) -> int:
        """Width of one of the four task preview columns."""
        gap = 6 * 2 * self.COLUMNS  # padx=6 on each side of each column
        return max(160, (total_width - gap) // self.COLUMNS)

    def _apply_preview_width(self, width: int):
        self._preview_resize_job = None
        self._preview_width = width
        cell_width = self._cell_preview_width(width)
        photos: list[ImageTk.PhotoImage | None] = []
        for index, source in enumerate(self.preview_sources):
            photo = self._render_preview(source, cell_width)
            photos.append(photo)
            label = self.preview_labels[index]
            if photo is None:
                label.configure(
                    image="",
                    text="No preview",
                    width=max(8, cell_width // 10),
                    height=max(4, (cell_width * 3) // (8 * 16)),
                )
            else:
                label.configure(image=photo, text="", width=0, height=0)
            if index < len(self.card_title_labels):
                self.card_title_labels[index].configure(
                    wraplength=max(80, cell_width - self._px(56, self._ui_scale))
                )
        self.preview_photos = photos
        self._refresh_tutorial_previews(cell_width)

    def _mousewheel(self, event):
        if event.delta:
            self.canvas.yview_scroll(-int(event.delta / 120), "units")

    def _tutorial_snapshot_path(self, part_index: int | None = None) -> Path | None:
        names: list[str] = []
        if part_index is not None:
            part_n = part_index + 1
            if getattr(self, "_keyboard_tutorial", lambda: False)():
                stems = ("buttons", "placement", "base", "household")
                names.append(f"scene_snapshot_kb_{stems[part_index]}.png")
                names.append(f"scene_snapshot_kb_part{part_n}.png")
            # Household Part 4 opens on the stove; prefer a dedicated still.
            if part_n == 4:
                names.append("scene_snapshot_part4_household.png")
            names.append(f"scene_snapshot_part{part_n}.png")
        names.append("scene_snapshot.png")
        directories = (TUTORIAL_DIR, DEMO_DIR / "tutorial_empty")
        for directory in directories:
            for name in names:
                path = directory / name
                if path.exists():
                    return path
        return None

    def _load_tutorial_source(self, part_index: int | None = None) -> Image.Image | None:
        path = self._tutorial_snapshot_path(part_index)
        if path is None:
            return None
        try:
            with Image.open(path) as source:
                source.seek(0)
                return source.convert("RGB")
        except Exception:
            return None

    def _refresh_tutorial_previews(self, width: int | None = None) -> None:
        cell_width = int(
            width if width is not None else self._cell_preview_width(self._preview_width)
        )
        photos: list[ImageTk.PhotoImage | None] = []
        for index, source in enumerate(self.tutorial_sources):
            photo = self._render_preview(source, cell_width)
            photos.append(photo)
            if index >= len(self.tutorial_preview_labels):
                continue
            label = self.tutorial_preview_labels[index]
            if photo is None:
                label.configure(
                    image="",
                    text="No preview",
                    width=max(8, cell_width // 10),
                    height=max(4, (cell_width * 3) // (8 * 16)),
                )
            else:
                label.configure(image=photo, text="", width=0, height=0)
        self.tutorial_photos = photos

    def _add_tutorial_section(self):
        """Same four-column tutorial card as the base GUI."""
        card = tk.Frame(
            self.page,
            bg=CARD_BG,
            highlightbackground=CARD_BORDER,
            highlightthickness=2,
        )
        card.pack(fill="x", padx=self.CARD_PAD, pady=12)

        card_header = tk.Frame(card, bg=CARD_BG)
        card_header.pack(fill="x", padx=self.PREVIEW_SIDE_PAD, pady=(14, 8))
        self.tutorial_index_label = tk.Label(
            card_header,
            text="00",
            bg=PLAY_BLUE,
            fg="white",
            font=("Sans", 17, "bold"),
            padx=13,
            pady=6,
        )
        self.tutorial_index_label.pack(side="left", padx=(0, 14))
        self.tutorial_title_label = tk.Label(
            card_header,
            text="Tutorial",
            bg=CARD_BG,
            fg=TEXT_PRIMARY,
            anchor="w",
            font=("Sans", 27, "bold"),
        )
        self.tutorial_title_label.pack(side="left", fill="x", expand=True)
        self.tutorial_badge_label = tk.Label(
            card_header,
            text="4 PARTS",
            bg=CARD_BG,
            fg="#7fb6dc",
            font=("Sans", 12, "bold"),
        )
        self.tutorial_badge_label.pack(side="right")

        grid = tk.Frame(card, bg=CARD_BG)
        grid.pack(fill="x", padx=self.PREVIEW_SIDE_PAD, pady=(0, 16))
        self.tutorial_sources = [
            self._load_tutorial_source(index) for index in range(len(TUTORIAL_PARTS))
        ]
        cell_width = self._cell_preview_width(self._preview_width)
        self.tutorial_photos = []

        for index, (label, _script, hint) in enumerate(TUTORIAL_PARTS):
            col = tk.Frame(grid, bg=CARD_BG)
            col.pack(side="left", expand=True, fill="both", padx=6)

            photo = self._render_preview(self.tutorial_sources[index], cell_width)
            self.tutorial_photos.append(photo)
            preview_label = tk.Label(
                col,
                image=photo,
                text="No preview" if photo is None else "",
                bg=CARD_BG,
                fg=TEXT_SECONDARY,
                font=("Sans", 12),
                bd=0,
                highlightthickness=0,
            )
            if photo is None:
                preview_label.configure(
                    width=max(8, cell_width // 10),
                    height=max(4, (cell_width * 3) // (8 * 16)),
                )
            preview_label.pack(fill="x", pady=(0, 8))
            self.tutorial_preview_labels.append(preview_label)

            button = RoundedButton(
                col,
                text=label,
                command=lambda i=index: self.play_or_stop_tutorial(i),
                bg=PLAY_BLUE,
                activebackground=PLAY_BLUE_ACTIVE,
                font=("Sans", 16, "bold"),
                width=80,
                height=70,
                radius=26,
                on_enter=lambda h=hint, l=label: self._show_tutorial_hint(l, h),
                on_leave=self._clear_task_hint,
            )
            button.pack(fill="x")
            self.tutorial_buttons.append(button)
        self._refresh_tutorial_mode()

    def _keyboard_tutorial(self) -> bool:
        return log_controller_tag(str(self.control.get() or "robot")) == "keyboard"

    def _tutorial_parts(self):
        return KEYBOARD_TUTORIAL_PARTS if self._keyboard_tutorial() else TUTORIAL_PARTS

    def _tutorial_part_locked(self, index: int) -> bool:
        if not self._keyboard_tutorial():
            return False
        return index == 2

    def _reload_task_preview_sources(self) -> None:
        sources: list[Image.Image | None] = []
        for orig in self._visible_task_indices:
            task = TASKS[orig][1]
            sources.append(self._load_preview_source(task))
        self.preview_sources = sources

    def _on_control_change(self, _event=None):
        if self.child is not None:
            return
        self._reload_task_preview_sources()
        self._refresh_tutorial_mode()
        self._apply_preview_width(self._preview_width)

    def _refresh_tutorial_mode(self):
        parts = self._tutorial_parts()
        self.tutorial_sources = [
            self._load_tutorial_source(index) for index in range(len(parts))
        ]
        for index, button in enumerate(self.tutorial_buttons):
            label, _script, hint = parts[index]
            locked = self._tutorial_part_locked(index)
            button.configure(
                text=label,
                state="disabled" if locked else "normal",
                bg=LOCKED_GRAY if locked else PLAY_BLUE,
                activebackground=LOCKED_GRAY if locked else PLAY_BLUE_ACTIVE,
                disabledbackground=LOCKED_GRAY,
                disabledforeground="#b4bac2",
            )
            button._on_enter_cb = lambda h=hint, l=label, lk=locked: self._show_tutorial_hint(
                l, "Not available in the household GUI." if lk else h
            )
        self._refresh_tutorial_previews()

    def _show_tutorial_hint(self, label: str, hint: str):
        if self.child is not None:
            return
        self.status.configure(text=f"Tutorial · {label}: {hint}", fg=HINT_FG)

    def _preview_path(self, task):
        directory = DEMO_DIR / task
        names = ("scene_snapshot.png", "default_sidebyside.gif")
        if self._keyboard_tutorial():
            names = ("scene_snapshot_kb.png",) + names
        for name in names:
            preferred = directory / name
            if preferred.exists():
                return preferred
        if directory.exists():
            for pattern in ("*.png", "*.jpg", "*.jpeg", "*.gif"):
                match = next(directory.glob(pattern), None)
                if match is not None:
                    return match
        return None

    def _load_preview_source(self, task) -> Image.Image | None:
        path = self._preview_path(task)
        if path is None:
            return None
        try:
            with Image.open(path) as source:
                source.seek(0)
                return source.convert("RGB")
        except Exception:
            return None

    def _render_preview(self, source: Image.Image | None, width: int) -> ImageTk.PhotoImage | None:
        if source is None:
            return None
        aspect = source.width / max(source.height, 1)
        height = max(1, int(round(width / aspect)))
        image = source.resize((width, height), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(image)

    def _add_category_header(self, title: str):
        header = tk.Label(
            self.page,
            text=title,
            bg=PAGE_BG,
            fg=TEXT_PRIMARY,
            anchor="w",
            font=("Sans", 16, "bold"),
        )
        header.pack(fill="x", padx=self.CARD_PAD, pady=(18, 2))
        self.category_header_labels.append(header)

    def _add_task_row(self, indices):
        """One card holding up to four household tasks, matching the base GUI grid."""
        card = tk.Frame(
            self.page,
            bg=CARD_BG,
            highlightbackground=CARD_BORDER,
            highlightthickness=2,
        )
        card.pack(fill="x", padx=self.CARD_PAD, pady=12)
        grid = tk.Frame(card, bg=CARD_BG)
        grid.pack(fill="x", padx=self.PREVIEW_SIDE_PAD, pady=(14, 16))
        for index in indices:
            label, task, _script_name = TASKS[index]
            self._add_task_cell(grid, index, label, task)

    def _add_task_cell(self, grid, index, label, task):
        col = tk.Frame(grid, bg=CARD_BG)
        col.pack(side="left", expand=True, fill="both", padx=6)

        cell_header = tk.Frame(col, bg=CARD_BG)
        cell_header.pack(fill="x", pady=(0, 8))
        index_label = tk.Label(
            cell_header,
            text=f"{index + 1:02d}",
            bg=PLAY_BLUE,
            fg="white",
            font=("Sans", 17, "bold"),
            padx=13,
            pady=6,
        )
        index_label.pack(side="left", padx=(0, 8))
        title = tk.Label(
            cell_header,
            text=label,
            bg=CARD_BG,
            fg=TEXT_PRIMARY,
            anchor="w",
            justify="left",
            font=("Sans", 15, "bold"),
        )
        title.pack(side="left", fill="x", expand=True)
        for widget in (index_label, title):
            widget.bind("<Enter>", lambda _e, t=task, l=label: self._show_task_hint(t, l))
            widget.bind("<Leave>", lambda _e: self._clear_task_hint())
        self.card_index_labels.append(index_label)
        self.card_title_labels.append(title)

        cell_width = self._cell_preview_width(self._preview_width)
        source = self._load_preview_source(task)
        photo = self._render_preview(source, cell_width)
        self.preview_sources.append(source)
        self.preview_photos.append(photo)
        preview_label = tk.Label(
            col,
            image=photo,
            text="No preview" if photo is None else "",
            bg=CARD_BG,
            fg=TEXT_SECONDARY,
            font=("Sans", 12),
            bd=0,
            highlightthickness=0,
            cursor="hand2",
        )
        if photo is None:
            preview_label.configure(
                width=max(8, cell_width // 10),
                height=max(4, (cell_width * 3) // (8 * 16)),
            )
        preview_label.pack(fill="x", pady=(0, 8))
        preview_label.bind(
            "<Enter>",
            lambda _e, t=task, l=label: self._show_task_hint(t, l),
        )
        preview_label.bind("<Leave>", lambda _e: self._clear_task_hint())
        self.preview_labels.append(preview_label)

        button = RoundedButton(
            col,
            text="Play",
            command=lambda i=index: self.play_or_stop(i),
            bg=PLAY_BLUE,
            activebackground=PLAY_BLUE_ACTIVE,
            font=("Sans", 16, "bold"),
            width=80,
            height=70,
            radius=26,
            on_enter=lambda t=task, l=label: self._show_task_hint(t, l),
            on_leave=self._clear_task_hint,
        )
        button.pack(fill="x")
        self.task_buttons.append(button)

    def _set_status(self, text: str, fg: str, *, sticky: bool = False):
        """Update the status bar; ``sticky`` also becomes the post-hover restore text."""
        self.status.configure(text=text, fg=fg)
        if sticky:
            self._idle_status = text
            self._idle_status_fg = fg

    def _show_task_hint(self, task: str, label: str):
        """Show the README task blurb while the pointer is over preview / Play."""
        if self.child is not None:
            return
        desc = task_description(task)
        text = f"{label}: {desc}" if desc else label
        self.status.configure(text=text, fg=HINT_FG)

    def _clear_task_hint(self):
        if self.child is not None:
            return
        self.status.configure(text=self._idle_status, fg=self._idle_status_fg)

    def play_or_stop(self, index):
        if experiment_mode() and self._slot_locked("household", TASKS[index][1]):
            self._set_status(
                "Play limit reached. Choose another remaining task.",
                "#e6a15c",
                sticky=True,
            )
            return
        if self.child is not None:
            if self._run_uninterruptible():
                self._refuse_recording_interrupt()
                return
            if self.active_index == index and self.active_tutorial is None:
                self._stop_task("Task stopped. Select another task when ready.")
            else:
                self._set_status(
                    "Stop the running session before starting another.",
                    "#e6a15c",
                    sticky=True,
                )
            return
        self._start_task(index)

    def play_or_stop_tutorial(self, index: int):
        if self._tutorial_part_locked(index) and self.child is None:
            self._set_status(
                "Base keyboard tutorial is only in the base GUI.",
                "#e6a15c",
                sticky=True,
            )
            return
        if self.child is not None:
            if self._run_uninterruptible():
                self._refuse_recording_interrupt()
                return
            if self.active_tutorial == index:
                self._stop_task("Tutorial stopped. Select a part or task when ready.")
            else:
                self._set_status(
                    "Stop the running session before starting another.",
                    "#e6a15c",
                    sticky=True,
                )
            return
        self._start_tutorial(index)

    def _run_uninterruptible(self) -> bool:
        """HDF5 write after the viewer closes cannot be stopped from the launcher."""
        return bool(
            self.child is not None
            and getattr(self, "_recording_run", False)
            and getattr(self, "_recording_phase", False)
        )

    def _refuse_recording_interrupt(self) -> None:
        self._set_status(
            "Recording data. This step cannot be interrupted.",
            "#e6a15c",
            sticky=True,
        )

    def _active_run_button(self) -> "RoundedButton | None":
        try:
            if self.active_tutorial is not None:
                return self.tutorial_buttons[int(self.active_tutorial)]
            if self.active_index is None:
                return None
            display = self._visible_task_indices.index(self.active_index)
            return self.task_buttons[display]
        except (IndexError, ValueError, TypeError):
            return None

    def _enter_recording_phase(self) -> None:
        if getattr(self, "_recording_phase", False):
            return
        if not getattr(self, "_recording_run", False):
            return
        self._recording_phase = True
        self._mark_running_buttons()
        button = self._active_run_button()
        if button is not None:
            self._style_active_run_button(button)
        note = "Recording data. This step cannot be interrupted."
        self._run_status_base = note
        self._set_status(note, "#70d6a2", sticky=True)

    def _style_active_run_button(self, button: "RoundedButton") -> None:
        if self._run_uninterruptible():
            button.configure(
                state="disabled",
                text="Recording",
                bg=RECORDING_GREEN,
                activebackground=RECORDING_GREEN,
                disabledbackground=RECORDING_GREEN,
                disabledforeground="white",
            )
            self.exit_button.configure(state="disabled")
            return
        button.configure(
            state="normal",
            text="Stop",
            bg="#b06a20",
            activebackground="#d0842b",
        )
        self.exit_button.configure(state="normal")

    def _mark_running_buttons(self):
        """Disable every Play control except the active Stop button.

        After the viewer closes, an experiment HDF5 write has no Stop control.
        """
        lock_stop = self._run_uninterruptible()
        for i, button in enumerate(self.task_buttons):
            orig = self._visible_task_indices[i]
            is_active = self.active_index == orig and self.active_tutorial is None
            button.configure(
                state="normal" if is_active and not lock_stop else "disabled"
            )
        for part_index, button in enumerate(self.tutorial_buttons):
            is_active = self.active_tutorial == part_index
            button.configure(
                state="normal" if is_active and not lock_stop else "disabled"
            )

    def _start_tutorial(self, index: int):
        label, script_stem, hint = self._tutorial_parts()[index]
        script = TUTORIAL_DIR / f"{script_stem}.py"
        if not script.exists():
            messagebox.showerror("Tutorial unavailable", f"Missing launcher:\n{script}")
            return
        try:
            seed = resolve_seed(self.seed_entry.get())
        except ValueError as exc:
            messagebox.showerror("Invalid seed", str(exc))
            self.seed_entry.focus_set()
            return

        control_mode = str(self.control.get() or "robot")
        kb = self._keyboard_tutorial()
        if bool(self.show_briefing.get()):
            if kb:
                summaries = (
                    "1-1/1-2: three colored keys with lamps (numbers, then mouse). "
                    "2-1/2-2: move the bowl over both red circles (arrows, then mouse). "
                    "3-1/3-2: on/off key. 4-1/4-2: hold the push button.",
                    "Click the green pad to move the cup fully onto it, then click the apple "
                    "and the box to pack it (same pick-and-place as pack fruits).",
                    "Open the base GUI for this part.",
                    "Click the moving ball to stop it, click the stove knob to turn "
                    "the lamp on then off, then click the green zone and tilt the board "
                    "with the arrow keys.",
                )
                summary = summaries[index]
                task_name = "tutorial_keyboard"
                script_for_brief = TUTORIAL_DIR / "_kb_run.py"
            elif index == 0:
                summary = (
                    "Empty table with both arms. Test arm selection (1 / 2 / 3), "
                    "then press V to switch camera views."
                )
                task_name = "tutorial_empty"
                script_for_brief = TUTORIAL_DIR / "_run.py"
            elif index == 1:
                summary = (
                    "Empty table with both arms. Practice the base teleop keys "
                    "on the selected (green) arm."
                )
                task_name = "tutorial_empty"
                script_for_brief = TUTORIAL_DIR / "_run.py"
            elif index == 2:
                summary = (
                    "Four basic actions on the left side of the table, one at a time: "
                    "pick up a cube, hold a spring button, toggle an on/off switch, "
                    "then push a box to a green line."
                )
                task_name = "tutorial_empty"
                script_for_brief = TUTORIAL_DIR / "_run.py"
            else:
                summary = (
                    "Two advanced household actions on the left side of the table, "
                    "one at a time: twist a stove knob, then press a multi-stage "
                    "force key."
                )
                task_name = "tutorial_empty"
                script_for_brief = TUTORIAL_DIR / "_run.py"
            instruction = "Follow the instructions to complete the tutorial."
            briefing = build_briefing_text(
                label=f"Tutorial · {label}",
                task=task_name,
                scenario_label=label,
                scenario_desc=hint,
                summary=summary,
                control_mode=control_mode,
                script_path=script_for_brief,
            )
            briefing["instruction"] = instruction
            if not show_task_briefing(self, briefing):
                self._set_status(
                    "Instructions cancelled. Select a tutorial part when ready.",
                    TEXT_SECONDARY,
                    sticky=True,
                )
                return

        try:
            child_env = os.environ.copy()
            child_env.setdefault(
                "PYTHONWARNINGS",
                "ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning",
            )
            self._prepare_result_file()
            child_env[TASK_RESULT_ENV] = str(self.result_file)
            stamp_child_env(child_env, controller=control_mode, seed=seed)
            self._run_meta = {
                "suite": "tutorial",
                "task": script_stem,
                "task_label": f"Tutorial · {label}",
                "scenario": None,
                "controller": control_mode,
                "seed": seed,
                "started_at": time.perf_counter(),
            }
            command = [
                sys.executable,
                str(script),
                "--config",
                "demo_dynamic",
                "--seed",
                str(seed),
                "--control",
                control_mode,
                "--part",
                str(index + 1),
                "--suite",
                "household",
            ]
            # Practice only: never record HDF5 / preview video for tutorials.
            child_env["ROBODYNA_SAVE_VIDEO"] = "0"
            child_env.pop("ROBODYNA_RECORD_DATA", None)
            child_env.pop("ROBODYNA_RECORD_CONFIG", None)
            child_env["ROBODYNA_EXPORT_LEROBOT"] = "0"
            self._recording_run = False
            self.child = subprocess.Popen(
                command, cwd=ROOT, start_new_session=True, env=child_env
            )
        except Exception as exc:
            self._remove_result_file()
            messagebox.showerror("Could not start tutorial", str(exc))
            self.child = None
            self._recording_run = False
            return

        self.active_index = None
        self.active_tutorial = index
        self._mark_running_buttons()
        self.control.configure(state="disabled")
        self.seed_entry.configure(state="disabled")
        self._set_option_checks("disabled")
        self._style_active_run_button(self.tutorial_buttons[index])
        run_text = (
            f"Running Tutorial / {label} with seed {seed}. "
            "Close its viewer or press Stop."
        )
        run_text = f"{run_text}{self._capture_run_note()}"
        self._run_status_base = run_text
        self._shown_episode_condition = None
        self._set_status(run_text, "#70d6a2", sticky=True)

    def _start_task(self, index):
        label, task, script_name = TASKS[index]
        script = SCRIPT_DIR / script_name
        if not script.exists():
            messagebox.showerror("Task unavailable", f"Missing launcher:\n{script}")
            return
        try:
            seed = self._resolve_launch_seed("household", task)
        except ValueError as exc:
            messagebox.showerror("Invalid seed", str(exc))
            self.seed_entry.focus_set()
            return
        control_mode = str(self.control.get() or "robot")
        if bool(self.show_briefing.get()):
            briefing = build_briefing_text(
                label=label,
                task=task,
                scenario_label=None,
                scenario_desc=None,
                summary=task_description(task),
                control_mode=control_mode,
                script_path=script,
                seed=seed,
            )
            if not show_task_briefing(self, briefing):
                self._set_status(
                    "Instructions cancelled. Select a task when ready.",
                    TEXT_SECONDARY,
                    sticky=True,
                )
                return
        command = [
            sys.executable,
            str(script),
            "--seed",
            str(seed),
            "--control",
            control_mode,
        ]
        try:
            self._prepare_result_file()
            child_env = os.environ.copy()
            child_env[TASK_RESULT_ENV] = str(self.result_file)
            stamp_child_env(child_env, controller=control_mode, seed=seed)
            self._run_meta = {
                "suite": "household",
                "task": task,
                "task_label": label,
                "scenario": None,
                "controller": control_mode,
                "seed": seed,
                "started_at": time.perf_counter(),
            }
            self._apply_record_launch(
                child_env, command, suite="household", task=task
            )
            self.child = subprocess.Popen(
                command, cwd=ROOT, start_new_session=True, env=child_env
            )
        except Exception as exc:
            self._remove_result_file()
            messagebox.showerror("Could not start task", str(exc))
            self.child = None
            self._recording_run = False
            return
        self.active_index = index
        self.active_tutorial = None
        self._mark_running_buttons()
        self._apply_completed_locks(keep_active=True)
        self.control.configure(state="disabled")
        self.seed_entry.configure(state="disabled")
        self._set_option_checks("disabled")
        display = self._visible_task_indices.index(index)
        self._style_active_run_button(self.task_buttons[display])
        run_text = (
            f"Running {label} with seed {seed}. Close its viewer or press Stop to return."
        )
        run_text = f"{run_text}{self._capture_run_note()}"
        self._run_status_base = run_text
        self._shown_episode_condition = None
        self._set_status(run_text, "#70d6a2", sticky=True)

    def _poll_child(self):
        if self.child is not None:
            code = self.child.poll()
            if code is not None:
                run_meta = self._run_meta
                self.child = None
                self.active_index = None
                self.active_tutorial = None
                self._recording_run = False
                self._recording_phase = False
                payload = self._read_result_payload()
                reason = None
                recorded = record_status_note(payload)
                ps_note = ps_status_note(payload)
                if isinstance(payload, dict):
                    reason = payload_fail_reason(payload) or None
                self._run_status_base = None
                self._shown_episode_condition = None
                self._record_experiment_play(run_meta, payload, exit_code=code)
                self._run_meta = None
                self._remove_result_file()
                self._reset_task_buttons()
                # Tutorials are practice — no SUCCESS/FAILURE, including viewer
                # close crashes (often exit -11 / SIGSEGV).
                if isinstance(run_meta, dict) and run_meta.get("suite") == "tutorial":
                    self._set_status(
                        "Tutorial completed. Select another part or task below.",
                        TEXT_SECONDARY,
                        sticky=True,
                    )
                elif code == 0:
                    self._set_status(
                        f"Task result: SUCCESS.{ps_note}{recorded} Select another task below.",
                        "#70d6a2",
                        sticky=True,
                    )
                elif code == 10:
                    msg = "Task result: FAILURE"
                    if reason:
                        msg = f"{msg} ({reason})"
                    self._set_status(
                        f"{msg}.{ps_note}{recorded} Select another task below.",
                        "#e6a15c",
                        sticky=True,
                    )
                elif code == 2:
                    self._set_status(
                        f"Task closed before a result was reached.{ps_note}{recorded}",
                        TEXT_SECONDARY,
                        sticky=True,
                    )
                else:
                    self._set_status(
                        f"Task result: ERROR (exit status {code}). Check the terminal.{recorded}",
                        "#e6a15c",
                        sticky=True,
                    )
                if code in (0, 10) and bool(self.show_metrics.get()):
                    show_trial_metrics(self, payload, run_meta)
            else:
                payload = self._read_result_payload()
                if isinstance(payload, dict) and payload.get("recording"):
                    self._enter_recording_phase()
                if not getattr(self, "_recording_phase", False):
                    cond = self._read_result_condition()
                    if cond and cond != getattr(self, "_shown_episode_condition", None):
                        self._shown_episode_condition = cond
                        base = getattr(self, "_run_status_base", None) or "Running task."
                        self._set_status(f"{base}  Condition: {cond}", "#70d6a2", sticky=True)
        self.after(250, self._poll_child)

    def _reset_task_buttons(self):
        self._recording_run = False
        self._recording_phase = False
        self.exit_button.configure(state="normal")
        if experiment_mode():
            self._apply_experiment_protocol()
        else:
            self.control.configure(state="readonly")
            self.seed_entry.configure(state="normal")
            self._set_option_checks("normal")
        for button in self.task_buttons:
            button.configure(
                state="normal",
                text="Play",
                bg=PLAY_BLUE,
                activebackground=PLAY_BLUE_ACTIVE,
                disabledbackground=LOCKED_GRAY,
                disabledforeground="#b4bac2",
            )
        for index, button in enumerate(self.tutorial_buttons):
            button.configure(
                state="normal",
                text=self._tutorial_parts()[index][0],
                bg=PLAY_BLUE,
                activebackground=PLAY_BLUE_ACTIVE,
                disabledbackground=LOCKED_GRAY,
                disabledforeground="#b4bac2",
            )
        self._apply_completed_locks()
        self._refresh_tutorial_mode()

    def _resolve_launch_seed(self, suite: str, task: str, scenario: str | None = None) -> int:
        """Protocol seed list/int (global or per-slot), else the seed field / random."""
        if experiment_mode():
            cfg = self._experiment_cfg or load_experiment_config()
            self._experiment_cfg = cfg
            if suite == "tutorial":
                index = int(getattr(self, "_tutorial_seed_index", 0) or 0)
                self._tutorial_seed_index = index + 1
            else:
                index = terminal_play_count(suite, task, scenario)
            picked = cfg.pick_seed(
                play_index=index,
                suite=suite,
                task=task,
                scenario=scenario,
            )
            if picked is not None:
                return picked
            # Protocol null / empty for this slot → random (ignore locked field text).
            return secrets.randbelow(RANDOM_SEED_MAX + 1)
        return resolve_seed(self.seed_entry.get())

    def _apply_experiment_protocol(self):
        """Lock record / video / controller / seed from the experiment protocol."""
        cfg = self._experiment_cfg or load_experiment_config()
        self._experiment_cfg = cfg
        self.record_data.set(bool(cfg.record_data))
        self.save_video.set(bool(cfg.save_video))
        self.control.set(session_control_mode())
        self.control.configure(state="disabled")
        self.seed_entry.configure(state="normal")
        self.seed_entry.delete(0, "end")
        display = cfg.seed_display()
        if display:
            self.seed_entry.insert(0, display)
        self.seed_entry.configure(state="disabled")
        self.briefing_check.configure(state="normal")
        self.record_check.configure(state="disabled")
        self.video_check.configure(state="disabled")
        self.metrics_check.configure(state="normal")

    def _slot_locked(self, suite: str, task: str, scenario: str | None = None) -> bool:
        cfg = self._experiment_cfg or load_experiment_config()
        self._experiment_cfg = cfg
        return is_slot_locked(
            suite,
            task,
            scenario,
            max_plays=cfg.max_plays(suite, task, scenario),
        )

    def _slot_button_text(self, suite: str, task: str, default_text: str, scenario: str | None = None) -> str:
        if not experiment_mode():
            return default_text
        cfg = self._experiment_cfg or load_experiment_config()
        self._experiment_cfg = cfg
        limit = cfg.max_plays(suite, task, scenario)
        if limit is None or int(limit) <= 1:
            return default_text
        used = terminal_play_count(suite, task, scenario)
        if used <= 0:
            return default_text
        return f"{default_text} ({used}/{limit})"

    def _apply_completed_locks(self, *, keep_active: bool = False):
        """Gray out tasks that have used up their experiment play budget."""
        if not experiment_mode():
            return
        cfg = self._experiment_cfg or load_experiment_config()
        self._experiment_cfg = cfg
        for i, button in enumerate(self.task_buttons):
            orig = self._visible_task_indices[i]
            if keep_active and self.active_index == orig:
                continue
            task = TASKS[orig][1]
            if self._slot_locked("household", task):
                limit = cfg.max_plays("household", task)
                skipped = limit is not None and int(limit) <= 0
                button.configure(
                    state="disabled",
                    text="Play" if skipped else slot_success_label("household", task),
                    bg=LOCKED_GRAY,
                    activebackground=LOCKED_GRAY,
                    disabledbackground=LOCKED_GRAY,
                    disabledforeground="#b4bac2",
                )
                continue
            if keep_active:
                continue
            button.configure(
                state="normal",
                text=self._slot_button_text("household", task, "Play"),
                bg=PLAY_BLUE,
                activebackground=PLAY_BLUE_ACTIVE,
                disabledbackground=LOCKED_GRAY,
                disabledforeground="#b4bac2",
            )

    def _record_experiment_play(self, run_meta, payload, *, exit_code=None, stopped=False):
        if not experiment_mode() or not isinstance(run_meta, dict):
            return
        # Tutorials are unlimited practice — never write them to exp_logs.
        if run_meta.get("suite") != "household":
            return
        started = run_meta.get("started_at")
        wall_fallback = None
        if isinstance(started, (int, float)):
            wall_fallback = time.perf_counter() - float(started)
        cfg = self._experiment_cfg or load_experiment_config()
        self._experiment_cfg = cfg
        task = str(run_meta.get("task") or "")
        append_play(
            suite="household",
            task=task,
            task_label=run_meta.get("task_label"),
            scenario=None,
            controller=run_meta.get("controller"),
            seed=run_meta.get("seed"),
            exit_code=exit_code,
            payload=payload if isinstance(payload, dict) else None,
            stopped=stopped,
            wall_fallback_s=wall_fallback,
            write_entry=cfg.should_log("household", task),
        )

    def _prepare_result_file(self):
        self._remove_result_file()
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="robodyna_household_result_",
            delete=False,
        )
        handle.close()
        self.result_file = Path(handle.name)

    def _read_result_payload(self) -> dict | None:
        path = self.result_file
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return None
        return data if isinstance(data, dict) else None

    def _read_result_detail(self) -> str | None:
        data = self._read_result_payload()
        if not data:
            return None
        detail = data.get("detail")
        if isinstance(detail, str):
            detail = detail.strip()
            return detail or None
        return None

    def _read_result_condition(self) -> str | None:
        data = self._read_result_payload()
        if not data:
            return None
        cond = data.get("condition")
        if isinstance(cond, str):
            cond = cond.strip()
            return cond or None
        return None

    def _remove_result_file(self):
        path = self.result_file
        self.result_file = None
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def _stop_task(self, status=None):
        if self._run_uninterruptible():
            self._refuse_recording_interrupt()
            return
        child = self.child
        run_meta = self._run_meta
        payload = self._read_result_payload()
        if child is None:
            self._remove_result_file()
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
            self.active_index = None
            self.active_tutorial = None
            self._recording_run = False
            self._recording_phase = False
            self._record_experiment_play(run_meta, payload, stopped=True)
            self._run_meta = None
            self._remove_result_file()
            self._reset_task_buttons()
        if status:
            self._set_status(status, "#e6a15c", sticky=True)

    def exit_app(self):
        if self._run_uninterruptible():
            messagebox.showinfo(
                "Recording in progress",
                "Data is being recorded and cannot be interrupted. "
                "Wait until the write finishes.",
            )
            return
        self._stop_task()
        self.destroy()


def main():
    HouseholdTaskLauncher().mainloop()


if __name__ == "__main__":
    main()

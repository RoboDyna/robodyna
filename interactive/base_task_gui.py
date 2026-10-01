#!/usr/bin/env python3
"""Scrollable graphical launcher for the dynamic interactive tasks."""
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

import yaml
from PIL import Image, ImageTk

_INTERACTIVE_DIR = Path(__file__).resolve().parent
if str(_INTERACTIVE_DIR) not in sys.path:
    sys.path.insert(0, str(_INTERACTIVE_DIR))
_REPO_ROOT = _INTERACTIVE_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
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
from task_config.scenario_overrides import (  # noqa: E402
    BASE_SCENARIOS,
    SCENARIO_OVERRIDES,
    apply_base_scenario,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent / "base"
TUTORIAL_DIR = Path(__file__).resolve().parent / "Tutorial"
CONFIG_DIR = ROOT / "task_config"
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

TASKS = (
    ("Catch Marbles Trapdoors", "catch_marbles_trapdoors"),
    ("Catch Ramp Ball", "catch_ramp_ball"),
    ("Catch Cuboid", "catch_cuboid"),
    ("Catch Shelf Marble", "catch_shelf_marble"),
    ("Catch Valley Ball", "catch_valley_ball"),
    ("Stop Valley Ball", "stop_valley_ball"),
    ("Cook Meat", "cook_meat"),
    ("Cook Meat Timer", "cook_meat_timer"),
    ("Put Cup Belt", "put_cup_belt"),
    ("Dispense Gummy", "dispense_gummy"),
    ("Punch Dual Holes", "punch_dual_holes"),
    ("Save Goal", "save_goal"),
    ("Hit Target", "hit_target"),
    ("Load Train", "load_train"),
    ("Marble Shelf Maze", "marble_shelf_maze"),
    ("Pack Fruits", "pack_fruits"),
    ("Pick Ripe Apple", "pick_ripe_apple"),
    ("Place Block Belt", "place_block_belt"),
    ("Play Billiard", "play_billiard"),
    ("Control Quality", "control_quality"),
    ("Drop Ball Hole", "drop_ball_hole"),
    ("Sort Apples Belt", "sort_apples_belt"),
    ("Whack Moles", "whack_moles"),
)

TUTORIAL_PARTS = (
    ("Selection", "tutorial_part1", "Select arms (1, 2, 3) then switch camera (V)."),
    ("Basic control", "tutorial_part2", "Move with arrows, E/Q, R/T, F/G, then Space."),
    ("Basic actions", "tutorial_part3", "Grasp, hold-button, switch, then push a box."),
    ("Advance actions", "tutorial_part4", "Catch a rolling ball, then pick up a mallet."),
)

KEYBOARD_TUTORIAL_PARTS = (
    ("Buttons", "tutorial_kb_buttons", "Number keys, arrows+bowl, on/off, then push — keyboard then mouse."),
    ("Placement", "tutorial_kb_placement", "Click to place a cup on the green pad, then pack an apple in the box."),
    ("Base", "tutorial_kb_base", "Move the bowl, open a two-key gate, then aim the stick."),
    ("Household", "tutorial_kb_household", "Household keyboard tutorial — open the household GUI."),
)

SCENARIOS = BASE_SCENARIOS
SCENARIO_LABELS = {
    "default": "Default",
    "opt1": "Opt 1",
    "opt2": "Opt 2",
    "opt1+2": "Opt 1+2",
}

PLAY_BLUE = "#3182bd"
PLAY_BLUE_ACTIVE = "#4295d0"
LOCKED_GRAY = "#59616b"
RECORDING_GREEN = "#2a7a52"
GATED_RED = "#e34a33"
PAGE_BG = GUI_PAGE_BG
HEADER_BG = GUI_PAGE_BG
HEADER_FG = GUI_INK
HEADER_MUTED = GUI_MUTED
CARD_BG = "#202c38"
CARD_BORDER = "#405367"
TEXT_PRIMARY = "#f4f8fb"
TEXT_SECONDARY = "#aebdca"
RANDOM_SEED_MAX = 100000

# Tasks whose README row lives under a different name (or is shared).
README_TASK_ALIASES: dict[str, str] = {}


def load_condition_descriptions(readme_path: Path = README_PATH) -> dict[str, dict[str, str]]:
    """Parse README task-table ``<sub>…</sub>`` blurbs into per-condition text.

    Each task row is:
      task summary, Default, Opt 1, Opt 2, Opt 1+2
    """
    if not readme_path.exists():
        return {}
    text = readme_path.read_text(encoding="utf-8")
    row_re = re.compile(r"^\|\s*\*\*`([a-z0-9_]+)`\*\*.*$", re.MULTILINE)
    sub_re = re.compile(r"<sub>(.*?)</sub>", re.DOTALL)
    out: dict[str, dict[str, str]] = {}
    for match in row_re.finditer(text):
        task = match.group(1)
        line_end = text.find("\n", match.start())
        line = text[match.start() : line_end if line_end >= 0 else None]
        subs = [re.sub(r"\s+", " ", s).strip() for s in sub_re.findall(line)]
        if len(subs) < 5:
            continue
        out[task] = {
            "summary": subs[0],
            "default": subs[1],
            "opt1": subs[2],
            "opt2": subs[3],
            "opt1+2": subs[4],
        }
    return out


CONDITION_DESCRIPTIONS = load_condition_descriptions()


def condition_description(task: str, scenario: str) -> str:
    """README blurb for ``task`` / ``scenario``, or empty if unavailable."""
    key = README_TASK_ALIASES.get(task, task)
    return str(CONDITION_DESCRIPTIONS.get(key, {}).get(scenario, "") or "")


def task_summary(task: str) -> str:
    """README task-summary blurb, or empty if unavailable."""
    key = README_TASK_ALIASES.get(task, task)
    return str(CONDITION_DESCRIPTIONS.get(key, {}).get("summary", "") or "")


def resolve_seed(value: str) -> int:
    """Return a fixed entered seed, or a fresh random seed for a blank value."""
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


def build_scenario_config(task: str, scenario: str) -> dict:
    """Load the standard interactive config and apply one scenario's flags."""
    config_path = CONFIG_DIR / "demo_dynamic.yml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    apply_base_scenario(config, task, scenario)
    return config


class RoundedButton(tk.Canvas):
    """Canvas-backed button with rounded corners."""

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
        radius=24,
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
        self._text = text
        self._font = font
        self._shape = None
        self._on_enter_cb = on_enter
        self._on_leave_cb = on_leave
        self.bind("<Configure>", self._redraw)
        self.bind("<Enter>", self._enter)
        self.bind("<Leave>", self._leave)
        self.bind("<ButtonRelease-1>", self._click)

    def _rounded_polygon(self, width, height):
        radius = min(self.radius, width // 2, height // 2)
        return (
            radius, 1, width - radius, 1, width - 1, 1, width - 1, radius,
            width - 1, height - radius, width - 1, height - 1,
            width - radius, height - 1, radius, height - 1, 1, height - 1,
            1, height - radius, 1, radius, 1, 1,
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
        self.create_text(
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


class InteractiveTaskLauncher(tk.Tk):
    # Head-camera stills are ~4:3. Width seeds the first layout; height follows
    # each image's native aspect so previews are not letterboxed.
    IMAGE_SIZE = (1400, 1050)
    CARD_PAD = 8
    PREVIEW_SIDE_PAD = 28
    DESIGN_SIZE = (1600, 1000)
    UI_SCALE_MIN = 0.55
    UI_SCALE_MAX = 1.15

    def __init__(self):
        # className becomes StartupWMClass for the Ubuntu dock .desktop entry.
        super().__init__(className=GUI_WM_CLASS["interactive"])
        self.title("Base Tasks")
        self.geometry("1600x1000")
        self.minsize(900, 640)
        self.configure(bg=PAGE_BG)
        setup_gui_app_icon(
            self,
            suite="interactive",
            script_path=Path(__file__),
        )
        self.protocol("WM_DELETE_WINDOW", self.exit_app)

        self.child: subprocess.Popen | None = None
        self.active_selection: tuple | None = None
        self.temporary_config: Path | None = None
        self.result_file: Path | None = None
        self._run_meta: dict | None = None
        self._recording_run = False
        self._recording_phase = False
        # Per task: one source/photo/label per scenario (Default / Opt1 / Opt2 / Opt1+2).
        self.preview_sources: list[list[Image.Image | None]] = []
        self.preview_photos: list[list[ImageTk.PhotoImage | None]] = []
        self.preview_labels: list[list[tk.Label]] = []
        self.task_buttons: list[list[RoundedButton]] = []
        self.card_index_labels: list[tk.Label] = []
        self.card_title_labels: list[tk.Label] = []
        self.card_badge_labels: list[tk.Label] = []
        self.category_header_labels: list[tk.Label] = []
        self._preview_resize_job: str | None = None
        self._ui_scale_job: str | None = None
        self._preview_width = self.IMAGE_SIZE[0]
        self._ui_scale = 1.0
        self._header_layout_key: tuple | None = None
        self.tutorial_sources: list[Image.Image | None] = []
        self.tutorial_photos: list[ImageTk.PhotoImage | None] = []
        self.tutorial_preview_labels: list[tk.Label] = []
        self.tutorial_buttons: list[RoundedButton] = []
        self._visible_task_indices: list[int] = list(range(len(TASKS)))
        self._experiment_cfg = load_experiment_config() if experiment_mode() else None
        self._assignment = load_user_assignment() if self._experiment_cfg is not None else None
        if self._experiment_cfg is not None:
            assignment = self._assignment or {}
            grouped = self._experiment_cfg.grouped_visible_indices(
                "base",
                len(TASKS),
                assigned=assignment.get("base_tasks"),
                picks=assignment.get("base_picks"),
            )
            self._visible_task_indices = [
                index for _label, idxs in grouped for index in idxs
            ]
        self._idle_status = (
            f"Tutorial (4 parts)  ·  {len(self._visible_task_indices)} tasks  |  "
            "Hover a scenario key for its README description."
        )
        self._idle_status_fg = HEADER_MUTED

        self._build_ui()
        if experiment_mode():
            user = os.environ.get("ROBODYNA_EXPERIMENT_USER", "").strip()
            title = f"Base Tasks  ·  {user}" if user else "Base Tasks  ·  Experiment"
            self.title(title)
            self.title_label.configure(text=title)
            self.subtitle_label.configure(
                text=(
                    "Play Default, Opt 1, and Opt 2 first. Opt 1+2 unlocks after those "
                    "are finished. Finished scenarios stay gray and cannot be replayed."
                )
            )
            self.exit_button.configure(text="Back")
            n_vis = len(self._visible_task_indices)
            self._idle_status = (
                f"Experiment  ·  {user or 'participant'}  ·  {n_vis} tasks  |  "
                "Completed scenarios are gray. Tutorial is always available."
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
            text="Base Tasks",
            bg=HEADER_BG,
            fg=HEADER_FG,
            anchor="w",
            justify="left",
            font=("Sans", 34, "bold"),
        )
        self.title_label.pack(anchor="w")
        self.subtitle_label = tk.Label(
            self.heading,
            text="Choose a tutorial part or a task and one of its four scenario variants.",
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
            wraplength=1500,
            font=("Sans", 19),
        )
        self.status.pack(fill="x", padx=34, pady=(2, 12))

        outer = tk.Frame(self, bg=PAGE_BG)
        outer.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.canvas = tk.Canvas(outer, bg=PAGE_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.page = tk.Frame(self.canvas, bg=PAGE_BG)
        self.page_window = self.canvas.create_window((0, 0), window=self.page, anchor="nw")
        self.page.bind("<Configure>", self._update_scroll_region)
        self.canvas.bind("<Configure>", self._resize_page)
        self.canvas.bind_all("<MouseWheel>", self._cuboidwheel)
        self.canvas.bind_all("<Button-4>", lambda _event: self.canvas.yview_scroll(-3, "units"))
        self.canvas.bind_all("<Button-5>", lambda _event: self.canvas.yview_scroll(3, "units"))

        self._add_tutorial_section()
        if self._experiment_cfg is not None:
            assignment = self._assignment or {}
            grouped = self._experiment_cfg.grouped_visible_indices(
                "base",
                len(TASKS),
                assigned=assignment.get("base_tasks"),
                picks=assignment.get("base_picks"),
            )
            for section, idxs in grouped:
                if section:
                    self._add_category_header(section)
                for index in idxs:
                    label, task = TASKS[index]
                    self._add_task_card(index, label, task)
        else:
            for index in self._visible_task_indices:
                label, task = TASKS[index]
                self._add_task_card(index, label, task)

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

    def _apply_ui_scale(self):
        self._ui_scale_job = None
        if getattr(self, "title_label", None) is None:
            return
        scale = self._compute_ui_scale()
        if abs(scale - self._ui_scale) < 0.02:
            # Width-only change near the same scale: refresh wrap + header stack.
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

        idx_padx = self._px(13, s)
        idx_pady = self._px(6, s)
        for label in self.card_index_labels:
            label.configure(
                font=self._scaled_font(17, "bold", s),
                padx=idx_padx,
                pady=idx_pady,
            )
        for label in self.card_title_labels:
            label.configure(
                font=self._scaled_font(27, "bold", s),
                wraplength=max(160, self.winfo_width() - self._px(280, s)),
            )
        for label in self.card_badge_labels:
            label.configure(font=self._scaled_font(12, "bold", s))
        for label in self.category_header_labels:
            label.configure(font=self._scaled_font(16, "bold", s))

        btn_h = self._px(70, s)
        btn_radius = self._px(26, s)
        btn_font = self._scaled_font(16, "bold", s)
        for row in self.task_buttons:
            for button in row:
                button.configure(font=btn_font, height=btn_h, radius=btn_radius)
        for button in self.tutorial_buttons:
            button.configure(font=btn_font, height=btn_h, radius=btn_radius)

        self._relayout_header(s)

    @staticmethod
    def _px(value: float, scale: float) -> int:
        return max(1, int(round(value * scale)))

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
        """Width of one of the four scenario preview columns."""
        gap = 6 * 2 * len(SCENARIOS)  # padx=6 on each side of each column
        return max(160, (total_width - gap) // len(SCENARIOS))

    def _apply_preview_width(self, width: int):
        self._preview_resize_job = None
        self._preview_width = width
        cell_width = self._cell_preview_width(width)
        for task_index, sources in enumerate(self.preview_sources):
            photos: list[ImageTk.PhotoImage | None] = []
            for scen_index, source in enumerate(sources):
                photo = self._render_preview(source, cell_width)
                photos.append(photo)
                label = self.preview_labels[task_index][scen_index]
                if photo is None:
                    label.configure(
                        image="",
                        text="No preview",
                        width=max(8, cell_width // 10),
                        height=max(4, (cell_width * 3) // (8 * 16)),
                    )
                else:
                    label.configure(image=photo, text="", width=0, height=0)
            self.preview_photos[task_index] = photos
        self._refresh_tutorial_previews(cell_width)

    def _cuboidwheel(self, event):
        if event.delta:
            self.canvas.yview_scroll(-int(event.delta / 120), "units")

    def _preview_path(self, task: str, scenario: str = "default") -> Path | None:
        directory = DEMO_DIR / task
        kb = self._keyboard_tutorial()
        names = (
            *(
                (
                    f"scene_snapshot_kb_{scenario}.png",
                    "scene_snapshot_kb.png" if scenario == "default" else None,
                )
                if kb
                else ()
            ),
            f"scene_snapshot_{scenario}.png",
            "scene_snapshot.png" if scenario == "default" else None,
            "default_sidebyside.gif" if scenario == "default" else None,
            f"{scenario}_sidebyside.gif",
            f"*{scenario}*sidebyside.gif",
        )
        for name in names:
            if not name:
                continue
            if "*" in name:
                match = next(directory.glob(name), None) if directory.exists() else None
                if match is not None:
                    return match
                continue
            preferred = directory / name
            if preferred.exists():
                return preferred
        if scenario == "default" and directory.exists():
            patterns = ("default*sidebyside.gif", "*.gif", "*.png", "*.jpg", "*.jpeg")
            for pattern in patterns:
                match = next(directory.glob(pattern), None)
                if match is not None:
                    return match
        return None

    def _load_preview_source(self, task: str, scenario: str = "default") -> Image.Image | None:
        path = self._preview_path(task, scenario)
        if path is None and scenario != "default":
            # Fall back to the default still so the grid stays populated.
            path = self._preview_path(task, "default")
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
        # Fill the column width at the image's native aspect ratio (no letterbox).
        aspect = source.width / max(source.height, 1)
        height = max(1, int(round(width / aspect)))
        image = source.resize((width, height), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(image)

    def _tutorial_snapshot_path(self, part_index: int | None = None) -> Path | None:
        names: list[str] = []
        kb = self._keyboard_tutorial()
        if part_index is not None:
            if kb:
                stems = ("buttons", "placement", "base", "household")
                names.append(f"scene_snapshot_kb_{stems[part_index]}.png")
                names.append(f"scene_snapshot_kb_part{part_index + 1}.png")
            names.append(f"scene_snapshot_part{part_index + 1}.png")
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
        cell_width = int(width if width is not None else self._cell_preview_width(self._preview_width))
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

    def _add_tutorial_section(self):
        """Same four-column card as other base tasks; index 00."""
        card = tk.Frame(
            self.page,
            bg=CARD_BG,
            highlightbackground=CARD_BORDER,
            highlightthickness=2,
        )
        card.pack(fill="x", padx=self.CARD_PAD, pady=12)

        card_header = tk.Frame(card, bg=CARD_BG)
        card_header.pack(fill="x", padx=self.PREVIEW_SIDE_PAD, pady=(14, 8))
        index_label = tk.Label(
            card_header,
            text="00",
            bg=PLAY_BLUE,
            fg="white",
            font=("Sans", 17, "bold"),
            padx=13,
            pady=6,
        )
        index_label.pack(side="left", padx=(0, 14))
        title_label = tk.Label(
            card_header,
            text="Tutorial",
            bg=CARD_BG,
            fg=TEXT_PRIMARY,
            anchor="w",
            font=("Sans", 27, "bold"),
        )
        title_label.pack(side="left", fill="x", expand=True)
        badge_label = tk.Label(
            card_header,
            text="4 PARTS",
            bg=CARD_BG,
            fg="#7fb6dc",
            font=("Sans", 12, "bold"),
        )
        badge_label.pack(side="right")
        self.card_index_labels.append(index_label)
        self.card_title_labels.append(title_label)
        self.card_badge_labels.append(badge_label)

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
                on_leave=self._clear_condition_hint,
            )
            button.pack(fill="x")
            self.tutorial_buttons.append(button)

    def _keyboard_tutorial(self) -> bool:
        return log_controller_tag(str(self.control.get() or "robot")) == "keyboard"

    def _tutorial_parts(self):
        return KEYBOARD_TUTORIAL_PARTS if self._keyboard_tutorial() else TUTORIAL_PARTS

    def _tutorial_part_locked(self, index: int) -> bool:
        if not self._keyboard_tutorial():
            return False
        # Base GUI: Household is listed but not playable.
        return index == 3

    def _reload_task_preview_sources(self) -> None:
        sources: list[list[Image.Image | None]] = []
        for orig in self._visible_task_indices:
            task = TASKS[orig][1]
            sources.append(
                [self._load_preview_source(task, scenario) for scenario in SCENARIOS]
            )
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
            if index < len(self.tutorial_preview_labels):
                # Rebind hint via a fresh enter callback by storing on the button.
                button._on_enter_cb = lambda h=hint, l=label, lk=locked: self._show_tutorial_hint(
                    l, "Not available in the base GUI." if lk else h
                )
        self._refresh_tutorial_previews()

    def _show_tutorial_hint(self, label: str, hint: str):
        if self.child is not None:
            return
        self.status.configure(text=f"Tutorial · {label}: {hint}", fg="#7fb6dc")

    def _add_task_card(self, index: int, label: str, task: str):
        card = tk.Frame(
            self.page,
            bg=CARD_BG,
            highlightbackground=CARD_BORDER,
            highlightthickness=2,
        )
        card.pack(fill="x", padx=self.CARD_PAD, pady=12)

        card_header = tk.Frame(card, bg=CARD_BG)
        card_header.pack(fill="x", padx=self.PREVIEW_SIDE_PAD, pady=(14, 8))
        index_label = tk.Label(
            card_header,
            text=f"{index + 1:02d}",
            bg=PLAY_BLUE,
            fg="white",
            font=("Sans", 17, "bold"),
            padx=13,
            pady=6,
        )
        index_label.pack(side="left", padx=(0, 14))
        title_label = tk.Label(
            card_header,
            text=label,
            bg=CARD_BG,
            fg=TEXT_PRIMARY,
            anchor="w",
            font=("Sans", 27, "bold"),
        )
        title_label.pack(side="left", fill="x", expand=True)
        badge_text = "4 SCENARIOS"
        cfg = self._experiment_cfg
        if cfg is not None:
            cats = cfg.categories_for_number(index + 1)
            if len(cats) > 1:
                badge_text = "4 SCENARIOS · also " + " · ".join(cats[1:])
        badge_label = tk.Label(
            card_header,
            text=badge_text,
            bg=CARD_BG,
            fg="#7fb6dc",
            font=("Sans", 12, "bold"),
        )
        badge_label.pack(side="right")
        self.card_index_labels.append(index_label)
        self.card_title_labels.append(title_label)
        self.card_badge_labels.append(badge_label)

        # Four columns: screenshot on top, matching scenario key underneath.
        grid = tk.Frame(card, bg=CARD_BG)
        grid.pack(fill="x", padx=self.PREVIEW_SIDE_PAD, pady=(0, 16))
        cell_width = self._cell_preview_width(self._preview_width)
        sources: list[Image.Image | None] = []
        photos: list[ImageTk.PhotoImage | None] = []
        labels: list[tk.Label] = []
        buttons: list[RoundedButton] = []
        for scenario in SCENARIOS:
            col = tk.Frame(grid, bg=CARD_BG)
            col.pack(side="left", expand=True, fill="both", padx=6)

            source = self._load_preview_source(task, scenario)
            photo = self._render_preview(source, cell_width)
            sources.append(source)
            photos.append(photo)
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
            labels.append(preview_label)

            button = RoundedButton(
                col,
                text=SCENARIO_LABELS[scenario],
                command=lambda i=index, s=scenario: self.play_or_stop(i, s),
                bg=PLAY_BLUE,
                activebackground=PLAY_BLUE_ACTIVE,
                font=("Sans", 16, "bold"),
                width=80,
                height=70,
                radius=26,
                on_enter=lambda t=task, s=scenario, l=label: self._show_condition_hint(t, s, l),
                on_leave=self._clear_condition_hint,
            )
            button.pack(fill="x")
            buttons.append(button)

        self.preview_sources.append(sources)
        self.preview_photos.append(photos)
        self.preview_labels.append(labels)
        self.task_buttons.append(buttons)

    def _set_status(self, text: str, fg: str, *, sticky: bool = False):
        """Update the status bar; ``sticky`` also becomes the post-hover restore text."""
        self.status.configure(text=text, fg=fg)
        if sticky:
            self._idle_status = text
            self._idle_status_fg = fg

    def _show_condition_hint(self, task: str, scenario: str, label: str):
        """Show the README condition blurb while the pointer is over a scenario key."""
        if self.child is not None:
            return
        desc = condition_description(task, scenario)
        scenario_label = SCENARIO_LABELS[scenario]
        if desc:
            text = f"{label} · {scenario_label}: {desc}"
        else:
            text = f"{label} · {scenario_label}"
        self.status.configure(text=text, fg="#7fb6dc")

    def _clear_condition_hint(self):
        if self.child is not None:
            return
        self.status.configure(text=self._idle_status, fg=self._idle_status_fg)

    def play_or_stop(self, index: int, scenario: str):
        task = TASKS[index][1]
        if experiment_mode() and self._combo_gated(task, scenario):
            self._set_status(
                "Finish Default, Opt 1, and Opt 2 before Opt 1+2.",
                "#e6a15c",
                sticky=True,
            )
            return
        if experiment_mode() and self._slot_locked("base", task, scenario):
            self._set_status(
                "Play limit reached. Choose another remaining scenario.",
                "#e6a15c",
                sticky=True,
            )
            return
        if self.child is not None:
            if self._run_uninterruptible():
                self._refuse_recording_interrupt()
                return
            if self.active_selection == (index, scenario):
                self._stop_task("Task stopped. Select another scenario when ready.")
            else:
                self._set_status(
                    "Stop the running task before starting another.",
                    "#e6a15c",
                    sticky=True,
                )
            return
        self._start_task(index, scenario)

    def play_or_stop_tutorial(self, index: int):
        if self._tutorial_part_locked(index) and self.child is None:
            self._set_status(
                "Household keyboard tutorial is only in the household GUI.",
                "#e6a15c",
                sticky=True,
            )
            return
        if self.child is not None:
            if self._run_uninterruptible():
                self._refuse_recording_interrupt()
                return
            if self.active_selection == ("tutorial", index):
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
        sel = self.active_selection
        if not isinstance(sel, tuple) or not sel:
            return None
        try:
            if sel[0] == "tutorial":
                return self.tutorial_buttons[int(sel[1])]
            orig, scenario = sel
            display = self._visible_task_indices.index(orig)
            return self.task_buttons[display][SCENARIOS.index(scenario)]
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
        for display_i, row in enumerate(self.task_buttons):
            orig = self._visible_task_indices[display_i]
            for button_scenario, button in zip(SCENARIOS, row):
                is_active = self.active_selection == (orig, button_scenario)
                button.configure(
                    state="normal" if is_active and not lock_stop else "disabled"
                )
        for part_index, button in enumerate(self.tutorial_buttons):
            is_active = self.active_selection == ("tutorial", part_index)
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
                    "Move the bowl onto a red circle with arrows plus Space, then with the mouse. "
                    "Next, hold both arrow keys to open the gate. Then click the green circle "
                    "to place the stick tip and rotate with the arrows.",
                    "Open the household GUI for this part.",
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
                    "Two advanced actions on the left side of the table, one at a time: "
                    "catch a rolling ball, then pick up a mallet."
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
                "base",
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

        self.active_selection = ("tutorial", index)
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

    def _write_temporary_config(self, task: str, scenario: str) -> str:
        config = build_scenario_config(task, scenario)
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=".interactive_gui_",
            suffix=".yml",
            dir=CONFIG_DIR,
            delete=False,
        )
        try:
            yaml.safe_dump(config, handle, sort_keys=False)
        finally:
            handle.close()
        self.temporary_config = Path(handle.name)
        return self.temporary_config.stem

    def _start_task(self, index: int, scenario: str):
        label, task = TASKS[index]
        script = SCRIPT_DIR / f"interactive_{task}.py"
        if not script.exists():
            messagebox.showerror("Task unavailable", f"Missing launcher:\n{script}")
            return
        try:
            seed = self._resolve_launch_seed("base", task, scenario)
        except ValueError as exc:
            messagebox.showerror("Invalid seed", str(exc))
            self.seed_entry.focus_set()
            return

        control_mode = str(self.control.get() or "robot")
        if bool(self.show_briefing.get()):
            briefing = build_briefing_text(
                label=label,
                task=task,
                scenario_label=SCENARIO_LABELS.get(scenario, scenario),
                scenario_desc=condition_description(task, scenario),
                summary=task_summary(task),
                control_mode=control_mode,
                script_path=script,
                scenario=scenario,
                seed=seed,
            )
            if not show_task_briefing(self, briefing):
                self._set_status(
                    "Instructions cancelled. Select a scenario when ready.",
                    TEXT_SECONDARY,
                    sticky=True,
                )
                return

        try:
            config_name = self._write_temporary_config(task, scenario)
            child_env = os.environ.copy()
            child_env.setdefault(
                "PYTHONWARNINGS",
                "ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning",
            )
            # Authoritative scenario for tasks that read ROBODYNA_SCENARIO
            # (pack_fruits applies Opt1/Opt2 flags from this even if a stale
            # temp yml is missing the new keys).
            child_env["ROBODYNA_SCENARIO"] = scenario
            self._prepare_result_file()
            child_env[TASK_RESULT_ENV] = str(self.result_file)
            stamp_child_env(
                child_env, controller=control_mode, seed=seed, scenario=scenario
            )
            self._run_meta = {
                "suite": "base",
                "task": task,
                "task_label": label,
                "scenario": scenario,
                "controller": control_mode,
                "seed": seed,
                "started_at": time.perf_counter(),
            }
            command = [
                sys.executable,
                str(script),
                "--config",
                config_name,
                "--seed",
                str(seed),
                "--control",
                control_mode,
            ]
            # Prefer an explicit CLI scenario when the interactive script
            # understands it (pack_fruits); unknown flags are avoided below.
            if task == "pack_fruits":
                command.extend(["--scenario", scenario])
            self._apply_record_launch(
                child_env, command, suite="base", task=task, scenario=scenario
            )
            self.child = subprocess.Popen(
                command, cwd=ROOT, start_new_session=True, env=child_env
            )
        except Exception as exc:
            self._remove_temporary_config()
            self._remove_result_file()
            messagebox.showerror("Could not start task", str(exc))
            self.child = None
            self._recording_run = False
            return

        self.active_selection = (index, scenario)
        self._mark_running_buttons()
        self.control.configure(state="disabled")
        self.seed_entry.configure(state="disabled")
        self._set_option_checks("disabled")
        display = self._visible_task_indices.index(index)
        active_button = self.task_buttons[display][SCENARIOS.index(scenario)]
        self._style_active_run_button(active_button)
        desc = condition_description(task, scenario)
        run_text = (
            f"Running {label} / {SCENARIO_LABELS[scenario]} with seed {seed}. "
            "Close its viewer or press Stop."
        )
        if desc:
            run_text = f"{run_text}  ({desc})"
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
                self.active_selection = None
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
                self._remove_temporary_config()
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
                # Match household_task_gui: 0=SUCCESS, 10=FAILURE, 2=closed early.
                elif code == 0:
                    self._set_status(
                        f"Task result: SUCCESS.{ps_note}{recorded} Select another scenario below.",
                        "#70d6a2",
                        sticky=True,
                    )
                elif code == 10:
                    msg = "Task result: FAILURE"
                    if reason:
                        msg = f"{msg} ({reason})"
                    self._set_status(
                        f"{msg}.{ps_note}{recorded} Select another scenario below.",
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
                # While running, surface the episode-specific condition once available.
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
        for row in self.task_buttons:
            for scenario, button in zip(SCENARIOS, row):
                button.configure(
                    state="normal",
                    text=SCENARIO_LABELS[scenario],
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

    def _slot_finished(self, suite: str, task: str, scenario: str | None = None) -> bool:
        """Play budget used up, or at least one terminal play if the slot is unlimited."""
        cfg = self._experiment_cfg or load_experiment_config()
        self._experiment_cfg = cfg
        limit = cfg.max_plays(suite, task, scenario)
        if is_slot_locked(suite, task, scenario, max_plays=limit):
            return True
        if limit is None:
            return terminal_play_count(suite, task, scenario) >= 1
        return False

    def _combo_gated(self, task: str, scenario: str | None) -> bool:
        """Opt 1+2 stays closed until Default, Opt 1, and Opt 2 are finished."""
        if not experiment_mode() or scenario != "opt1+2":
            return False
        return not all(
            self._slot_finished("base", task, prereq)
            for prereq in ("default", "opt1", "opt2")
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
        """Gray out scenarios that have used up their experiment play budget."""
        if not experiment_mode():
            return
        cfg = self._experiment_cfg or load_experiment_config()
        self._experiment_cfg = cfg
        for display_i, row in enumerate(self.task_buttons):
            orig = self._visible_task_indices[display_i]
            task = TASKS[orig][1]
            for scenario, button in zip(SCENARIOS, row):
                if keep_active and self.active_selection == (orig, scenario):
                    continue
                if self._slot_locked("base", task, scenario):
                    limit = cfg.max_plays("base", task, scenario)
                    skipped = limit is not None and int(limit) <= 0
                    button.configure(
                        state="disabled",
                        text=(
                            SCENARIO_LABELS[scenario]
                            if skipped
                            else slot_success_label("base", task, scenario)
                        ),
                        bg=LOCKED_GRAY,
                        activebackground=LOCKED_GRAY,
                        disabledbackground=LOCKED_GRAY,
                        disabledforeground="#b4bac2",
                    )
                    continue
                if self._combo_gated(task, scenario):
                    button.configure(
                        state="disabled",
                        text=SCENARIO_LABELS[scenario],
                        bg=GATED_RED,
                        activebackground=GATED_RED,
                        disabledbackground=GATED_RED,
                        disabledforeground="white",
                    )
                    continue
                if keep_active:
                    continue
                button.configure(
                    state="normal",
                    text=self._slot_button_text(
                        "base", task, SCENARIO_LABELS[scenario], scenario
                    ),
                    bg=PLAY_BLUE,
                    activebackground=PLAY_BLUE_ACTIVE,
                    disabledbackground=LOCKED_GRAY,
                    disabledforeground="#b4bac2",
                )

    def _record_experiment_play(self, run_meta, payload, *, exit_code=None, stopped=False):
        if not experiment_mode() or not isinstance(run_meta, dict):
            return
        # Tutorials are unlimited practice — never write them to exp_logs.
        if run_meta.get("suite") != "base":
            return
        started = run_meta.get("started_at")
        wall_fallback = None
        if isinstance(started, (int, float)):
            wall_fallback = time.perf_counter() - float(started)
        cfg = self._experiment_cfg or load_experiment_config()
        self._experiment_cfg = cfg
        task = str(run_meta.get("task") or "")
        scenario = run_meta.get("scenario")
        append_play(
            suite="base",
            task=task,
            task_label=run_meta.get("task_label"),
            scenario=scenario,
            controller=run_meta.get("controller"),
            seed=run_meta.get("seed"),
            exit_code=exit_code,
            payload=payload if isinstance(payload, dict) else None,
            stopped=stopped,
            wall_fallback_s=wall_fallback,
            write_entry=cfg.should_log("base", task, scenario),
        )

    def _remove_temporary_config(self):
        path = self.temporary_config
        self.temporary_config = None
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def _prepare_result_file(self):
        self._remove_result_file()
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="robodyna_interactive_result_",
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
            self._remove_temporary_config()
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
            self.active_selection = None
            self._recording_run = False
            self._recording_phase = False
            self._record_experiment_play(run_meta, payload, stopped=True)
            self._run_meta = None
            self._remove_temporary_config()
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
    InteractiveTaskLauncher().mainloop()


if __name__ == "__main__":
    main()

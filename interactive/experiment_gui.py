#!/usr/bin/env python3
"""Human-experiment launcher: login, pre/post surveys, then base / household GUIs."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

_INTERACTIVE_DIR = Path(__file__).resolve().parent
if str(_INTERACTIVE_DIR) not in sys.path:
    sys.path.insert(0, str(_INTERACTIVE_DIR))

from _task_briefing import (  # noqa: E402
    GUI_INK,
    GUI_MUTED,
    GUI_PAGE_BG,
    apply_gui_logo,
    setup_gui_app_icon,
)
from base_task_gui import (  # noqa: E402
    LOCKED_GRAY,
    PLAY_BLUE,
    PLAY_BLUE_ACTIVE,
    RoundedButton,
    SCENARIOS,
    TASKS as BASE_TASKS,
)
from experiment_config import CONFIG_ENV, load_experiment_config  # noqa: E402
from experiment_logs import (  # noqa: E402
    ASSIGNED_TASKS,
    EXPERIMENT_ENV,
    EXPERIMENT_LOG_ENV,
    EXPERIMENT_USER_ENV,
    LOG_CONTROLLER_TAGS,
    POST_SURVEY_QUESTIONS,
    PRE_SURVEY_QUESTIONS,
    child_experiment_env,
    create_user,
    ensure_controller_log,
    find_user,
    iter_user_controller_logs,
    load_user_assignment,
    load_user_log,
    log_controller_tag,
    progress_counts,
    question_choices,
    save_post_survey,
    slugify_user_name,
    survey_complete,
    survey_missing_prompts,
)
from household_task_gui import TASKS as HOUSEHOLD_TASKS  # noqa: E402
from robodyna_gui import (  # noqa: E402
    ABOUT_ACCENT,
    BASE_ACCENT,
    BLURB_INK,
    BRAND_BASE,
    BRAND_EXPERIMENT,
    BRAND_HOUSEHOLD,
    EXPERIMENT_ACCENT,
    HEADING_INK,
    HOUSEHOLD_ACCENT,
    _hex,
    _shade,
)

ROOT = Path(__file__).resolve().parents[1]
PAGE_BG = GUI_PAGE_BG
HEADER_BG = GUI_PAGE_BG
CARD_BG = "#202c38"
CARD_BORDER = "#405367"
TEXT_PRIMARY = "#f4f8fb"
TEXT_SECONDARY = "#aebdca"
SURVEY_BG = "#ffffff"
SURVEY_FG = "#1A1A1A"
SURVEY_BORDER = "#d4d5db"
RANK_TILE_BORDER = "#c5cad3"
RANK_TILE_SELECTED_BG = "#eaf3fb"
# Post-experiment questionnaire card shares the hub GUI's About-box blue.
QUESTIONNAIRE_CARD_BG = _hex(ABOUT_ACCENT)
QUESTIONNAIRE_CARD_BORDER = _hex(_shade(ABOUT_ACCENT, 0.82))
QUESTIONNAIRE_BUTTON_BG = _hex(_shade(ABOUT_ACCENT, 0.62))
QUESTIONNAIRE_BUTTON_ACTIVE = _hex(_shade(ABOUT_ACCENT, 0.78))
# Name card shares the hub GUI's Experiments-box red.
NAME_CARD_BG = _hex(EXPERIMENT_ACCENT)
NAME_CARD_BORDER = _hex(_shade(EXPERIMENT_ACCENT, 0.82))
NAME_BUTTON_BG = _hex(BRAND_EXPERIMENT)
NAME_BUTTON_ACTIVE = _hex(_shade(BRAND_EXPERIMENT, 1.12))
NAME_STATUS_FG = _hex(_shade(BRAND_EXPERIMENT, 0.55))
NOTICE_FG = "#c0261a"
NOTICE_SECONDS = 5
QUESTIONNAIRE_LOCKED_NOTICE = (
    "Please complete the tasks with both controllers before starting the questionnaire."
)
SUITE_TITLE_FG = _hex(HEADING_INK)
SUITE_BLURB_FG = _hex(BLURB_INK)
BASE_CARD_BG = _hex(BASE_ACCENT)
BASE_CARD_BORDER = _hex(_shade(BASE_ACCENT, 0.82))
BASE_BUTTON_BG = _hex(BRAND_BASE)
BASE_BUTTON_ACTIVE = _hex(_shade(BRAND_BASE, 1.12))
HOUSEHOLD_CARD_BG = _hex(HOUSEHOLD_ACCENT)
HOUSEHOLD_CARD_BORDER = _hex(_shade(HOUSEHOLD_ACCENT, 0.82))
HOUSEHOLD_BUTTON_BG = _hex(BRAND_HOUSEHOLD)
HOUSEHOLD_BUTTON_ACTIVE = _hex(_shade(BRAND_HOUSEHOLD, 1.12))

CONTROLLER_DISPLAY = ("Robot", "Keyboard + mouse")
CONTROLLER_MODE = {
    "Robot": "robot",
    "Keyboard + mouse": "keyboard+mouse",
}
MODE_DISPLAY = {mode: label for label, mode in CONTROLLER_MODE.items()}
SCENARIO_REMAINING_LABELS = (
    ("default", "Default"),
    ("opt1", "Opt 1"),
    ("opt2", "Opt 2"),
    ("opt1+2", "Opt 1+2"),
)


class ExperimentLauncher(tk.Tk):
    DESIGN_SIZE = (920, 780)
    UI_SCALE_MIN = 0.7
    UI_SCALE_MAX = 1.15
    CHOICE_GAP = 22

    def __init__(self):
        super().__init__(className="Robodynaexperiment")
        self.title("RoboDyna Human Experiment")
        self.geometry("920x780")
        self.minsize(720, 620)
        self.configure(bg=PAGE_BG)
        setup_gui_app_icon(
            self,
            suite="experiment",
            script_path=Path(__file__),
        )
        self.protocol("WM_DELETE_WINDOW", self.exit_app)

        self.child: subprocess.Popen | None = None
        self.user_data: dict | None = None
        self._ui_scale = 1.0
        self._ui_scale_job: str | None = None
        self._screen = "name"
        self._survey_canvas = None
        self._subtitle_default = ""
        self._notice_job: str | None = None

        self.logo_bar = tk.Frame(self, bg=PAGE_BG)
        self.logo_bar.pack(fill="x", padx=24, pady=(16, 0))
        self.logo_label = tk.Label(self.logo_bar, bg=PAGE_BG, bd=0, highlightthickness=0)
        self.logo_label.pack(anchor="center")
        apply_gui_logo(self.logo_label, height=140)

        self.body = tk.Frame(self, bg=PAGE_BG)
        self.body.pack(fill="both", expand=True, padx=28, pady=(8, 20))

        self._build_name_screen()
        self._build_experience_screen()
        self._build_post_screen()
        self._build_suite_screen()
        self._show_screen("name")

        self.bind("<Configure>", self._on_root_configure)
        self.bind_all("<Button-4>", self._on_survey_wheel)
        self.bind_all("<Button-5>", self._on_survey_wheel)
        self.bind_all("<MouseWheel>", self._on_survey_wheel)
        self.after(0, self._apply_ui_scale)
        self.after(250, self._poll_child)

    def _choice_dot(
        self,
        parent,
        *,
        selected_fn,
        select_fn,
        text: str,
        watch_var=None,
        shape: str = "circle",
        bg: str = SURVEY_BG,
        fg: str = SURVEY_FG,
    ) -> tk.Frame:
        """Clickable choice with a large custom radio (circle) or checkbox (square)."""
        wrap = tk.Frame(parent, bg=bg, cursor="hand2")
        size = 28
        canvas = tk.Canvas(
            wrap,
            width=size,
            height=size,
            bg=bg,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        canvas.pack(side="left")
        label = tk.Label(
            wrap,
            text=text,
            bg=bg,
            fg=fg,
            activebackground=bg,
            activeforeground=fg,
            font=("Sans", 16, "bold"),
            cursor="hand2",
        )
        label.pack(side="left", padx=(10, 0))

        def paint(*_args):
            canvas.delete("all")
            outer = int(wrap._dot_size)
            canvas.configure(width=outer, height=outer)
            pad = max(2, outer // 12)
            ring = max(2, outer // 12)
            box = (pad, pad, outer - pad, outer - pad)
            if shape == "square":
                canvas.create_rectangle(*box, outline=fg, width=ring)
            else:
                canvas.create_oval(*box, outline=fg, width=ring)
            if selected_fn():
                inset = max(pad + ring + 1, outer // 4)
                inner = (inset, inset, outer - inset, outer - inset)
                if shape == "square":
                    canvas.create_rectangle(*inner, fill=fg, outline=fg)
                else:
                    canvas.create_oval(*inner, fill=fg, outline=fg)

        def select(_event=None):
            select_fn()
            paint()

        wrap._dot_size = size
        wrap._dot_canvas = canvas
        wrap._dot_label = label
        wrap._dot_paint = paint
        for widget in (wrap, canvas, label):
            widget.bind("<Button-1>", select)
        if watch_var is not None:
            watch_var.trace_add("write", paint)
        paint()
        return wrap

    def _card(self, parent, *, bg: str = CARD_BG, border: str = CARD_BORDER) -> tk.Frame:
        card = tk.Frame(
            parent,
            bg=bg,
            highlightbackground=border,
            highlightthickness=2,
        )
        return card

    def _header_with_exit(
        self, parent, *, bg: str = HEADER_BG, title_fg: str = GUI_INK
    ) -> tuple[tk.Frame, tk.Frame, RoundedButton]:
        """Page header with an Exit button on the right (same chrome on every screen)."""
        header = tk.Frame(parent, bg=bg, highlightbackground="#d4d5db", highlightthickness=1)
        header.pack(fill="x", pady=(0, 16))
        top = tk.Frame(header, bg=bg)
        top.pack(fill="x")
        exit_button = RoundedButton(
            top,
            text="Exit",
            command=self.exit_app,
            bg="#e34a33",
            activebackground="#eb6854",
            font=("Sans", 16, "bold"),
            width=140,
            height=64,
            radius=26,
        )
        exit_button.pack(side="right", padx=(8, 18), pady=14)
        inner = tk.Frame(top, bg=bg)
        inner.pack(side="left", fill="x", expand=True, padx=24, pady=18)
        inner._header_bg = bg
        inner._header_fg = title_fg
        return header, inner, exit_button

    def _build_name_screen(self):
        self.name_screen = tk.Frame(self.body, bg=PAGE_BG)
        _header, inner, self.name_exit = self._header_with_exit(self.name_screen)
        self.name_title = tk.Label(
            inner,
            text="Human Experiment",
            bg=HEADER_BG,
            fg=GUI_INK,
            anchor="w",
            font=("Sans", 30, "bold"),
        )
        self.name_title.pack(anchor="w")
        self.name_subtitle = tk.Label(
            inner,
            text="Enter your name to start or continue. Returning participants skip the pre-experiment survey.",
            bg=HEADER_BG,
            fg=GUI_MUTED,
            anchor="w",
            wraplength=780,
            justify="left",
            font=("Sans", 14),
        )
        self.name_subtitle.pack(anchor="w", pady=(4, 0))

        card = self._card(self.name_screen, bg=NAME_CARD_BG, border=NAME_CARD_BORDER)
        card.pack(fill="x")
        pad = tk.Frame(card, bg=NAME_CARD_BG)
        pad.pack(fill="x", padx=28, pady=28)
        self.name_caption = tk.Label(
            pad,
            text="Please enter your name",
            bg=NAME_CARD_BG,
            fg=SUITE_TITLE_FG,
            anchor="w",
            font=("Sans", 13, "bold"),
        )
        self.name_caption.pack(anchor="w")
        self.name_entry = tk.Entry(
            pad,
            font=("Sans", 18, "bold"),
            bg="#f7fafc",
            fg="#182633",
            insertbackground="#182633",
            relief="flat",
        )
        self.name_entry.pack(fill="x", ipady=10, pady=(8, 18))
        self.name_entry.bind("<Return>", lambda _e: self._submit_name())
        self.name_continue = RoundedButton(
            pad,
            text="Continue",
            command=self._submit_name,
            bg=NAME_BUTTON_BG,
            activebackground=NAME_BUTTON_ACTIVE,
            font=("Sans", 16, "bold"),
            width=220,
            height=64,
            radius=26,
        )
        self.name_continue.pack(anchor="w")
        self.name_status = tk.Label(
            pad,
            text="",
            bg=NAME_CARD_BG,
            fg=NAME_STATUS_FG,
            anchor="w",
            justify="left",
            wraplength=760,
            font=("Sans", 13),
        )
        self.name_status.pack(anchor="w", pady=(16, 0))

    def _scrollable_card(self, parent, *, bg: str = CARD_BG, border: str = CARD_BORDER):
        card = self._card(parent, bg=bg, border=border)
        body = tk.Frame(card, bg=bg)
        body.pack(fill="both", expand=True)
        canvas = tk.Canvas(body, bg=bg, highlightthickness=0, bd=0)
        scroll = ttk.Scrollbar(body, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=bg)
        window = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)

        def _inner_cfg(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _canvas_cfg(event):
            canvas.itemconfigure(window, width=max(1, event.width))

        inner.bind("<Configure>", _inner_cfg)
        canvas.bind("<Configure>", _canvas_cfg)
        canvas.pack(side="left", fill="both", expand=True, padx=(28, 0), pady=(24, 0))
        scroll.pack(side="right", fill="y", pady=(24, 0), padx=(0, 10))
        footer = tk.Frame(card, bg=bg)
        footer.pack(fill="x", padx=28, pady=(12, 24))
        return card, canvas, inner, footer

    def _scrollable_region(self, parent, *, bg: str = PAGE_BG):
        wrap = tk.Frame(parent, bg=bg)
        canvas = tk.Canvas(wrap, bg=bg, highlightthickness=0, bd=0)
        scroll = ttk.Scrollbar(wrap, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=bg)
        window = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)

        def _inner_cfg(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _canvas_cfg(event):
            canvas.itemconfigure(window, width=max(1, event.width))
            canvas.configure(scrollregion=canvas.bbox("all"))

        inner.bind("<Configure>", _inner_cfg)
        canvas.bind("<Configure>", _canvas_cfg)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        return wrap, canvas, inner

    def _on_survey_wheel(self, event):
        if self._screen not in ("experience", "post", "suite"):
            return
        canvas = self._survey_canvas
        if canvas is None:
            return
        if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            canvas.yview_scroll(-1, "units")
        else:
            canvas.yview_scroll(1, "units")

    def _make_question_var(self, question: dict):
        if question.get("rank"):
            return []
        if question.get("multi"):
            return {
                value: tk.BooleanVar(value=False)
                for value, _label in question_choices(question)
            }
        return tk.StringVar(value="")

    def _fill_rank_host(self, form: dict, block: dict, choices):
        host = block["host"]
        for child in host.winfo_children():
            child.destroy()
        block["dots"] = []
        block["tiles"] = []
        block["rank_captions"] = []
        question = block["question"]
        labels = {value: text for value, text in choices}
        allowed = [value for value, _text in choices]
        current = [value for value in list(form["vars"].get(question["key"]) or []) if value in labels]
        for value in allowed:
            if value not in current:
                current.append(value)
        form["vars"][question["key"]] = current
        selected = {"value": None}
        if not choices:
            return

        scale = self._ui_scale

        def fnt(size, weight=""):
            px = max(8, int(round(size * scale)))
            return ("Sans", px, weight) if weight else ("Sans", px)

        wraplength = max(360, int(self.winfo_width() or 920) - 100)
        help_label = tk.Label(
            host,
            text="Click a task, then use the arrows to move it. Most difficult at the top, easiest at the bottom.",
            bg=SURVEY_BG,
            fg=GUI_MUTED,
            anchor="w",
            justify="left",
            wraplength=wraplength,
            font=fnt(13),
        )
        help_label.pack(anchor="w", pady=(0, 8))
        top_label = tk.Label(
            host,
            text="Most difficult",
            bg=SURVEY_BG,
            fg=SURVEY_FG,
            anchor="w",
            font=fnt(12, "bold"),
        )
        top_label.pack(anchor="w", pady=(0, 4))
        list_host = tk.Frame(host, bg=SURVEY_BG)
        list_host.pack(anchor="w", fill="x")
        bot_label = tk.Label(
            host,
            text="Easiest",
            bg=SURVEY_BG,
            fg=SURVEY_FG,
            anchor="w",
            font=fnt(12, "bold"),
        )
        bot_label.pack(anchor="w", pady=(8, 0))
        block["rank_captions"] = [help_label, top_label, bot_label]

        def redraw():
            for child in list_host.winfo_children():
                child.destroy()
            block["tiles"] = []
            order = form["vars"][question["key"]]
            for index, value in enumerate(order):
                active = selected["value"] == value
                tile_bg = RANK_TILE_SELECTED_BG if active else SURVEY_BG
                tile = tk.Frame(
                    list_host,
                    bg=tile_bg,
                    highlightbackground=PLAY_BLUE if active else RANK_TILE_BORDER,
                    highlightthickness=2,
                    cursor="hand2",
                )
                tile.pack(anchor="w", fill="x", pady=(0, 6))
                inner = tk.Frame(tile, bg=tile_bg)
                inner.pack(fill="x", padx=12, pady=8)
                num = tk.Label(
                    inner,
                    text=str(index + 1),
                    bg=tile_bg,
                    fg=PLAY_BLUE if active else GUI_MUTED,
                    font=fnt(15, "bold"),
                    width=2,
                    anchor="w",
                    cursor="hand2",
                )
                num.pack(side="left")
                name = tk.Label(
                    inner,
                    text=labels.get(value, value),
                    bg=tile_bg,
                    fg=SURVEY_FG,
                    font=fnt(15, "bold"),
                    anchor="w",
                    cursor="hand2",
                )
                name.pack(side="left", fill="x", expand=True, padx=(8, 8))
                tile._rank_num = num
                tile._rank_name = name
                tile._rank_up = None
                tile._rank_down = None

                def select(_event=None, current=value):
                    selected["value"] = current
                    redraw()

                if active:
                    arrows = tk.Frame(inner, bg=tile_bg)
                    arrows.pack(side="right")
                    up_fg = SURVEY_FG if index > 0 else "#b7bec6"
                    down_fg = SURVEY_FG if index + 1 < len(order) else "#b7bec6"
                    up = tk.Label(
                        arrows,
                        text="▲",
                        bg=tile_bg,
                        fg=up_fg,
                        font=fnt(14, "bold"),
                        cursor="hand2" if index > 0 else "arrow",
                        padx=8,
                    )
                    down = tk.Label(
                        arrows,
                        text="▼",
                        bg=tile_bg,
                        fg=down_fg,
                        font=fnt(14, "bold"),
                        cursor="hand2" if index + 1 < len(order) else "arrow",
                        padx=8,
                    )
                    up.pack(side="left")
                    down.pack(side="left")
                    tile._rank_up = up
                    tile._rank_down = down

                    def move(_event=None, current=value, delta=0):
                        items = form["vars"][question["key"]]
                        here = items.index(current)
                        there = here + delta
                        if 0 <= there < len(items):
                            items[here], items[there] = items[there], items[here]
                            selected["value"] = current
                            redraw()
                        return "break"

                    if index > 0:
                        up.bind("<Button-1>", lambda e, current=value: move(e, current, -1))
                    if index + 1 < len(order):
                        down.bind("<Button-1>", lambda e, current=value: move(e, current, 1))
                for widget in (tile, inner, num, name):
                    widget.bind("<Button-1>", select)
                block["tiles"].append(tile)

        redraw()

    def _fill_choice_host(self, form: dict, block: dict, choices):
        host = block["host"]
        for child in host.winfo_children():
            child.destroy()
        block["dots"] = []
        block["rows"] = []
        block["flow_width"] = None
        question = block["question"]
        var = form["vars"][question["key"]]
        layout = question.get("layout") or "row"
        exclusive = question.get("exclusive")
        on_change = form.get("on_change")
        if not choices:
            return

        def add_dot(parent, value, text, watch):
            if question.get("multi"):
                def selected_fn(current=value):
                    return bool(var[current].get())

                def select_fn(current=value):
                    chosen = not bool(var[current].get())
                    if exclusive and current == exclusive:
                        for key, flag in var.items():
                            flag.set(bool(chosen) if key == exclusive else False)
                    else:
                        var[current].set(chosen)
                        if exclusive and chosen:
                            var[exclusive].set(False)
                    if callable(on_change):
                        on_change()

                return self._choice_dot(
                    parent,
                    selected_fn=selected_fn,
                    select_fn=select_fn,
                    text=text,
                    watch_var=watch,
                    shape="square",
                    bg=SURVEY_BG,
                    fg=SURVEY_FG,
                )
            def selected_fn(current=value):
                return var.get() == current

            def select_fn(current=value):
                var.set(current)
                if callable(on_change):
                    on_change()

            return self._choice_dot(
                parent,
                selected_fn=selected_fn,
                select_fn=select_fn,
                text=text,
                watch_var=watch,
                shape="circle",
                bg=SURVEY_BG,
                fg=SURVEY_FG,
            )

        if layout == "stack":
            for value, text in choices:
                row = tk.Frame(host, bg=SURVEY_BG)
                row.pack(anchor="w", pady=(0, 8))
                watch = var[value] if question.get("multi") else var
                dot = add_dot(row, value, text, watch)
                dot.pack(side="left")
                block["dots"].append(dot)
            return
        # Rows must exist before the choices so the reflowed choices stay on top.
        block["rows"] = [tk.Frame(host, bg=SURVEY_BG) for _ in choices]
        for value, text in choices:
            watch = var[value] if question.get("multi") else var
            block["dots"].append(add_dot(host, value, text, watch))
        if not block.get("flow_bound"):
            host.bind("<Configure>", lambda _event, b=block: self._schedule_choice_reflow(b))
            block["flow_bound"] = True
        self._reflow_choice_host(block)

    def _schedule_choice_reflow(self, block: dict):
        """Reflow choices only when the available width actually changed."""
        width = block["host"].winfo_width()
        if width == block.get("flow_width"):
            return
        job = block.get("flow_job")
        if job is not None:
            try:
                self.after_cancel(job)
            except (ValueError, tk.TclError):
                pass
        block["flow_job"] = self.after(40, lambda b=block: self._reflow_choice_host(b))

    def _reflow_choice_host(self, block: dict):
        """Lay choices out on one line, wrapping only when they do not fit."""
        block["flow_job"] = None
        dots = block.get("dots") or []
        question = block["question"]
        if not dots or (question.get("layout") or "row") != "row":
            return
        host = block["host"]
        try:
            host.update_idletasks()
        except tk.TclError:
            return
        block["flow_width"] = host.winfo_width()
        avail = block["flow_width"]
        if avail <= 1:
            avail = max(1, self.winfo_width() - 120)
        gap = max(8, int(round(self.CHOICE_GAP * self._ui_scale)))
        lines: list[list[tk.Frame]] = [[]]
        used = 0
        for dot in dots:
            need = max(dot.winfo_reqwidth(), 1)
            if lines[-1] and used + gap + need > avail:
                lines.append([dot])
                used = need
            else:
                used += need + (gap if lines[-1] else 0)
                lines[-1].append(dot)
        rows = block.get("rows") or []
        for index, row in enumerate(rows):
            row.pack_forget()
            if index < len(lines):
                row.pack(anchor="w", fill="x", pady=(0, 8))
        for line, row in zip(lines, rows):
            for position, dot in enumerate(line):
                dot.pack_forget()
                dot.pack(
                    in_=row,
                    side="left",
                    padx=(0, gap if position + 1 < len(line) else 0),
                )
                dot.lift()

    def _reflow_all_choices(self):
        for form in (getattr(self, "pre_form", None), getattr(self, "post_form", None)):
            if not form:
                continue
            for block in form["blocks"]:
                self._reflow_choice_host(block)

    def _populate_survey(self, inner, questions, *, on_change=None) -> dict:
        form = {
            "questions": questions,
            "vars": {},
            "blocks": [],
            "on_change": on_change,
        }
        for question in questions:
            form["vars"][question["key"]] = self._make_question_var(question)
            wrap = tk.Frame(inner, bg=SURVEY_BG)
            wrap.pack(anchor="w", fill="x", pady=(0, 18))
            label = tk.Label(
                wrap,
                text=question["prompt"],
                bg=SURVEY_BG,
                fg=SURVEY_FG,
                anchor="w",
                justify="left",
                wraplength=760,
                font=("Sans", 15, "bold"),
            )
            label.pack(anchor="w", pady=(0, 8))
            host = tk.Frame(wrap, bg=SURVEY_BG)
            host.pack(anchor="w", fill="x")
            block = {
                "question": question,
                "wrap": wrap,
                "label": label,
                "host": host,
                "dots": [],
                "tiles": [],
                "rank_captions": [],
            }
            form["blocks"].append(block)
            if question.get("rank"):
                self._fill_rank_host(form, block, question_choices(question))
            else:
                self._fill_choice_host(form, block, question_choices(question))
        return form

    def _survey_answers(self, form: dict) -> dict:
        answers = {}
        for question in form["questions"]:
            var = form["vars"][question["key"]]
            if question.get("rank"):
                answers[question["key"]] = list(var)
            elif question.get("multi"):
                answers[question["key"]] = [
                    value for value, flag in var.items() if flag.get()
                ]
            else:
                answers[question["key"]] = str(var.get() or "").strip()
        return answers

    def _clear_survey(self, form: dict):
        for question in form["questions"]:
            var = form["vars"][question["key"]]
            if question.get("rank"):
                form["vars"][question["key"]] = []
            elif question.get("multi"):
                for flag in var.values():
                    flag.set(False)
            else:
                var.set("")
        self._refresh_survey_visibility(form)

    def _refresh_survey_visibility(self, form: dict, extra_choices=None):
        answers = self._survey_answers(form)
        for block in form["blocks"]:
            block["wrap"].pack_forget()
        for block in form["blocks"]:
            question = block["question"]
            visible = True
            pred = question.get("visible_if")
            if callable(pred) and not pred(answers):
                visible = False
            if question.get("choices") == ASSIGNED_TASKS:
                if not question_choices(question, extra_choices):
                    visible = False
            if visible:
                block["wrap"].pack(anchor="w", fill="x", pady=(0, 18))
                continue
            var = form["vars"][question["key"]]
            if question.get("rank"):
                form["vars"][question["key"]] = []
            elif question.get("multi"):
                for flag in var.values():
                    flag.set(False)
            elif var.get():
                var.set("")

    def _iter_survey_dots(self):
        for form in (getattr(self, "pre_form", None), getattr(self, "post_form", None)):
            if not form:
                continue
            for block in form["blocks"]:
                yield from block["dots"]

    def _iter_survey_labels(self):
        for form in (getattr(self, "pre_form", None), getattr(self, "post_form", None)):
            if not form:
                continue
            for block in form["blocks"]:
                yield block["label"]

    def _assigned_task_choices(self) -> tuple[tuple[str, str], ...]:
        base_names, household_names = self._protocol_task_names()
        labels = {key: label for label, key in BASE_TASKS}
        labels.update({key: label for label, key, _script in HOUSEHOLD_TASKS})
        out = []
        seen = set()
        for key in list(base_names) + list(household_names):
            if key in seen:
                continue
            seen.add(key)
            out.append((key, labels.get(key, key)))
        return tuple(out)

    def _post_extra_choices(self) -> dict:
        return {ASSIGNED_TASKS: self._assigned_task_choices()}

    def _refresh_post_task_choices(self):
        extra = self._post_extra_choices()
        scale = self._ui_scale
        size = max(8, int(round(28 * scale)))
        font_px = max(8, int(round(16 * scale)))
        for block in self.post_form["blocks"]:
            question = block["question"]
            if question.get("choices") != ASSIGNED_TASKS:
                continue
            choices = question_choices(question, extra)
            if question.get("rank"):
                self._fill_rank_host(self.post_form, block, choices)
                continue
            var = self.post_form["vars"][question["key"]]
            allowed = [value for value, _label in choices]
            if question.get("multi"):
                previous = {
                    value for value, flag in (var or {}).items() if flag.get()
                }
                self.post_form["vars"][question["key"]] = {
                    value: tk.BooleanVar(value=value in previous) for value in allowed
                }
            else:
                current = str(var.get() or "")
                self.post_form["vars"][question["key"]] = tk.StringVar(
                    value=current if current in allowed else ""
                )
            self._fill_choice_host(self.post_form, block, choices)
            for dot in block["dots"]:
                dot._dot_size = size
                dot._dot_label.configure(font=("Sans", font_px, "bold"))
                dot._dot_paint()
            self._reflow_choice_host(block)
        self._refresh_survey_visibility(self.post_form, extra_choices=extra)

    def _build_experience_screen(self):
        self.exp_screen = tk.Frame(self.body, bg=PAGE_BG)
        _header, inner, self.exp_exit = self._header_with_exit(self.exp_screen)
        self.exp_title = tk.Label(
            inner,
            text="A few questions",
            bg=HEADER_BG,
            fg=GUI_INK,
            anchor="w",
            font=("Sans", 30, "bold"),
        )
        self.exp_title.pack(anchor="w")
        self.exp_subtitle = tk.Label(
            inner,
            text="Asked once for new participants. Answers are stored in your experiment log.",
            bg=HEADER_BG,
            fg=GUI_MUTED,
            anchor="w",
            wraplength=780,
            justify="left",
            font=("Sans", 14),
        )
        self.exp_subtitle.pack(anchor="w", pady=(4, 0))

        card, canvas, body, footer = self._scrollable_card(
            self.exp_screen, bg=SURVEY_BG, border=SURVEY_BORDER
        )
        card.pack(fill="both", expand=True)
        self.exp_canvas = canvas
        self.pre_form = self._populate_survey(
            body,
            PRE_SURVEY_QUESTIONS,
            on_change=lambda: self._refresh_survey_visibility(self.pre_form),
        )
        self._refresh_survey_visibility(self.pre_form)
        self.exp_continue = RoundedButton(
            footer,
            text="Continue",
            command=self._submit_experience,
            bg=PLAY_BLUE,
            activebackground=PLAY_BLUE_ACTIVE,
            font=("Sans", 16, "bold"),
            width=220,
            height=64,
            radius=26,
        )
        self.exp_continue.pack(anchor="w")

    def _build_post_screen(self):
        self.post_screen = tk.Frame(self.body, bg=PAGE_BG)
        _header, inner, self.post_exit = self._header_with_exit(self.post_screen)
        self.post_title = tk.Label(
            inner,
            text="After the experiment",
            bg=HEADER_BG,
            fg=GUI_INK,
            anchor="w",
            font=("Sans", 30, "bold"),
        )
        self.post_title.pack(anchor="w")
        self.post_subtitle = tk.Label(
            inner,
            text="Please answer these questions now that you have finished the assigned tasks.",
            bg=HEADER_BG,
            fg=GUI_MUTED,
            anchor="w",
            wraplength=780,
            justify="left",
            font=("Sans", 14),
        )
        self.post_subtitle.pack(anchor="w", pady=(4, 0))

        card, canvas, body, footer = self._scrollable_card(
            self.post_screen, bg=SURVEY_BG, border=SURVEY_BORDER
        )
        card.pack(fill="both", expand=True)
        self.post_canvas = canvas
        self.post_form = self._populate_survey(body, POST_SURVEY_QUESTIONS)
        self.post_continue = RoundedButton(
            footer,
            text="Finish",
            command=self._submit_post,
            bg=PLAY_BLUE,
            activebackground=PLAY_BLUE_ACTIVE,
            font=("Sans", 16, "bold"),
            width=220,
            height=64,
            radius=26,
        )
        self.post_continue.pack(side="left")
        self.post_back = RoundedButton(
            footer,
            text="Back",
            command=lambda: self._show_screen("suite"),
            bg="#6b7c8d",
            activebackground="#7d8e9e",
            font=("Sans", 16, "bold"),
            width=140,
            height=64,
            radius=26,
        )
        self.post_back.pack(side="left", padx=(12, 0))

    def _build_suite_screen(self):
        self.suite_screen = tk.Frame(self.body, bg=PAGE_BG)
        _header, inner, self.suite_exit = self._header_with_exit(self.suite_screen)
        self.suite_title = tk.Label(
            inner,
            text="Choose a suite",
            bg=HEADER_BG,
            fg=GUI_INK,
            anchor="w",
            font=("Sans", 30, "bold"),
        )
        self.suite_title.pack(anchor="w")
        self.suite_subtitle = tk.Label(
            inner,
            text="",
            bg=HEADER_BG,
            fg=GUI_MUTED,
            anchor="w",
            wraplength=620,
            justify="left",
            font=("Sans", 14),
        )
        self.suite_subtitle.pack(anchor="w", pady=(4, 0))
        control_row = tk.Frame(inner, bg=HEADER_BG)
        control_row.pack(anchor="w", fill="x", pady=(12, 0))
        self.controller_caption = tk.Label(
            control_row,
            text="Controller",
            bg=HEADER_BG,
            fg=GUI_MUTED,
            anchor="w",
            font=("Sans", 13, "bold"),
        )
        self.controller_caption.pack(side="left")
        self.controller_combo = ttk.Combobox(
            control_row,
            values=CONTROLLER_DISPLAY,
            state="readonly",
            width=22,
            font=("Sans", 13, "bold"),
        )
        self.controller_combo.set("Robot")
        self.controller_combo.pack(side="left", padx=(10, 0))
        self.controller_combo.bind("<<ComboboxSelected>>", self._on_controller_change)

        body, canvas, inner = self._scrollable_region(self.suite_screen)
        body.pack(fill="both", expand=True)
        self.suite_canvas = canvas

        self.remaining_stats = tk.Label(
            inner,
            text="",
            bg=PAGE_BG,
            fg=GUI_INK,
            anchor="w",
            justify="left",
            wraplength=820,
            font=("Sans", 13, "bold"),
        )
        self.remaining_stats.pack(fill="x", pady=(0, 12))

        grid = tk.Frame(inner, bg=PAGE_BG)
        grid.pack(fill="x")
        self.questionnaire_card = self._suite_card(
            grid,
            title="Post-experiment Questionnaire",
            blurb="Please fill in the questionnaire to share your experience about the experiment.",
            command=self._open_questionnaire,
            button_text="Start",
            bg=QUESTIONNAIRE_CARD_BG,
            border=QUESTIONNAIRE_CARD_BORDER,
            title_fg=SUITE_TITLE_FG,
            blurb_fg=SUITE_BLURB_FG,
            button_bg=LOCKED_GRAY,
            button_active=LOCKED_GRAY,
            show_progress=False,
        )
        self.questionnaire_card._button.configure(
            state="disabled",
            disabledbackground=LOCKED_GRAY,
        )
        self.questionnaire_card._button.bind(
            "<ButtonRelease-1>", self._on_questionnaire_click, add="+"
        )
        self.questionnaire_card.pack(fill="x", pady=(0, 12))
        self.base_card = self._suite_card(
            grid,
            title="Base Tasks",
            blurb="Dynamic table tasks with Default / Opt 1 / Opt 2 / Opt 1+2.",
            command=lambda: self._launch_suite("base"),
            bg=BASE_CARD_BG,
            border=BASE_CARD_BORDER,
            title_fg=SUITE_TITLE_FG,
            blurb_fg=SUITE_BLURB_FG,
            button_bg=BASE_BUTTON_BG,
            button_active=BASE_BUTTON_ACTIVE,
            progress_fg=BASE_BUTTON_BG,
        )
        self.base_card.pack(fill="x", pady=(0, 12))
        self.household_card = self._suite_card(
            grid,
            title="Household Tasks",
            blurb="Kitchen / office tasks with per-episode randomization.",
            command=lambda: self._launch_suite("household"),
            bg=HOUSEHOLD_CARD_BG,
            border=HOUSEHOLD_CARD_BORDER,
            title_fg=SUITE_TITLE_FG,
            blurb_fg=SUITE_BLURB_FG,
            button_bg=HOUSEHOLD_BUTTON_BG,
            button_active=HOUSEHOLD_BUTTON_ACTIVE,
            progress_fg=HOUSEHOLD_BUTTON_BG,
        )
        self.household_card.pack(fill="x")
        self.switch_user = tk.Button(
            inner,
            text="Switch user",
            command=self._switch_user,
            bg=PAGE_BG,
            fg=GUI_MUTED,
            activebackground=PAGE_BG,
            activeforeground=GUI_INK,
            bd=0,
            highlightthickness=0,
            font=("Sans", 12, "underline"),
            cursor="hand2",
        )
        self.switch_user.pack(anchor="w", pady=(14, 8))

    def _suite_card(
        self,
        parent,
        *,
        title: str,
        blurb: str,
        command,
        bg: str = CARD_BG,
        border: str = CARD_BORDER,
        title_fg: str = TEXT_PRIMARY,
        blurb_fg: str = TEXT_SECONDARY,
        button_text: str = "Open",
        button_bg: str = PLAY_BLUE,
        button_active: str = PLAY_BLUE_ACTIVE,
        show_progress: bool = True,
        progress_fg: str = "#7fb6dc",
    ) -> tk.Frame:
        card = self._card(parent, bg=bg, border=border)
        pad = tk.Frame(card, bg=bg)
        pad.pack(fill="x", padx=24, pady=20)
        title_label = tk.Label(
            pad,
            text=title,
            bg=bg,
            fg=title_fg,
            anchor="w",
            font=("Sans", 22, "bold"),
        )
        title_label.pack(anchor="w")
        blurb_label = tk.Label(
            pad,
            text=blurb,
            bg=bg,
            fg=blurb_fg,
            anchor="w",
            wraplength=780,
            justify="left",
            font=("Sans", 13),
        )
        blurb_label.pack(anchor="w", pady=(4, 10))
        progress = None
        if show_progress:
            progress = tk.Label(
                pad,
                text="",
                bg=bg,
                fg=progress_fg,
                anchor="w",
                font=("Sans", 13, "bold"),
            )
            progress.pack(anchor="w", pady=(0, 12))
        button = RoundedButton(
            pad,
            text=button_text,
            command=command,
            bg=button_bg,
            activebackground=button_active,
            font=("Sans", 16, "bold"),
            width=180,
            height=60,
            radius=26,
        )
        button.pack(anchor="w")
        card._pad = pad
        card._title = title_label
        card._blurb = blurb_label
        card._progress = progress
        card._button = button
        return card

    def _style_suite_card(
        self,
        card,
        *,
        bg: str,
        border: str,
        title_fg: str,
        blurb_fg: str,
    ):
        card.configure(bg=bg, highlightbackground=border)
        card._pad.configure(bg=bg)
        card._title.configure(bg=bg, fg=title_fg)
        card._blurb.configure(bg=bg, fg=blurb_fg)
        if card._progress is not None:
            card._progress.configure(bg=bg)

    def _show_screen(self, name: str):
        self._screen = name
        for screen in (self.name_screen, self.exp_screen, self.post_screen, self.suite_screen):
            screen.pack_forget()
        if name == "name":
            self._survey_canvas = None
            self.name_screen.pack(fill="both", expand=True)
            self.after(50, lambda: self.name_entry.focus_set())
        elif name == "experience":
            self._survey_canvas = self.exp_canvas
            self.exp_screen.pack(fill="both", expand=True)
        elif name == "post":
            self._survey_canvas = self.post_canvas
            self._refresh_post_task_choices()
            self.post_screen.pack(fill="both", expand=True)
        else:
            self._survey_canvas = self.suite_canvas
            self._refresh_suite_progress()
            self.suite_screen.pack(fill="both", expand=True)
            self.after(0, lambda: self.suite_canvas.configure(scrollregion=self.suite_canvas.bbox("all")))

    def _submit_name(self):
        name = self.name_entry.get().strip()
        if not name:
            self.name_status.configure(text="Please enter your name.", fg=NAME_STATUS_FG)
            self.name_entry.focus_set()
            return
        if not slugify_user_name(name):
            self.name_status.configure(text="Please use letters or numbers in your name.", fg=NAME_STATUS_FG)
            return
        existing = find_user(name)
        if existing:
            self._activate_user(existing)
            self._show_screen("suite")
            return
        self._pending_name = name
        self.name_status.configure(text="")
        self._clear_survey(self.pre_form)
        self._show_screen("experience")

    def _submit_experience(self):
        answers = self._survey_answers(self.pre_form)
        missing = survey_missing_prompts(PRE_SURVEY_QUESTIONS, answers)
        if missing:
            messagebox.showinfo(
                "Please answer every question",
                "Select an option for each question before continuing.",
            )
            return
        name = getattr(self, "_pending_name", "").strip()
        if not name:
            self._show_screen("name")
            return
        data = create_user(name, answers)
        self._activate_user(data)
        self._show_screen("suite")

    def _submit_post(self):
        if self.user_data is None:
            self._show_screen("name")
            return
        extra = self._post_extra_choices()
        answers = self._survey_answers(self.post_form)
        missing = survey_missing_prompts(
            POST_SURVEY_QUESTIONS, answers, extra_choices=extra
        )
        if missing:
            messagebox.showinfo(
                "Please answer every question",
                "Select an option for each question before continuing.",
            )
            return
        log = save_post_survey(self.user_data, answers, extra_choices=extra)
        self.user_data = log
        self._show_screen("suite")

    def _selected_control_mode(self) -> str:
        label = str(self.controller_combo.get() or "Robot").strip()
        return CONTROLLER_MODE.get(label, "robot")

    def _session_controller(self) -> str:
        return log_controller_tag(self._selected_control_mode())

    def _set_controller_combo(self, mode_or_tag: str | None):
        tag = log_controller_tag(mode_or_tag or "robot")
        mode = "keyboard+mouse" if tag == "keyboard" else "robot"
        self.controller_combo.set(MODE_DISPLAY.get(mode, "Robot"))

    def _on_controller_change(self, _event=None):
        if self.child is not None:
            messagebox.showinfo(
                "Task still running",
                "Close the task window before switching controllers.",
            )
            tag = str((self.user_data or {}).get("controller") or "robot")
            self._set_controller_combo(tag)
            return
        if self.user_data is None:
            return
        self._bind_session_log(self.user_data)
        self._refresh_suite_progress()

    def _bind_session_log(self, data: dict | None = None) -> dict:
        """Point this session at ``user_robot.json`` or ``user_keyboard.json``."""
        source = data if data is not None else self.user_data or {}
        display = str(source.get("user_name") or source.get("user_id") or "user")
        slug = str(source.get("user_id") or slugify_user_name(display))
        mode = self._selected_control_mode()
        tag = log_controller_tag(mode)
        log = ensure_controller_log(
            slug,
            display=display,
            experience=source.get("experience"),
            controller=tag,
            template=source,
        )
        os.environ[EXPERIMENT_ENV] = "1"
        os.environ[EXPERIMENT_USER_ENV] = display
        os.environ["ROBODYNA_CONTROL"] = mode
        os.environ[EXPERIMENT_LOG_ENV] = str(log.get("log_path") or "")
        self.user_data = log
        return log

    def _activate_user(self, data: dict):
        self._set_controller_combo(str((data or {}).get("controller") or "robot"))
        log = self._bind_session_log(data)
        display = str(log.get("user_name") or log.get("user_id") or "user")
        tag = str(log.get("controller") or self._session_controller())
        control_label = MODE_DISPLAY.get(
            "keyboard+mouse" if tag == "keyboard" else "robot",
            "Robot",
        )
        self.suite_title.configure(text=f"Welcome back, {display}" if log.get("plays") else f"Hello, {display}")
        self._set_suite_subtitle(
            f"{control_label} session — progress is stored in user_{tag}.json. "
            "Finished items stay gray after you close this window."
        )

    def _protocol_task_names(self) -> tuple[list[str], list[str]]:
        cfg = load_experiment_config()
        assignment = load_user_assignment()
        assigned_base = (assignment or {}).get("base_tasks")
        assigned_hh = (assignment or {}).get("household_tasks")
        base_idx = cfg.visible_indices("base", len(BASE_TASKS), assigned=assigned_base)
        household_idx = cfg.visible_indices(
            "household", len(HOUSEHOLD_TASKS), assigned=assigned_hh
        )
        return (
            [BASE_TASKS[i][1] for i in base_idx],
            [HOUSEHOLD_TASKS[i][1] for i in household_idx],
        )

    def _remaining_summary(self, data: dict | None = None, counts: dict | None = None) -> str:
        log = data if data is not None else load_user_log()
        cfg = load_experiment_config()
        if counts is None:
            base_names, household_names = self._protocol_task_names()
            counts = progress_counts(
                log,
                base_task_names=base_names,
                household_task_names=household_names,
                n_scenarios=len(SCENARIOS),
                cfg=cfg,
            )
        parts: list[str] = []
        all_done = True
        any_slot = False
        scenario_done = counts.get("scenario_done") or {}
        scenario_total = counts.get("scenario_total") or {}
        for key, label in SCENARIO_REMAINING_LABELS:
            total = int(scenario_total.get(key, 0) or 0)
            done = int(scenario_done.get(key, 0) or 0)
            if total <= 0:
                continue
            any_slot = True
            left = max(0, total - done)
            if left:
                all_done = False
            parts.append(f"{label} {left} left ({done}/{total})")
        household_total = int(counts.get("household_total") or 0)
        household_done = int(counts.get("household_done") or 0)
        if household_total > 0:
            any_slot = True
            left = max(0, household_total - household_done)
            if left:
                all_done = False
            parts.append(f"Household {left} left ({household_done}/{household_total})")
        if not any_slot:
            return "No limited scenarios in this protocol."
        if all_done:
            if self._should_collect_post():
                return (
                    "All protocol scenarios are complete for this controller. "
                    "Use the Post-experiment Questionnaire card to share your experience."
                )
            return "All protocol scenarios are complete for this controller."
        return "  ·  ".join(parts)

    def _protocol_complete(self, data: dict | None = None) -> bool:
        log = data if data is not None else load_user_log()
        cfg = load_experiment_config()
        base_names, household_names = self._protocol_task_names()
        counts = progress_counts(
            log,
            base_task_names=base_names,
            household_task_names=household_names,
            n_scenarios=len(SCENARIOS),
            cfg=cfg,
        )
        total = int(counts.get("base_total") or 0) + int(counts.get("household_total") or 0)
        if total <= 0:
            return False
        return (
            int(counts.get("base_done") or 0) >= int(counts.get("base_total") or 0)
            and int(counts.get("household_done") or 0) >= int(counts.get("household_total") or 0)
        )

    def _post_saved(self, data: dict | None = None) -> bool:
        extra = self._post_extra_choices()
        source = data if data is not None else self.user_data or {}
        slug = str(source.get("user_id") or slugify_user_name(source.get("user_name") or ""))
        logs = list(iter_user_controller_logs(slug)) if slug else []
        if not logs and source.get("post_survey"):
            return survey_complete(
                POST_SURVEY_QUESTIONS, source.get("post_survey"), extra_choices=extra
            )
        return any(
            survey_complete(POST_SURVEY_QUESTIONS, log.get("post_survey"), extra_choices=extra)
            for _tag, _path, log in logs
        )

    def _both_controllers_complete(self) -> bool:
        """True only after every assigned slot is done on robot and keyboard+mouse."""
        if self.user_data is None:
            return False
        slug = str(self.user_data.get("user_id") or "")
        if not slug:
            return False
        logs = {tag: log for tag, _path, log in iter_user_controller_logs(slug)}
        return all(
            tag in logs and self._protocol_complete(logs[tag])
            for tag in LOG_CONTROLLER_TAGS
        )

    def _should_collect_post(self) -> bool:
        if self.user_data is None or self._post_saved():
            return False
        return self._both_controllers_complete()

    def _refresh_suite_progress(self):
        if self.user_data:
            self._bind_session_log(self.user_data)
        log = load_user_log()
        if log:
            self.user_data = log
        cfg = load_experiment_config()
        base_names, household_names = self._protocol_task_names()
        counts = progress_counts(
            log,
            base_task_names=base_names,
            household_task_names=household_names,
            n_scenarios=len(SCENARIOS),
            cfg=cfg,
        )
        base_total = counts["base_total"]
        household_total = counts["household_total"]
        if cfg.suite_enabled("base"):
            self.base_card._blurb.configure(
                text=(
                    f"{len(base_names)} sampled dynamic-table tasks "
                    f"(one from each skill group, {cfg.base_scenarios_per_experiment} total), "
                    "each with Default / Opt 1 / Opt 2 / Opt 1+2."
                )
            )
        else:
            self.base_card._blurb.configure(
                text="Not part of this protocol (base_scenarios_per_experiment: 0)."
            )
        if cfg.suite_enabled("household"):
            self.household_card._blurb.configure(
                text=(
                    f"{len(household_names)} sampled kitchen / office tasks "
                    f"(1 easy + 1 hard, {cfg.household_scenarios_per_experiment} total)."
                )
            )
        else:
            self.household_card._blurb.configure(
                text="Not part of this protocol (household_scenarios_per_experiment: 0)."
            )
        base_left = max(0, int(base_total) - int(counts["base_done"]))
        household_left = max(0, int(household_total) - int(counts["household_done"]))
        self.base_card._progress.configure(
            text=(
                f"{counts['base_done']} / {base_total} scenarios completed"
                + (f"  ·  {base_left} left" if base_total else "")
            )
        )
        self.household_card._progress.configure(
            text=(
                f"{counts['household_done']} / {household_total} tasks completed"
                + (f"  ·  {household_left} left" if household_total else "")
            )
        )
        remaining = self._remaining_summary(log, counts=counts)
        self.remaining_stats.configure(text=remaining)
        self._refresh_questionnaire_card()
        display = ""
        if self.user_data:
            display = str(self.user_data.get("user_name") or "")
        if display:
            plays = bool((log or {}).get("plays"))
            tag = str((log or {}).get("controller") or self._session_controller())
            control_label = MODE_DISPLAY.get(
                "keyboard+mouse" if tag == "keyboard" else "robot",
                "Robot",
            )
            self.suite_title.configure(text=f"Welcome back, {display}" if plays else f"Hello, {display}")
            self._set_suite_subtitle(
                f"{control_label} session — progress is stored in user_{tag}.json. "
                "Finished items stay gray after you close this window."
            )
        if counts["base_done"] >= base_total > 0:
            self.base_card._button.configure(text="Review")
        else:
            self.base_card._button.configure(text="Open")
        if counts["household_done"] >= household_total > 0:
            self.household_card._button.configure(text="Review")
        else:
            self.household_card._button.configure(text="Open")
        brand = {
            "base": (BASE_BUTTON_BG, BASE_BUTTON_ACTIVE),
            "household": (HOUSEHOLD_BUTTON_BG, HOUSEHOLD_BUTTON_ACTIVE),
        }
        for suite, card in (("base", self.base_card), ("household", self.household_card)):
            if cfg.suite_enabled(suite):
                bg, active = brand[suite]
                card._button.configure(state="normal", bg=bg, activebackground=active)
            else:
                card._progress.configure(text="")
                card._button.configure(state="disabled", text="Skipped")

    def _launch_suite(self, suite: str):
        if self.child is not None:
            messagebox.showinfo("Already open", "Close the task window before opening another suite.")
            return
        if not load_experiment_config().suite_enabled(suite):
            messagebox.showinfo(
                "Not in this protocol",
                f"{suite}_scenarios_per_experiment is 0, so no tasks are assigned "
                "for this suite.",
            )
            return
        if self.user_data is None:
            self._show_screen("name")
            return
        self._bind_session_log(self.user_data)
        script = ROOT / "interactive" / (
            "base_task_gui.py" if suite == "base" else "household_task_gui.py"
        )
        if not script.exists():
            messagebox.showerror("Unavailable", f"Missing launcher:\n{script}")
            return
        child_env = child_experiment_env()
        cfg_path = os.environ.get(CONFIG_ENV, "").strip()
        if cfg_path:
            child_env[CONFIG_ENV] = cfg_path
        try:
            self.child = subprocess.Popen(
                [sys.executable, str(script)],
                cwd=ROOT,
                start_new_session=True,
                env=child_env,
            )
        except Exception as exc:
            messagebox.showerror("Could not start suite", str(exc))
            self.child = None
            return
        self.withdraw()

    def _poll_child(self):
        if self.child is not None:
            code = self.child.poll()
            if code is not None:
                self.child = None
                try:
                    self.deiconify()
                    self.lift()
                    self.focus_force()
                except tk.TclError:
                    pass
                self._refresh_suite_progress()
        self.after(250, self._poll_child)

    def _switch_user(self):
        if self.child is not None:
            messagebox.showinfo("Task still running", "Close the task window before switching users.")
            return
        self.user_data = None
        os.environ.pop(EXPERIMENT_USER_ENV, None)
        os.environ.pop(EXPERIMENT_LOG_ENV, None)
        os.environ.pop("ROBODYNA_CONTROL", None)
        self.name_entry.delete(0, "end")
        self.name_status.configure(text="")
        self._show_screen("name")

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
        if abs(scale - self._ui_scale) < 0.02:
            wrap = max(360, width - 80)
            self.name_subtitle.configure(wraplength=wrap)
            self.exp_subtitle.configure(wraplength=wrap)
            self.post_subtitle.configure(wraplength=wrap)
            self.suite_subtitle.configure(wraplength=max(280, wrap - 160))
            self.remaining_stats.configure(wraplength=max(360, width - 80))
            for label in self._iter_survey_labels():
                label.configure(wraplength=max(360, width - 100))
            if getattr(self, "suite_canvas", None) is not None:
                self.suite_canvas.configure(scrollregion=self.suite_canvas.bbox("all"))
            return
        self._ui_scale = scale

        def font(size, weight=""):
            px = max(8, int(round(size * scale)))
            return ("Sans", px, weight) if weight else ("Sans", px)

        def px(value):
            return max(1, int(round(value * scale)))

        apply_gui_logo(self.logo_label, height=px(140))
        self.name_title.configure(font=font(30, "bold"))
        self.name_subtitle.configure(font=font(14), wraplength=max(360, width - 80))
        self.name_caption.configure(font=font(13, "bold"))
        self.name_entry.configure(font=font(18, "bold"))
        self.name_continue.configure(font=font(16, "bold"), width=px(220), height=px(64), radius=px(26))
        self.name_status.configure(font=font(13), wraplength=max(360, width - 100))
        self.exp_title.configure(font=font(30, "bold"))
        self.exp_subtitle.configure(font=font(14), wraplength=max(360, width - 80))
        self.post_title.configure(font=font(30, "bold"))
        self.post_subtitle.configure(font=font(14), wraplength=max(360, width - 80))
        for label in self._iter_survey_labels():
            label.configure(font=font(15, "bold"), wraplength=max(360, width - 100))
        for choice in self._iter_survey_dots():
            choice._dot_size = px(28)
            choice._dot_label.configure(font=font(16, "bold"))
            choice._dot_paint()
        self._reflow_all_choices()
        for form in (getattr(self, "pre_form", None), getattr(self, "post_form", None)):
            if not form:
                continue
            for block in form["blocks"]:
                captions = block.get("rank_captions") or []
                if len(captions) >= 1:
                    captions[0].configure(font=font(13), wraplength=max(360, width - 100))
                for caption in captions[1:]:
                    caption.configure(font=font(12, "bold"))
                for tile in block.get("tiles") or []:
                    tile._rank_num.configure(font=font(15, "bold"))
                    tile._rank_name.configure(font=font(15, "bold"))
                    if tile._rank_up is not None:
                        tile._rank_up.configure(font=font(14, "bold"))
                    if tile._rank_down is not None:
                        tile._rank_down.configure(font=font(14, "bold"))
        self.exp_continue.configure(font=font(16, "bold"), width=px(220), height=px(64), radius=px(26))
        self.post_continue.configure(font=font(16, "bold"), width=px(220), height=px(64), radius=px(26))
        self.post_back.configure(font=font(16, "bold"), width=px(140), height=px(64), radius=px(26))
        self.suite_title.configure(font=font(30, "bold"))
        self.suite_subtitle.configure(font=font(14), wraplength=max(280, width - 240))
        self.controller_caption.configure(font=font(13, "bold"))
        self.controller_combo.configure(font=font(13, "bold"))
        self.remaining_stats.configure(font=font(13, "bold"), wraplength=max(360, width - 80))
        self.suite_exit.configure(font=font(16, "bold"), width=px(140), height=px(64), radius=px(26))
        self.name_exit.configure(font=font(16, "bold"), width=px(140), height=px(64), radius=px(26))
        self.exp_exit.configure(font=font(16, "bold"), width=px(140), height=px(64), radius=px(26))
        self.post_exit.configure(font=font(16, "bold"), width=px(140), height=px(64), radius=px(26))
        for card in (self.questionnaire_card, self.base_card, self.household_card):
            card._title.configure(font=font(22, "bold"))
            card._blurb.configure(font=font(13), wraplength=max(360, width - 100))
            if card._progress is not None:
                card._progress.configure(font=font(13, "bold"))
            card._button.configure(font=font(16, "bold"), width=px(180), height=px(60), radius=px(26))
        self.switch_user.configure(font=font(12, "underline"))
        if getattr(self, "suite_canvas", None) is not None:
            self.suite_canvas.configure(scrollregion=self.suite_canvas.bbox("all"))

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

    def _set_suite_subtitle(self, text: str):
        """Remember the standard subtitle; a flashing notice owns the label until it clears."""
        self._subtitle_default = text
        if self._notice_job is None:
            self.suite_subtitle.configure(text=text, fg=GUI_MUTED)

    def _flash_suite_notice(self, message: str, seconds: int = NOTICE_SECONDS):
        if self._notice_job is not None:
            self.after_cancel(self._notice_job)
        self.suite_subtitle.configure(text=message, fg=NOTICE_FG)
        self._notice_job = self.after(seconds * 1000, self._clear_suite_notice)

    def _clear_suite_notice(self):
        self._notice_job = None
        self.suite_subtitle.configure(text=self._subtitle_default, fg=GUI_MUTED)

    def _on_questionnaire_click(self, _event=None):
        """RoundedButton swallows clicks while disabled, so explain the gray Start here."""
        if self.questionnaire_card._button.button_state == "normal":
            return
        self._flash_suite_notice(QUESTIONNAIRE_LOCKED_NOTICE)

    def _lock_questionnaire_card(self, card):
        self._style_suite_card(
            card,
            bg=QUESTIONNAIRE_CARD_BG,
            border=QUESTIONNAIRE_CARD_BORDER,
            title_fg=SUITE_TITLE_FG,
            blurb_fg=SUITE_BLURB_FG,
        )
        card._button.configure(
            text="Start",
            state="disabled",
            bg=LOCKED_GRAY,
            activebackground=LOCKED_GRAY,
            disabledbackground=LOCKED_GRAY,
        )

    def _refresh_questionnaire_card(self):
        card = getattr(self, "questionnaire_card", None)
        if card is None:
            return
        if not self._should_collect_post():
            self._lock_questionnaire_card(card)
            return
        self._style_suite_card(
            card,
            bg=QUESTIONNAIRE_CARD_BG,
            border=QUESTIONNAIRE_CARD_BORDER,
            title_fg=SUITE_TITLE_FG,
            blurb_fg=SUITE_BLURB_FG,
        )
        card._button.configure(
            text="Start",
            state="normal",
            bg=QUESTIONNAIRE_BUTTON_BG,
            activebackground=QUESTIONNAIRE_BUTTON_ACTIVE,
        )

    def _open_questionnaire(self):
        if not self._should_collect_post():
            return
        if self.child is not None:
            messagebox.showinfo(
                "Task still running",
                "Close the task window before opening the questionnaire.",
            )
            return
        if self.user_data is None:
            self._show_screen("name")
            return
        self._show_screen("post")

    def exit_app(self):
        self._stop_child()
        self.destroy()


def main():
    ExperimentLauncher().mainloop()


if __name__ == "__main__":
    main()

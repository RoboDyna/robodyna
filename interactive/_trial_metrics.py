"""Format and display the metrics emitted by one interactive trial."""
from __future__ import annotations

import json
import math
import tkinter as tk
from tkinter import scrolledtext, ttk
from typing import Any, Mapping


def _label(key: str) -> str:
    """Turn a machine-readable metric key into a compact display label."""
    replacements = {
        "s": "s",
        "ms": "MS",
        "rc": "RC",
        "ps": "PS",
    }
    return " ".join(
        replacements.get(part, part.capitalize())
        for part in str(key).replace("_", " ").split()
    )


def _value(value: Any) -> str:
    """Render JSON-like metric values without hiding missing data."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def format_trial_metrics(
    payload: Mapping[str, Any] | None,
    run_meta: Mapping[str, Any] | None = None,
) -> str:
    """Return the human-readable body for the post-trial metrics dialog."""
    payload = payload if isinstance(payload, Mapping) else {}
    run_meta = run_meta if isinstance(run_meta, Mapping) else {}
    metrics = payload.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    partial_detail = metrics.get("partial_score_detail")
    partial_detail = partial_detail if isinstance(partial_detail, Mapping) else {}
    task_metrics = metrics.get("task_metrics")
    task_metrics = task_metrics if isinstance(task_metrics, Mapping) else {}

    task = run_meta.get("task") or metrics.get("task") or "Unknown task"
    scenario = run_meta.get("scenario") or metrics.get("option_label")
    seed = run_meta.get("seed", metrics.get("seed"))
    ok = payload.get("ok", metrics.get("success"))
    if ok is True:
        result = "SUCCESS"
    elif ok is False:
        result = "FAILURE"
    else:
        result = "No terminal result"

    summary = [
        ("Task", task),
        ("Result", result),
        ("Scenario", scenario),
        ("Seed", seed),
        ("Simulation time (s)", metrics.get("total_time_sim_s")),
        ("Wall time (s)", metrics.get("wall_s")),
        ("Steps", metrics.get("steps")),
        ("Partial score", metrics.get("partial_score")),
        ("Route completion", metrics.get("route_completion")),
        ("Manipulation score", metrics.get("manipulation_score")),
        ("Penalty factor", metrics.get("total_penalty_factor")),
    ]
    lines = ["Trial summary"]
    lines.extend(f"{name}: {_value(value)}" for name, value in summary if value is not None)

    fail_reason = payload.get("fail_reason") or metrics.get("fail_reason")
    detail = payload.get("detail")
    if ok is False and fail_reason:
        lines.append(f"Fail reason: {_value(fail_reason)}")
    elif detail:
        lines.append(f"Detail: {_value(detail)}")
    elif fail_reason:
        lines.append(f"Fail reason: {_value(fail_reason)}")

    if partial_detail:
        lines.append("")
        lines.append("Partial-score calculation")
        base = partial_detail.get("base_progress")
        final = partial_detail.get("final_score")
        if base is not None:
            lines.append(f"Base completed progress: {_value(base)}")
        actions = partial_detail.get("completed_actions")
        if isinstance(actions, Mapping) and actions:
            lines.append("Completed actions:")
            lines.extend(
                f"  {_label(key)}: {_value(value)}" for key, value in actions.items()
            )
        penalties = partial_detail.get("penalties")
        if isinstance(penalties, (list, tuple)) and penalties:
            lines.append("Error deductions:")
            for penalty in penalties:
                if isinstance(penalty, Mapping):
                    name = _label(penalty.get("event", "error"))
                    count = _value(penalty.get("count", 1))
                    factor = _value(penalty.get("factor"))
                    lines.append(f"  {name}: {count} ×, factor {factor}")
        elif base is not None:
            lines.append("Error deductions: none")
        if final is not None:
            lines.append(f"Final partial score: {_value(final)}")

    lines.append("")
    lines.append("Task metrics")
    if task_metrics:
        lines.extend(f"{_label(key)}: {_value(value)}" for key, value in task_metrics.items())
    else:
        lines.append("No task-specific metrics were returned for this trial.")
    return "\n".join(lines)


def show_trial_metrics(
    parent: tk.Misc,
    payload: Mapping[str, Any] | None,
    run_meta: Mapping[str, Any] | None = None,
) -> None:
    """Show a modal, scrollable post-trial metrics dialog."""
    dialog = tk.Toplevel(parent)
    dialog.title("Trial metrics")
    dialog.transient(parent)
    dialog.minsize(520, 400)
    dialog.geometry("660x560")

    frame = ttk.Frame(dialog, padding=14)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Trial metrics", font=("Sans", 16, "bold")).pack(
        anchor="w", pady=(0, 8)
    )
    ttk.Label(
        frame,
        text="Metrics are recorded at the terminal task result.",
    ).pack(anchor="w", pady=(0, 10))

    text = scrolledtext.ScrolledText(
        frame,
        wrap="word",
        height=24,
        font=("Monospace", 11),
        padx=10,
        pady=10,
    )
    text.insert("1.0", format_trial_metrics(payload, run_meta))
    text.configure(state="disabled")
    text.pack(fill="both", expand=True)

    def close() -> None:
        try:
            dialog.grab_release()
        except tk.TclError:
            pass
        dialog.destroy()

    buttons = ttk.Frame(frame)
    buttons.pack(fill="x", pady=(10, 0))
    ttk.Button(buttons, text="Close", command=close).pack(side="right")
    dialog.protocol("WM_DELETE_WINDOW", close)
    dialog.bind("<Escape>", lambda _event: close())
    dialog.grab_set()
    dialog.focus_set()
    parent.wait_window(dialog)

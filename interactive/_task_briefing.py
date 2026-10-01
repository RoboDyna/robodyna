"""Pre-start briefing dialog for interactive / household task GUIs."""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tkinter as tk
from pathlib import Path

from _briefing_copy import (
    TASK_BRIEFING,
    episode_amount,
    fill_amount_placeholders,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTRUCTION_CATALOG = REPO_ROOT / "task_config" / "task_instructions.json"
_TEXTURE_DIR = REPO_ROOT / "assets" / "dyna_textures"
# Same Robo/Dyna stamp mark as control_quality tiles (cropped + upright).
APP_ICON_PATH = _TEXTURE_DIR / "robodyna_app_icon.png"
_APP_ICON_SIZES = (16, 32, 48, 64, 128, 256, 512)
# XDG icon name installed under ~/.local/share/icons/hicolor/.../apps/
APP_ICON_NAME = "robodyna"

# Per-suite variants (wordmark + suite word in the suite color); regenerate with
# ``python script/make_gui_app_icons.py``.
SUITE_ICONS = {
    "interactive": (_TEXTURE_DIR / "robodyna_app_icon_base.png", "robodyna-base"),
    "household": (_TEXTURE_DIR / "robodyna_app_icon_hh.png", "robodyna-hh"),
    "experiment": (_TEXTURE_DIR / "robodyna_app_icon_exp.png", "robodyna-exp"),
    # Hub launcher: Robo/Dyna with one brand color per letter.
    "gui": (_TEXTURE_DIR / "robodyna_app_icon_gui.png", "robodyna-gui"),
}
# Set by ``setup_gui_app_icon`` so later windows (briefing dialog) match the suite.
_active_icon_path = APP_ICON_PATH

# Sampled from docs/robodyna_logo.png corners (RGB 238, 238, 242).
GUI_PAGE_BG = "#eeeef2"
GUI_INK = "#002d56"
GUI_MUTED = "#5a6a7c"
LOGO_PATHS = (
    REPO_ROOT / "docs" / "robodyna_logo.png",
)

# Tk lowercases everything after the first letter for WM_CLASS class, so pass the
# final GNOME StartupWMClass string here (must match the .desktop file).
GUI_WM_CLASS = {
    "interactive": "Robodynainteractive",
    "household": "Robodynahousehold",
    "experiment": "Robodynaexperiment",
    "gui": "Robodynagui",
}


def _xdg_data_home() -> Path:
    raw = os.environ.get("XDG_DATA_HOME", "").strip()
    if raw:
        return Path(raw)
    return Path.home() / ".local" / "share"


def suite_icon(suite: str) -> tuple[Path, str]:
    """Return ``(png path, XDG icon name)`` for a suite, falling back to the stamp."""
    path, name = SUITE_ICONS.get(suite, (APP_ICON_PATH, APP_ICON_NAME))
    if not path.is_file():
        return APP_ICON_PATH, APP_ICON_NAME
    return path, name


def install_ubuntu_dock_icon(
    *,
    desktop_id: str,
    name: str,
    comment: str,
    script_path: Path,
    wm_class: str,
    python_exe: str | None = None,
    icon_path: Path | None = None,
    icon_name: str = APP_ICON_NAME,
) -> Path | None:
    """Install the stamp PNG + a .desktop entry so Ubuntu's dock shows it.

    GNOME ignores Tk ``iconphoto`` for the dock; it uses ``Icon=`` from a
    matching ``.desktop`` via ``StartupWMClass``.
    """
    source = icon_path or APP_ICON_PATH
    if not source.is_file():
        return None
    try:
        from PIL import Image
    except ImportError:
        return None

    data_home = _xdg_data_home()
    icon_root = data_home / "icons" / "hicolor"
    apps_dir = data_home / "applications"
    apps_dir.mkdir(parents=True, exist_ok=True)

    try:
        master = Image.open(source).convert("RGBA")
    except OSError:
        return None

    for size in _APP_ICON_SIZES:
        dest_dir = icon_root / f"{size}x{size}" / "apps"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{icon_name}.png"
        master.resize((size, size), Image.Resampling.LANCZOS).save(dest, optimize=True)

    exe = python_exe or sys.executable
    script = script_path.resolve()
    desktop_path = apps_dir / f"{desktop_id}.desktop"
    # Absolute Icon= path is the most reliable fallback if the theme cache lags.
    icon_abs = (icon_root / "256x256" / "apps" / f"{icon_name}.png").resolve()
    desktop_path.write_text(
        "\n".join(
            [
                "[Desktop Entry]",
                "Version=1.0",
                "Type=Application",
                f"Name={name}",
                f"Comment={comment}",
                f"Exec={exe} {script}",
                f"Icon={icon_abs}",
                "Terminal=false",
                "Categories=Science;Education;",
                f"StartupWMClass={wm_class}",
                "StartupNotify=true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    desktop_path.chmod(desktop_path.stat().st_mode | 0o111)

    # Refresh caches when tools exist (best-effort; dock still works without).
    for cmd in (
        ["update-desktop-database", str(apps_dir)],
        ["gtk-update-icon-cache", "-f", "-t", str(icon_root)],
    ):
        if shutil.which(cmd[0]) is None:
            continue
        try:
            subprocess.run(cmd, check=False, capture_output=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            pass
    return desktop_path


def apply_window_icon(window: tk.Misc, icon_path: Path | None = None) -> None:
    """Set the window decoration icon (title bar). Dock icon needs .desktop install."""
    source = icon_path or _active_icon_path
    if not source.is_file():
        return
    try:
        from PIL import Image, ImageTk
    except ImportError:
        return
    try:
        master = Image.open(source).convert("RGBA")
    except OSError:
        return
    photos: list = []
    for size in (16, 32, 48, 64, 128, 256):
        im = master.resize((size, size), Image.Resampling.LANCZOS)
        photos.append(ImageTk.PhotoImage(im, master=window))
    if not photos:
        return
    try:
        window.iconphoto(True, *photos)
    except tk.TclError:
        return
    # Keep PhotoImage refs alive for the lifetime of the window.
    existing = getattr(window, "_robodyna_icon_photos", None)
    if isinstance(existing, list):
        existing.extend(photos)
    else:
        window._robodyna_icon_photos = photos


def setup_gui_app_icon(
    window: tk.Tk,
    *,
    suite: str,
    script_path: Path,
) -> None:
    """Install Ubuntu dock branding and apply the in-window icon."""
    global _active_icon_path
    wm_class = GUI_WM_CLASS.get(suite, "Robodyna")
    icon_path, icon_name = suite_icon(suite)
    _active_icon_path = icon_path
    if suite == "household":
        name = "RoboDyna Household Tasks"
        comment = "Household interactive task launcher"
        desktop_id = "robodyna-household-tasks"
    elif suite == "experiment":
        name = "RoboDyna Human Experiment"
        comment = "Human-experiment task launcher"
        desktop_id = "robodyna-experiment"
    elif suite == "gui":
        name = "RoboDyna"
        comment = "Shared launcher for the base, household, and experiment GUIs"
        desktop_id = "robodyna-gui"
    else:
        name = "RoboDyna Base Tasks"
        comment = "Dynamic interactive task launcher"
        desktop_id = "robodyna-interactive-tasks"

    install_ubuntu_dock_icon(
        desktop_id=desktop_id,
        name=name,
        comment=comment,
        script_path=script_path,
        wm_class=wm_class,
        icon_path=icon_path,
        icon_name=icon_name,
    )
    apply_window_icon(window, icon_path)


def apply_gui_logo(label: tk.Label, *, height: int) -> None:
    """Put the docs-owned RoboDyna logo on ``label``, scaled to ``height`` pixels."""
    height = max(28, int(height))
    cache = getattr(label, "_robodyna_logo", None)
    if isinstance(cache, dict) and cache.get("height") == height and cache.get("photo") is not None:
        return
    try:
        from PIL import Image, ImageTk
    except ImportError:
        return
    path = next((p for p in LOGO_PATHS if p.is_file()), None)
    if path is None:
        return
    try:
        source = Image.open(path).convert("RGB")
    except OSError:
        return
    src_w, src_h = source.size
    if src_h <= 0:
        return
    width = max(1, int(round(src_w * (height / src_h))))
    photo = ImageTk.PhotoImage(
        source.resize((width, height), Image.Resampling.LANCZOS),
        master=label,
    )
    label.configure(image=photo)
    label._robodyna_logo = {"height": height, "photo": photo}


# Warm slate + copper accent — distinct from the launcher chrome, not purple/cream.
PAGE_BG = "#0e141c"
PANEL_BG = "#151d28"
INK = "#f2ebe3"
MUTED = "#9aa8b5"
FAINT = "#5d6b78"
LINE = "#2a3644"
ACCENT = "#d08a4c"
ACCENT_SOFT = "#3a2a1c"
KEY_BG = "#1c2733"
KEY_FG = "#f6d7b0"
START_BG = "#d08a4c"
START_ACTIVE = "#e09a5c"
START_FG = "#1a120c"
CANCEL_BG = "#243040"
CANCEL_ACTIVE = "#314255"
CANCEL_FG = "#d7e0e8"

# Keep in sync with ``print_mode_controls`` shared teleop block in
# ``interactive/_interactive_common.py`` (UniversalRobotControls).
_SHARED_ROBOT = """\
Arrow keys — move selected arm(s) in world XY
E / Q — raise / lower selected arm(s)
F / G — tip gripper left / right (world Y)
R / T — yaw gripper CCW / CW (world Z)
1 / 2 / 3 — select left / right / both arms (selected gripper turns green)
O — return selected arm(s) to original position
Space — open / close selected gripper(s)
V — cycle view: head_camera ↔ gripper(s)
Escape — give up (closes the viewer, counts as a failure)
"""

_GRIPPER_TOGGLE_HELP = "Space — open / close selected gripper(s)"
_VIEW_HELP = "V — cycle view: head_camera ↔ gripper(s)"
_ESCAPE_HELP = "Escape — give up (closes the viewer, counts as a failure)"

_NOTE_RED = "#c0392b"


def load_task_instruction(task: str) -> str:
    """Return the one canonical instruction from ``task_config``."""
    if not INSTRUCTION_CATALOG.exists():
        return ""
    try:
        data = json.loads(INSTRUCTION_CATALOG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return ""
    tasks = data.get("tasks")
    if not isinstance(tasks, dict):
        return ""
    return str(tasks.get(str(task).strip().replace("-", "_")) or "").strip()


def _extract_string_constant(source: str, name: str) -> str:
    """Pull a module-level string assignment ``NAME = \"\"\"...\"\"\"`` from source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    return ""
                if isinstance(value, str):
                    return value.strip("\n")
    pattern = re.compile(
        rf"{re.escape(name)}\s*=\s*(?P<q>\"\"\"|''')(?P<body>.*?)(?P=q)",
        re.DOTALL,
    )
    match = pattern.search(source)
    if not match:
        return ""
    return match.group("body").strip("\n")


def _join_prose_continuations(text: str) -> str:
    """Join wrapped prose lines from household ROBOT/KEYBOARD banners."""
    joined: list[str] = []
    buf = ""
    buf_is_binding = False
    for raw in str(text or "").splitlines():
        s = raw.strip()
        if not s:
            if buf:
                joined.append(buf)
                buf = ""
                buf_is_binding = False
            continue
        keyish = _looks_like_key_binding(s) or bool(
            re.match(r"^[\w() /+]{1,28}\s{2,}\S", s)
            or re.match(
                r"^(Space|F|G)\s+opens?/closes?\b",
                s,
                re.IGNORECASE,
            )
        )
        # Only wrap mid-sentence prose; never append onto a finished key row.
        if (
            buf
            and not keyish
            and not buf_is_binding
            and not buf.rstrip().endswith((".", "!", "?"))
        ):
            buf = f"{buf} {s}"
            continue
        if buf:
            joined.append(buf)
        buf = s
        buf_is_binding = bool(keyish)
    if buf:
        joined.append(buf)
    return "\n".join(joined)


def _looks_like_key_binding(line: str) -> bool:
    """True for control rows like ``Space — …`` / ``Arrow keys    …``."""
    s = line.strip()
    if re.match(r"^Tip\b", s, re.IGNORECASE):
        return True
    if re.match(r"^mouse\s+click\b", s, re.IGNORECASE):
        return True
    if re.match(r"^Note\b", s, re.IGNORECASE):
        return True
    # Known teleop / viewer keys (and a few task-specific tokens like "(hit)").
    return bool(
        re.match(
            r"^(?:Arrow keys|Arrows(?:\s*/\s*E\s*/\s*Q)?|Left\s*/\s*Right|"
            r"Up\s*/\s*Down|E\s*/\s*Q|Z\s*/\s*X|F\s*/\s*G|R\s*/\s*T|1\s*/\s*2\s*/\s*3|"
            r"O|Space|F|G|V|Escape|Note|\(hit\)|"
            r"[A-Z0-9](?:\s*/\s*[A-Z0-9])*)"
            r"(?:\s{2,}|\s+[—–-]\s+|:\s+)\S",
            s,
            re.IGNORECASE,
        )
    )


def _normalize_control_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in _join_prose_continuations(text).splitlines():
        line = raw.strip()
        if not line:
            continue
        # Collapse "Key    description" spacing into "Key — description".
        # Do this even when an em-dash appears later in the description.
        m = re.match(r"^(.+?)\s{2,}(.+)$", line)
        if m and len(m.group(1).strip()) <= 28 and _looks_like_key_binding(
            f"{m.group(1).strip()}  {m.group(2).strip()}"
        ):
            line = f"{m.group(1).strip()} — {m.group(2).strip()}"
        # Household prose: "Space opens/closes the gripper only — …"
        m = re.match(
            r"^(Space|F|G)\s+opens?/closes?\s+(?:the\s+)?gripper\b(.*)$",
            line,
            re.IGNORECASE,
        )
        if m:
            rest = m.group(2).strip(" —-")
            line = _GRIPPER_TOGGLE_HELP
            if rest:
                line = f"{_GRIPPER_TOGGLE_HELP} {rest}"
        elif not _looks_like_key_binding(line):
            line = f"Tip — {line}"
        lines.append(line)
    return lines


def _line_documents_key(lines: list[str], key: str) -> bool:
    """True when a control line already documents ``key`` as a binding."""
    key_u = key.strip().upper()
    for ln in lines:
        s = ln.strip()
        s_u = s.upper()
        if (
            s_u.startswith(f"{key_u} ")
            or s_u.startswith(f"{key_u}:")
            or s_u.startswith(f"{key_u}\t")
            or s_u.startswith(f"{key_u} —")
            or s_u.startswith(f"{key_u} -")
        ):
            return True
        if f"{key_u}:" in s_u or f"| {key_u}:" in s_u or f"|{key_u}:" in s_u:
            return True
    return False


def _is_gripper_toggle_help_line(line: str) -> bool:
    """True for leftover F/G or Space lines that document open/close gripper."""
    s = line.strip()
    s_u = s.upper()
    # F/G now tip the wrist; don't treat the shared tilt row as a gripper toggle.
    if s_u.startswith("F / G") or s_u.startswith("F/G"):
        return False
    if not (
        s.startswith("F ")
        or s.startswith("F:")
        or s.startswith("F\t")
        or s.startswith("F —")
        or s.startswith("G ")
        or s.startswith("G:")
        or s.startswith("G\t")
        or s.startswith("G —")
        or s.startswith("Space ")
        or s.startswith("Space:")
        or s.startswith("Space\t")
        or s.startswith("Space —")
    ):
        return False
    low = line.lower()
    return "gripper" in low and (
        "open" in low or "close" in low or "grasp" in low or "pinch" in low
    )


def _is_shared_robot_teleop_help_line(line: str) -> bool:
    """True for task-banner rows that only restate universal robot teleop keys."""
    s = line.strip()
    if not s:
        return False
    s_u = s.upper()
    if s_u.startswith("ARROWS / E / Q") or s_u.startswith("ARROWS/E/Q"):
        return True
    prefixes = (
        "ARROW KEYS",
        "ARROWS ",
        "ARROWS\t",
        "ARROWS:",
        "E / Q",
        "E/Q ",
        "E/Q\t",
        "E/Q:",
        "Z / X",
        "Z/X ",
        "Z/X\t",
        "Z/X:",
        "F / G",
        "F/G ",
        "F/G\t",
        "F/G:",
        "R / T",
        "R/T ",
        "R/T\t",
        "R/T:",
        "1 / 2 / 3",
        "1/2/3",
        "O ",
        "O\t",
        "O:",
        "O —",
        "O -",
    )
    return any(s_u.startswith(p) for p in prefixes)


def _shared_teleop_is_generic(line: str) -> bool:
    """True when a shared-teleop-prefixed line just restates the universal help."""
    _, desc = _split_control_line(line)
    low = desc.lower()
    generics = (
        "move selected arm",
        "raise / lower",
        "tip gripper",
        "yaw gripper",
        "select left",
        "turns green",
        "return selected",
        "original position",
        "move in z",
        "move in xy",
        "world xy",
        "world y",
        "world z",
        "teleop the selected",
    )
    if low.strip() in {"teleop", "teleop."}:
        return True
    return any(g in low for g in generics)


def _is_view_help_line(line: str) -> bool:
    low = line.lower()
    s = line.strip()
    return (
        "toggle view" in low
        or "cycle view" in low
        or "gripper view" in low
        or s.startswith("V —")
        or s.startswith("V:")
        or s.startswith("V ")
    )


def _rewrite_gripper_toggle_lines(lines: list[str]) -> list[str]:
    """Collapse F/G/Space gripper-toggle rows; keep task-specific Space wording."""
    rewritten: list[str] = []
    saw_space_grip = False
    for ln in lines:
        if not _is_gripper_toggle_help_line(ln):
            # Task Space lines that aren't generic open/close still count.
            if ln.strip().upper().startswith("SPACE") and saw_space_grip:
                continue
            if ln.strip().upper().startswith("SPACE") and not saw_space_grip:
                if "—" not in ln and " - " not in ln:
                    m = re.match(r"^Space\s{2,}(.+)$", ln.strip(), re.IGNORECASE)
                    rewritten.append(
                        f"Space — {m.group(1).strip()}" if m else ln
                    )
                else:
                    rewritten.append(ln)
                saw_space_grip = True
                continue
            rewritten.append(ln)
            continue
        if saw_space_grip:
            continue
        if ln.strip().upper().startswith("SPACE"):
            if "—" not in ln and " - " not in ln:
                m = re.match(r"^Space\s{2,}(.+)$", ln.strip(), re.IGNORECASE)
                rewritten.append(
                    f"Space — {m.group(1).strip()}" if m else ln
                )
            else:
                rewritten.append(ln)
        else:
            rewritten.append(_GRIPPER_TOGGLE_HELP)
        saw_space_grip = True
    return rewritten


def _dedupe_lines(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def load_task_controls(script_path: Path, mode: str) -> str:
    """Return keyboard/robot control help from an interactive launcher script."""
    if not script_path.exists():
        return ""
    try:
        source = script_path.read_text(encoding="utf-8")
    except OSError:
        return ""
    mode_raw = str(mode or "").strip().lower().replace("_", "+").replace(" ", "")
    mode = "robot" if mode_raw == "robot" else "keyboard"
    # ``keyboard+mouse`` (and aliases) use the keyboard banner.
    if mode == "robot":
        body = (
            _extract_string_constant(source, "CONTROLS_ROBOT")
            or _extract_string_constant(source, "ROBOT")
        )
        task_lines: list[str] = []
        for ln in _normalize_control_lines(body):
            if _is_view_help_line(ln) or ln.strip().lower().startswith("escape"):
                continue
            if _is_shared_robot_teleop_help_line(ln):
                if _shared_teleop_is_generic(ln):
                    continue
                # Keep task-specific teleop nuance as a tip (e.g. keeper nudge).
                _, desc = _split_control_line(ln)
                task_lines.append(f"Tip — {desc}" if desc else ln)
                continue
            task_lines.append(ln)
        task_lines = _rewrite_gripper_toggle_lines(task_lines)

        shared = list(_normalize_control_lines(_SHARED_ROBOT))
        # Prefer task-specific Space wording over the generic shared line.
        if _line_documents_key(task_lines, "Space"):
            shared = [
                ln for ln in shared if not (
                    _is_gripper_toggle_help_line(ln)
                    or ln.strip().upper().startswith("SPACE")
                )
            ]
            insert_at = next(
                (i for i, ln in enumerate(shared) if _is_view_help_line(ln)),
                len(shared),
            )
            space_line = next(
                ln
                for ln in task_lines
                if _is_gripper_toggle_help_line(ln)
                or ln.strip().upper().startswith("SPACE")
            )
            shared[insert_at:insert_at] = [space_line]
            task_lines = [
                ln
                for ln in task_lines
                if not (
                    _is_gripper_toggle_help_line(ln)
                    or ln.strip().upper().startswith("SPACE")
                )
            ]

        core = [
            ln
            for ln in _dedupe_lines(shared + task_lines)
            if not _is_view_help_line(ln)
            and not ln.strip().lower().startswith("escape")
        ]
        return "\n".join(core + [_VIEW_HELP, _ESCAPE_HELP])

    body = (
        _extract_string_constant(source, "CONTROLS_KEYBOARD")
        or _extract_string_constant(source, "KEYBOARD")
    )
    lines = _rewrite_gripper_toggle_lines(_normalize_control_lines(body))
    # Keyboard+mouse: no arms / no gripper view — omit Space gripper toggle and V.
    extras = [_ESCAPE_HELP]
    for extra in extras:
        if extra.lower() not in {x.lower() for x in lines}:
            lines.append(extra)
    core = [
        ln
        for ln in _dedupe_lines(lines)
        if not _is_view_help_line(ln)
        and not ln.strip().lower().startswith("escape")
        and not _is_gripper_toggle_help_line(ln)
    ]
    # Keep task-specific Space (pour, stove, …); drop generic gripper Space.
    core = [
        ln for ln in core
        if not (
            ln.strip().upper().startswith("SPACE")
            and ("gripper" in ln.lower() or "open / close" in ln.lower())
        )
    ]
    return "\n".join(core + extras)


def _is_mouse_click_key(key: str) -> bool:
    k = re.sub(r"\s+", " ", key.strip().lower())
    return k in {"mouse click", "mouse"} or k.startswith("mouse click")


def _format_catalog_controls(rows: list) -> str:
    lines: list[str] = []
    for row in rows or []:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        key, desc = str(row[0]).strip(), str(row[1]).strip()
        if _is_mouse_click_key(key):
            key = "mouse click"
            low = desc.lower()
            if low.startswith("mouse click"):
                desc = desc[len("mouse click") :].lstrip(" —-")
            elif low.startswith("click "):
                pass
        lines.append(f"{key} — {desc}" if desc else key)
    return "\n".join(lines)


def build_briefing_text(
    *,
    label: str,
    task: str,
    scenario_label: str | None,
    scenario_desc: str | None,
    summary: str | None,
    control_mode: str,
    script_path: Path,
    scenario: str | None = None,
    seed: int | None = None,
) -> dict[str, object]:
    """Assemble instruction / controls / notes / tips for the dialog."""
    del summary  # Objectives removed; README summary is not shown.
    spec = TASK_BRIEFING.get(task) or {}
    amount = episode_amount(task, seed)

    def _amt(text: str) -> str:
        return fill_amount_placeholders(str(text or ""), amount)

    instruction = _amt(load_task_instruction(task))
    option_heading = str(scenario_label or scenario or "").strip()
    option_body = _amt(str(scenario_desc or "").strip())
    success_line = ""
    if spec.get("success"):
        success_line = _amt(spec["success"]).strip()

    mode_raw = str(control_mode or "").strip().lower().replace("_", "+").replace(" ", "")
    is_robot = mode_raw == "robot"

    notes: list = []
    if not is_robot:
        extra_notes = list(spec.get("notes") or [])
        extra_tips = list(spec.get("tips") or [])
        for item in extra_notes + extra_tips:
            if isinstance(item, dict):
                notes.append({
                    **item,
                    "text": _amt(item.get("text") or ""),
                    "red": list(item.get("red") or []) + ([amount] if amount else []),
                })
            else:
                notes.append(_amt(item))

    if is_robot:
        # Full shared teleop key bindings; no task-specific robot tips.
        controls = "\n".join(_normalize_control_lines(_SHARED_ROBOT))
        display_mode = "robot"
    else:
        kb = spec.get("keyboard")
        controls = (
            _format_catalog_controls(kb)
            if kb
            else load_task_controls(script_path, control_mode)
        )
        has_mouse = any(
            isinstance(row, (list, tuple))
            and len(row) >= 1
            and _is_mouse_click_key(str(row[0]))
            for row in (kb or [])
        )
        display_mode = "keyboard+mouse" if has_mouse else "keyboard"

    return {
        "title": label,
        "task": task,
        "scenario_label": scenario_label or "",
        "instruction": instruction or "No written instruction found for this task.",
        "option_heading": option_heading,
        "option_body": option_body,
        "option_line": (
            f"{option_heading}: {option_body}".strip(": ") if option_body else ""
        ),
        "success_line": success_line,
        "objectives": [],
        "controls": controls or "No control help found for this launcher.",
        "notes": notes,
        "control_mode": display_mode,
    }


def _split_control_line(line: str) -> tuple[str, str]:
    for sep in (" — ", " - ", ": "):
        if sep in line:
            left, right = line.split(sep, 1)
            return left.strip(), right.strip()
    return line.strip(), ""


def _draw_header_wash(canvas: tk.Canvas, width: int, height: int) -> None:
    """Soft vertical wash for the hero band (no flat single-color slab)."""
    canvas.delete("wash")
    steps = max(24, height // 3)
    for i in range(steps):
        t = i / max(steps - 1, 1)
        # Deep navy → warmer slate near the copper accent.
        r = int(14 + (28 - 14) * t)
        g = int(20 + (32 - 20) * t)
        b = int(28 + (36 - 28) * t)
        # Slight copper lift at the bottom edge.
        r = min(255, int(r + 28 * (t**2)))
        g = min(255, int(g + 10 * (t**2)))
        color = f"#{r:02x}{g:02x}{b:02x}"
        y0 = int(height * i / steps)
        y1 = int(height * (i + 1) / steps) + 1
        canvas.create_rectangle(0, y0, width, y1, outline="", fill=color, tags="wash")
    # Thin accent rule along the bottom.
    canvas.create_rectangle(
        0, height - 3, width, height, outline="", fill=ACCENT, tags="wash"
    )


def show_task_briefing(parent: tk.Tk | tk.Toplevel, briefing: dict) -> bool:
    """Modal briefing screen. Return True to start the task, False to cancel."""
    dialog = tk.Toplevel(parent)
    dialog.title("Task instructions")
    dialog.configure(bg=PAGE_BG)
    apply_window_icon(dialog)
    dialog.transient(parent)
    dialog.grab_set()
    dialog.resizable(True, True)

    parent.update_idletasks()
    width, height = 980, 760
    px = parent.winfo_rootx() + max(0, (parent.winfo_width() - width) // 2)
    py = parent.winfo_rooty() + max(0, (parent.winfo_height() - height) // 2)
    dialog.geometry(f"{width}x{height}+{px}+{py}")
    dialog.minsize(780, 620)

    result = {"ok": False}
    title = str(briefing.get("title") or briefing.get("task") or "Task")
    scenario = str(briefing.get("scenario_label") or "")
    mode = str(briefing.get("control_mode") or "robot")
    instruction = str(briefing.get("instruction") or "")
    option_heading = str(briefing.get("option_heading") or "").strip()
    option_body = str(briefing.get("option_body") or "").strip()
    if not option_body:
        option_body = str(briefing.get("option_line") or "").strip()
        if option_body.lower().startswith("option ") and ":" in option_body:
            option_body = option_body.split(":", 1)[-1].strip()
    success_line = str(briefing.get("success_line") or "").strip()
    notes = briefing.get("notes") or []
    control_lines = _normalize_control_lines(str(briefing.get("controls") or ""))

    # ---- Hero ----
    hero = tk.Canvas(dialog, height=168, bg=PAGE_BG, highlightthickness=0, bd=0)
    hero.pack(fill="x")

    def _paint_hero(_event=None):
        w = max(hero.winfo_width(), 2)
        h = max(hero.winfo_height(), 2)
        _draw_header_wash(hero, w, h)
        hero.delete("copy")
        pad_x = 40
        y = 28
        hero.create_text(
            pad_x,
            y,
            text="INSTRUCTIONS",
            anchor="w",
            fill=ACCENT,
            font=("Georgia", 11, "bold"),
            tags="copy",
        )
        y += 28
        hero.create_text(
            pad_x,
            y,
            text=title,
            anchor="nw",
            fill=INK,
            font=("Georgia", 30, "bold"),
            width=w - pad_x * 2 - 160,
            tags="copy",
        )
        # Mode chip on the right.
        chip = f"{mode.upper()} CONTROL"
        chip_w = 8 * len(chip) + 28
        chip_x1 = w - 40
        chip_x0 = chip_x1 - chip_w
        hero.create_rectangle(
            chip_x0,
            28,
            chip_x1,
            56,
            outline=ACCENT,
            fill=ACCENT_SOFT,
            width=1,
            tags="copy",
        )
        hero.create_text(
            (chip_x0 + chip_x1) / 2,
            42,
            text=chip,
            fill=KEY_FG,
            font=("Sans", 10, "bold"),
            tags="copy",
        )
        if scenario:
            y = 118
            hero.create_text(
                pad_x,
                y,
                text=scenario,
                anchor="w",
                fill=MUTED,
                font=("Sans", 13),
                tags="copy",
            )

    hero.bind("<Configure>", _paint_hero)

    # ---- Scrollable body ----
    shell = tk.Frame(dialog, bg=PAGE_BG)
    shell.pack(fill="both", expand=True)

    canvas = tk.Canvas(shell, bg=PAGE_BG, highlightthickness=0, bd=0)
    scroll = tk.Scrollbar(shell, orient="vertical", command=canvas.yview, width=10)
    canvas.configure(yscrollcommand=scroll.set)
    canvas.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")

    inner = tk.Frame(canvas, bg=PAGE_BG)
    window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

    def _sync_scroll(_event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _sync_width(event):
        canvas.itemconfigure(window_id, width=event.width)

    inner.bind("<Configure>", _sync_scroll)
    canvas.bind("<Configure>", _sync_width)

    def _wheel(event):
        if event.delta:
            canvas.yview_scroll(-int(event.delta / 120), "units")
            return "break"
        return None

    def _wheel_up(_event):
        canvas.yview_scroll(-3, "units")
        return "break"

    def _wheel_down(_event):
        canvas.yview_scroll(3, "units")
        return "break"

    for widget in (dialog, canvas, inner):
        widget.bind("<MouseWheel>", _wheel)
        widget.bind("<Button-4>", _wheel_up)
        widget.bind("<Button-5>", _wheel_down)

    content = tk.Frame(inner, bg=PAGE_BG)
    content.pack(fill="both", expand=True, padx=40, pady=(22, 12))

    def _section(parent_frame: tk.Frame, heading: str) -> tk.Frame:
        block = tk.Frame(parent_frame, bg=PAGE_BG)
        block.pack(fill="x", pady=(0, 26))
        head = tk.Frame(block, bg=PAGE_BG)
        head.pack(fill="x", pady=(0, 10))
        tk.Label(
            head,
            text=heading.upper(),
            bg=PAGE_BG,
            fg=MUTED,
            font=("Sans", 11, "bold"),
        ).pack(side="left")
        tk.Frame(block, bg=LINE, height=1).pack(fill="x", pady=(0, 12))
        return block

    def _pack_highlighted(parent_frame: tk.Frame, text: str, red_phrases: list[str]) -> None:
        row = tk.Frame(parent_frame, bg=PAGE_BG)
        row.pack(anchor="w", fill="x")
        remaining = text
        font = ("Sans", 13)
        while remaining:
            hit_at = None
            hit_phrase = ""
            low = remaining.lower()
            for phrase in red_phrases or []:
                p = str(phrase).strip()
                if not p:
                    continue
                idx = low.find(p.lower())
                if idx >= 0 and (hit_at is None or idx < hit_at):
                    hit_at = idx
                    hit_phrase = remaining[idx : idx + len(p)]
            if hit_at is None:
                tk.Label(
                    row, text=remaining, bg=PAGE_BG, fg=INK, font=font,
                    justify="left", wraplength=860, anchor="w",
                ).pack(side="left")
                break
            if hit_at:
                tk.Label(
                    row, text=remaining[:hit_at], bg=PAGE_BG, fg=INK, font=font,
                    justify="left", anchor="w",
                ).pack(side="left")
            tk.Label(
                row, text=hit_phrase, bg=PAGE_BG, fg=_NOTE_RED,
                font=("Sans", 13, "bold"), justify="left", anchor="w",
            ).pack(side="left")
            remaining = remaining[hit_at + len(hit_phrase) :]

    # Instruction
    instr = _section(content, "Instruction")
    tk.Label(
        instr,
        text=instruction,
        bg=PAGE_BG,
        fg=INK,
        font=("Georgia", 16),
        justify="left",
        wraplength=860,
        anchor="w",
    ).pack(anchor="w", fill="x")

    if success_line:
        success_body = success_line
        if success_body.lower().startswith("success condition:"):
            success_body = success_body.split(":", 1)[-1].strip()
        succ = _section(content, "Success condition")
        tk.Label(
            succ,
            text=success_body,
            bg=PAGE_BG,
            fg=INK,
            font=("Georgia", 16),
            justify="left",
            wraplength=860,
            anchor="w",
        ).pack(anchor="w", fill="x")

    if option_body:
        opt = _section(content, "Option")
        if option_heading:
            tk.Label(
                opt,
                text=option_heading,
                bg=PAGE_BG,
                fg=ACCENT,
                font=("Sans", 13, "bold"),
                justify="left",
                anchor="w",
            ).pack(anchor="w", fill="x")
        tk.Label(
            opt,
            text=option_body,
            bg=PAGE_BG,
            fg=INK,
            font=("Georgia", 16),
            justify="left",
            wraplength=860,
            anchor="w",
        ).pack(anchor="w", fill="x", pady=(4 if option_heading else 0, 0))

    # Controls — key chips + descriptions.
    ctrl = _section(content, "Controls")
    grid = tk.Frame(ctrl, bg=PAGE_BG)
    grid.pack(fill="x")
    for line in control_lines:
        key, desc = _split_control_line(line)
        if _is_mouse_click_key(key):
            key = "mouse click"
            low = desc.lower()
            if low.startswith("mouse click"):
                desc = desc[len("mouse click") :].lstrip(" —-")
        row = tk.Frame(grid, bg=PAGE_BG)
        row.pack(fill="x", pady=4)
        key_lbl = tk.Label(
            row,
            text=key,
            bg=KEY_BG,
            fg=KEY_FG,
            font=("Sans", 11, "bold"),
            padx=10,
            pady=5,
        )
        key_lbl.pack(side="left")
        if desc:
            tk.Label(
                row,
                text=desc,
                bg=PAGE_BG,
                fg=MUTED,
                font=("Sans", 13),
                justify="left",
                wraplength=700,
                anchor="w",
            ).pack(side="left", padx=(12, 0), fill="x", expand=True)

    if notes:
        note_sec = _section(content, "Notes")
        for item in notes:
            if isinstance(item, dict):
                _pack_highlighted(
                    note_sec, str(item.get("text") or ""), list(item.get("red") or [])
                )
            else:
                tk.Label(
                    note_sec,
                    text=str(item),
                    bg=PAGE_BG,
                    fg=INK,
                    font=("Sans", 13),
                    justify="left",
                    wraplength=860,
                    anchor="w",
                ).pack(anchor="w", fill="x")

    # ---- Footer ----
    footer_wrap = tk.Frame(dialog, bg=PAGE_BG)
    footer_wrap.pack(fill="x")
    tk.Frame(footer_wrap, bg=LINE, height=1).pack(fill="x")
    footer = tk.Frame(footer_wrap, bg=PAGE_BG)
    footer.pack(fill="x", padx=40, pady=16)

    tk.Label(
        footer,
        text="Enter to start   ·   Esc to cancel",
        bg=PAGE_BG,
        fg=FAINT,
        font=("Sans", 11),
    ).pack(side="left")

    def _finish(ok: bool):
        result["ok"] = bool(ok)
        dialog.grab_release()
        dialog.destroy()

    cancel = tk.Button(
        footer,
        text="Not now",
        command=lambda: _finish(False),
        bg=CANCEL_BG,
        activebackground=CANCEL_ACTIVE,
        fg=CANCEL_FG,
        activeforeground=CANCEL_FG,
        relief="flat",
        borderwidth=0,
        font=("Sans", 13, "bold"),
        padx=18,
        pady=11,
        cursor="hand2",
        highlightthickness=0,
    )
    cancel.pack(side="right", padx=(10, 0))

    start = tk.Button(
        footer,
        text="Start task  →",
        command=lambda: _finish(True),
        bg=START_BG,
        activebackground=START_ACTIVE,
        fg=START_FG,
        activeforeground=START_FG,
        relief="flat",
        borderwidth=0,
        font=("Sans", 13, "bold"),
        padx=22,
        pady=11,
        cursor="hand2",
        highlightthickness=0,
    )
    start.pack(side="right")

    dialog.bind("<Escape>", lambda _e: _finish(False))
    dialog.bind("<Return>", lambda _e: _finish(True))
    dialog.protocol("WM_DELETE_WINDOW", lambda: _finish(False))
    dialog.after(50, _paint_hero)
    start.focus_set()
    parent.wait_window(dialog)
    return bool(result["ok"])

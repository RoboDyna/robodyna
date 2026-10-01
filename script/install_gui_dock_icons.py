#!/usr/bin/env python3
"""Install RoboDyna GUI icons for the Ubuntu dock (GNOME .desktop + hicolor).

Tk ``iconphoto`` alone does not change the Ubuntu taskbar icon. GNOME matches a
running window's WM_CLASS to a ``.desktop`` ``StartupWMClass`` and uses that
entry's ``Icon=``.

Run once (or rely on the GUIs, which install on launch)::

    python script/install_gui_dock_icons.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "interactive"))

from _task_briefing import GUI_WM_CLASS, install_ubuntu_dock_icon, suite_icon  # noqa: E402

GUIS = (
    (
        "gui",
        "robodyna-gui",
        "RoboDyna",
        "Shared launcher for the base, household, and experiment GUIs",
        "robodyna_gui.py",
    ),
    (
        "interactive",
        "robodyna-interactive-tasks",
        "RoboDyna Base Tasks",
        "Dynamic interactive task launcher",
        "base_task_gui.py",
    ),
    (
        "household",
        "robodyna-household-tasks",
        "RoboDyna Household Tasks",
        "Household interactive task launcher",
        "household_task_gui.py",
    ),
    (
        "experiment",
        "robodyna-experiment",
        "RoboDyna Human Experiment",
        "Human-experiment task launcher",
        "experiment_gui.py",
    ),
)


def main() -> int:
    py = sys.executable
    paths = []
    for suite, desktop_id, name, comment, script in GUIS:
        icon_path, icon_name = suite_icon(suite)
        paths.append(
            install_ubuntu_dock_icon(
                desktop_id=desktop_id,
                name=name,
                comment=comment,
                script_path=ROOT / "interactive" / script,
                wm_class=GUI_WM_CLASS[suite],
                python_exe=py,
                icon_path=icon_path,
                icon_name=icon_name,
            )
        )
    for p in paths:
        print(f"installed {p}")
    print("Restart the GUI (or log out/in if the dock still shows a generic icon).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

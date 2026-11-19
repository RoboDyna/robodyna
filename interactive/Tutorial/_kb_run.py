"""Shared launcher for keyboard+mouse tutorial parts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_INTERACTIVE = _HERE.parent
sys.path.insert(0, str(_INTERACTIVE))
sys.path.insert(0, str(_HERE))
from _interactive_common import (  # noqa: E402
    add_robot_motion_arg,
    bootstrap_repo,
    configure_task,
    parse_control_arg,
    prepare_interactive_control,
    print_banner,
    print_instructions,
    print_mode_controls,
)

bootstrap_repo()

PART_TITLES = {
    "buttons": "Buttons",
    "placement": "Placement",
    "base": "Base",
    "household": "Household",
}

CONTROLS = """
  Keyboard substages ignore the mouse; mouse substages ignore the keyboard.
  Esc               close the viewer
"""


def main(part: str | None = None) -> int:
    parser = argparse.ArgumentParser(description="Keyboard tutorial viewer")
    parser.add_argument("--config", default="demo_dynamic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--part",
        default=part or "buttons",
        choices=("buttons", "placement", "base", "household", "1", "2", "3", "4"),
    )
    parser.add_argument("--suite", default="base", choices=("base", "household"))
    add_robot_motion_arg(parser)
    args = parser.parse_args()
    parse_control_arg(args)
    raw = str(args.part)
    part_id = {
        "1": "buttons",
        "2": "placement",
        "3": "base",
        "4": "household",
    }.get(raw, raw)
    title = PART_TITLES[part_id]

    from envs.tutorial_keyboard import tutorial_keyboard

    env = tutorial_keyboard()
    env.setup_demo(
        **configure_task("tutorial_keyboard", args.config, args.seed, use_robot=False)
    )
    prepare_interactive_control(env, "keyboard+mouse")

    print_mode_controls(
        "tutorial_keyboard",
        "keyboard+mouse",
        keyboard=CONTROLS,
        robot=CONTROLS,
    )
    print_banner(
        f"Keyboard tutorial · {title}",
        [
            f"config: {args.config}  |  seed: {args.seed}  |  suite: {args.suite}",
            "Follow the top-left overlay. Keyboard and mouse are taught separately.",
            "Esc — close the viewer",
        ],
    )
    print_instructions("Follow the overlay instructions.")
    from _kb_coach import run_keyboard_part

    return run_keyboard_part(env, part_id)


if __name__ == "__main__":
    raise SystemExit(main())

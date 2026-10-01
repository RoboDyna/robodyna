#!/usr/bin/env python3
"""Interactive household task: cook_food."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:
    from _interactive_common import make_parser, run_task

KEYBOARD = """
  Food starts in the pan.
  Click the cooktop knob or press Space to turn the stove on/off.
"""
ROBOT = """
  Space opens/closes the gripper only — no automatic food grasp.
  Grasp food with teleop, then close on the cooktop knob and twist to shut off.
"""

if __name__ == "__main__":
    a = make_parser("cook_food", __doc__)
    result = run_task("cook_food", a.parse_args(), KEYBOARD, ROBOT)
    raise SystemExit(10 if result is False else 0 if result is True else 2)

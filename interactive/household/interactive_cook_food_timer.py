#!/usr/bin/env python3
"""Interactive household task: cook_food_timer."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:
    from _interactive_common import make_parser, run_task

KEYBOARD = """
  Food starts in the pan. The pie timer follows the stove.
  Click the cooktop knob or press Space to turn the stove on/off.
"""
ROBOT = """
  Space opens/closes the gripper only — no automatic food grasp.
  Grasp food with teleop, then twist the cooktop knob to shut off; timer runs while cooking.
"""

if __name__ == "__main__":
    a = make_parser("cook_food_timer", __doc__)
    result = run_task("cook_food_timer", a.parse_args(), KEYBOARD, ROBOT)
    raise SystemExit(10 if result is False else 0 if result is True else 2)

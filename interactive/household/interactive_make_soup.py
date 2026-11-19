#!/usr/bin/env python3
"""Interactive household task: make_soup."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:
    from _interactive_common import make_parser, run_task

KEYBOARD = """
  Click to place the board (top-center at the click, 2 cm above the pot rim).
  Left / Right arrows tilt the board.
"""
ROBOT = """
  Space opens/closes the gripper only — no automatic board grasp.
  Close on the board handle to pick it up, carry over the pot, tip with F/G to pour.
"""

if __name__ == "__main__":
    a = make_parser("make_soup", __doc__)
    result = run_task("make_soup", a.parse_args(), KEYBOARD, ROBOT)
    raise SystemExit(10 if result is False else 0 if result is True else 2)

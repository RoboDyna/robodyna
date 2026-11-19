#!/usr/bin/env python3
"""Interactive household task: clean_table."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:
    from _interactive_common import make_parser, run_task

KEYBOARD = """
  Sponge hovers 5 cm above the table.
  Click the table to press the pad onto that spot (stains wipe as usual).
"""
ROBOT = """
  Space opens/closes the gripper only — no automatic sponge grasp.
  Grippers start open — select the sponge-side arm (2 if mug is right).
  Approach the small top cube open, then pinch; lower the yellow pad onto a stain to wipe.
"""

if __name__ == "__main__":
    a = make_parser("clean_table", __doc__)
    result = run_task("clean_table", a.parse_args(), KEYBOARD, ROBOT)
    raise SystemExit(10 if result is False else 0 if result is True else 2)

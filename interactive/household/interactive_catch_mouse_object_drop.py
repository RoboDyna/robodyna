#!/usr/bin/env python3
"""Interactive household task: catch_mouse_object_drop."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:
    from _interactive_common import make_parser, run_task

KEYBOARD = """
  Object drop starts as soon as the viewer is ready.
  Click the table once to place the basket; later clicks do nothing.
"""
ROBOT = """
  Space opens/closes the gripper only — no automatic basket grasp.
  Close on the basket handle, carry under the landing, then open to place.
"""

if __name__ == "__main__":
    a = make_parser("catch_mouse_object_drop", __doc__)
    result = run_task("catch_mouse_object_drop", a.parse_args(), KEYBOARD, ROBOT)
    raise SystemExit(10 if result is False else 0 if result is True else 2)

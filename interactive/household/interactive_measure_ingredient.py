#!/usr/bin/env python3
"""Interactive household task: measure_ingredient."""
try:
    from ._interactive_common import make_parser, print_instructions, run_task
except ImportError:
    from _interactive_common import make_parser, print_instructions, run_task

KEYBOARD = """
  Click the table to move the jar there. Space turns the nozzle on/off.
  Success is checked after the key turns OFF.
"""
ROBOT = """
  Push the jar under the nozzle, then press the green key to fill.
  Press again to stop at the target ring. Success is scored after OFF.
  Spilling outside the jar fails.
"""


def _post_setup(env):
    # Allow physical key presses for the whole interactive session.
    env._ignore_tab = False
    env._interactive_universal_controls = True
    # Viewer treats a solid transmission cylinder as a filled volume — swap to
    # a hollow glass shell so the rising oil level is visible from outside.
    env.use_viewer_hollow_jar()
    # Unlock jar so the closed gripper can shove it under the nozzle.
    env.enable_interactive_jar_push()
    print_instructions("Push jar under nozzle; Z-press green key.")


if __name__ == "__main__":
    a = make_parser("measure_ingredient", __doc__)
    args = a.parse_args()
    # Keep real transmission glass on the jar (do NOT force plain_glass).
    # Caller may still pass plain_glass=true for other materials if needed.
    result = run_task(
        "measure_ingredient", args, KEYBOARD, ROBOT, post_setup=_post_setup
    )
    raise SystemExit(10 if result is False else 0 if result is True else 2)

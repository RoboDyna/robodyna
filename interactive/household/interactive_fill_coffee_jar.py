#!/usr/bin/env python3
"""Interactive household task: fill_coffee_jar."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:
    from _interactive_common import make_parser, run_task

KEYBOARD = """
  Press 1 / 2 / 3 / 4 for fill force levels 1–4.
"""
ROBOT = """
  Select one arm, move above the blue key, then lower with Q to press and raise with E to release.
  Coffee dispenses only from measured key pressure.
"""


def _post_setup(env):
    # Viewer treats a solid transmission cylinder as a filled volume — swap to
    # a hollow glass shell so the rising coffee fill column is visible from outside.
    env.use_viewer_hollow_jar()


if __name__ == "__main__":
    a = make_parser("fill_coffee_jar", __doc__)
    args = a.parse_args()
    # Interactive SAPIEN viewer: plain alpha glass for dispenser panels.
    if not any(str(x).startswith("plain_glass=") for x in (args.task_arg or [])):
        args.task_arg.append("plain_glass=true")
    result = run_task(
        "fill_coffee_jar", args, KEYBOARD, ROBOT, post_setup=_post_setup
    )
    raise SystemExit(10 if result is False else 0 if result is True else 2)

#!/usr/bin/env python3
"""Interactive household task: pour_beer."""
try:
    from ._interactive_common import make_parser, print_instructions, run_task
except ImportError:
    from _interactive_common import make_parser, print_instructions, run_task

KEYBOARD = """
  Click and hold the red circle on the tap, or hold Space, to pour. Release to stop.
  Click the finish bell to score.
"""
ROBOT = """
  Lower onto the fancy tap button and hold to pour; lift to stop.
  Foam % of the stream rises the longer you hold. Overflow fails.
  When finished pouring, click the finish bell beside the tap to score.
  EE Z is capped over the key so Q cannot dive through it.
"""


def _post_setup(env):
    # Viewer treats a solid transmission cylinder as a filled volume — swap to
    # a hollow glass shell so the rising beer level is visible from outside.
    env.use_viewer_hollow_mug()
    print_instructions(
        "Hold button to pour; click finish bell to score; EE Z capped on key."
    )


if __name__ == "__main__":
    a = make_parser("pour_beer", __doc__)
    args = a.parse_args()
    result = run_task(
        "pour_beer", args, KEYBOARD, ROBOT, post_setup=_post_setup
    )
    raise SystemExit(10 if result is False else 0 if result is True else 2)

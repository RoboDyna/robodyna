#!/usr/bin/env python3
"""Interactive household task: trap_bug."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:  # direct ``python interactive/household/...py`` execution
    from _interactive_common import make_parser, run_task

KEYBOARD = """
  Bug starts running as soon as the viewer is ready.
  Click the table: trap teleports 4 cm above that spot and drops.
"""
ROBOT = """
  Bug starts running as soon as the viewer is ready.
  Select the trap-side arm (1/2), approach from above with the gripper open,
  close until the fingers pinch the outer walls (about half-open — not crushed
  shut) — trap latches and lifts. Space again to fully open and drop.
"""

if __name__ == "__main__":
    a = make_parser("trap_bug", __doc__)
    args = a.parse_args()
    # Interactive SAPIEN viewer: simple alpha box (no glass transmission/IOR).
    if not any(str(x).startswith("plain_trap=") for x in (args.task_arg or [])):
        args.task_arg.append("plain_trap=true")

    result = run_task("trap_bug", args, KEYBOARD, ROBOT)
    raise SystemExit(10 if result is False else 0 if result is True else 2)

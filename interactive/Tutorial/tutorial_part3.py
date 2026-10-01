#!/usr/bin/env python3
"""Tutorial part 3 — grasp, hold-button, on/off switch, then push a box."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _run import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(part=3))

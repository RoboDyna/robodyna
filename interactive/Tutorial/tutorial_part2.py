#!/usr/bin/env python3
"""Tutorial part 2 — base teleop controls (arrows, E/Q, R/T, F/G, Space)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _run import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(part=2))

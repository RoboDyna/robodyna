#!/usr/bin/env python3
"""Keyboard tutorial — Base (bowl+trigger, two-key gate, stick aim)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _kb_run import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(part="base"))

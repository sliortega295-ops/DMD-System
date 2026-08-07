#!/usr/bin/env python3
"""Repository-local entrypoint; equivalent to the installed ``distillctl`` command."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from distillschedule.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

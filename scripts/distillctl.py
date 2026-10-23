#!/usr/bin/env python3
"""Repository-local entrypoint; equivalent to the installed ``distillctl`` command."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
# runtime/ is an intentionally retained and packaged compatibility namespace.
# Match the installed command even when invoked from another working directory.
for path in (ROOT, SRC):
    if str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))

from distillschedule.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

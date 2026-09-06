#!/usr/bin/env python3
"""Thin repo entry point for the canonical workload registry; no model imports."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from distillschedule.workloads.best_known.cli import main
if __name__ == "__main__":
    raise SystemExit(main())

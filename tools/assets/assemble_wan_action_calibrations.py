#!/usr/bin/env python3
"""Assemble existing native component receipts. Never launch or repeat GPU runs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from distillschedule.search.wan_action_model_adapter import assemble_calibrations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-result", type=Path, action="append", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assemble_calibrations(args.calibration_result, output=args.output,
                                   expected_head=args.expected_head, root=ROOT)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

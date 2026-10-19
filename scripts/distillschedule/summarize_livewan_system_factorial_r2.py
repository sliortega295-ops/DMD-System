#!/usr/bin/env python3
"""Collect one R2 native allocation, never synthetic timing as a speed result."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from distillschedule.runtime.livewan_crossscale_worker import FormalTrace, atomic_json
from distillschedule.runtime.livewan_system_factorial_collect import collect


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--arm", choices=("baseline", "only_A", "full", "only_A_with_capacity_offload"), required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    try:
        value = collect(a.run_root, arm=a.arm, expected_head=a.expected_head,
                        trace=FormalTrace.load(a.trace, expected_cycles=4))
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        value = dict(status="FAIL", errors=[repr(error)], native_descriptive_timing_eligible=False,
                     cycle_2_seconds=None, C2_C4_mean_seconds=None)
    atomic_json(a.output, value)
    return 0 if value["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

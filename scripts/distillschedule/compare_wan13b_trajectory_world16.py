#!/usr/bin/env python3
"""Compare completed Wan1.3B world16 trajectory arms and their numeric anchors."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from distillschedule.evidence.trajectory_coarse import (
    compare_trajectories,
    load_trajectory_jsonl,
    write_trajectory_report,
)
from distillschedule.evidence.trajectory_state_sketch import (
    ALL_RANKS_SCHEMA_VERSION as STATE_SKETCH_SCHEMA_VERSION,
    compare_numeric_state_sketches,
)


COMPONENTS = (
    "generator_trainable",
    "critic_trainable",
    "generator_adam",
    "critic_adam",
    "generator_ema",
)


def _anchor_cycles(cycles: int) -> tuple[int, ...]:
    if cycles == 1:
        return (1,)
    if cycles == 2:
        return (1, 2)
    if cycles == 5:
        return (1, 5)
    if cycles == 50:
        return (1, 10, 25, 50)
    raise ValueError("trajectory cycles must be one of 1, 2, 5, or 50")


def _load_rows(run_root: Path) -> list[Any]:
    rows = []
    for rank in range(16):
        rows.extend(
            load_trajectory_jsonl(
                run_root / "execution" / "trajectory" / f"rank_{rank:02d}.jsonl"
            )
        )
    return rows


def _load_anchor(run_root: Path, cycle: int) -> dict[str, Any]:
    path = (
        run_root
        / "execution"
        / "trajectory"
        / "anchors"
        / f"cycle_{cycle:02d}_global.json"
    )
    value = json.loads(path.read_text())
    if value.get("schema_version") != "distillschedule.wan13b_trajectory_state_anchor.v2":
        raise ValueError(f"unsupported Wan trajectory anchor schema at {path}")
    recorded = value.get("anchor_sha256")
    canonical = dict(value)
    canonical.pop("anchor_sha256", None)
    actual = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if recorded != actual:
        raise ValueError(f"numeric state anchor SHA mismatch at {path}")
    if value.get("cycle") != cycle or set(value.get("components", {})) != set(COMPONENTS):
        raise ValueError(f"numeric state anchor identity/inventory changed at {path}")
    for component in COMPONENTS:
        sketch = value["components"][component].get("numeric_sketch", {})
        if sketch.get("schema_version") != STATE_SKETCH_SCHEMA_VERSION:
            raise ValueError(f"{component} has unsupported numeric state schema at {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run-root", type=Path, required=True)
    parser.add_argument("--candidate-run-root", type=Path, required=True)
    parser.add_argument("--cycles", type=int, choices=(1, 2, 5, 50), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-plot", action="store_true",
                        help="Write all numerical evidence without importing matplotlib; gates unchanged")
    args = parser.parse_args()

    reference_rows = _load_rows(args.reference_run_root)
    candidate_rows = _load_rows(args.candidate_run_root)
    state_comparisons = []
    anchors = _anchor_cycles(args.cycles)
    for cycle in anchors:
        reference = _load_anchor(args.reference_run_root, cycle)
        candidate = _load_anchor(args.candidate_run_root, cycle)
        for component in COMPONENTS:
            state_comparisons.append(
                compare_numeric_state_sketches(
                    reference["components"][component]["numeric_sketch"],
                    candidate["components"][component]["numeric_sketch"],
                    cycle=cycle,
                    component=component,
                )
            )

    result = compare_trajectories(
        reference_rows,
        candidate_rows,
        state_comparisons=state_comparisons,
        expected_cycles=args.cycles,
        expected_world_size=16,
        comparison_kind="semantic",
        required_state_cycles=anchors,
    )
    paths = write_trajectory_report(result, args.output_dir, render_png=not args.no_plot)
    print(
        json.dumps(
            {
                "status": result["status"],
                "contract": result["contract"],
                "plot_status": "NOT_RUN_REQUESTED" if args.no_plot else "PASS",
                "outputs": {key: str(path) for key, path in paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.get("complete") and result.get("contract", {}).get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

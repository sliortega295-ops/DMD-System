#!/usr/bin/env python3
"""Replay known Full5 anchors and simulate candidate plans from a profile DB."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

from distillschedule.planner.calibration import calibration_coverage
from distillschedule.planner.profile_db import ProfileDB
from distillschedule.planner.schema import load_plan, load_profile
from distillschedule.planner.simulator import PlanSimulator, replay_decomposed_observation


def convert(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: convert(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, tuple):
        return [convert(item) for item in value]
    if isinstance(value, dict):
        return {str(key): convert(item) for key, item in value.items()}
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--plan", action="append", default=[])
    parser.add_argument("--no-interpolation", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    profile = load_profile(args.profile)
    db = ProfileDB(profile)
    simulator = PlanSimulator(db)
    result = {
        "schema_version": "distillschedule.planner_simulation_report.v0",
        "profile": str(Path(args.profile).resolve()),
        "known_plan_component_replays": [
            replay_decomposed_observation(value)
            for value in profile.plans
            if value.phase_seconds
        ],
        "calibration_coverage": calibration_coverage(db),
        "simulations": [],
        "claim_boundary": {
            "component_replay_is_not_leave_one_out_validation": True,
            "unseen_plan_prediction_requires_primitive_coverage": True,
            "profiled_runs_are_not_formal_timing": True,
        },
    }
    for path in args.plan:
        plan = load_plan(path)
        result["simulations"].append(
            convert(
                simulator.simulate(
                    plan,
                    allow_interpolation=not args.no_interpolation,
                )
            )
        )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Construct the official Generator on CPU and census ZeRO-3 granularity."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.critic_pool_helios import build_helios_pool_model
from runtime.zero3_module_granularity import (
    census_zero3_module_granularity,
    selected_preorder_leaf_records,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-root", required=True)
    parser.add_argument("--precast-generator-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=int, action="append", default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model, metadata = build_helios_pool_model(
        role="generator",
        model_root=args.generator_root,
        subfolder="transformer_ode",
        precast_model_root=args.precast_generator_root,
    )
    records = census_zero3_module_granularity(model)
    finite = [record for record in records if record.granularity < sys.maxsize]
    by_type: dict[str, list[int]] = defaultdict(list)
    for record in finite:
        by_type[record.module_type].append(record.granularity)
    thresholds = sorted(set(args.threshold))
    payload = {
        "schema_version": "distillschedule.zero3_module_granularity.v1",
        "status": "PASS",
        "hostname": platform.node(),
        "pid": os.getpid(),
        "generator_metadata": metadata,
        "record_count": len(records),
        "finite_record_count": len(finite),
        "minimum_effective_threshold": min(
            (record.granularity for record in finite), default=None
        ),
        "type_summary": {
            key: {
                "count": len(values),
                "min": min(values),
                "median": sorted(values)[len(values) // 2],
                "max": max(values),
            }
            for key, values in sorted(by_type.items())
        },
        "thresholds": {
            str(threshold): {
                "selected_count": len(selected := selected_preorder_leaf_records(
                    records, threshold=threshold
                )),
                "selected_parameter_numel_sum": sum(
                    record.parameter_numel for record in selected
                ),
                "selected": [record.to_dict() for record in selected],
            }
            for threshold in thresholds
        },
        "records": [
            record.to_dict()
            for record in sorted(records, key=lambda value: (value.granularity, value.name))
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded)
    print(json.dumps({
        "status": "PASS",
        "output": str(args.output),
        "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "minimum_effective_threshold": payload["minimum_effective_threshold"],
        "finite_record_count": len(finite),
    }, indent=2))


if __name__ == "__main__":
    main()

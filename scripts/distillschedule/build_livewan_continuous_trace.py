#!/usr/bin/env python3
"""Derive an immutable three-cycle trace from the validated Full5 trace.

Cycle zero preserves every registered random stream from the input trace.
Cycles one and two intentionally reuse the same 80 source samples and control
shapes, but receive fresh operation-keyed RNG streams.  Reuse is explicit in
the schema so this diagnostic cannot be mistaken for a 240-sample training run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def derived_seed(base_seed: int, operation_key: str, stream: str) -> int:
    payload = f"{base_seed}/{operation_key}/{stream}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=3)
    args = parser.parse_args()
    if args.cycles != 3:
        raise SystemExit("C21 is preregistered for exactly three continuous cycles")
    source = json.loads(args.source_trace.resolve().read_text(encoding="utf-8"))
    if int(source.get("steps", -1)) != 5 or len(source.get("rows", ())) != 80:
        raise SystemExit("source is not the validated world16 Full5 trace")
    rows_by_address = {
        (int(row["logical_step"]), int(row["logical_slot"])): row
        for row in source["rows"]
    }
    if len(rows_by_address) != 80:
        raise SystemExit("source trace is not a step-slot bijection")
    base_seed = int(source["seed"])
    rows: list[dict[str, Any]] = []
    for cycle in range(args.cycles):
        for cycle_step in range(5):
            for slot in range(16):
                source_row = dict(rows_by_address[(cycle_step, slot)])
                global_step = cycle * 5 + cycle_step
                operation_key = (
                    f"cycle={cycle}/step={cycle_step}/"
                    f"global_step={global_step}/slot={slot}"
                )
                if cycle == 0:
                    seeds = dict(source_row["seeds"])
                else:
                    seeds = {
                        name: derived_seed(base_seed, operation_key, name)
                        for name in sorted(source_row["seeds"])
                    }
                source_row.update(
                    logical_step=global_step,
                    cycle_index=cycle,
                    cycle_step=cycle_step,
                    operation_key=operation_key,
                    seeds=seeds,
                    generator_update=cycle_step == 0,
                    critic_update=True,
                )
                rows.append(source_row)
    value = {
        key: source[key]
        for key in (
            "livewan_commit", "wan21_commit", "seed", "world_size",
            "global_batch", "prompts", "source_sample_count",
        )
    }
    value.update(
        schema_version=2,
        kind="systems_only_continuous_operation_keyed_livewan_sfdmd",
        claim_boundary=(
            "Three continuous Full5 cycles for pipeline steady-state diagnosis; "
            "cycle0 preserves the validated trace RNG, later cycles reuse source "
            "samples with fresh operation-keyed RNG; not a convergence result."
        ),
        steps=args.cycles * 5,
        cycles=args.cycles,
        cycle_length=5,
        generator_updates=args.cycles,
        critic_updates=args.cycles * 5,
        source_reuse_across_cycles=True,
        unique_samples_consumed=80,
        rank_qualified_row_count=len(rows),
        rows=rows,
    )
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value["trace_sha256"] = hashlib.sha256(canonical).hexdigest()
    atomic_json(args.output.resolve(), value)
    print(json.dumps({
        "status": "PASS", "output": str(args.output.resolve()),
        "cycles": args.cycles, "rows": len(rows),
        "trace_sha256": value["trace_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

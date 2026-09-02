#!/usr/bin/env python3
"""Derive a source-reuse global-B32 systems trace from validated C21 B16.

The first 16 slots preserve C21 exactly. Slots 16--31 reuse the corresponding
source sample but receive distinct operation keys and RNG streams. This is a
pipeline-throughput diagnostic, not a training-quality 160-sample cycle.
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
    args = parser.parse_args()
    source = json.loads(args.source_trace.resolve().read_text(encoding="utf-8"))
    recorded_source_sha = str(source.get("trace_sha256", ""))
    source_without_sha = dict(source)
    source_without_sha.pop("trace_sha256", None)
    actual_source_sha = hashlib.sha256(
        json.dumps(source_without_sha, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if recorded_source_sha != actual_source_sha:
        raise SystemExit("source C21 trace payload SHA mismatch")
    if (
        int(source.get("cycles", -1)) != 3
        or int(source.get("global_batch", -1)) != 16
        or len(source.get("rows", ())) != 240
        or source.get("source_reuse_across_cycles") is not True
    ):
        raise SystemExit("source is not the validated continuous C21 B16 trace")
    rows_by_address = {
        (int(row["logical_step"]), int(row["logical_slot"])): row
        for row in source["rows"]
    }
    if len(rows_by_address) != 240:
        raise SystemExit("source C21 trace is not a step-slot bijection")
    base_seed = int(source["seed"])
    rows: list[dict[str, Any]] = []
    for global_step in range(15):
        cycle = global_step // 5
        cycle_step = global_step % 5
        for slot in range(32):
            source_slot = slot % 16
            row = dict(rows_by_address[(global_step, source_slot)])
            operation_key = (
                f"cycle={cycle}/step={cycle_step}/"
                f"global_step={global_step}/slot={slot}"
            )
            if slot >= 16:
                row["seeds"] = {
                    name: derived_seed(base_seed, operation_key, name)
                    for name in sorted(row["seeds"])
                }
            row.update(
                logical_slot=slot,
                operation_key=operation_key,
                source_reuse_within_step=slot >= 16,
                source_slot=source_slot,
            )
            rows.append(row)
    value = {
        key: source[key]
        for key in (
            "livewan_commit",
            "wan21_commit",
            "seed",
            "world_size",
            "prompts",
            "source_sample_count",
            "steps",
            "cycles",
            "cycle_length",
            "generator_updates",
            "critic_updates",
        )
    }
    value.update(
        schema_version=3,
        kind="systems_only_continuous_b32_source_reuse_livewan_sfdmd",
        claim_boundary=(
            "Three continuous global-B32 systems cycles. Each step reuses the "
            "16 validated source samples once with distinct operation-keyed RNG; "
            "valid only for pipeline bubble/throughput diagnosis, not convergence."
        ),
        global_batch=32,
        source_reuse_across_cycles=True,
        source_reuse_within_step=True,
        unique_samples_consumed=80,
        unique_source_samples_per_cycle=80,
        rank_qualified_samples_per_cycle=160,
        rank_qualified_row_count=len(rows),
        training_quality_claimable=False,
        source_trace_sha256=actual_source_sha,
        rows=rows,
    )
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value["trace_sha256"] = hashlib.sha256(canonical).hexdigest()
    atomic_json(args.output.resolve(), value)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(args.output.resolve()),
                "rows": len(rows),
                "trace_sha256": value["trace_sha256"],
                "source_reuse_within_step": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

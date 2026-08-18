#!/usr/bin/env python3
"""Build a frozen MiniMax native checkpoint-boundary offload manifest.

The input is a no-copy block-input census from one rank/role.  Selection is a
contiguous prefix so the earliest boundaries are retained on CPU for the
longest portion of backward.  The resulting manifest is an experiment input,
not a timing result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from runtime.lightx2v_checkpoint_boundary_offload import (
    PRODUCTION_BLOCK_COUNT,
    sha256_file,
    write_boundary_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--target-bytes",
        type=int,
        default=3 * 1024**3 + 20 * 1024**2,
        help="default is approximately 3.02 GiB (2.02 GiB failed allocation + 1 GiB reserve)",
    )
    parser.add_argument("--max-blocks", type=int, default=16)
    parser.add_argument("--dmd-system-head", required=True)
    parser.add_argument("--upstream-commit", required=True)
    args = parser.parse_args()
    if args.target_bytes <= 0 or args.max_blocks <= 0:
        raise ValueError("target and max-blocks must be positive")
    rows: list[dict[str, object]] = []
    with args.census.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            rows.append(value)
    by_index: dict[int, dict[str, object]] = {}
    for row in rows:
        index = int(row["block_index"])
        if index in by_index and by_index[index] != row:
            raise RuntimeError(f"duplicate boundary {index} has different metadata")
        by_index[index] = row
    if sorted(by_index) != list(range(PRODUCTION_BLOCK_COUNT)):
        raise RuntimeError(
            "census must contain exactly contiguous block indices 0..49, got "
            f"{sorted(by_index)}"
        )
    boundaries = [by_index[index] for index in range(PRODUCTION_BLOCK_COUNT)]
    selected: list[int] = []
    cumulative = 0
    for row in boundaries[: args.max_blocks]:
        index = int(row["block_index"])
        cumulative += int(row["storage_bytes"])
        selected.append(index)
        if cumulative >= args.target_bytes:
            break
    if cumulative < args.target_bytes:
        raise RuntimeError(
            f"first {args.max_blocks} boundaries provide only {cumulative} bytes, "
            f"below target {args.target_bytes}"
        )
    output = write_boundary_manifest(
        args.output,
        boundaries=boundaries,
        selected_block_indices=selected,
        source={
            "census": str(args.census.resolve()),
            "census_sha256": sha256_file(args.census),
            "target_bytes": args.target_bytes,
            "selected_storage_bytes": cumulative,
            "max_blocks": args.max_blocks,
            "dmd_system_head": args.dmd_system_head,
            "upstream_commit": args.upstream_commit,
        },
    )
    print(json.dumps({"manifest": str(output), "selected": selected, "selected_storage_bytes": cumulative}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

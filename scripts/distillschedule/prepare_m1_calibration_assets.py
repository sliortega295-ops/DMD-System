#!/usr/bin/env python3
"""Build rank-qualified CAL-A bundles and exact-count dataset shells.

The operation trace remains the source authority. Dataset shells exist only so
the official StatefulDataLoader has one sample per rank and therefore resumes
checkpoint-10 at an epoch boundary for worlds 1/2/4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from runtime.operation_trace import OperationTrace, write_operation_trace_bundle_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    trace = OperationTrace.load(args.source_trace)
    source_files = sorted(args.source_dataset.glob("*.pt"))
    if len(source_files) < max(args.worlds):
        raise ValueError(
            f"source dataset has {len(source_files)} files; needs {max(args.worlds)}"
        )
    result: dict[str, Any] = {
        "schema_version": "distillschedule.m1_calibration_assets.v1",
        "source_trace": str(args.source_trace.resolve()),
        "trace_hash": trace.trace_hash,
        "source_dataset": str(args.source_dataset.resolve()),
        "worlds": {},
        "guardrail": (
            "Dataset files only advance the official loader. Operation-keyed CAL-A "
            "tensors remain the authoritative model inputs."
        ),
    }
    for world in args.worlds:
        trace_root = args.output_root / "traces" / f"cal_a_rank{world}"
        data_root = args.output_root / "datasets" / f"cal_a_rank{world}"
        if trace_root.exists() or data_root.exists():
            raise FileExistsError(
                f"refusing to overwrite calibration assets for world {world}"
            )
        trace_root.mkdir(parents=True)
        data_root.mkdir(parents=True)
        entries = []
        for rank in range(world):
            rank_path = trace_root / f"rank_{rank:03d}"
            relative_target = os.path.relpath(args.source_trace.resolve(), trace_root)
            rank_path.symlink_to(relative_target, target_is_directory=True)
            entries.append(
                {
                    "rank": rank,
                    "path": rank_path.name,
                    "trace_hash": trace.trace_hash,
                }
            )
        bundle = write_operation_trace_bundle_manifest(trace_root, entries)
        files = []
        for rank, source in enumerate(source_files[:world]):
            destination = data_root / f"sample_{rank:03d}.pt"
            os.link(source, destination)
            files.append(
                {
                    "rank": rank,
                    "path": str(destination),
                    "bytes": destination.stat().st_size,
                    "sha256": _sha256(destination),
                }
            )
        result["worlds"][str(world)] = {
            "trace_root": str(trace_root),
            "bundle_hash": bundle["bundle_hash"],
            "trace_hashes": [entry["trace_hash"] for entry in bundle["ranks"]],
            "dataset_root": str(data_root),
            "dataset_file_count": len(files),
            "dataset_files": files,
        }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-trace", required=True, type=Path)
    parser.add_argument("--source-dataset", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--worlds", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    if sorted(set(args.worlds)) != args.worlds or any(
        value not in {1, 2, 4} for value in args.worlds
    ):
        raise ValueError("worlds must be a sorted unique subset of 1, 2, 4")
    result = prepare(args)
    print(
        json.dumps(
            {
                "status": "PASS",
                "manifest": str(args.manifest),
                "worlds": list(result["worlds"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

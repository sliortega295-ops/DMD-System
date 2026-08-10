#!/usr/bin/env python3
"""Validate that Nsight captured the exact GAN Full5 training range."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from distillschedule.runtime.nsys_range import validate_nsys_receipts


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt-dir", required=True, type=Path)
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = validate_nsys_receipts(
            args.receipt_dir.expanduser().resolve(),
            expected_world_size=args.world_size,
        )
    except (OSError, ValueError, TypeError) as exc:
        result = {
            "schema_version": "distillschedule.gan_nsys_receipt_bundle.v1",
            "status": "FAIL",
            "diagnostic_only": True,
            "formal_speedup_eligible": False,
            "error": str(exc),
        }
        if args.output:
            atomic_json(args.output.expanduser().resolve(), result)
        print(json.dumps(result, sort_keys=True))
        return 2
    if args.output:
        atomic_json(args.output.expanduser().resolve(), result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

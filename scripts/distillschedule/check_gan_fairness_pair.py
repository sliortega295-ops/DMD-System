#!/usr/bin/env python3
"""Compare actual baseline/pipeline GAN stochastic-consumption receipts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from distillschedule.runtime.gan_fairness import compare_gan_fairness_receipts


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--pipeline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = compare_gan_fairness_receipts(
            load(args.baseline.expanduser().resolve()),
            load(args.pipeline.expanduser().resolve()),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = {
            "schema_version": "distillschedule.gan_fairness_pair.v1",
            "status": "FAIL",
            "claimable": False,
            "error": str(exc),
        }
        atomic_json(args.output.expanduser().resolve(), result)
        print(json.dumps(result, sort_keys=True))
        return 2
    result = dict(result)
    result["claimable"] = True
    atomic_json(args.output.expanduser().resolve(), result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

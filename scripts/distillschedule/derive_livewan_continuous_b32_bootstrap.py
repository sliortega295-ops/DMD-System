#!/usr/bin/env python3
"""Bind the validated LiveWan bootstrap to a B32 source-reuse trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bootstrap", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.source_bootstrap.resolve().read_text(encoding="utf-8"))
    trace = json.loads(args.trace.resolve().read_text(encoding="utf-8"))
    recorded_bootstrap_sha = str(value.get("bootstrap_sha256", ""))
    source_without_sha = dict(value)
    source_without_sha.pop("bootstrap_sha256", None)
    actual_bootstrap_sha = hashlib.sha256(
        json.dumps(source_without_sha, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if recorded_bootstrap_sha != actual_bootstrap_sha:
        raise SystemExit("source bootstrap payload SHA mismatch")
    recorded_trace_sha = str(trace.get("trace_sha256", ""))
    trace_without_sha = dict(trace)
    trace_without_sha.pop("trace_sha256", None)
    actual_trace_sha = hashlib.sha256(
        json.dumps(trace_without_sha, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if recorded_trace_sha != actual_trace_sha:
        raise SystemExit("continuous B32 trace payload SHA mismatch")
    if value.get("kind") != "systems_only_livewan_crossscale_bootstrap":
        raise SystemExit("source bootstrap kind changed")
    if (
        int(trace.get("cycles", -1)) != 3
        or int(trace.get("global_batch", -1)) != 32
        or trace.get("source_reuse_within_step") is not True
        or not trace.get("trace_sha256")
    ):
        raise SystemExit("continuous B32 trace is incomplete")
    value.pop("bootstrap_sha256", None)
    value["trace"] = {
        "path": str(args.trace.resolve()),
        "sha256": trace["trace_sha256"],
    }
    value["continuous_diagnostic"] = {
        "cycles": 3,
        "global_batch": 32,
        "source_reuse_across_cycles": True,
        "source_reuse_within_step": True,
        "training_quality_claimable": False,
        "initial_versions": dict(value["initial_versions"]),
        "expected_final_versions": {"generator": 3, "critic": 15, "ema": 0},
        "source_bootstrap_sha256": actual_bootstrap_sha,
    }
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value["bootstrap_sha256"] = hashlib.sha256(canonical).hexdigest()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=output.name + ".", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output),
                "bootstrap_sha256": value["bootstrap_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

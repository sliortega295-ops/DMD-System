#!/usr/bin/env python3
"""Bind the existing systems-only bootstrap to a continuous trace."""

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
    if value.get("kind") != "systems_only_livewan_crossscale_bootstrap":
        raise SystemExit("source bootstrap kind changed")
    cycles = int(trace.get("cycles", -1))
    if not 1 <= cycles <= 50 or not trace.get("trace_sha256"):
        raise SystemExit("continuous trace is incomplete")
    value.pop("bootstrap_sha256", None)
    value["trace"] = {
        "path": str(args.trace.resolve()),
        "sha256": trace["trace_sha256"],
    }
    value["continuous_diagnostic"] = {
        "cycles": cycles,
        "source_reuse_across_cycles": cycles > 1,
        "initial_versions": dict(value["initial_versions"]),
        "expected_final_versions": {
            "generator": cycles,
            "critic": cycles * 5,
            "ema": 0,
        },
    }
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value["bootstrap_sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=args.output.name + ".", dir=args.output.resolve().parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, args.output.resolve())
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(json.dumps({"status": "PASS", "output": str(args.output.resolve()),
                      "bootstrap_sha256": value["bootstrap_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

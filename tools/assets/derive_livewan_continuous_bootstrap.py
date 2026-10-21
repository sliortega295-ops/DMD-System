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
    parser.add_argument(
        "--source-receipt",
        type=Path,
        help=(
            "Optional immutable source receipt to bind into the derived bootstrap. "
            "When omitted, the receipt embedded in the source bootstrap is revalidated."
        ),
    )
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.source_bootstrap.resolve().read_text(encoding="utf-8"))
    trace = json.loads(args.trace.resolve().read_text(encoding="utf-8"))
    if value.get("kind") != "systems_only_livewan_crossscale_bootstrap":
        raise SystemExit("source bootstrap kind changed")
    source_item = dict(value.get("source_receipt", {}))
    source_path = (
        args.source_receipt.resolve()
        if args.source_receipt is not None
        else Path(str(source_item.get("path", ""))).expanduser().resolve()
    )
    if not source_path.is_file():
        raise SystemExit(f"source receipt does not exist: {source_path}")
    source_value = json.loads(source_path.read_text(encoding="utf-8"))
    if source_value.get("status") != "PASS":
        raise SystemExit("source receipt is not PASS")
    recorded_source_sha = str(source_value.get("source_receipt_sha256", ""))
    canonical_source = dict(source_value)
    canonical_source.pop("source_receipt_sha256", None)
    actual_source_sha = hashlib.sha256(
        json.dumps(canonical_source, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if not recorded_source_sha or recorded_source_sha != actual_source_sha:
        raise SystemExit("source receipt payload SHA mismatch")
    if args.source_receipt is None and source_item.get("sha256") != actual_source_sha:
        raise SystemExit("source bootstrap embeds a stale source receipt SHA")
    value["source_receipt"] = {
        "path": str(source_path),
        "sha256": actual_source_sha,
    }
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

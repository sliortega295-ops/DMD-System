#!/usr/bin/env python3
"""Create the immutable systems-only bootstrap shared by both placements."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def checkpoint_files(root: Path) -> list[Path]:
    rows = sorted(root.glob("diffusion_pytorch_model*.safetensors"))
    if not rows:
        raise SystemExit(f"no diffusion_pytorch_model*.safetensors under {root}")
    return rows


def inventory(root: Path) -> dict[str, Any]:
    files = checkpoint_files(root)
    rows = [
        {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in files
    ]
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {"root": str(root.resolve()), "files": rows, "sha256": hashlib.sha256(canonical).hexdigest()}


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


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--student-base", required=True, type=Path)
    value.add_argument("--critic-base", required=True, type=Path)
    value.add_argument("--teacher-base", required=True, type=Path)
    value.add_argument("--student-state", type=Path)
    value.add_argument("--trace", required=True, type=Path)
    value.add_argument("--source-receipt", required=True, type=Path)
    value.add_argument("--output", required=True, type=Path)
    value.add_argument("--critic-lora-seed", type=int, default=2026083001)
    return value


def main() -> int:
    args = parser().parse_args()
    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    source = json.loads(args.source_receipt.read_text(encoding="utf-8"))
    if trace.get("trace_sha256") is None or source.get("status") != "PASS":
        raise SystemExit("trace/source receipt is incomplete")
    student_state = None
    if args.student_state is not None:
        path = args.student_state.expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"student state does not exist: {path}")
        student_state = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    value: dict[str, Any] = {
        "schema_version": 1,
        "kind": "systems_only_livewan_crossscale_bootstrap",
        "claim_boundary": "Not an official upstream continuation checkpoint.",
        "student_base": inventory(args.student_base.expanduser().resolve()),
        "critic_base": inventory(args.critic_base.expanduser().resolve()),
        "teacher_base": inventory(args.teacher_base.expanduser().resolve()),
        "student_state": student_state,
        "critic_lora": {
            "initializer": "LiveWan.inject_lora.rank32",
            "seed": args.critic_lora_seed,
            "state": "deterministic_initializer_shared_by_both_arms",
        },
        "optimizer": {
            "generator_step": 0,
            "critic_step": 0,
            "moments": "materialize_zero_state_before_timer",
        },
        "initial_versions": {"generator": 0, "critic": 0, "ema": 0},
        "ema": {"full5_updates": 0, "validation": "separate"},
        "trace": {"path": str(args.trace.resolve()), "sha256": trace["trace_sha256"]},
        "source_receipt": {
            "path": str(args.source_receipt.resolve()),
            "sha256": source["source_receipt_sha256"],
        },
    }
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value["bootstrap_sha256"] = hashlib.sha256(canonical).hexdigest()
    atomic_json(args.output.expanduser().resolve(), value)
    print(json.dumps({"status": "PASS", "output": str(args.output.resolve()),
                      "bootstrap_sha256": value["bootstrap_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

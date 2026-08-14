#!/usr/bin/env python3
"""Chunked comparison of Candidate-B DP12 logical optimizer partitions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch


C_RANKS = (2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15)
STATE_NAMES = ("master", "exp_avg", "exp_avg_sq")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(root: Path, phase: str, rank: int) -> dict[str, Any]:
    path = root / "execution" / "state" / phase / f"rank_{rank:02d}" / "manifest.json"
    value = json.loads(path.read_text())
    expected_step = 10 if phase == "initial" else 15
    if value.get("schema_version") != "distillschedule.critic_pool_rank_state.v1":
        raise ValueError(f"{path}: schema changed")
    if value.get("phase") != phase or int(value.get("rank", -1)) != rank:
        raise ValueError(f"{path}: identity changed")
    if int(value.get("optimizer_step", -1)) != expected_step or int(value.get("scheduler_step", -1)) != expected_step:
        raise ValueError(f"{path}: version changed")
    return value


def _empty() -> dict[str, Any]:
    return {"n": 0, "ref2": 0.0, "test2": 0.0, "diff2": 0.0, "dot": 0.0, "max_abs": 0.0, "exact": 0}


def _add(acc: dict[str, Any], left: torch.Tensor, right: torch.Tensor) -> None:
    a, b = left.double(), right.double()
    difference = b - a
    acc["n"] += int(a.numel())
    acc["ref2"] += float(torch.dot(a, a))
    acc["test2"] += float(torch.dot(b, b))
    acc["diff2"] += float(torch.dot(difference, difference))
    acc["dot"] += float(torch.dot(a, b))
    acc["max_abs"] = max(acc["max_abs"], float(difference.abs().max()))
    acc["exact"] += int(torch.eq(left, right).sum())


def _finish(acc: dict[str, Any]) -> dict[str, Any]:
    return {
        **acc,
        "relative_l2": math.sqrt(acc["diff2"]) / max(math.sqrt(acc["ref2"]), 1e-300),
        "cosine": acc["dot"] / max(math.sqrt(acc["ref2"] * acc["test2"]), 1e-300),
        "exact_fraction": acc["exact"] / max(acc["n"], 1),
    }


def _compare(reference: Path, candidate: Path, phase: str) -> dict[str, Any]:
    accumulators = {name: _empty() for name in STATE_NAMES}
    input_sha_exact = True
    for rank in C_RANKS:
        left_manifest = _manifest(reference, phase, rank)
        right_manifest = _manifest(candidate, phase, rank)
        for name in STATE_NAMES:
            left_record = left_manifest["records"][name]
            right_record = right_manifest["records"][name]
            for field in ("dtype", "shape", "numel", "bytes"):
                if left_record.get(field) != right_record.get(field):
                    raise ValueError(f"{phase} rank {rank} {name}: {field} changed")
            left_path = Path(left_record["path"])
            right_path = Path(right_record["path"])
            if _sha256(left_path) != left_record["sha256"] or _sha256(right_path) != right_record["sha256"]:
                raise ValueError(f"{phase} rank {rank} {name}: payload SHA changed")
            input_sha_exact &= left_record["sha256"] == right_record["sha256"]
            numel = int(left_record["numel"])
            left = torch.from_file(str(left_path), shared=False, size=numel, dtype=torch.float32)
            right = torch.from_file(str(right_path), shared=False, size=numel, dtype=torch.float32)
            chunk = 8 * 1024 * 1024
            for lower in range(0, numel, chunk):
                length = min(chunk, numel - lower)
                _add(accumulators[name], left.narrow(0, lower, length), right.narrow(0, lower, length))
            del left, right
    return {
        "input_sha_exact": input_sha_exact,
        **{name: _finish(value) for name, value in accumulators.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    result = {
        "schema_version": "distillschedule.routeb_candidate_b_state_pair.v1",
        "reference_run": args.reference_root.name,
        "candidate_run": args.candidate_root.name,
        "critic": {
            "initial": _compare(args.reference_root, args.candidate_root, "initial"),
            "final": _compare(args.reference_root, args.candidate_root, "final"),
        },
    }
    result["status"] = "PASS" if result["critic"]["initial"]["input_sha_exact"] else "FAIL"
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

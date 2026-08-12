#!/usr/bin/env python3
"""Direct tensor comparison for Candidate-A ZeRO-1 checkpoint shards."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
from typing import Any

import torch


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def _state_tensors(value: dict[str, Any]) -> dict[str, torch.Tensor]:
    optimizer = value["optimizer_state_dict"]
    state = optimizer["base_optimizer_state"]["state"][0]
    return {
        "master": optimizer["single_partition_of_fp32_groups"][0],
        "exp_avg": state["exp_avg"],
        "exp_avg_sq": state["exp_avg_sq"],
    }


def _empty() -> dict[str, Any]:
    return {"n": 0, "ref2": 0.0, "test2": 0.0, "diff2": 0.0, "dot": 0.0, "max_abs": 0.0, "exact": 0}


def _add(acc: dict[str, Any], reference: torch.Tensor, candidate: torch.Tensor) -> None:
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ValueError("checkpoint shard tensor metadata changed")
    left, right = reference.double(), candidate.double()
    difference = right - left
    acc["n"] += reference.numel()
    acc["ref2"] += float(torch.dot(left, left))
    acc["test2"] += float(torch.dot(right, right))
    acc["diff2"] += float(torch.dot(difference, difference))
    acc["dot"] += float(torch.dot(left, right))
    acc["max_abs"] = max(acc["max_abs"], float(difference.abs().max()))
    acc["exact"] += int(torch.eq(reference, candidate).sum())


def _finish(acc: dict[str, Any]) -> dict[str, Any]:
    return {
        **acc,
        "relative_l2": math.sqrt(acc["diff2"]) / max(math.sqrt(acc["ref2"]), 1e-300),
        "cosine": acc["dot"] / max(math.sqrt(acc["ref2"] * acc["test2"]), 1e-300),
        "exact_fraction": acc["exact"] / acc["n"],
    }


def _component(reference: Path, candidate: Path, name: str) -> dict[str, Any]:
    accumulators = {key: _empty() for key in ("master", "exp_avg", "exp_avg_sq")}
    for rank in range(8):
        filename = f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
        left_path = reference / name / "pytorch_model" / filename if name else reference / "pytorch_model" / filename
        right_path = candidate / name / "pytorch_model" / filename if name else candidate / "pytorch_model" / filename
        left, right = _load(left_path), _load(right_path)
        left_state, right_state = _state_tensors(left), _state_tensors(right)
        for key in accumulators:
            _add(accumulators[key], left_state[key], right_state[key])
        del left, right, left_state, right_state
        gc.collect()
    return {key: _finish(value) for key, value in accumulators.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    result = {
        "schema_version": "distillschedule.zero1_checkpoint_pair.v2",
        "generator": _component(args.reference, args.candidate, ""),
        "critic": _component(args.reference, args.candidate, "critic"),
    }
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

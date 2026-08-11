#!/usr/bin/env python3
"""Fail closed on all cheap N2 inputs before constructing a 14B model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from runtime.critic_pool_checkpoint import PoolTargetPackage
from runtime.gst_pool_contract import GeneratorOffloadCandidate, NegativePromptAsset
from runtime.gst_pool_trace import validate_gst_trace_bundle
from runtime.precast_checkpoint import (
    HELIOS_KEEP_IN_FP32_MODULES,
    validate_precast_checkpoint,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("generator", "score"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--contract-digest", required=True)
    parser.add_argument("--trace-bundle", type=Path, required=True)
    parser.add_argument("--expected-trace-bundle-hash", required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-subfolder", required=True)
    parser.add_argument("--precast-root", type=Path, required=True)
    parser.add_argument("--target-package", type=Path, required=True)
    parser.add_argument("--target-native-root", type=Path)
    parser.add_argument("--negative-prompt-asset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _native_generator_inventory(root: Path) -> dict[str, Any]:
    checkpoint = root / "checkpoint-11" / "pytorch_model"
    model_state = checkpoint / "mp_rank_00_model_states.pt"
    optimizer_states = sorted(
        checkpoint.glob("bf16_zero_pp_rank_*_mp_rank_00_optim_states.pt")
    )
    if not model_state.is_file() or model_state.stat().st_size <= 0:
        raise ValueError("target native package lacks the Generator model state")
    if len(optimizer_states) != 8 or any(path.stat().st_size <= 0 for path in optimizer_states):
        raise ValueError("target native package lacks eight Generator ZeRO-1 shards")
    return {
        "checkpoint": str(checkpoint),
        "model_state_bytes": model_state.stat().st_size,
        "optimizer_shard_count": len(optimizer_states),
        "optimizer_shard_bytes": [path.stat().st_size for path in optimizer_states],
    }


def main() -> int:
    args = _parser().parse_args()
    candidate = GeneratorOffloadCandidate.load(
        args.candidate_manifest,
        expected_contract_digest=args.contract_digest,
    )
    if candidate.run_id != args.run_id:
        raise ValueError("input validation run ID differs from candidate")
    trace = validate_gst_trace_bundle(args.trace_bundle)
    if trace["bundle_hash"] != args.expected_trace_bundle_hash:
        raise ValueError("rank-qualified trace-bundle identity changed")
    package = PoolTargetPackage.load(args.target_package)
    precast = validate_precast_checkpoint(
        args.precast_root,
        expected_target_dtype=torch.bfloat16,
        expected_keep_in_fp32_modules=HELIOS_KEEP_IN_FP32_MODULES,
        expected_source_dir=args.model_root.resolve() / args.model_subfolder,
        verify_shard_hashes=False,
    )
    role_specific: dict[str, Any]
    if args.role == "generator":
        if args.target_native_root is None:
            raise ValueError("Generator validation requires a native G3 package")
        role_specific = {
            "native_generator": _native_generator_inventory(
                args.target_native_root.resolve()
            )
        }
    else:
        if args.negative_prompt_asset is None:
            raise ValueError("Score validation requires the official negative prompt")
        negative = NegativePromptAsset.load(args.negative_prompt_asset)
        role_specific = {
            "negative_prompt_manifest_hash": negative.manifest["manifest_hash"],
            "negative_prompt_tensor_sha256": negative.manifest["payload"][
                "tensor_sha256"
            ],
        }
    result = {
        "schema_version": "distillschedule.gst_pool_input_validation.v1",
        "status": "PASS",
        "role": args.role,
        "run_id": args.run_id,
        "contract_digest": candidate.contract_digest,
        "candidate_manifest_sha256": candidate.manifest_sha256,
        "streamed_blocks": list(candidate.streamed_blocks),
        "trace": trace,
        "target_manifest_hash": package.manifest["manifest_hash"],
        "target_versions": {"generator": 3, "student": 11},
        "precast_source": precast["source"]["path"],
        "precast_tensor_bytes": precast["artifact"]["tensor_bytes"],
        **role_specific,
    }
    _write(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

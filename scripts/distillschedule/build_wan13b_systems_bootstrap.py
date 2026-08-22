#!/usr/bin/env python3
"""Build one immutable real-Wan1.3B LoRA/optimizer bootstrap.

The resulting artifact is shared by the resident and world16-FSDP2 arms.  Base
weights remain in the pinned Diffusers model directory; this artifact stores
only the two deterministic FP32 LoRA inventories and declares zero-materialized
Adam state plus the initial logical G2/C10/EMA10 versions.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distillschedule.runtime.wan13b_upper_bound import (
    LORA_RANK,
    WAN13B_BLOCK_COUNT,
    WAN13B_PYRAMID_PATCH_SEED,
    file_sha256,
)
from runtime.critic_pool_helios import build_helios_pool_model
from runtime.official_baseline import suspend_transformers_zero3_init, trim_host_allocator


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _model_files(model_root: Path) -> list[Path]:
    transformer = model_root / "transformer"
    required = transformer / "config.json"
    if not required.is_file():
        raise FileNotFoundError(required)
    files = [required]
    index = transformer / "diffusion_pytorch_model.safetensors.index.json"
    single = transformer / "diffusion_pytorch_model.safetensors"
    if index.is_file():
        files.append(index)
        value = json.loads(index.read_text())
        names = sorted(set(str(name) for name in value["weight_map"].values()))
        files.extend(transformer / name for name in names)
    elif single.is_file():
        files.append(single)
    else:
        raise FileNotFoundError("Wan1.3B transformer safetensors are missing")
    if not all(path.is_file() for path in files):
        raise FileNotFoundError("Wan1.3B weight index references missing shards")
    return files


def _source_inventory(model_root: Path) -> tuple[list[dict[str, Any]], str]:
    rows = [
        {
            "path": str(path.relative_to(model_root)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in _model_files(model_root)
    ]
    return rows, hashlib.sha256(_canonical(rows)).hexdigest()


def _component(
    *, model_root: Path, role: str, seed: int, output: Path
) -> dict[str, Any]:
    from safetensors.torch import save_file

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        with suspend_transformers_zero3_init():
            model, metadata = build_helios_pool_model(
                role=role,
                model_root=model_root,
                subfolder="transformer",
                precast_model_root=None,
                lora_rank=LORA_RANK,
                enable_gan=False,
                chunked_residual_storage="cuda_out_of_place",
                missing_pyramid_patch_seed=WAN13B_PYRAMID_PATCH_SEED,
            )
    if int(metadata["block_count"]) != WAN13B_BLOCK_COUNT:
        raise AssertionError(
            f"{role} is not Wan1.3B: {metadata['block_count']} blocks"
        )
    trainables = {
        name: parameter.detach().cpu().float().contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not trainables:
        raise RuntimeError(f"{role} has no LoRA trainables")
    payload = output / f"{role}_trainables.safetensors"
    save_file(
        trainables,
        str(payload),
        metadata={
            "role": role,
            "model": "Wan2.1-T2V-1.3B-Diffusers",
            "lora_rank": str(LORA_RANK),
            "seed": str(seed),
        },
    )
    result = {
        "seed": seed,
        "trainable_payload": payload.name,
        "sha256": file_sha256(payload),
        "bytes": payload.stat().st_size,
        "parameter_names": list(trainables),
        "trainable_tensor_count": len(trainables),
        "trainable_numel": sum(int(value.numel()) for value in trainables.values()),
        "trainable_dtype": "torch.float32",
        "optimizer_state": "ZERO_ADAM_M_AND_V_MATERIALIZED_BEFORE_TIMING",
        "optimizer_step": 2 if role == "generator" else 10,
        "scheduler_step": 2 if role == "generator" else 10,
        "loader_metadata": metadata,
    }
    del trainables, model
    gc.collect()
    trim_host_allocator()
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--model-root", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--expected-head", required=True)
    value.add_argument("--generator-seed", type=int, default=20260820)
    value.add_argument("--critic-seed", type=int, default=20260821)
    return value


def main() -> int:
    args = parser().parse_args()
    if len(args.expected_head) != 40:
        raise ValueError("--expected-head must be a full Git SHA")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.model_root.is_dir():
        raise FileNotFoundError(args.model_root)
    actual_head = os.popen(f"git -C {ROOT!s} rev-parse HEAD").read().strip()
    if actual_head != args.expected_head:
        raise ValueError("DMD-System source HEAD changed")
    args.output.mkdir(parents=True)
    source_files, source_hash = _source_inventory(args.model_root.resolve())
    components = {
        "generator": _component(
            model_root=args.model_root.resolve(),
            role="generator",
            seed=args.generator_seed,
            output=args.output,
        ),
        "critic": _component(
            model_root=args.model_root.resolve(),
            role="critic",
            seed=args.critic_seed,
            output=args.output,
        ),
    }
    manifest: dict[str, Any] = {
        "schema_version": "distillschedule.wan13b_bootstrap.v1",
        "status": "PASS",
        "systems_only": True,
        "official_training_checkpoint": False,
        "claim_boundary": (
            "real Wan1.3B base weights with deterministic systems-only LoRA/Adam/EMA state"
        ),
        "dmd_system_head": actual_head,
        "model_root": str(args.model_root.resolve()),
        "model_source": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "model_files": source_files,
        "model_identity_sha256": source_hash,
        "block_count": WAN13B_BLOCK_COUNT,
        "lora_rank": LORA_RANK,
        "components": components,
        "ema": {
            "source": "generator_trainables",
            "device": "cuda",
            "optimization_step": 10,
            "decay": 0.999,
            "update_after_step": 0,
        },
        "logical_initial_versions": {"generator": 2, "critic": 10, "ema": 10},
    }
    manifest["manifest_hash"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    _atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps({"status": "PASS", "manifest_hash": manifest["manifest_hash"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build an immutable GAN-active steps1000-1004 trace from a validated Full5 trace.

This is a systems-only workload: source-shaped tensors and rollout controls are
copied exactly from source steps10-14, while every GAN-specific stochastic value
is sampled once from an operation-keyed RNG stream and stored explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch


SOURCE_STEPS = tuple(range(10, 15))
TARGET_STEPS = tuple(range(1000, 1005))
STEP_MAP = dict(zip(SOURCE_STEPS, TARGET_STEPS))


def _seed(base: int, rank: int, step: int, key: str) -> int:
    digest = hashlib.sha256(f"{base}:{rank}:{step}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _randn(shape: tuple[int, ...], *, seed: int, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=dtype)


def _crop(*, seed: int, height: int = 48, width: int = 80, scale: float = 0.5) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    h2, w2 = int(height * scale), int(width * scale)
    top = torch.randint(0, height - h2 + 1, (1,), generator=generator)
    left = torch.randint(0, width - w2 + 1, (1,), generator=generator)
    return torch.stack((top, left), dim=1).to(torch.int64)


def _rewrite_step(value: str) -> str:
    for source, target in STEP_MAP.items():
        value = value.replace(f"step/{source}/", f"step/{target}/")
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_rank(
    *,
    source_root: Path,
    output_root: Path,
    rank: int,
    base_seed: int,
    operation_trace_module: Any,
) -> dict[str, Any]:
    source = operation_trace_module.OperationTrace.load(source_root)
    writer = operation_trace_module.OperationTraceWriter(
        classification="SYSTEMS_ONLY_GAN_BOOTSTRAP: steady-state training runtime with precomputed source-shaped features",
        source={
            **dict(source.manifest.get("source", {})),
            "systems_only_gan_bootstrap": True,
            "source_trace_hash": source.trace_hash,
            "source_steps": list(SOURCE_STEPS),
            "logical_steps": list(TARGET_STEPS),
            "gan_rng_base_seed": base_seed,
            "rank": rank,
        },
    )

    selected: dict[str, torch.Tensor] = {}
    roles: dict[str, str] = {}
    packings: dict[str, str] = {}
    for key, tensor in source.tensors.items():
        keep = not key.startswith("step/") or any(key.startswith(f"step/{step}/") for step in SOURCE_STEPS)
        if not keep:
            continue
        target_key = _rewrite_step(key)
        selected[target_key] = tensor
        descriptor = source.manifest["tensors"][key]
        roles[target_key] = str(descriptor["role"])
        packings[target_key] = str(descriptor["packing"])

    latent_shape = (1, 16, 9, 48, 80)
    extra: dict[str, tuple[torch.Tensor, str]] = {}
    for target_step in TARGET_STEPS:
        prefix = f"step/{target_step}/critic"
        for role in ("gan_d_fake_crop", "gan_d_real_crop", "gan_d_r1_crop"):
            extra[f"{prefix}/{role}"] = (
                _crop(seed=_seed(base_seed, rank, target_step, role)),
                role,
            )
        extra[f"{prefix}/r1_epsilon"] = (
            _randn(latent_shape, seed=_seed(base_seed, rank, target_step, "r1_epsilon"), dtype=torch.bfloat16),
            "r1_epsilon_standard_normal",
        )
    extra["step/1000/generator/gan_g_noise"] = (
        _randn(latent_shape, seed=_seed(base_seed, rank, 1000, "gan_g_noise"), dtype=torch.bfloat16),
        "gan_g_noise",
    )
    extra["step/1000/generator/gan_g_crop"] = (
        _crop(seed=_seed(base_seed, rank, 1000, "gan_g_crop")),
        "gan_g_crop",
    )

    for key in sorted(selected):
        writer.add_tensor(key, selected[key], role=roles[key], packing=packings[key])
    for key in sorted(extra):
        tensor, role = extra[key]
        writer.add_tensor(key, tensor, role=role)
    registered_keys = set(selected) | set(extra)

    for operation in source.manifest.get("operations", []):
        control = dict(operation.get("control", {}))
        global_step = control.get("global_step")
        if isinstance(global_step, int) and global_step not in SOURCE_STEPS:
            continue
        tensor_keys = [_rewrite_step(str(key)) for key in operation.get("tensor_keys", [])]
        if any(key not in registered_keys for key in tensor_keys):  # fail closed for filtered steps
            continue
        key = _rewrite_step(str(operation["key"]))
        if isinstance(global_step, int):
            control["global_step"] = STEP_MAP[global_step]
            if operation.get("operation") == "stage3_branch":
                control["gan"] = True
        writer.add_operation(
            key,
            component=str(operation.get("component", "trace")),
            operation=str(operation.get("operation", "trace")),
            tensor_keys=tensor_keys,
            control=control,
        )

    writer.add_operation(
        "step/1000/generator/gan_g",
        component="generator_score",
        operation="gan_g_stochastic_inputs",
        tensor_keys=["step/1000/generator/gan_g_noise", "step/1000/generator/gan_g_crop"],
        control={"global_step": 1000, "r2_active": False},
    )
    for step in TARGET_STEPS:
        prefix = f"step/{step}/critic"
        writer.add_operation(
            f"{prefix}/gan_d",
            component="score_critic",
            operation="gan_d_r1_stochastic_inputs",
            tensor_keys=[
                f"{prefix}/gan_d_fake_crop",
                f"{prefix}/gan_d_real_crop",
                f"{prefix}/gan_d_r1_crop",
                f"{prefix}/r1_epsilon",
            ],
            control={"global_step": step, "r1_active": True, "r2_active": False},
        )

    contract = dict(source.manifest.get("contract", {}))
    contract.update(
        {
            "systems_only_gan_bootstrap": True,
            "logical_steps": list(TARGET_STEPS),
            "source_steps": list(SOURCE_STEPS),
            "gan_start_step": 1000,
            "r1_weight": 100.0,
            "r2_weight": 0.0,
            "gan_stochastic_values": "operation_keyed_explicit_replay",
        }
    )
    trace = writer.write(output_root, contract=contract)
    return {
        "rank": rank,
        "path": output_root.name,
        "sample_id": source.manifest.get("source", {}).get("sample_id", rank),
        "sample_index": source.manifest.get("source", {}).get("sample_index", rank),
        "trace_hash": trace.trace_hash,
        "source_trace_hash": source.trace_hash,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--helios-root", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=2026081001)
    args = parser.parse_args(argv)
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable trace {output}")
    sys.path.insert(0, str(args.helios_root.resolve(strict=True)))
    from runtime import operation_trace

    bundle = operation_trace.load_operation_trace_bundle_manifest(source)
    if bundle is None or len(bundle["ranks"]) != 16:
        raise ValueError("source must be a validated rank16 trace bundle")
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    for entry in bundle["ranks"]:
        rank = int(entry["rank"])
        entries.append(
            build_rank(
                source_root=source / str(entry["path"]),
                output_root=output / f"rank{rank}",
                rank=rank,
                base_seed=args.base_seed,
                operation_trace_module=operation_trace,
            )
        )
    target_bundle = operation_trace.write_operation_trace_bundle_manifest(output, entries)
    manifest = {
        "schema_version": "distillschedule.systems_only_gan_trace.v1",
        "status": "PASS",
        "source": str(source),
        "source_bundle_hash": bundle["bundle_hash"],
        "output": str(output),
        "output_bundle_hash": target_bundle["bundle_hash"],
        "base_seed": args.base_seed,
        "source_steps": list(SOURCE_STEPS),
        "logical_steps": list(TARGET_STEPS),
        "world_size": 16,
        "r2_active": False,
    }
    manifest["manifest_digest"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    path = output / "systems_only_gan_trace_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**manifest, "manifest_sha256": _sha256_file(path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export the lightweight post-Step10-A state required by Candidate B.

The authoritative Candidate-A transition checkpoint contains two full 14B
DeepSpeed model-state files.  Candidate B does not need a Generator optimizer
during steps 10--14, so replicating or repeatedly mmap-reading that 29-GiB
file is unnecessary.  This exporter extracts only the exact BF16 G3
trainables, hardlinks EMA10/RNG, and binds them to the existing C10 logical
package.  It never changes model state and refuses to overwrite an output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from runtime.critic_pool_checkpoint import (
    CANDIDATE_B_TRANSITION_SCHEMA,
    LogicalCheckpointPackage,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(path: Path, chunk_bytes: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def _load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-transition-manifest-sha256", required=True)
    parser.add_argument("--critic-logical-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    source = args.transition_checkpoint.resolve(strict=True)
    transition_manifest_path = source / "transition_manifest.json"
    observed_transition_sha = _sha256(transition_manifest_path)
    if observed_transition_sha != args.expected_transition_manifest_sha256:
        raise ValueError("source transition manifest SHA-256 changed")
    transition = json.loads(transition_manifest_path.read_text())
    expected_versions = {
        "generator_version": 3,
        "critic_version": 10,
        "ema_optimization_step": 10,
    }
    mismatch = {
        key: (transition.get(key), value)
        for key, value in expected_versions.items()
        if transition.get(key) != value
    }
    if mismatch:
        raise ValueError(f"source transition is not G3/C10/EMA10: {mismatch}")

    critic = LogicalCheckpointPackage.load(args.critic_logical_package)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    model_path = source / "pytorch_model" / "mp_rank_00_model_states.pt"
    model_state = _load(model_path)
    if int(model_state.get("global_steps", -1)) != 3:
        raise ValueError("source Generator model state is not G3")
    groups = model_state.get("param_shapes")
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError("source Generator trainable shape inventory changed")
    shapes = OrderedDict(groups[0])
    module = model_state.get("module")
    if not isinstance(module, dict):
        raise ValueError("source Generator module state is missing")

    payload_path = output / "generator" / "module.bf16"
    payload_path.parent.mkdir()
    records: dict[str, Any] = {}
    offset = 0
    with payload_path.open("wb") as handle:
        for name, shape in shapes.items():
            tensor = module.get(name)
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"source Generator trainable is missing: {name}")
            expected_shape = tuple(int(value) for value in shape)
            if tuple(tensor.shape) != expected_shape or tensor.dtype != torch.bfloat16:
                raise ValueError(
                    f"source Generator trainable changed: {name} "
                    f"{tuple(tensor.shape)} {tensor.dtype}"
                )
            value = tensor.detach().cpu().contiguous()
            value.view(torch.uint16).numpy().reshape(-1).tofile(handle)
            records[name] = {
                "shape": list(expected_shape),
                "numel": int(value.numel()),
                "offset_elements": offset,
            }
            offset += int(value.numel())
            del value
    payload = {
        "path": payload_path.relative_to(output).as_posix(),
        "bytes": payload_path.stat().st_size,
        "sha256": _sha256(payload_path),
        "dtype": "torch.bfloat16",
    }
    if payload["bytes"] != offset * 2:
        raise AssertionError("G3 BF16 payload size changed")

    ema_source = source / "model_ema"
    ema_manifest = json.loads((ema_source / "sparse_ema_manifest.json").read_text())
    if int(ema_manifest.get("optimization_step", -1)) != 10:
        raise ValueError("source sparse EMA is not EMA10")
    ema_files = []
    for path in sorted(ema_source.iterdir()):
        if not path.is_file():
            continue
        destination = output / "model_ema" / path.name
        _link(path, destination)
        ema_files.append(
            {
                "path": destination.relative_to(output).as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
            }
        )

    rng: dict[str, Any] = {}
    for rank in range(16):
        source_path = source / f"random_states_{rank}.pkl"
        destination = output / "rng" / f"rank_{rank:02d}.pt"
        _link(source_path, destination)
        rng[str(rank)] = {
            "path": destination.relative_to(output).as_posix(),
            "bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
        }

    manifest: dict[str, Any] = {
        "schema_version": CANDIDATE_B_TRANSITION_SCHEMA,
        "status": "PASS",
        "execution_boundary": "post_step10_generator_update_pre_critic_update",
        "source_transition_checkpoint": str(source),
        "source_transition_manifest_sha256": observed_transition_sha,
        "critic_logical_package": str(critic.root.resolve()),
        "critic_logical_manifest_hash": critic.manifest_hash,
        "versions": {
            "generator": 3,
            "critic": 10,
            "ema_optimization_step": 10,
        },
        "generator": {
            "parameter_names": list(shapes),
            "records": records,
            "logical_numel": offset,
            "payload": payload,
            "optimizer_policy": "BACKING_ONLY_IN_SOURCE_TRANSITION_CHECKPOINT",
        },
        "ema": {"files": ema_files, "optimization_step": 10},
        "rng": rng,
        "guardrails": [
            "Candidate B starts after the already validated Candidate-A Step10-A transition.",
            "C10 master/moments come from the frozen logical package and are repartitioned to DP12 outside timing.",
            "This package does not claim DP8-to-DP12 trajectory equivalence; it seeds both Candidate-B schedules identically.",
        ],
    }
    manifest["manifest_hash"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    _atomic_json(output / "manifest.json", manifest)
    result = {
        "schema_version": "distillschedule.routeb_candidate_b_transition_export.v1",
        "status": "PASS",
        "output": str(output),
        "manifest_hash": manifest["manifest_hash"],
        "generator_payload_bytes": payload["bytes"],
        "generator_payload_sha256": payload["sha256"],
        "ema_file_count": len(ema_files),
        "rng_file_count": len(rng),
    }
    _atomic_json(args.result, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

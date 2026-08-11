#!/usr/bin/env python3
"""Add the frozen systems-only GAN heads to a validated Score pre-cast asset.

The official Score backbone pre-cast predates GAN-head construction.  GAN-on
model construction therefore has 130 additional parameters before DeepSpeed
resume can load the shared bootstrap.  This builder hard-links the immutable
backbone shards and publishes BF16 GAN-head shards derived from the exact FP32
master artifact in ``systems_only_gan_bootstrap.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import save_file


PRECAST_MANIFEST = "distillschedule_precast_manifest.json"
INDEX = "diffusion_pytorch_model.safetensors.index.json"
MARKER = "systems_only_gan_precast.json"
SCHEMA = "distillschedule.systems_only_gan_precast.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _group_name(parameter_name: str) -> str:
    if parameter_name.startswith("gan_final_head."):
        return "gan_final_head"
    fields = parameter_name.split(".")
    if len(fields) >= 2 and fields[0] == "gan_heads":
        return f"gan_heads_{fields[1]}"
    raise ValueError(f"not an official GAN-head parameter: {parameter_name}")


def build_systems_gan_precast(
    *, base_precast: str | Path, bootstrap_checkpoint: str | Path, output: str | Path
) -> dict[str, Any]:
    base = Path(base_precast).resolve()
    bootstrap = Path(bootstrap_checkpoint).resolve()
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")

    base_manifest_path = base / PRECAST_MANIFEST
    base_index_path = base / INDEX
    bootstrap_marker_path = bootstrap / "systems_only_gan_bootstrap.json"
    base_manifest = json.loads(base_manifest_path.read_text())
    index = json.loads(base_index_path.read_text())
    bootstrap_marker = json.loads(bootstrap_marker_path.read_text())
    if bootstrap_marker.get("classification") != "systems_only_gan_bootstrap":
        raise ValueError("bootstrap is not classified systems_only_gan_bootstrap")
    head = bootstrap_marker.get("gan_head_manifest")
    if not isinstance(head, dict) or not isinstance(head.get("records"), list):
        raise ValueError("bootstrap GAN-head manifest is missing")
    head_path = bootstrap / str(head["path"])
    if sha256_file(head_path) != head.get("sha256"):
        raise ValueError("bootstrap GAN-head FP32 master hash mismatch")

    records = head["records"]
    names = [str(item["name"]) for item in records]
    if len(names) != len(set(names)) or len(names) != int(head["parameter_count"]):
        raise ValueError("bootstrap GAN-head parameter inventory is malformed")
    if set(names) & set(index["weight_map"]):
        raise ValueError("base pre-cast already contains GAN heads")

    staging = destination.parent / (
        f".{destination.name}.incomplete-{os.getpid()}-{time.time_ns()}"
    )
    shutil.copytree(base, staging, copy_function=link_or_copy)
    try:
        raw = torch.from_file(
            str(head_path),
            shared=False,
            size=int(head["logical_numel"]),
            dtype=torch.float32,
        )
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in records:
            grouped[_group_name(str(item["name"]))].append(item)

        weight_map = dict(index["weight_map"])
        shard_records: list[dict[str, Any]] = []
        added_bytes = 0
        for group in sorted(grouped):
            filename = f"distillschedule_{group}_bf16.safetensors"
            state: dict[str, torch.Tensor] = {}
            for item in grouped[group]:
                name = str(item["name"])
                offset = int(item["offset_elements"])
                numel = int(item["numel"])
                shape = tuple(int(value) for value in item["shape"])
                value = raw[offset : offset + numel].reshape(shape).to(torch.bfloat16)
                state[name] = value.contiguous()
                weight_map[name] = filename
                added_bytes += numel * 2
            shard_path = staging / filename
            save_file(
                state,
                str(shard_path),
                metadata={
                    "format": "pt",
                    "distillschedule_schema": SCHEMA,
                    "bootstrap_artifact_hash": str(bootstrap_marker["artifact_hash"]),
                },
            )
            shard_records.append(
                {
                    "file": filename,
                    "sha256": sha256_file(shard_path),
                    "size_bytes": shard_path.stat().st_size,
                    "tensor_count": len(state),
                }
            )
            del state
        del raw

        updated_index = dict(index)
        updated_index["weight_map"] = dict(sorted(weight_map.items()))
        updated_index["metadata"] = dict(index.get("metadata") or {})
        updated_index["metadata"]["total_size"] = (
            int(base_manifest["artifact"]["tensor_bytes"]) + added_bytes
        )
        atomic_json(staging / INDEX, updated_index)

        updated_manifest = json.loads(json.dumps(base_manifest))
        artifact = updated_manifest["artifact"]
        artifact["index_sha256"] = sha256_file(staging / INDEX)
        artifact["tensor_count"] = int(artifact["tensor_count"]) + len(records)
        artifact["tensor_bytes"] = int(artifact["tensor_bytes"]) + added_bytes
        dtype_counts = Counter(artifact["dtype_tensor_counts"])
        dtype_counts["bfloat16"] += len(records)
        artifact["dtype_tensor_counts"] = dict(sorted(dtype_counts.items()))
        artifact["shards"] = list(artifact["shards"]) + shard_records
        artifact["systems_only_gan_supplement"] = {
            "bootstrap_artifact_hash": bootstrap_marker["artifact_hash"],
            "head_fp32_sha256": head["sha256"],
            "head_parameter_count": len(records),
            "head_logical_numel": int(head["logical_numel"]),
        }
        atomic_json(staging / PRECAST_MANIFEST, updated_manifest)

        marker = {
            "schema_version": SCHEMA,
            "status": "PASS",
            "classification": "systems_only_gan_precast",
            "base_precast": str(base),
            "base_manifest_sha256": sha256_file(base_manifest_path),
            "bootstrap_checkpoint": str(bootstrap),
            "bootstrap_artifact_hash": bootstrap_marker["artifact_hash"],
            "head_fp32_sha256": head["sha256"],
            "head_parameter_count": len(records),
            "head_logical_numel": int(head["logical_numel"]),
            "added_tensor_bytes": added_bytes,
            "shards": shard_records,
            "index_sha256": sha256_file(staging / INDEX),
            "precast_manifest_sha256": sha256_file(staging / PRECAST_MANIFEST),
        }
        marker["artifact_hash"] = canonical_digest(marker)
        atomic_json(staging / MARKER, marker)
        os.replace(staging, destination)
        return marker
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-precast", required=True, type=Path)
    parser.add_argument("--bootstrap-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    result = build_systems_gan_precast(
        base_precast=args.base_precast,
        bootstrap_checkpoint=args.bootstrap_checkpoint,
        output=args.output,
    )
    if args.manifest:
        atomic_json(args.manifest, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

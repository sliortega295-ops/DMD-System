#!/usr/bin/env python3
"""Build one immutable systems-only GAN checkpoint shared by baseline/pipeline.

The source checkpoint supplies real Generator/Critic LoRA, Adam, scheduler and
EMA tensors.  Official GAN-head architecture is added once with deterministic
initialization; its FP32 master and zero Adam moments are materialized before
timing.  The directory is named checkpoint-1000 solely to select the official
1000-1004 control window and is explicitly *not* an official continuation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


SCHEMA = "distillschedule.systems_only_gan_bootstrap.v1"
ALIGNMENT = 32  # DeepSpeed ZeRO-1: 2 BF16 elements * DP16.


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    shape: tuple[int, ...]
    init: str
    fan_in: int | None = None

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


def gan_head_specs() -> tuple[ParameterSpec, ...]:
    specs: list[ParameterSpec] = []
    for prefix, input_channels in [
        ("gan_heads.5", 5120),
        ("gan_heads.15", 5120),
        ("gan_heads.25", 5120),
        ("gan_heads.35", 5120),
        ("gan_final_head", 16),
    ]:
        layers = [
            (0, input_channels, (3, 3, 3), 1),
            (3, 768, (4, 4, 4), 4),
            (6, 768, (4, 4, 4), 7),
            (9, 768, (3, 3, 3), 10),
            (12, 768, (3, 3, 3), 13),
            (15, 768, (1, 3, 3), 16),
        ]
        for index, incoming, kernel, norm_index in layers:
            fan_in = incoming * math.prod(kernel)
            specs.extend(
                [
                    ParameterSpec(f"{prefix}.head3d.{index}.weight", (768, incoming, *kernel), "uniform", fan_in),
                    ParameterSpec(f"{prefix}.head3d.{index}.bias", (768,), "uniform", fan_in),
                    ParameterSpec(f"{prefix}.head3d.{norm_index}.weight", (768,), "ones"),
                    ParameterSpec(f"{prefix}.head3d.{norm_index}.bias", (768,), "zeros"),
                ]
            )
        specs.extend(
            [
                ParameterSpec(f"{prefix}.head3d.20.weight", (1, 768), "uniform", 768),
                ParameterSpec(f"{prefix}.head3d.20.bias", (1,), "uniform", 768),
            ]
        )
    names = [item.name for item in specs]
    if len(names) != len(set(names)):
        raise AssertionError("duplicate GAN-head parameter names")
    return tuple(specs)


def canonical_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key_seed(base_seed: int, name: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{base_seed}:{name}".encode()).digest()[:8], "little") % (2**63 - 1)


def offsets(shapes: Mapping[str, Sequence[int]]) -> tuple[dict[str, tuple[int, int]], int]:
    result: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, shape in shapes.items():
        count = math.prod(int(value) for value in shape)
        result[name] = (cursor, count)
        cursor += count
    return result, cursor


def engine_state_root(engine: Path) -> Path:
    return engine / (engine / "latest").read_text().strip()


def optimizer_files(engine: Path) -> dict[int, Path]:
    root = engine_state_root(engine)
    result = {}
    for path in root.glob("bf16_zero_pp_rank_*_mp_rank_00_optim_states.pt"):
        rank = int(path.name.split("_rank_")[1].split("_")[0])
        result[rank] = path
    if sorted(result) != list(range(16)):
        raise ValueError(f"expected ZeRO-1 ranks0-15 under {root}, got {sorted(result)}")
    return result


def model_state_path(engine: Path) -> Path:
    return engine_state_root(engine) / "mp_rank_00_model_states.pt"


def torch_load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def copy_or_link(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def write_head_master(path: Path, specs: Sequence[ParameterSpec], *, base_seed: int) -> dict[str, Any]:
    total = sum(item.numel for item in specs)
    raw = np.memmap(path, dtype=np.dtype("<f4"), mode="w+", shape=(total,))
    cursor = 0
    records: list[dict[str, Any]] = []
    chunk = 8 * 1024 * 1024
    for spec in specs:
        view = raw[cursor : cursor + spec.numel]
        if spec.init == "zeros":
            view[:] = 0.0
        elif spec.init == "ones":
            view[:] = 1.0
        elif spec.init == "uniform":
            assert spec.fan_in
            bound = 1.0 / math.sqrt(spec.fan_in)
            generator = torch.Generator(device="cpu").manual_seed(key_seed(base_seed, spec.name))
            for lower in range(0, spec.numel, chunk):
                upper = min(spec.numel, lower + chunk)
                values = torch.empty(upper - lower, dtype=torch.float32).uniform_(
                    -bound, bound, generator=generator
                )
                view[lower:upper] = values.numpy()
        else:  # pragma: no cover
            raise ValueError(spec.init)
        view.flush()
        digest = hashlib.sha256()
        digest.update(json.dumps({"name": spec.name, "shape": spec.shape, "dtype": "float32"}, sort_keys=True).encode())
        digest.update(memoryview(view).cast("B"))
        records.append(
            {
                "name": spec.name,
                "shape": list(spec.shape),
                "numel": spec.numel,
                "offset_elements": cursor,
                "init": spec.init,
                "seed": key_seed(base_seed, spec.name) if spec.init == "uniform" else None,
                "sha256": digest.hexdigest(),
            }
        )
        cursor += spec.numel
    raw.flush()
    del raw
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "parameter_count": len(specs),
        "logical_numel": total,
        "records": records,
    }


def fragment_mapping(
    *,
    shapes: Mapping[str, Sequence[int]],
    rank: int,
    partition_numel: int,
    address_type: type[Any],
) -> OrderedDict[str, Any]:
    locations, total = offsets(shapes)
    lower, upper = rank * partition_numel, min(total, (rank + 1) * partition_numel)
    mapping: OrderedDict[str, Any] = OrderedDict()
    for name, (start, count) in locations.items():
        left, right = max(lower, start), min(upper, start + count)
        if left < right:
            mapping[name] = address_type(start=left - lower, numel=right - left)
    return mapping


def replace_critic_model(
    *,
    source_checkpoint: Path,
    target_checkpoint: Path,
    specs: Sequence[ParameterSpec],
    head_master: Path,
) -> dict[str, Any]:
    source_path = model_state_path(source_checkpoint / "critic")
    target_path = model_state_path(target_checkpoint / "critic")
    payload = torch_load(source_path)
    shapes = OrderedDict((name, torch.Size(shape)) for name, shape in payload["param_shapes"][0].items())
    old_total = sum(math.prod(tuple(shape)) for shape in shapes.values())
    module = payload["module"]
    head_raw = torch.from_file(str(head_master), shared=False, size=sum(item.numel for item in specs), dtype=torch.float32)
    cursor = 0
    for spec in specs:
        if spec.name in module or spec.name in shapes:
            raise ValueError(f"source checkpoint already contains {spec.name}")
        values = head_raw[cursor : cursor + spec.numel].reshape(spec.shape).to(torch.bfloat16)
        module[spec.name] = values.contiguous()
        shapes[spec.name] = torch.Size(spec.shape)
        cursor += spec.numel
    payload["param_shapes"] = [shapes]
    temporary = target_path.with_name(target_path.name + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, target_path)
    return {
        "path": str(target_path),
        "bytes": target_path.stat().st_size,
        "sha256": sha256_file(target_path),
        "old_logical_numel": old_total,
        "new_logical_numel": old_total + cursor,
        "parameter_count": len(shapes),
        "head_parameter_count": len(specs),
    }


def build_combined_raw(
    *, logical_package: Path, head_master: Path, build_dir: Path, head_numel: int
) -> dict[str, Path]:
    critic = json.loads((logical_package / "manifest.json").read_text())["components"]["critic"]
    old_total = int(critic["logical_numel"])
    total = old_total + head_numel
    outputs: dict[str, Path] = {}
    for key, source_key in [
        ("master", "master"),
        ("first_moment", "first_moment"),
        ("second_moment", "second_moment"),
    ]:
        source = logical_package / critic["payloads"][source_key]["path"]
        if sha256_file(source) != critic["payloads"][source_key]["sha256"]:
            raise ValueError(f"logical package payload hash mismatch: {source}")
        target = build_dir / f"critic_{key}.f32"
        raw = np.memmap(target, dtype=np.dtype("<f4"), mode="w+", shape=(total,))
        source_raw = np.memmap(source, dtype=np.dtype("<f4"), mode="r", shape=(old_total,))
        raw[:old_total] = source_raw
        raw[old_total:] = 0.0
        if key == "master":
            head_raw = np.memmap(head_master, dtype=np.dtype("<f4"), mode="r", shape=(head_numel,))
            raw[old_total:] = head_raw
            del head_raw
        raw.flush()
        del source_raw, raw
        outputs[key] = target
    return outputs


def replace_critic_optimizer(
    *,
    source_checkpoint: Path,
    target_checkpoint: Path,
    shapes: Mapping[str, Sequence[int]],
    raw_paths: Mapping[str, Path],
) -> dict[str, Any]:
    _, total = offsets(shapes)
    aligned = ((total + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
    partition_numel = aligned // 16
    padding = aligned - total
    source_template = optimizer_files(source_checkpoint / "critic")[0]
    target_paths = optimizer_files(target_checkpoint / "critic")
    template_probe = torch_load(source_template)
    mappings = template_probe["optimizer_state_dict"]["param_slice_mappings"]
    address_type = type(next(iter(mappings[0].values())))
    raw = {
        key: torch.from_file(str(path), shared=False, size=total, dtype=torch.float32)
        for key, path in raw_paths.items()
    }
    files = []
    for rank, target_path in target_paths.items():
        payload = torch_load(source_template)
        optimizer = payload["optimizer_state_dict"]
        state = optimizer["base_optimizer_state"]["state"][0]
        lower = rank * partition_numel
        upper = min(total, lower + partition_numel)
        optimizer["single_partition_of_fp32_groups"] = [raw["master"][lower:upper].clone()]
        state["exp_avg"] = raw["first_moment"][lower:upper].clone()
        state["exp_avg_sq"] = raw["second_moment"][lower:upper].clone()
        optimizer["param_slice_mappings"] = [
            fragment_mapping(
                shapes=shapes,
                rank=rank,
                partition_numel=partition_numel,
                address_type=address_type,
            )
        ]
        optimizer["partition_count"] = [16]
        optimizer["group_paddings"] = [padding if rank == 15 else 0]
        if isinstance(payload.get("ds_config"), dict):
            payload["ds_config"]["train_batch_size"] = 16
        temporary = target_path.with_name(target_path.name + f".tmp-{os.getpid()}")
        torch.save(payload, temporary)
        os.replace(temporary, target_path)
        files.append(
            {
                "rank": rank,
                "path": str(target_path),
                "elements": upper - lower,
                "padding": padding if rank == 15 else 0,
                "bytes": target_path.stat().st_size,
                "sha256": sha256_file(target_path),
            }
        )
    return {
        "zero_stage": 1,
        "world_size": 16,
        "logical_numel": total,
        "aligned_numel": aligned,
        "partition_numel": partition_numel,
        "last_rank_padding": padding,
        "optimizer_step": 10,
        "head_moments": "materialized_zero",
        "files": files,
    }


def patch_ema(target_checkpoint: Path, *, optimization_step: int) -> dict[str, Any]:
    path = target_checkpoint / "model_ema" / "sparse_ema_manifest.json"
    value = json.loads(path.read_text())
    source_step = int(value["optimization_step"])
    update_after = int(value["update_after_step"])
    step = max(0, optimization_step - update_after - 1)
    decay = 0.0 if step <= 0 else max(0.0, min((1 + step) / (10 + step), float(value["decay"])))
    value["optimization_step"] = int(optimization_step)
    value["cur_decay_value"] = float(decay)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "source_optimization_step": source_step,
        "logical_optimization_step": optimization_step,
        "cur_decay_value": decay,
        "tensor_payloads_unchanged": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--logical-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026081000)
    parser.add_argument("--ema-optimization-step", type=int, default=800)
    args = parser.parse_args(argv)
    source = args.source_checkpoint.resolve(strict=True)
    logical = args.logical_package.resolve(strict=True)
    output = args.output.resolve()
    if output.name != "checkpoint-1000":
        raise ValueError("systems bootstrap output must be named checkpoint-1000 for official control flow")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable bootstrap {output}")
    logical_manifest_path = logical / "manifest.json"
    logical_manifest = json.loads(logical_manifest_path.read_text())
    if logical_manifest.get("schema_version") != "distillschedule.logical_checkpoint.v3":
        raise ValueError("expected logical checkpoint v3")
    if Path(logical_manifest["source_checkpoint"]).resolve() != source:
        raise ValueError("logical package and native source checkpoint differ")
    if int(logical_manifest["global_step"]) != 10:
        raise ValueError("systems bootstrap currently requires validated logical step10")

    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    shutil.copytree(source, temporary, copy_function=copy_or_link)
    build_dir = temporary / ".systems_gan_build"
    bootstrap_dir = temporary / "gan_bootstrap"
    build_dir.mkdir()
    bootstrap_dir.mkdir()
    specs = gan_head_specs()
    head_manifest = write_head_master(
        bootstrap_dir / "gan_heads_master.f32", specs, base_seed=args.seed
    )
    model = replace_critic_model(
        source_checkpoint=source,
        target_checkpoint=temporary,
        specs=specs,
        head_master=Path(head_manifest["path"]),
    )
    raw_paths = build_combined_raw(
        logical_package=logical,
        head_master=Path(head_manifest["path"]),
        build_dir=build_dir,
        head_numel=int(head_manifest["logical_numel"]),
    )
    model_payload = torch_load(Path(model["path"]))
    shapes = OrderedDict((name, list(shape)) for name, shape in model_payload["param_shapes"][0].items())
    optimizer = replace_critic_optimizer(
        source_checkpoint=source,
        target_checkpoint=temporary,
        shapes=shapes,
        raw_paths=raw_paths,
    )
    for path in raw_paths.values():
        path.unlink()
    build_dir.rmdir()
    ema = patch_ema(temporary, optimization_step=args.ema_optimization_step)

    head_manifest["path"] = str(Path(head_manifest["path"]).relative_to(temporary))
    model["path"] = str(Path(model["path"]).relative_to(temporary))
    for item in optimizer["files"]:
        item["path"] = str(Path(item["path"]).relative_to(temporary))
    ema["path"] = str(Path(ema["path"]).relative_to(temporary))

    source_manifest_sha = sha256_file(logical_manifest_path)
    modified = {
        "critic_model": {key: model[key] for key in ("path", "bytes", "sha256")},
        "critic_optimizer": [
            {key: item[key] for key in ("rank", "bytes", "sha256")} for item in optimizer["files"]
        ],
        "ema_manifest": ema,
        "gan_heads": {key: head_manifest[key] for key in ("path", "bytes", "sha256")},
    }
    artifact_basis = {
        "source_logical_manifest_sha256": source_manifest_sha,
        "source_logical_manifest_hash": logical_manifest["manifest_hash"],
        "modified_files": modified,
        "logical_control_steps": list(range(1000, 1005)),
        "seed": args.seed,
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "classification": "systems_only_gan_bootstrap",
        "source_checkpoint": str(source),
        "source_checkpoint_logical_step": 10,
        "source_logical_package": str(logical),
        "source_logical_manifest_sha256": source_manifest_sha,
        "source_parameter_versions": {
            "generator": logical_manifest["components"]["generator"]["parameter_version"],
            "critic": logical_manifest["components"]["critic"]["parameter_version"],
            "ema": logical_manifest["ema"]["optimization_step"],
        },
        "logical_control_window": [1000, 1001, 1002, 1003, 1004],
        "logical_schedule_versions": {"generator": 200, "critic": 1000, "ema": 800},
        "official_checkpoint_1000": False,
        "source_faithful_continuation": False,
        "gan_head_architecture": "official Helios Discriminator3DHead; hooks 5,15,25,35 plus final",
        "gan_head_initialization": "per-parameter fixed-seed PyTorch CPU uniform(-1/sqrt(fan_in),+1/sqrt(fan_in)); GroupNorm ones/zeros",
        "gan_head_seed": args.seed,
        "gan_head_manifest": head_manifest,
        "critic_model": model,
        "critic_optimizer": optimizer,
        "generator_optimizer_materialized_from_source": True,
        "critic_lora_optimizer_materialized_from_source": True,
        "gan_head_optimizer_materialized": True,
        "scheduler_state": "unchanged_from_source_step10",
        "ema": ema,
        "modified_files": modified,
        "artifact_hash_method": "sha256(canonical source logical identity + generated/modified content hashes)",
        "artifact_hash": canonical_digest(artifact_basis),
    }
    marker = temporary / "systems_only_gan_bootstrap.json"
    marker.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    # Paths in the marker were written through the temporary directory.  They
    # are diagnostic only; the immutable artifact identity is content based.
    print(json.dumps({"status": "PASS", "output": str(output), "artifact_hash": manifest["artifact_hash"], "marker_sha256": sha256_file(output / marker.name)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

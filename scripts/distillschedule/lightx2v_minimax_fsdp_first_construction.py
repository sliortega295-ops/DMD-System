#!/usr/bin/env python3
"""Gate A/B for the shared MiniMax-H3 CPU-load -> FSDP2 construction path.

This is intentionally a construction-only probe.  It does not build an
optimizer, consume the dataset, or execute a training step.  The same model
construction order is used by the native and DistillSchedule entry points;
the opt-in environment flag is checked by the upstream LightX2V model wrapper.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", choices=("A", "B"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dmd-system-head", required=True)
    parser.add_argument("--upstream-commit", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    return parser


def _rss_bytes() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _cgroup_memory_current() -> int | None:
    for path in (Path("/sys/fs/cgroup/memory.current"), Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            return int(path.read_text().strip())
        except (OSError, ValueError):
            continue
    return None


def _storage_bytes(parameters) -> tuple[int, int, int]:
    """Return logical, CUDA-resident and CPU-resident parameter bytes.

    ``named_parameters`` removes aliases, while the storage-pointer set keeps
    tied parameters from being counted twice when a model exposes duplicates.
    """
    logical = 0
    cuda = 0
    cpu = 0
    seen: set[tuple[int, int]] = set()
    for parameter in parameters:
        try:
            storage = parameter.untyped_storage()
            key = (int(storage.data_ptr()), int(storage.nbytes()))
        except (RuntimeError, AttributeError):
            key = (id(parameter), int(parameter.numel() * parameter.element_size()))
        if key in seen:
            continue
        seen.add(key)
        value = key[1]
        logical += value
        if parameter.is_cuda:
            cuda += value
        elif parameter.device.type == "cpu":
            cpu += value
    return logical, cuda, cpu


def _snapshot(model: Any, label: str, started: float) -> dict[str, Any]:
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    denoiser = getattr(model, "transformer", None)
    if denoiser is None:
        logical = cuda_param = cpu_param = 0
        fsdp2 = False
    else:
        logical, cuda_param, cpu_param = _storage_bytes(denoiser.parameters())
        fsdp2 = bool(model.is_fsdp2_wrapped())
    return {
        "label": label,
        "elapsed_s": time.monotonic() - started,
        "hbm": {
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "driver_free_bytes": int(free),
            "driver_total_bytes": int(total),
        },
        "host": {
            "rss_bytes": _rss_bytes(),
            "cgroup_memory_current_bytes": _cgroup_memory_current(),
        },
        "parameters": {
            "logical_storage_bytes": logical,
            "cuda_storage_bytes": cuda_param,
            "cpu_storage_bytes": cpu_param,
            "fsdp2_wrapped": fsdp2,
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _role_lora(config: dict[str, Any], role: str) -> dict[str, Any] | None:
    section = config.get("training", {}).get(role, {})
    if section.get("train_type") != "lora":
        return None
    value = dict(section.get("lora", {}))
    value["rank"] = int(value["rank"])
    value["alpha"] = int(value["alpha"])
    return value


def _build_role(config: dict[str, Any], role: str, reference: Any | None, started: float) -> tuple[Any, list[dict[str, Any]]]:
    from lightx2v_train.model_zoo import build_model
    from lightx2v_train.runtime.parallel import apply_parallel

    model = build_model(config)
    records = [_snapshot(model, f"{role}.after_build", started)]
    model.load_components(transformer_only=reference is not None, reference_model=reference)
    records.append(_snapshot(model, f"{role}.after_cpu_load", started))

    lora = _role_lora(config, role)
    if lora is not None:
        model.add_lora(lora["rank"], lora["alpha"], lora.get("target_modules"))
        model.set_lora_trainable()
    else:
        model.denoiser_module().requires_grad_(False)
        model.set_denoiser_eval()
    records.append(_snapshot(model, f"{role}.after_lora_or_freeze", started))

    apply_parallel(model, config)
    records.append(_snapshot(model, f"{role}.after_fully_shard", started))

    if bool(config.get("training", {}).get("gradient_checkpointing", True)) and lora is not None:
        model.enable_gradient_checkpointing()
    records.append(_snapshot(model, f"{role}.after_checkpointing", started))
    return model, records


def _write_rank_receipt(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    args = _parser().parse_args()
    if os.environ.get("LIGHTX2V_MINIMAX_FSDP_FIRST_INIT") != "1":
        raise RuntimeError("Gate A/B requires LIGHTX2V_MINIMAX_FSDP_FIRST_INIT=1")
    if not os.environ.get("LIGHTX2V_BENCHMARK_SOURCE_ROOT"):
        raise RuntimeError("Gate A/B requires a prepared benchmark source root")

    from lightx2v_train.runtime import cleanup_distributed, init_distributed, load_config

    config = load_config(args.config)
    torch.cuda.reset_peak_memory_stats()
    init_distributed(config)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    started = time.monotonic()
    roles = ["student"] if args.gate == "A" else ["student", "fake", "teacher"]
    models: list[Any] = []
    all_records: list[dict[str, Any]] = []
    error: str | None = None
    try:
        reference = None
        for index, role in enumerate(roles):
            model, records = _build_role(config, role, reference, started)
            models.append(model)
            reference = models[0]
            all_records.extend(records)
            if index + 1 < len(roles):
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                if dist.is_initialized():
                    dist.barrier()

        if dist.is_initialized():
            dist.barrier()
    except Exception as exc:  # receipt is still written by the launcher on failure
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        receipt = {
            "schema_version": "distillschedule.lightx2v_minimax_fsdp_first_construction.v1",
            "gate": args.gate,
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
            "dmd_system_head": args.dmd_system_head,
            "lightx2v_upstream_commit": args.upstream_commit,
            "source_root": str(args.source_root.resolve()),
            "source_receipt_sha256": _sha256(args.source_root / ".distillschedule_source_receipt") if (args.source_root / ".distillschedule_source_receipt").is_file() else None,
            "config_sha256": _sha256(args.config),
            "fsdp_first_init": os.environ.get("LIGHTX2V_MINIMAX_FSDP_FIRST_INIT"),
            "profile_diagnostics": os.environ.get("LIGHTX2V_PROFILE_DIAGNOSTICS", "off"),
            "roles": roles,
            "records": all_records,
            "final_hbm": {
                "allocated_bytes": int(torch.cuda.memory_allocated()),
                "reserved_bytes": int(torch.cuda.memory_reserved()),
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "driver_free_bytes": int(torch.cuda.mem_get_info()[0]),
            },
            "host_rss_bytes": _rss_bytes(),
            "cgroup_memory_current_bytes": _cgroup_memory_current(),
            "error": error,
        }
        _write_rank_receipt(args.output_dir / f"rank_{rank:03d}.json", receipt)
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            cleanup_distributed()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

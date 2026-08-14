#!/usr/bin/env python3
"""Four resident Student plus four resident Teacher ranks for N2 Step15-A."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from runtime.critic_pool_checkpoint import (
    PoolTargetPackage,
    load_logical_master_into_trainables,
)
from runtime.critic_pool_helios import build_helios_pool_model
from runtime.gst_pool_contract import (
    GeneratorOffloadCandidate,
    NegativePromptAsset,
    source_identity,
)
from runtime.gst_pool_exchange import (
    GRADIENT_SCHEMA,
    STUDENT_SCHEMA,
    atomic_torch_save,
    exchange_paths,
    lane_source_ids,
    validate_input_payload,
    wait_for_file,
)
from runtime.gst_pool_helios import (
    construct_dmd_gradient,
    prepare_dmd_score_batch,
    score_forward,
)
from runtime.gst_pool_trace import load_gst_trace_entry
from runtime.operation_trace import tensor_sha256


def _parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--run-id", required=True)
    result.add_argument("--candidate-manifest", type=Path, required=True)
    result.add_argument("--contract-digest", required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--exchange-dir", type=Path, required=True)
    result.add_argument("--rendezvous-dir", type=Path, required=True)
    result.add_argument("--trace-bundle", type=Path, required=True)
    result.add_argument("--score-root", type=Path, required=True)
    result.add_argument("--precast-score-root", type=Path, required=True)
    result.add_argument("--target-package", type=Path, required=True)
    result.add_argument("--negative-prompt-asset", type=Path, required=True)
    result.add_argument("--timeout-seconds", type=float, default=7200.0)
    result.add_argument("--construction-wave", type=int, default=4)
    return result


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _snapshot(device: torch.device, stage: str) -> dict[str, Any]:
    free, total = torch.cuda.mem_get_info(device)
    return {
        "stage": stage,
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "driver_free_bytes": int(free),
        "driver_total_bytes": int(total),
    }


def _ready(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(f"pid={os.getpid()}\ntime={time.time()}\n")
    os.replace(temporary, path)


def _resident_inventory(model: torch.nn.Module) -> dict[str, int]:
    parameters = list(model.parameters())
    non_cuda = [parameter for parameter in parameters if parameter.device.type != "cuda"]
    if non_cuda:
        raise AssertionError(f"resident Score has {len(non_cuda)} non-CUDA parameters")
    return {
        "parameter_count": len(parameters),
        "cuda_bytes": sum(p.numel() * p.element_size() for p in parameters),
    }


def _construct(
    args: argparse.Namespace,
    *,
    rank: int,
    role: str,
    package: PoolTargetPackage,
    device: torch.device,
):
    model = None
    metadata = None
    for begin in range(0, 8, args.construction_wave):
        if begin <= rank < begin + args.construction_wave:
            model, metadata = build_helios_pool_model(
                role="critic" if role == "student" else "teacher",
                model_root=args.score_root,
                subfolder="transformer",
                precast_model_root=args.precast_score_root,
            )
            if role == "student":
                restore = load_logical_master_into_trainables(
                    model,
                    package,
                    component="critic",
                    verify_payload_sha256=(rank == 0),
                )
                model.enable_adapters()
                model.requires_grad_(False)
                metadata["logical_restore"] = restore
                metadata["parameter_version"] = 11
            else:
                metadata["parameter_version"] = "official_frozen_base"
            model.to(device)
            model.train()
        dist.barrier()
    if model is None or metadata is None:
        raise AssertionError("Score construction wave skipped a rank")
    return model, metadata


def _load_input(
    args: argparse.Namespace,
    *,
    source_id: int,
    entry,
) -> tuple[dict[str, Any], torch.Tensor, float]:
    path = exchange_paths(args.exchange_dir, source_id)["input"]
    waited = wait_for_file(path, args.timeout_seconds)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    x0 = validate_input_payload(
        payload,
        run_id=args.run_id,
        source_id=source_id,
        trace_identity_sha256=entry.identity_sha256,
    )
    return payload, x0, waited


def main() -> int:
    args = _parser().parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    if dist.get_world_size() != 8 or rank != local_rank:
        raise RuntimeError("N2 Score executor requires one local world8")
    if args.construction_wave <= 0 or 8 % args.construction_wave:
        raise ValueError("construction wave must divide eight ranks")
    role = "student" if rank < 4 else "teacher"
    lane = rank if role == "student" else rank - 4
    output = args.output_dir / role / f"rank_{rank:02d}.json"
    result: dict[str, Any] = {
        "schema_version": "distillschedule.gst_pool_score_rank.v1",
        "status": "RUNNING",
        "run_id": args.run_id,
        "rank": rank,
        "role": role,
        "lane": lane,
        "student_residency": "resident",
        "teacher_residency": "resident",
        "samples": [],
        "memory": [],
    }
    try:
        candidate = GeneratorOffloadCandidate.load(
            args.candidate_manifest,
            expected_contract_digest=args.contract_digest,
        )
        if candidate.run_id != args.run_id:
            raise ValueError("Score run ID differs from the immutable candidate")
        result["candidate_manifest_sha256"] = candidate.manifest_sha256
        result["contract_digest"] = candidate.contract_digest
        package = PoolTargetPackage.load(args.target_package)
        negative = NegativePromptAsset.load(
            args.negative_prompt_asset,
            verify_payload_sha256=(rank == 4),
        )
        model, metadata = _construct(
            args,
            rank=rank,
            role=role,
            package=package,
            device=device,
        )
        result["initialization"] = {
            "model": metadata,
            "resident": _resident_inventory(model),
            "negative_prompt_manifest_hash": negative.manifest["manifest_hash"],
            "negative_prompt_tensor_sha256": negative.manifest["payload"][
                "tensor_sha256"
            ],
        }
        negative_embedding = negative.embedding.to(
            device=device, dtype=torch.bfloat16
        )
        result["memory"].append(_snapshot(device, "resident_ready"))
        dist.barrier()
        if rank == 0:
            _ready(args.rendezvous_dir / "ST_READY")
        wait_for_file(args.rendezvous_dir / "G_READY", args.timeout_seconds)
        dist.barrier()

        for source_id in lane_source_ids(lane):
            entry = load_gst_trace_entry(
                args.trace_bundle, source_id=source_id, device=device
            )
            payload, x0_cpu, input_wait = _load_input(
                args, source_id=source_id, entry=entry
            )
            x0 = x0_cpu.to(device=device, dtype=torch.bfloat16)
            batch = prepare_dmd_score_batch(
                generated_x0=x0, entry=entry, device=device
            )
            paths = exchange_paths(args.exchange_dir, source_id)
            started = time.perf_counter()
            if role == "student":
                fake = score_forward(
                    model,
                    batch,
                    condition=batch["encoder_hidden_states"],
                    adapter_enabled=True,
                )
                torch.cuda.synchronize(device)
                fake_cpu = fake.cpu()
                student_payload = {
                    "schema_version": STUDENT_SCHEMA,
                    "execution_graph": "target_no_gan_gst_pool",
                    "run_id": args.run_id,
                    **source_identity(source_id),
                    "logical_step": 15,
                    "generator_version": 3,
                    "student_version": 11,
                    "trace_identity_sha256": entry.identity_sha256,
                    "x0_sha256": payload["x0_sha256"],
                    "noisy_sha256": tensor_sha256(batch["hidden_states"]),
                    "fake_flow_cond": fake_cpu,
                    "fake_flow_cond_sha256": tensor_sha256(fake_cpu),
                }
                atomic_torch_save(paths["student"], student_payload)
                record = {
                    "source_id": source_id,
                    "trace_identity_sha256": entry.identity_sha256,
                    "x0_sha256": payload["x0_sha256"],
                    "fake_flow_cond_sha256": student_payload[
                        "fake_flow_cond_sha256"
                    ],
                    "input_wait_seconds": input_wait,
                    "compute_seconds": time.perf_counter() - started,
                }
                del fake, fake_cpu, student_payload
            else:
                real_cond = score_forward(
                    model,
                    batch,
                    condition=batch["encoder_hidden_states"],
                    adapter_enabled=False,
                )
                real_uncond = score_forward(
                    model,
                    batch,
                    condition=negative_embedding,
                    adapter_enabled=False,
                )
                student_wait = wait_for_file(
                    paths["student"], args.timeout_seconds
                )
                student = torch.load(
                    paths["student"], map_location="cpu", weights_only=False
                )
                expected = {
                    "schema_version": STUDENT_SCHEMA,
                    "execution_graph": "target_no_gan_gst_pool",
                    "run_id": args.run_id,
                    **source_identity(source_id),
                    "logical_step": 15,
                    "generator_version": 3,
                    "student_version": 11,
                    "trace_identity_sha256": entry.identity_sha256,
                    "x0_sha256": payload["x0_sha256"],
                    "noisy_sha256": tensor_sha256(batch["hidden_states"]),
                }
                for key, value in expected.items():
                    if student.get(key) != value:
                        raise ValueError(f"Student result mismatch for {key}")
                fake_cpu = student["fake_flow_cond"]
                if tensor_sha256(fake_cpu) != student.get(
                    "fake_flow_cond_sha256"
                ):
                    raise ValueError("Student fake-flow digest mismatch")
                fake = fake_cpu.to(device=device, dtype=torch.bfloat16)
                gradient, auxiliary = construct_dmd_gradient(
                    batch=batch,
                    fake_flow_cond=fake,
                    real_flow_cond=real_cond,
                    real_flow_uncond=real_uncond,
                )
                torch.cuda.synchronize(device)
                gradient_cpu = gradient.cpu()
                gradient_payload = {
                    "schema_version": GRADIENT_SCHEMA,
                    "execution_graph": "target_no_gan_gst_pool",
                    "run_id": args.run_id,
                    **source_identity(source_id),
                    "logical_step": 15,
                    "generator_version": 3,
                    "student_version": 11,
                    "teacher_version": "official_frozen_base",
                    "trace_identity_sha256": entry.identity_sha256,
                    "x0_sha256": payload["x0_sha256"],
                    "input_gradient": gradient_cpu,
                    "input_gradient_sha256": tensor_sha256(gradient_cpu),
                    "normalizer": [
                        float(value)
                        for value in auxiliary["normalizer"].float().flatten().cpu()
                    ],
                }
                atomic_torch_save(paths["gradient"], gradient_payload)
                record = {
                    "source_id": source_id,
                    "trace_identity_sha256": entry.identity_sha256,
                    "x0_sha256": payload["x0_sha256"],
                    "fake_flow_cond_sha256": student[
                        "fake_flow_cond_sha256"
                    ],
                    "input_gradient_sha256": gradient_payload[
                        "input_gradient_sha256"
                    ],
                    "input_wait_seconds": input_wait,
                    "student_wait_seconds": student_wait,
                    "compute_seconds": time.perf_counter() - started,
                }
                del (
                    real_cond,
                    real_uncond,
                    student,
                    fake_cpu,
                    fake,
                    gradient,
                    gradient_cpu,
                    gradient_payload,
                    auxiliary,
                )
            record["resident"] = _resident_inventory(model)
            result["samples"].append(record)
            del entry, payload, x0_cpu, x0, batch
            torch.cuda.empty_cache()
        if len(result["samples"]) != 4:
            raise AssertionError("each resident Score lane must process four samples")
        result["memory"].append(_snapshot(device, "complete"))
        result["status"] = "PASS"
        _atomic_json(output, result)
        dist.barrier()
        return 0
    except BaseException as exc:
        is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or (
            "out of memory" in str(exc).lower()
        )
        result.update(
            {
                "status": "OOM" if is_oom else "FAIL",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        with contextlib.suppress(BaseException):
            result["memory"].append(_snapshot(device, "failure"))
        _atomic_json(output, result)
        return 42 if is_oom else 1
    finally:
        with contextlib.suppress(BaseException):
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())

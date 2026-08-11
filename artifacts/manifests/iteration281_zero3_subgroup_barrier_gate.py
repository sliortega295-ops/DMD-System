#!/usr/bin/env python3
"""Real DeepSpeed gate for ZeRO-3 inside a strict default-world subgroup."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from runtime.critic_pool import StudentDPGroupAdapter
from runtime.deepspeed_backend import (
    build_client_optimizer_offload_overrides,
    build_zero_optimization_config,
    deepspeed_subgroup_default_barriers,
)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import deepspeed

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 4:
        raise AssertionError(f"gate requires world4, got {world}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    control = dist.new_group(ranks=[0, 1, 2, 3], backend="gloo")
    dp = dist.new_group(ranks=[0, 1], backend="nccl")
    singleton_groups = {
        member: dist.new_group(ranks=[member], backend="nccl")
        for member in range(4)
    }
    dist.barrier(group=control)

    value = {
        "rank": rank,
        "world_size": world,
        "dp_members": [0, 1],
        "role": "zero3" if rank < 2 else "nonmember_control_wait",
    }
    if rank < 2:
        model = torch.nn.Sequential(
            torch.nn.Linear(16, 32, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(32, 8, bias=False),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        adapter = StudentDPGroupAdapter(
            data_parallel_group=dp,
            data_parallel_ranks=(0, 1),
            global_rank=rank,
            singleton_model_parallel_group=singleton_groups[rank],
        )
        zero = build_zero_optimization_config(
            stage=3,
            reduce_bucket_mib=1,
            allgather_bucket_mib=1,
            overlap_comm=False,
            contiguous_gradients=False,
            stage3_param_persistence_threshold=0,
            optimizer_offload_device="cpu",
            optimizer_offload_pin_memory=False,
        )
        config = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "train_batch_size": 2,
            "fp16": {"enabled": False},
            "bf16": {"enabled": True},
            "zero_allow_untested_optimizer": True,
            "zero_optimization": zero,
        }
        config.update(build_client_optimizer_offload_overrides("cpu"))
        with deepspeed_subgroup_default_barriers(dp) as audit:
            engine, _, _, _ = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                lr_scheduler=scheduler,
                config=config,
                mpu=adapter,
                dist_init_required=False,
            )
        if audit["redirected_bare_barriers"] < 3:
            raise AssertionError(audit)
        dp_world_size = getattr(engine, "dp_world_size")
        dp_world_size = int(dp_world_size() if callable(dp_world_size) else dp_world_size)
        if int(engine.zero_optimization_stage()) != 3:
            raise AssertionError("not ZeRO-3")
        if dp_world_size != 2:
            raise AssertionError(f"wrong DP size: {dp_world_size}")
        value.update(
            {
                "status": "ZERO3_SUBGROUP_READY",
                "redirected_bare_barriers": audit["redirected_bare_barriers"],
                "explicit_barriers": audit["explicit_barriers"],
                "zero_stage": int(engine.zero_optimization_stage()),
                "dp_world_size": dp_world_size,
            }
        )
    else:
        value["status"] = "CONTROL_WAIT_READY"

    # Nonmembers deliberately reach this default-world control fence while the
    # DP2 members initialize ZeRO-3.  Completion proves no ZeRO bare barrier
    # escaped to the default world4 NCCL group.
    dist.barrier(group=control)
    value["world4_flush"] = True
    atomic_json(args.output_dir / f"rank_{rank:02d}.json", value)
    dist.barrier(group=control)
    if rank == 0:
        values = [
            json.loads((args.output_dir / f"rank_{member:02d}.json").read_text())
            for member in range(4)
        ]
        summary = {
            "schema_version": "distillschedule.zero3_subgroup_barrier_gate.v1",
            "status": "PASS",
            "world_size": 4,
            "dp_members": [0, 1],
            "nonmembers": [2, 3],
            "redirected_bare_barriers": [
                item["redirected_bare_barriers"] for item in values[:2]
            ],
            "all_world4_flushed": all(item["world4_flush"] for item in values),
        }
        atomic_json(args.output_dir / "summary.json", summary)
    dist.barrier(group=control)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

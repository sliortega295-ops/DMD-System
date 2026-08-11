#!/usr/bin/env python3
"""DP2 ZeRO-3 gate with rank-divergent logical exits and padded physical calls."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from helios.utils.utils_helios_post import collective_padded_denoising_slots
from runtime.critic_pool import StudentDPGroupAdapter
from runtime.deepspeed_backend import (
    build_client_optimizer_offload_overrides,
    build_zero_optimization_config,
    deepspeed_subgroup_default_barriers,
)
from runtime.official_baseline import trainable_checksum


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
    if dist.get_world_size() != 4:
        raise AssertionError("gate requires world4")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    control = dist.new_group(ranks=[0, 1, 2, 3], backend="gloo")
    dp = dist.new_group(ranks=[0, 1], backend="nccl")
    singleton = {
        member: dist.new_group(ranks=[member], backend="nccl")
        for member in range(4)
    }
    dist.barrier(group=control)
    value = {"rank": rank, "role": "zero3" if rank < 2 else "control"}

    if rank < 2:
        torch.manual_seed(20260810)
        model = torch.nn.Sequential(
            torch.nn.Linear(16, 32, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(32, 16, bias=False),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        adapter = StudentDPGroupAdapter(
            data_parallel_group=dp,
            data_parallel_ranks=(0, 1),
            global_rank=rank,
            singleton_model_parallel_group=singleton[rank],
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
        atomic_json(args.output_dir / f"stage_rank_{rank:02d}.json", {"stage": "before_initialize", "rank": rank})
        with deepspeed_subgroup_default_barriers(dp) as audit:
            engine, _, _, _ = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                lr_scheduler=scheduler,
                config=config,
                mpu=adapter,
                dist_init_required=False,
            )
        atomic_json(args.output_dir / f"stage_rank_{rank:02d}.json", {"stage": "after_initialize", "rank": rank})
        before = trainable_checksum(engine.module)
        atomic_json(args.output_dir / f"stage_rank_{rank:02d}.json", {"stage": "after_checksum", "rank": rank})
        logical_exit = 0 if rank == 0 else 2
        physical_exit = 2
        x = torch.full((1, 16), float(rank + 1), device=device, dtype=torch.bfloat16)
        selected = None
        modes = []
        for physical_index, _timestep, is_padding, is_selected in collective_padded_denoising_slots(
            [1000, 750, 500],
            logical_exit_flag=logical_exit,
            physical_exit_flag=physical_exit,
        ):
            atomic_json(
                args.output_dir / f"stage_rank_{rank:02d}.json",
                {"stage": "before_forward", "rank": rank, "physical_index": physical_index, "is_padding": is_padding, "is_selected": is_selected},
            )
            if is_padding:
                with torch.no_grad():
                    discarded = engine(x.detach())
                del discarded
                modes.append("padding")
            elif is_selected:
                selected = engine(x)
                modes.append("selected")
            else:
                with torch.no_grad():
                    x = engine(x)
                modes.append("prefix")
            atomic_json(
                args.output_dir / f"stage_rank_{rank:02d}.json",
                {"stage": "after_forward", "rank": rank, "physical_index": physical_index, "modes": modes},
            )
        if selected is None:
            raise AssertionError("selected graph missing")
        loss = selected.float().square().mean()
        atomic_json(args.output_dir / f"stage_rank_{rank:02d}.json", {"stage": "before_backward", "rank": rank, "modes": modes})
        engine.backward(loss)
        atomic_json(args.output_dir / f"stage_rank_{rank:02d}.json", {"stage": "after_backward", "rank": rank, "modes": modes})
        engine.step()
        atomic_json(args.output_dir / f"stage_rank_{rank:02d}.json", {"stage": "after_step", "rank": rank, "modes": modes})
        torch.cuda.synchronize(device)
        after = trainable_checksum(engine.module)
        if int(engine.global_steps) != 1 or before == after:
            raise AssertionError("padded gate did not perform one optimizer update")
        value.update(
            {
                "status": "PASS",
                "logical_exit": logical_exit,
                "physical_exit": physical_exit,
                "modes": modes,
                "physical_calls": len(modes),
                "selected_calls": modes.count("selected"),
                "global_steps": int(engine.global_steps),
                "loss": float(loss.detach().cpu()),
                "trainable_before": before,
                "trainable_after": after,
                "redirected_bare_barriers": audit["redirected_bare_barriers"],
            }
        )
    else:
        value["status"] = "CONTROL_WAIT"

    dist.barrier(group=control)
    value["world4_flush"] = True
    atomic_json(args.output_dir / f"rank_{rank:02d}.json", value)
    dist.barrier(group=control)
    if rank == 0:
        rows = [
            json.loads((args.output_dir / f"rank_{member:02d}.json").read_text())
            for member in range(4)
        ]
        summary = {
            "schema_version": "distillschedule.zero3_padded_subgroup_gate.v1",
            "status": "PASS",
            "dp_members": [0, 1],
            "logical_exits": [rows[0]["logical_exit"], rows[1]["logical_exit"]],
            "physical_calls": [rows[0]["physical_calls"], rows[1]["physical_calls"]],
            "selected_calls": [rows[0]["selected_calls"], rows[1]["selected_calls"]],
            "modes": [rows[0]["modes"], rows[1]["modes"]],
            "global_steps": [rows[0]["global_steps"], rows[1]["global_steps"]],
            "all_world4_flushed": all(row["world4_flush"] for row in rows),
        }
        atomic_json(args.output_dir / "summary.json", summary)
    dist.barrier(group=control)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

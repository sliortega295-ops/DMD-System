#!/usr/bin/env python3
"""Two-node phase-world gate: local default ZeRO-3 -> global target world."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from helios.utils.utils_helios_post import collective_padded_denoising_slots
from runtime.critic_pool import StudentDPGroupAdapter
from runtime.deepspeed_backend import (
    build_client_optimizer_offload_overrides,
    build_zero_optimization_config,
)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def wait_for(path: Path, timeout_s: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(path)
        time.sleep(0.05)


def checksum(parameters) -> dict[str, float | int]:
    total = 0.0
    square = 0.0
    numel = 0
    for parameter in parameters:
        value = parameter.detach().float()
        total += float(value.sum().cpu())
        square += float(value.square().sum().cpu())
        numel += value.numel()
    return {"sum": total, "square": square, "numel": numel}


def zero_config(stage: int, train_batch_size: int) -> dict[str, Any]:
    zero = build_zero_optimization_config(
        stage=stage,
        reduce_bucket_mib=1,
        allgather_bucket_mib=1,
        overlap_comm=False,
        contiguous_gradients=False,
        stage3_param_persistence_threshold=0,
        optimizer_offload_device="cpu",
        optimizer_offload_pin_memory=False,
    )
    value = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "train_batch_size": train_batch_size,
        "fp16": {"enabled": False},
        "bf16": {"enabled": True},
        "zero_allow_untested_optimizer": True,
        "zero_optimization": zero,
    }
    value.update(build_client_optimizer_offload_overrides("cpu"))
    return value


def make_model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32, bias=False),
        torch.nn.GELU(),
        torch.nn.Linear(32, 16, bias=False),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--phase-b-master-addr", required=True)
    parser.add_argument("--phase-b-master-port", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import deepspeed

    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 2:
        raise AssertionError("each Phase-A node must own an independent default world2")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
    if dist.get_world_size() != 2 or dist.get_rank() != local_rank:
        raise AssertionError("Phase-A default world is not node-local world2")

    global_rank = args.node_rank * 2 + local_rank
    output = args.output_dir / f"rank_{global_rank:02d}.json"
    value: dict[str, Any] = {
        "status": "RUNNING",
        "node_rank": args.node_rank,
        "local_rank": local_rank,
        "global_rank": global_rank,
        "phase_a_world": 2,
        "phase_a_default_rank": dist.get_rank(),
    }
    atomic_json(output, value)

    engine = None
    gathered: list[torch.Tensor] | None = None
    if args.node_rank == 0:
        torch.manual_seed(20260810)
        model = make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        engine, _, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=zero_config(3, 2),
            dist_init_required=False,
        )
        logical_exit = 0 if local_rank == 0 else 2
        x = torch.full((1, 16), float(local_rank + 1), device=device, dtype=torch.bfloat16)
        selected = None
        modes: list[str] = []
        for _index, _timestep, is_padding, is_selected in collective_padded_denoising_slots(
            [1000, 750, 500], logical_exit_flag=logical_exit, physical_exit_flag=2
        ):
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
        if selected is None:
            raise AssertionError("Phase-A selected graph missing")
        engine.backward(selected.float().square().mean())
        engine.step()
        if int(engine.global_steps) != 1:
            raise AssertionError("Phase-A G version did not advance exactly once")
        gathered = [] if local_rank == 0 else None
        for parameter in engine.module.parameters():
            full = engine.optimizer.get_full_hp_param(parameter)
            if local_rank == 0:
                gathered.append(full.detach().to(device="cpu", dtype=torch.bfloat16).contiguous())
            del full
        value["phase_a"] = {
            "role": "generator_zero3",
            "zero_stage": int(engine.zero_optimization_stage()),
            "modes": modes,
            "global_steps": int(engine.global_steps),
        }
    else:
        value["phase_a"] = {"role": "score_teacher_control"}

    dist.barrier()
    if local_rank == 0:
        atomic_json(args.output_dir / f"phase_a_node_{args.node_rank}.ready", {
            "node_rank": args.node_rank, "status": "READY"
        })
        wait_for(args.output_dir / f"phase_a_node_{1-args.node_rank}.ready")
    dist.barrier()

    if engine is not None:
        engine.destroy()
        engine.module = None
        engine.optimizer = None
        engine.lr_scheduler = None
        engine = None
    torch.cuda.synchronize(device)
    dist.barrier()
    dist.destroy_process_group()
    if dist.is_initialized():
        raise AssertionError("Phase-A default process group survived destroy")

    os.environ["RANK"] = str(global_rank)
    os.environ["WORLD_SIZE"] = "4"
    os.environ["MASTER_ADDR"] = args.phase_b_master_addr
    os.environ["MASTER_PORT"] = str(args.phase_b_master_port)
    # torchrun sets this to True because its agent owns the Phase-A rendezvous.
    # Phase-B intentionally uses a new port and needs global rank 0 to create
    # that TCPStore, so retaining the flag makes every rank a client and no
    # server exists. Removing only this launcher hint does not alter tensor or
    # collective semantics; the explicit rank/world contract below is frozen.
    elastic_agent_store = os.environ.pop("TORCHELASTIC_USE_AGENT_STORE", None)
    value["phase_transition"] = {
        "torchelastic_use_agent_store_before": elastic_agent_store,
        "torchelastic_use_agent_store_after": os.environ.get("TORCHELASTIC_USE_AGENT_STORE"),
    }
    atomic_json(output, value)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://{args.phase_b_master_addr}:{args.phase_b_master_port}",
        rank=global_rank,
        world_size=4,
        timeout=datetime.timedelta(seconds=120),
    )
    if dist.get_world_size() != 4 or dist.get_rank() != global_rank:
        raise AssertionError("Phase-B default world4 rank mapping changed")
    # Node 1 did not construct a DeepSpeed engine in Phase A, so its process
    # has no cdb wrapper yet. Attach all ranks to the already-created torch
    # world before checking or constructing the target ZeRO-1 subgroup.
    deepspeed.init_distributed(dist_backend="nccl", dist_init_required=False)
    if int(deepspeed.comm.get_world_size()) != 4:
        raise AssertionError("DeepSpeed comm backend retained the Phase-A world")

    generator_ranks = [0, 2]
    critic_ranks = [1, 3]
    generator_group = dist.new_group(ranks=generator_ranks, backend="nccl")
    critic_group = dist.new_group(ranks=critic_ranks, backend="nccl")
    singleton = {r: dist.new_group(ranks=[r], backend="nccl") for r in range(4)}
    dist.barrier()

    if global_rank in generator_ranks:
        torch.manual_seed(20260810)
        target = make_model().to(device=device, dtype=torch.bfloat16)
        trainables = list(target.parameters())
        if global_rank == 0:
            if gathered is None or len(gathered) != len(trainables):
                raise AssertionError("rank0 lost gathered G1 logical state")
            with torch.no_grad():
                for destination, source in zip(trainables, gathered):
                    destination.copy_(source.to(device=device))
        for parameter in trainables:
            dist.broadcast(parameter.data, src=0, group=generator_group)
        torch.cuda.synchronize(device)
        local_checksum = checksum(trainables)
        checksum_tensor = torch.tensor(
            [local_checksum["sum"], local_checksum["square"]],
            dtype=torch.float64,
            device=device,
        )
        gathered_checksums = [torch.zeros_like(checksum_tensor) for _ in generator_ranks]
        dist.all_gather(gathered_checksums, checksum_tensor, group=generator_group)
        if not torch.equal(gathered_checksums[0], gathered_checksums[1]):
            raise AssertionError("G1 redistribution differs across target Generator ranks")
        value["phase_b"] = {
            "role": "resident_generator",
            "generator_version": 1,
            "checksum": local_checksum,
        }
    else:
        torch.manual_seed(20260811)
        model = make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        adapter = StudentDPGroupAdapter(
            data_parallel_group=critic_group,
            data_parallel_ranks=critic_ranks,
            global_rank=global_rank,
            singleton_model_parallel_group=singleton[global_rank],
        )
        critic_engine, _, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=zero_config(1, 2),
            mpu=adapter,
            dist_init_required=False,
        )
        if int(critic_engine.zero_optimization_stage()) != 1:
            raise AssertionError("Phase-B Critic did not initialize ZeRO-1")
        if int(critic_engine.dp_world_size) != 2:
            raise AssertionError("Phase-B Critic DP group changed")
        x = torch.full((1, 16), float(global_rank + 1), device=device, dtype=torch.bfloat16)
        loss = critic_engine(x).float().square().mean()
        critic_engine.backward(loss)
        critic_engine.step()
        if int(critic_engine.global_steps) != 1:
            raise AssertionError("Phase-B Critic did not commit once")
        value["phase_b"] = {
            "role": "critic_zero1",
            "zero_stage": int(critic_engine.zero_optimization_stage()),
            "critic_version": 1,
            "dp_world_size": int(critic_engine.dp_world_size),
        }
        critic_engine.destroy()

    dist.barrier()
    value.update({"status": "PASS", "phase_b_world": 4, "world4_flush": True})
    atomic_json(output, value)
    dist.barrier()
    if global_rank == 0:
        rows = [json.loads((args.output_dir / f"rank_{r:02d}.json").read_text()) for r in range(4)]
        atomic_json(args.output_dir / "summary.json", {
            "schema_version": "distillschedule.phase_reinitialization_gate.v1",
            "status": "PASS",
            "phase_a_worlds": {"node0": 2, "node1": 2},
            "phase_b_world": 4,
            "roles": [row["phase_b"]["role"] for row in rows],
            "all_flushed": all(row["world4_flush"] for row in rows),
            "generator_versions": [rows[r]["phase_b"]["generator_version"] for r in generator_ranks],
            "critic_versions": [rows[r]["phase_b"]["critic_version"] for r in critic_ranks],
        })
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

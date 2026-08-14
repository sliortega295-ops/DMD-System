#!/usr/bin/env python3
"""Bounded single-node gate for the Phase-B communicator-collapse candidate.

This is a tiny real-NCCL/DeepSpeed test, not a Helios performance result.  It
checks two proposed representation-transition changes before the 14B executor
may use them:

* disjoint Generator->Critic P2P routes share the default NCCL world; and
* ZeRO-1 DP6 represents model-parallel world size one without singleton NCCL
  process groups.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from runtime.critic_pool import StudentDPGroupAdapter
from runtime.deepspeed_backend import (
    build_client_optimizer_offload_overrides,
    build_zero_optimization_config,
)

WORLD_SIZE = 8
GENERATOR_RANKS = (0, 1)
CRITIC_RANKS = (2, 3, 4, 5, 6, 7)
ROUTES = {0: (2, 3, 4), 1: (5, 6, 7)}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _config() -> dict[str, Any]:
    zero = build_zero_optimization_config(
        stage=1,
        reduce_bucket_mib=1,
        allgather_bucket_mib=1,
        overlap_comm=False,
        contiguous_gradients=False,
        stage3_param_persistence_threshold=0,
        optimizer_offload_device="cpu",
        optimizer_offload_pin_memory=False,
    )
    config: dict[str, Any] = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "train_batch_size": len(CRITIC_RANKS),
        "fp16": {"enabled": False},
        "bf16": {"enabled": True},
        "communication_data_type": "fp32",
        "gradient_clipping": 1.0,
        "zero_allow_untested_optimizer": True,
        "zero_optimization": zero,
    }
    config.update(build_client_optimizer_offload_overrides("cpu"))
    return config


def _model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32, bias=False),
        torch.nn.GELU(),
        torch.nn.Linear(32, 8, bias=False),
    )


def _producer_for(critic_rank: int) -> int:
    matches = [source for source, targets in ROUTES.items() if critic_rank in targets]
    if len(matches) != 1:
        raise AssertionError(f"critic rank {critic_rank} has invalid producer mapping")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import deepspeed

    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=180))
    rank = dist.get_rank()
    if dist.get_world_size() != WORLD_SIZE:
        raise AssertionError("communicator-collapse gate requires one world8")
    local_rank = int(os.environ["LOCAL_RANK"])
    if rank != local_rank:
        raise AssertionError("single-node gate requires global rank == local rank")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    group_started = time.perf_counter()
    control = dist.new_group(
        ranks=list(range(WORLD_SIZE)),
        backend="gloo",
        timeout=datetime.timedelta(seconds=180),
    )
    generator_group = dist.new_group(ranks=list(GENERATOR_RANKS), backend="nccl")
    critic_group = dist.new_group(ranks=list(CRITIC_RANKS), backend="nccl")
    dist.barrier(group=control)
    # A default-world NCCL collective must initialize the full communicator
    # before role-asymmetric unbatched P2P begins.  Without this fence the first
    # two-rank operation can enter lazy communicator creation on only its peers.
    dist.barrier(group=dist.group.WORLD)
    group_create_seconds = time.perf_counter() - group_started

    result: dict[str, Any] = {
        "schema_version": "distillschedule.phaseb_communicator_collapse_gate.v1",
        "status": "RUNNING",
        "rank": rank,
        "world_size": WORLD_SIZE,
        "group_create_seconds": group_create_seconds,
        "created_nccl_subgroups": 2,
        "pair_groups_created": 0,
        "singleton_groups_created": 0,
        "p2p_group": "DEFAULT_WORLD",
        "default_world_nccl_initialized_collectively": True,
    }
    _atomic_json(args.output_dir / f"rank_{rank:02d}.json", result)

    engine = None
    if rank in CRITIC_RANKS:
        torch.manual_seed(20260810)
        model = _model()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, betas=(0.0, 0.999), weight_decay=1e-3
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        adapter = StudentDPGroupAdapter(
            data_parallel_group=critic_group,
            data_parallel_ranks=CRITIC_RANKS,
            global_rank=rank,
            singleton_model_parallel_group=None,
        )
        initialize_started = time.perf_counter()
        engine, _, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=_config(),
            mpu=adapter,
            dist_init_required=False,
        )
        initialize_seconds = time.perf_counter() - initialize_started
        if int(engine.zero_optimization_stage()) != 1:
            raise AssertionError("gate Critic is not ZeRO-1")
        if int(engine.dp_world_size) != len(CRITIC_RANKS):
            raise AssertionError("gate Critic DP world changed")
        if engine.optimizer.model_parallel_group is not None:
            raise AssertionError("ZeRO-1 unexpectedly materialized an MP process group")
        if int(engine.optimizer.model_parallel_world_size) != 1:
            raise AssertionError("ZeRO-1 MP world size is not one")
        result["zero1"] = {
            "dp_world_size": int(engine.dp_world_size),
            "model_parallel_group": None,
            "model_parallel_world_size": int(
                engine.optimizer.model_parallel_world_size
            ),
            "initialize_seconds": initialize_seconds,
        }
    else:
        marker = torch.tensor([rank + 1.0], device=device)
        dist.all_reduce(marker, group=generator_group)
        if float(marker.cpu()) != 3.0:
            raise AssertionError("Generator subgroup communication failed")

    # Fail if DeepSpeed initialization escaped from DP6 to the default world8.
    dist.barrier(group=control)

    p2p_started = time.perf_counter()
    if rank in GENERATOR_RANKS:
        works = []
        retained = []
        for destination in ROUTES[rank]:
            value = rank * 100 + destination
            header = torch.tensor([rank, destination, value], dtype=torch.int64, device=device)
            payload = torch.full((1, 16), float(value), dtype=torch.bfloat16, device=device)
            retained.extend((header, payload))
            works.extend(
                (
                    dist.isend(header, dst=destination, group=dist.group.WORLD),
                    dist.isend(payload, dst=destination, group=dist.group.WORLD),
                )
            )
        for work in works:
            work.wait()
        result["p2p"] = {
            "sent": len(ROUTES[rank]),
            "destinations": list(ROUTES[rank]),
        }
    else:
        source = _producer_for(rank)
        header = torch.empty(3, dtype=torch.int64, device=device)
        payload = torch.empty((1, 16), dtype=torch.bfloat16, device=device)
        dist.recv(header, src=source, group=dist.group.WORLD)
        dist.recv(payload, src=source, group=dist.group.WORLD)
        expected = source * 100 + rank
        observed_header = [int(item) for item in header.cpu().tolist()]
        if observed_header != [source, rank, expected]:
            raise AssertionError("default-world P2P header mismatch")
        if not torch.equal(payload, torch.full_like(payload, float(expected))):
            raise AssertionError("default-world P2P payload mismatch")
        loss = engine(payload).float().square().mean()
        engine.backward(loss)
        engine.step()
        if int(engine.global_steps) != 1:
            raise AssertionError("ZeRO-1 Critic did not commit exactly once")
        result["p2p"] = {
            "received": 1,
            "source": source,
            "header": observed_header,
        }
        result["zero1"]["global_steps"] = int(engine.global_steps)
    torch.cuda.synchronize(device)
    result["p2p_and_update_seconds"] = time.perf_counter() - p2p_started

    dist.barrier(group=control)
    result["global_flush"] = True
    result["status"] = "PASS"
    _atomic_json(args.output_dir / f"rank_{rank:02d}.json", result)
    dist.barrier(group=control)

    if rank == 0:
        ranks = [
            json.loads((args.output_dir / f"rank_{member:02d}.json").read_text())
            for member in range(WORLD_SIZE)
        ]
        summary = {
            "schema_version": "distillschedule.phaseb_communicator_collapse_gate_summary.v1",
            "status": "PASS" if all(item["status"] == "PASS" for item in ranks) else "FAIL",
            "world_size": WORLD_SIZE,
            "generator_ranks": list(GENERATOR_RANKS),
            "critic_ranks": list(CRITIC_RANKS),
            "created_nccl_subgroups": 2,
            "pair_groups_created": 0,
            "singleton_groups_created": 0,
            "default_world_p2p_routes": sum(len(value) for value in ROUTES.values()),
            "critic_updates": sum(
                int(item.get("zero1", {}).get("global_steps", 0))
                for item in ranks
            ),
            "all_flushed": all(item["global_flush"] for item in ranks),
            "rank_files": [f"rank_{member:02d}.json" for member in range(WORLD_SIZE)],
        }
        if summary["critic_updates"] != len(CRITIC_RANKS):
            raise AssertionError("not every ZeRO-1 rank observed the single global update")
        _atomic_json(args.output_dir / "summary.json", summary)
    dist.barrier(group=control)
    if engine is not None:
        engine.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

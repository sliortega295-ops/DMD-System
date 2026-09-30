#!/usr/bin/env python3
"""Bounded world16 subgroup and two-wave P2P gate for LiveWan role pools.

This probe allocates no model weights.  It collectively initializes the default
world communicator before reproducing the exact rank pairing and message order
used by the 8G+8Score worker.  The test therefore catches split worlds, invalid
subgroups, or lazy NCCL communicator deadlocks before expensive construction.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import socket


WORLD_SIZE = 16
GENERATOR_RANKS = tuple(range(8))
SCORE_RANKS = tuple(range(8, 16))
GENERATOR_RANKS_6G10S = tuple(range(6))
SCORE_RANKS_6G10S = tuple(range(6, 16))
SCORE_NODE_GROUPS_6G10S = ((6, 7), tuple(range(8, 16)))
SCORE_LEADERS_6G10S = (6, 8)
TRACE_BALANCED_SLOT_PAIRS = (
    (0, 4), (1, 9), (2, 8), (3, 10),
    (5, 12), (6, 11), (7, 15), (13, 14),
)
TRACE_BALANCED_SLOT_GROUPS_B32 = tuple(
    (first, second, first + 16, second + 16)
    for first, second in TRACE_BALANCED_SLOT_PAIRS
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument(
        "--pipeline-variant",
        choices=(
            "b1", "b1-runahead", "b1-runahead-critic-b2",
            "b1-runahead-critic-b2-local-context",
            "b1-runahead-critic-b2-local-context-async32",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-context-prefetch",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-scale1",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-scale1-modulation",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-fused-cast-add",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-autocast-input-elision",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-autocast-input-elision-split-dmd-shadow",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-autocast-input-elision-split-dmd-shadow-fused-adam",
            "c15-score-rope",
            "c16-score-flash-attn2",
            "c18-generator-cross-kv-cache",
            "c19-host-runahead",
            "c20-generator-flash-attn2",
            "c21-continuous-b16",
            "c21-c1-continuous-b16",
            "c22-critic-checkpoint-rng",
            "c23-continuous-b32",
            "c24-continuous-b16-score-modulation",
            "c25-continuous-b16-score-rmsnorm", "c26-continuous-b16-critic-attention-checkpoint", "c27-continuous-b16-pinned-prompt-staging", "c28-continuous-b16-rollout-cast-cache", "c29-continuous-b16-generator-rms-epilogue", "c30-continuous-b16-generator-rms-critic-checkpoint", "c31-continuous-b16-generator-rms-cast-square", "c32-continuous-b16-generator-modulation", "c33-continuous-b16-generator-gate", "c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap",
            "c24-continuous-b16-score-modulation-b1-attribution",
            "phase-expanded-critic16",
            "phase-expanded-critic16-fused-lora-elision",
            "score-b2",
            "6g10score", "6g10score-node-local",
            "6g10score-node-local-fused",
        ),
        default="b1",
    )
    args = parser.parse_args()
    wave_indices = (
        tuple(range(4))
        if args.pipeline_variant == "c23-continuous-b32"
        else tuple(range(2))
    )
    waves = len(wave_indices)

    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != WORLD_SIZE:
        raise RuntimeError("transport probe requires world16")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=args.timeout_seconds))
    generator_ranks, score_ranks = (
        (GENERATOR_RANKS_6G10S, SCORE_RANKS_6G10S)
        if args.pipeline_variant in {
            "6g10score", "6g10score-node-local", "6g10score-node-local-fused"
        }
        else (GENERATOR_RANKS, SCORE_RANKS)
    )
    generator_group = dist.new_group(ranks=list(generator_ranks), backend="nccl")
    score_group = dist.new_group(ranks=list(score_ranks), backend="nccl")
    phase_expanded_critic16 = args.pipeline_variant in {
        "phase-expanded-critic16",
        "phase-expanded-critic16-fused-lora-elision",
    }
    critic_ranks = tuple(range(WORLD_SIZE)) if phase_expanded_critic16 else score_ranks
    critic_group = (
        dist.new_group(ranks=list(critic_ranks), backend="nccl")
        if phase_expanded_critic16
        else score_group
    )
    score_node_groups = []
    score_leader_group = None
    node_local_variant = args.pipeline_variant in {
        "6g10score-node-local", "6g10score-node-local-fused"
    }
    if node_local_variant:
        score_node_groups = [
            dist.new_group(ranks=list(ranks), backend="nccl")
            for ranks in SCORE_NODE_GROUPS_6G10S
        ]
        score_leader_group = dist.new_group(
            ranks=list(SCORE_LEADERS_6G10S), backend="nccl"
        )
    host_runahead_group = (
        dist.new_group(ranks=list(range(WORLD_SIZE)), backend="gloo")
        if args.pipeline_variant
        in {
            "c19-host-runahead",
            "c20-generator-flash-attn2",
            "c21-continuous-b16",
            "c21-c1-continuous-b16",
            "c22-critic-checkpoint-rng",
            "c23-continuous-b32",
            "c24-continuous-b16-score-modulation",
            "c25-continuous-b16-score-rmsnorm", "c26-continuous-b16-critic-attention-checkpoint", "c27-continuous-b16-pinned-prompt-staging", "c28-continuous-b16-rollout-cast-cache", "c29-continuous-b16-generator-rms-epilogue", "c30-continuous-b16-generator-rms-critic-checkpoint", "c31-continuous-b16-generator-rms-cast-square", "c32-continuous-b16-generator-modulation", "c33-continuous-b16-generator-gate", "c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap",
            "c24-continuous-b16-score-modulation-b1-attribution",
        }
        else None
    )
    critic_shadow_pair_group = None
    if args.pipeline_variant in {
        "c21-continuous-b16",
        "c21-c1-continuous-b16",
        "c22-critic-checkpoint-rng",
        "c23-continuous-b32",
        "c24-continuous-b16-score-modulation",
        "c25-continuous-b16-score-rmsnorm", "c26-continuous-b16-critic-attention-checkpoint", "c27-continuous-b16-pinned-prompt-staging", "c28-continuous-b16-rollout-cast-cache", "c29-continuous-b16-generator-rms-epilogue", "c30-continuous-b16-generator-rms-critic-checkpoint", "c31-continuous-b16-generator-rms-cast-square", "c32-continuous-b16-generator-modulation", "c33-continuous-b16-generator-gate", "c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap",
        "c24-continuous-b16-score-modulation-b1-attribution",
    }:
        pair_groups = [
            dist.new_group(ranks=[lane, lane + 8], backend="nccl")
            for lane in range(8)
        ]
        critic_shadow_pair_group = pair_groups[rank if rank < 8 else rank - 8]

    # Collectively initialize default world before asymmetric default-group P2P.
    dist.barrier()
    subgroup = generator_group if rank in generator_ranks else score_group
    subgroup_token = torch.tensor([rank + 1], dtype=torch.int64, device=device)
    dist.all_reduce(subgroup_token, group=subgroup)
    expected_subgroup = sum(
        value + 1 for value in (generator_ranks if rank in generator_ranks else score_ranks)
    )
    if int(subgroup_token.cpu()) != expected_subgroup:
        raise RuntimeError("role subgroup all-reduce mismatch")
    if host_runahead_group is not None:
        host_token = torch.ones(1, dtype=torch.int64, device="cpu")
        dist.all_reduce(host_token, group=host_runahead_group)
        if int(host_token.item()) != WORLD_SIZE:
            raise RuntimeError("host run-ahead Gloo all-reduce mismatch")
    critic_shadow_sync_canary = None
    if critic_shadow_pair_group is not None:
        lane = rank if rank < 8 else rank - 8
        source_rank = lane + 8
        value = torch.full(
            (4096,), float(source_rank + 1), dtype=torch.float32, device=device
        )
        if rank < 8:
            value.zero_()
        dist.broadcast(
            value, src=source_rank, group=critic_shadow_pair_group
        )
        if not torch.equal(
            value, torch.full_like(value, float(source_rank + 1))
        ):
            raise RuntimeError("paired Critic-shadow broadcast mismatch")
        critic_shadow_sync_canary = {
            "pair_ranks": [lane, lane + 8],
            "source_rank": source_rank,
            "dtype": "torch.float32",
            "elements": value.numel(),
            "status": "PASS",
        }

    async_reducer_canary = None
    if (
        args.pipeline_variant in {
            "b1-runahead-critic-b2-local-context-async32",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-context-prefetch",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-scale1",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-scale1-modulation",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-fused-cast-add",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-autocast-input-elision",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-autocast-input-elision-split-dmd-shadow",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-autocast-input-elision-split-dmd-shadow-fused-adam",
            "c15-score-rope",
            "c16-score-flash-attn2",
            "c18-generator-cross-kv-cache",
            "c19-host-runahead",
            "c20-generator-flash-attn2",
            "c21-continuous-b16",
            "c21-c1-continuous-b16",
            "c22-critic-checkpoint-rng",
            "c23-continuous-b32",
            "c24-continuous-b16-score-modulation",
            "c25-continuous-b16-score-rmsnorm", "c26-continuous-b16-critic-attention-checkpoint", "c27-continuous-b16-pinned-prompt-staging", "c28-continuous-b16-rollout-cast-cache", "c29-continuous-b16-generator-rms-epilogue", "c30-continuous-b16-generator-rms-critic-checkpoint", "c31-continuous-b16-generator-rms-cast-square", "c32-continuous-b16-generator-modulation", "c33-continuous-b16-generator-gate", "c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap",
            "c24-continuous-b16-score-modulation-b1-attribution",
        }
        and rank in critic_ranks
    ):
        from distillschedule.runtime.livewan_grad_reducer import (
            GradReadyAsyncGroupReducer,
        )

        parameters = [
            torch.nn.Parameter(
                torch.zeros(4096, dtype=torch.float32, device=device)
            )
            for _index in range(4)
        ]
        reducer = GradReadyAsyncGroupReducer(
            torch,
            [(f"canary_{index}", parameter) for index, parameter in enumerate(parameters)],
            group=critic_group,
            group_size=len(critic_ranks),
            bucket_bytes=16 * 1024,
        )
        reducer.zero_grad()
        local_value = float(critic_ranks.index(rank) + 1)
        canary_loss = sum(parameter.sum() * local_value for parameter in parameters)
        if args.pipeline_variant in {
            "c21-c1-continuous-b16",
            "c23-continuous-b32",
            "c24-continuous-b16-score-modulation",
            "c25-continuous-b16-score-rmsnorm", "c26-continuous-b16-critic-attention-checkpoint", "c27-continuous-b16-pinned-prompt-staging", "c28-continuous-b16-rollout-cast-cache", "c29-continuous-b16-generator-rms-epilogue", "c30-continuous-b16-generator-rms-critic-checkpoint", "c31-continuous-b16-generator-rms-cast-square", "c32-continuous-b16-generator-modulation", "c33-continuous-b16-generator-gate", "c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap",
            "c24-continuous-b16-score-modulation-b1-attribution",
        }:
            reducer.backward_accumulate(
                canary_loss * 0.5,
                accumulation_index=0,
                accumulation_count=2,
            )
            reducer.backward_accumulate(
                sum(parameter.sum() * local_value for parameter in parameters) * 0.5,
                accumulation_index=1,
                accumulation_count=2,
            )
        else:
            reducer.backward(canary_loss)
        async_reducer_canary = reducer.finish()
        expected_mean = sum(range(1, len(critic_ranks) + 1)) / len(critic_ranks)
        if any(
            not torch.equal(
                parameter.grad,
                torch.full_like(parameter.grad, expected_mean),
            )
            for parameter in parameters
        ):
            raise RuntimeError("async Score gradient mean mismatch")
        if async_reducer_canary["buckets"] != 4:
            raise RuntimeError("async Score gradient bucket inventory mismatch")
        async_reducer_canary["expected_mean"] = expected_mean
        del reducer, parameters

    critic_fsdp_canary = None
    if phase_expanded_critic16:
        # Match the new Critic placement semantics without constructing a model:
        # one world16 reduce-scatter SUM followed by the registered /16 mean.
        local = torch.full(
            (WORLD_SIZE * 4096,),
            float(rank + 1),
            dtype=torch.float32,
            device=device,
        )
        reduced = torch.empty(4096, dtype=torch.float32, device=device)
        dist.reduce_scatter_tensor(
            reduced, local, op=dist.ReduceOp.SUM, group=critic_group
        )
        reduced.mul_(1.0 / WORLD_SIZE)
        expected_mean = sum(range(1, WORLD_SIZE + 1)) / WORLD_SIZE
        if not torch.equal(reduced, torch.full_like(reduced, expected_mean)):
            raise RuntimeError("world16 FSDP-like Critic reduce-scatter mean mismatch")
        critic_fsdp_canary = {
            "backend": "FSDP2_REDUCE_SCATTER",
            "group_size": WORLD_SIZE,
            "reduction": "SUM_THEN_DIVIDE_GROUP_SIZE",
            "expected_mean": expected_mean,
            "output_bytes": reduced.numel() * reduced.element_size(),
        }

    node_local_collective = None
    if node_local_variant and rank in score_ranks:
        group_index = next(
            index for index, ranks in enumerate(SCORE_NODE_GROUPS_6G10S) if rank in ranks
        )
        local_ranks = SCORE_NODE_GROUPS_6G10S[group_index]
        local_group = score_node_groups[group_index]
        # Exercise a real FSDP-like 8 MiB shard all-gather, but only inside the
        # node-local Teacher mesh.  The failed Score10 path is never invoked.
        local_shard = torch.full(
            (4 * 1024 * 1024,), float(rank), dtype=torch.bfloat16, device=device
        )
        gathered = torch.empty(
            (len(local_ranks) * local_shard.numel(),),
            dtype=local_shard.dtype,
            device=device,
        )
        dist.all_gather_into_tensor(gathered, local_shard, group=local_group)
        for index, member in enumerate(local_ranks):
            observed = gathered[
                index * local_shard.numel() : (index + 1) * local_shard.numel()
            ]
            if not torch.equal(observed, torch.full_like(local_shard, float(member))):
                raise RuntimeError("node-local Teacher all-gather payload mismatch")

        # Exercise the exact Critic gradient tree: local SUM, two-leader SUM,
        # local broadcast, divide by ten.
        reduced = torch.full(
            (2 * 1024 * 1024,), float(rank + 1), dtype=torch.float32, device=device
        )
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=local_group)
        if rank in SCORE_LEADERS_6G10S:
            if score_leader_group is None:
                raise RuntimeError("Score leader group is missing")
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=score_leader_group)
        dist.broadcast(reduced, src=local_ranks[0], group=local_group)
        reduced.mul_(1.0 / len(score_ranks))
        expected_mean = sum(member + 1 for member in score_ranks) / len(score_ranks)
        if not torch.equal(reduced, torch.full_like(reduced, expected_mean)):
            raise RuntimeError("hierarchical Score mean mismatch")
        node_local_collective = {
            "teacher_mesh_ranks": list(local_ranks),
            "teacher_allgather_local_bytes": local_shard.numel() * local_shard.element_size(),
            "teacher_allgather_output_bytes": gathered.numel() * gathered.element_size(),
            "critic_sync_local_ranks": list(local_ranks),
            "critic_sync_leader_ranks": list(SCORE_LEADERS_6G10S),
            "critic_sync_buffer_bytes": reduced.numel() * reduced.element_size(),
            "large_score10_collective": False,
        }

    if args.pipeline_variant in {
        "6g10score", "6g10score-node-local", "6g10score-node-local-fused"
    }:
        generator_by_slot = {
            slot: generator_ranks[slot % len(generator_ranks)] for slot in range(WORLD_SIZE)
        }
        score_by_slot = {
            slot: score_ranks[slot % len(score_ranks)] for slot in range(WORLD_SIZE)
        }
        forward_ops = []
        reverse_ops = []
        received_headers: dict[int, torch.Tensor] = {}
        received_payloads: dict[int, torch.Tensor] = {}
        acknowledgements: dict[int, torch.Tensor] = {}
        local_slots: list[int] = []
        retained: list[torch.Tensor] = []
        for slot in range(WORLD_SIZE):
            generator_rank = generator_by_slot[slot]
            score_rank = score_by_slot[slot]
            if rank == generator_rank:
                header = torch.tensor(
                    [slot, generator_rank, score_rank], dtype=torch.int64, device=device
                )
                payload = torch.full(
                    (256,), float(slot), dtype=torch.bfloat16, device=device
                )
                acknowledgement = torch.empty(1, dtype=torch.int64, device=device)
                retained.extend((header, payload))
                acknowledgements[slot] = acknowledgement
                forward_ops.extend(
                    (
                        dist.P2POp(dist.isend, header, score_rank),
                        dist.P2POp(dist.isend, payload, score_rank),
                    )
                )
                reverse_ops.append(
                    dist.P2POp(dist.irecv, acknowledgement, score_rank)
                )
                local_slots.append(slot)
            elif rank == score_rank:
                header = torch.empty(3, dtype=torch.int64, device=device)
                payload = torch.empty(256, dtype=torch.bfloat16, device=device)
                acknowledgement = torch.tensor([slot], dtype=torch.int64, device=device)
                received_headers[slot] = header
                received_payloads[slot] = payload
                retained.append(acknowledgement)
                forward_ops.extend(
                    (
                        dist.P2POp(dist.irecv, header, generator_rank),
                        dist.P2POp(dist.irecv, payload, generator_rank),
                    )
                )
                reverse_ops.append(
                    dist.P2POp(dist.isend, acknowledgement, generator_rank)
                )
                local_slots.append(slot)
        for work in dist.batch_isend_irecv(forward_ops):
            work.wait()
        for slot, header in received_headers.items():
            expected = [slot, generator_by_slot[slot], score_by_slot[slot]]
            if [int(value) for value in header.cpu()] != expected:
                raise RuntimeError("routed P2P identity header mismatch")
            if not torch.equal(
                received_payloads[slot],
                torch.full_like(received_payloads[slot], float(slot)),
            ):
                raise RuntimeError("routed P2P payload mismatch")
        for work in dist.batch_isend_irecv(reverse_ops):
            work.wait()
        for slot, acknowledgement in acknowledgements.items():
            if int(acknowledgement.cpu()) != slot:
                raise RuntimeError("routed P2P acknowledgement mismatch")

        torch.cuda.synchronize(device)
        dist.barrier()
        receipt = {
            "status": "PASS",
            "rank": rank,
            "local_rank": local_rank,
            "hostname": socket.gethostname(),
            "pipeline_variant": args.pipeline_variant,
            "local_slots": local_slots,
            "subgroup_size": len(generator_ranks) if rank in generator_ranks else len(score_ranks),
            "default_world_initialized_collectively": True,
            "routed_batch_p2p": True,
            "node_local_collective": node_local_collective,
        }
        atomic_json(args.output_dir / f"rank_{rank:03d}.json", receipt)
        dist.barrier()
        if rank == 0:
            rows = [
                json.loads((args.output_dir / f"rank_{member:03d}.json").read_text())
                for member in range(WORLD_SIZE)
            ]
            atomic_json(
                args.output_dir / "summary.json",
                {
                    "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
                    "rank_count": len(rows),
                    "pipeline_variant": args.pipeline_variant,
                    "generator_ranks": list(generator_ranks),
                    "score_ranks": list(score_ranks),
                    "slot_count": sum(len(row["local_slots"]) for row in rows) // 2,
                    "node_local_collective_ranks": sum(
                        row.get("node_local_collective") is not None for row in rows
                    ),
                },
            )
        dist.barrier()
        dist.destroy_process_group()
        return 0

    peer = rank + 8 if rank < 8 else rank - 8
    lane = rank if rank < 8 else peer
    slot_pair = (
        (
            TRACE_BALANCED_SLOT_GROUPS_B32[lane]
            if args.pipeline_variant == "c23-continuous-b32"
            else TRACE_BALANCED_SLOT_PAIRS[lane]
        )
        if args.pipeline_variant in {
            "b1-runahead-critic-b2-local-context-async32-trace-balanced",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-context-prefetch",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-scale1",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-scale1-modulation",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-fused-cast-add",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-autocast-input-elision",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-autocast-input-elision-split-dmd-shadow",
            "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-autocast-input-elision-split-dmd-shadow-fused-adam",
            "c15-score-rope",
            "c16-score-flash-attn2",
            "c18-generator-cross-kv-cache",
            "c19-host-runahead",
            "c20-generator-flash-attn2",
            "c21-continuous-b16",
            "c21-c1-continuous-b16",
            "c22-critic-checkpoint-rng",
            "c23-continuous-b32",
            "c24-continuous-b16-score-modulation",
            "c25-continuous-b16-score-rmsnorm", "c26-continuous-b16-critic-attention-checkpoint", "c27-continuous-b16-pinned-prompt-staging", "c28-continuous-b16-rollout-cast-cache", "c29-continuous-b16-generator-rms-epilogue", "c30-continuous-b16-generator-rms-critic-checkpoint", "c31-continuous-b16-generator-rms-cast-square", "c32-continuous-b16-generator-modulation", "c33-continuous-b16-generator-gate", "c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap",
            "c24-continuous-b16-score-modulation-b1-attribution",
            "phase-expanded-critic16",
        }
        else (lane, lane + 8)
    )
    observed: list[int] = []
    observed_gradients: list[int] = []
    retained: list[torch.Tensor] = []
    deferred_gradient_work = None
    if rank < 8:
        for wave in wave_indices:
            slot = slot_pair[wave]
            header = torch.tensor([rank, peer, wave, slot], device=device)
            payload = torch.full((256,), float(slot), dtype=torch.bfloat16, device=device)
            retained.extend((header, payload))
            if wave == 0:
                dist.send(header, dst=peer)
                dist.send(payload, dst=peer)
                gradient = torch.empty(1, dtype=torch.int64, device=device)
                dist.recv(gradient, src=peer)
                observed_gradients.append(int(gradient.cpu()))
            else:
                works = (dist.isend(header, dst=peer), dist.isend(payload, dst=peer))
                for work in works:
                    work.wait()
                gradient = torch.empty(1, dtype=torch.int64, device=device)
                dist.recv(gradient, src=peer)
                observed_gradients.append(int(gradient.cpu()))
        if observed_gradients != [peer + 1000 + wave for wave in range(waves)]:
            raise RuntimeError("P2P returned-gradient identity mismatch")
    else:
        for wave in wave_indices:
            header = torch.empty(4, dtype=torch.int64, device=device)
            payload = torch.empty(256, dtype=torch.bfloat16, device=device)
            dist.recv(header, src=peer)
            dist.recv(payload, src=peer)
            expected = slot_pair[wave]
            if [int(value) for value in header.cpu()] != [peer, rank, wave, expected]:
                raise RuntimeError("P2P identity header mismatch")
            if not torch.equal(payload, torch.full_like(payload, float(expected))):
                raise RuntimeError("P2P payload mismatch")
            observed.append(expected)
            gradient = torch.tensor([rank + 1000 + wave], dtype=torch.int64, device=device)
            retained.append(gradient)
            if wave % 2 == 0:
                dist.send(gradient, dst=peer)
            else:
                # Match the production score-priority DAG: return gradient1
                # asynchronously, then allow independent Critic work to run
                # before waiting for the transport at the commit boundary.
                deferred_gradient_work = dist.isend(gradient, dst=peer)
                deferred_gradient_work.wait()
                deferred_gradient_work = None

    if deferred_gradient_work is not None:
        deferred_gradient_work.wait()

    host_runahead_canary = None
    if host_runahead_group is not None:
        from distillschedule.runtime.livewan_host_runahead import runahead_tag

        host_buffers = {
            (step, wave): torch.empty(
                256, dtype=torch.float32, device="cpu", pin_memory=True
            )
            for step in range(1, 5)
            for wave in wave_indices
        }
        if rank < 8:
            for (step, wave), buffer in host_buffers.items():
                buffer.fill_(float(rank * 100 + step * 10 + wave))
                dist.isend(
                    buffer,
                    dst=peer,
                    group=host_runahead_group,
                    tag=runahead_tag(step, wave, waves_per_step=waves),
                ).wait()
        else:
            works = {
                key: dist.irecv(
                    buffer,
                    src=peer,
                    group=host_runahead_group,
                    tag=runahead_tag(*key, waves_per_step=waves),
                )
                for key, buffer in host_buffers.items()
            }
            for key, work in works.items():
                work.wait()
                expected_value = float(peer * 100 + key[0] * 10 + key[1])
                if not torch.equal(
                    host_buffers[key], torch.full_like(host_buffers[key], expected_value)
                ):
                    raise RuntimeError(f"host run-ahead payload mismatch: {key}")
        host_runahead_canary = {
            "status": "PASS",
            "backend": "gloo",
            "messages": 4 * waves,
            "pinned_buffers": sum(int(value.is_pinned()) for value in host_buffers.values()),
            "payload_dtype": "torch.float32",
        }

    torch.cuda.synchronize(device)
    dist.barrier()
    receipt = {
        "status": "PASS",
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "peer": peer,
        "waves": waves,
        "subgroup_size": 8,
        "default_world_initialized_collectively": True,
        "score_gradient_1_async": True,
        "return_payload_semantics": (
            "teacher_velocity_float32"
            if args.pipeline_variant in {
                "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-autocast-input-elision-split-dmd-shadow",
                "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-autocast-input-elision-split-dmd-shadow-fused-adam",
                "c15-score-rope",
                "c16-score-flash-attn2",
                "c18-generator-cross-kv-cache",
                "c19-host-runahead",
                "c20-generator-flash-attn2",
                "c21-continuous-b16",
                "c21-c1-continuous-b16",
                "c22-critic-checkpoint-rng",
                "c23-continuous-b32",
                "c24-continuous-b16-score-modulation",
                "c25-continuous-b16-score-rmsnorm", "c26-continuous-b16-critic-attention-checkpoint", "c27-continuous-b16-pinned-prompt-staging", "c28-continuous-b16-rollout-cast-cache", "c29-continuous-b16-generator-rms-epilogue", "c30-continuous-b16-generator-rms-critic-checkpoint", "c31-continuous-b16-generator-rms-cast-square", "c32-continuous-b16-generator-modulation", "c33-continuous-b16-generator-gate", "c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap",
                "c24-continuous-b16-score-modulation-b1-attribution",
            }
            else "final_dmd_gradient_float32"
        ),
        "pipeline_variant": args.pipeline_variant,
        "slot_pair": list(slot_pair),
        "score_context_transport": (
            "trace_local_replay"
            if args.pipeline_variant in {
                "b1-runahead-critic-b2-local-context",
                "b1-runahead-critic-b2-local-context-async32",
                "b1-runahead-critic-b2-local-context-async32-trace-balanced",
                "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context",
                "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-context-prefetch",
                "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-scale1",
                "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-scale1-modulation",
                "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-fused-cast-add",
                "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-autocast-input-elision",
                "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-autocast-input-elision-split-dmd-shadow",
                "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-autocast-input-elision-split-dmd-shadow-fused-adam",
                "c15-score-rope",
                "c16-score-flash-attn2",
                "c18-generator-cross-kv-cache",
                "c19-host-runahead",
                "c20-generator-flash-attn2",
                "c21-continuous-b16",
                "c21-c1-continuous-b16",
                "c22-critic-checkpoint-rng",
                "c23-continuous-b32",
                "c24-continuous-b16-score-modulation",
                "c25-continuous-b16-score-rmsnorm", "c26-continuous-b16-critic-attention-checkpoint", "c27-continuous-b16-pinned-prompt-staging", "c28-continuous-b16-rollout-cast-cache", "c29-continuous-b16-generator-rms-epilogue", "c30-continuous-b16-generator-rms-critic-checkpoint", "c31-continuous-b16-generator-rms-cast-square", "c32-continuous-b16-generator-modulation", "c33-continuous-b16-generator-gate", "c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap",
                "c24-continuous-b16-score-modulation-b1-attribution",
                "phase-expanded-critic16",
                "phase-expanded-critic16-fused-lora-elision",
            }
            else "cuda_p2p_float32"
        ),
        "async_reducer_canary": async_reducer_canary,
        "critic_fsdp_canary": critic_fsdp_canary,
        "critic_group_size": len(critic_ranks) if rank in critic_ranks else 0,
        "phase_expanded_critic16": phase_expanded_critic16,
        "host_runahead_canary": host_runahead_canary,
        "critic_shadow_sync_canary": critic_shadow_sync_canary,
        "observed": observed,
        "observed_gradients": observed_gradients,
    }
    atomic_json(args.output_dir / f"rank_{rank:03d}.json", receipt)
    dist.barrier()
    if rank == 0:
        rows = [
            json.loads((args.output_dir / f"rank_{member:03d}.json").read_text())
            for member in range(WORLD_SIZE)
        ]
        observed_pairs = tuple(
            tuple(int(slot) for slot in rows[lane]["slot_pair"])
            for lane in range(8)
        )
        expected_pairs = (
            (
                TRACE_BALANCED_SLOT_GROUPS_B32
                if args.pipeline_variant == "c23-continuous-b32"
                else TRACE_BALANCED_SLOT_PAIRS
            )
            if args.pipeline_variant in {
                "b1-runahead-critic-b2-local-context-async32-trace-balanced",
                "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context",
                "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-context-prefetch",
                "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-scale1",
                "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-scale1-modulation",
                "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-fused-cast-add",
                "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-autocast-input-elision",
                "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-autocast-input-elision-split-dmd-shadow",
                "b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-autocast-input-elision-split-dmd-shadow-fused-adam",
                "c15-score-rope",
                "c16-score-flash-attn2",
                "c18-generator-cross-kv-cache",
                "c19-host-runahead",
                "c20-generator-flash-attn2",
                "c21-continuous-b16",
                "c21-c1-continuous-b16",
                "c22-critic-checkpoint-rng",
                "c23-continuous-b32",
                "c24-continuous-b16-score-modulation",
                "c25-continuous-b16-score-rmsnorm", "c26-continuous-b16-critic-attention-checkpoint", "c27-continuous-b16-pinned-prompt-staging", "c28-continuous-b16-rollout-cast-cache", "c29-continuous-b16-generator-rms-epilogue", "c30-continuous-b16-generator-rms-critic-checkpoint", "c31-continuous-b16-generator-rms-cast-square", "c32-continuous-b16-generator-modulation", "c33-continuous-b16-generator-gate", "c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap",
                "c24-continuous-b16-score-modulation-b1-attribution",
                "phase-expanded-critic16",
                "phase-expanded-critic16-fused-lora-elision",
            }
            else tuple((lane, lane + 8) for lane in range(8))
        )
        atomic_json(
            args.output_dir / "summary.json",
            {
                "status": (
                    "PASS"
                    if all(row["status"] == "PASS" for row in rows)
                    and observed_pairs == expected_pairs
                    and all(
                        tuple(rows[lane + 8]["slot_pair"]) == observed_pairs[lane]
                        for lane in range(8)
                    )
                    and (
                        not phase_expanded_critic16
                        or all(
                            row.get("critic_group_size") == WORLD_SIZE
                            and row.get("critic_fsdp_canary") is not None
                            for row in rows
                        )
                    )
                    and (
                        args.pipeline_variant
                        not in {
                            "c19-host-runahead",
                            "c20-generator-flash-attn2",
                            "c21-continuous-b16",
                            "c21-c1-continuous-b16",
                            "c22-critic-checkpoint-rng",
                            "c23-continuous-b32",
                            "c24-continuous-b16-score-modulation",
                            "c25-continuous-b16-score-rmsnorm", "c26-continuous-b16-critic-attention-checkpoint", "c27-continuous-b16-pinned-prompt-staging", "c28-continuous-b16-rollout-cast-cache", "c29-continuous-b16-generator-rms-epilogue", "c30-continuous-b16-generator-rms-critic-checkpoint", "c31-continuous-b16-generator-rms-cast-square", "c32-continuous-b16-generator-modulation", "c33-continuous-b16-generator-gate", "c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap",
                            "c24-continuous-b16-score-modulation-b1-attribution",
                        }
                        or all(
                            row.get("host_runahead_canary", {}).get("status") == "PASS"
                            and row.get("host_runahead_canary", {}).get("messages")
                            == 4 * waves
                            and row.get("host_runahead_canary", {}).get("pinned_buffers")
                            == 4 * waves
                            for row in rows
                        )
                    )
                    and (
                        args.pipeline_variant not in {
                            "c21-continuous-b16",
                            "c21-c1-continuous-b16",
                            "c22-critic-checkpoint-rng",
                            "c23-continuous-b32",
                            "c24-continuous-b16-score-modulation",
                            "c25-continuous-b16-score-rmsnorm", "c26-continuous-b16-critic-attention-checkpoint", "c27-continuous-b16-pinned-prompt-staging", "c28-continuous-b16-rollout-cast-cache", "c29-continuous-b16-generator-rms-epilogue", "c30-continuous-b16-generator-rms-critic-checkpoint", "c31-continuous-b16-generator-rms-cast-square", "c32-continuous-b16-generator-modulation", "c33-continuous-b16-generator-gate", "c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap",
                            "c24-continuous-b16-score-modulation-b1-attribution",
                        }
                        or all(
                            row.get("critic_shadow_sync_canary", {}).get("status")
                            == "PASS"
                            for row in rows
                        )
                    )
                    else "FAIL"
                ),
                "rank_count": len(rows),
                "generator_ranks": list(generator_ranks),
                "score_ranks": list(score_ranks),
                "waves": waves,
                "slot_pairs_by_lane": [list(pair) for pair in observed_pairs],
                "async_reducer_canary_ranks": sum(
                    row.get("async_reducer_canary") is not None for row in rows
                ),
                "host_runahead_canary_ranks": sum(
                    row.get("host_runahead_canary", {}).get("status") == "PASS"
                    for row in rows
                ),
            },
        )
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

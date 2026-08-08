#!/usr/bin/env python3
"""Cheap distributed handshake that must pass before Route-A model construction."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

from distillschedule.runtime.topology import TopologyRecord, validate_topology_records


def _env_int(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None:
        raise RuntimeError(f"required environment variable {name} is missing")
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, observed {raw!r}") from exc


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-nodes", type=int, default=2)
    parser.add_argument("--expected-local-world-size", type=int, default=8)
    parser.add_argument("--expected-world-size", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=None)
    args = parser.parse_args()

    try:
        import torch
        import torch.distributed as dist
    except ImportError as exc:
        print(f"topology_probe_import_error={exc}", file=sys.stderr)
        return 2

    initialized = False
    rank = -1
    try:
        rank = _env_int("RANK")
        world_size = _env_int("WORLD_SIZE")
        local_rank = args.local_rank if args.local_rank is not None else _env_int("LOCAL_RANK")
        local_world_size = _env_int("LOCAL_WORLD_SIZE")
        node_rank = _env_int("DISTILLSCHEDULE_NODE_RANK")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        cuda_device_count = torch.cuda.device_count()
        if not 0 <= local_rank < cuda_device_count:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} is outside visible CUDA device count {cuda_device_count}"
            )
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="gloo",
            timeout=timedelta(seconds=args.timeout_seconds),
        )
        initialized = True
        record = TopologyRecord(
            rank=rank,
            local_rank=local_rank,
            node_rank=node_rank,
            world_size=world_size,
            local_world_size=local_world_size,
            hostname=socket.gethostname(),
            cuda_device_index=torch.cuda.current_device(),
            cuda_device_count=cuda_device_count,
        )
        gathered: list[dict[str, object] | None] = [None] * world_size
        dist.all_gather_object(gathered, record.to_dict())

        # A tiny NCCL collective verifies that all 16 GPU ranks can communicate.
        nccl_group = dist.new_group(backend="nccl", timeout=timedelta(seconds=args.timeout_seconds))
        token = torch.ones(1, dtype=torch.int64, device=f"cuda:{local_rank}")
        dist.all_reduce(token, group=nccl_group)
        torch.cuda.synchronize(local_rank)
        if int(token.item()) != world_size:
            raise RuntimeError(
                f"NCCL all-reduce produced {int(token.item())}, expected {world_size}"
            )

        error: str | None = None
        summary_payload: dict[str, object] | None = None
        if rank == 0:
            try:
                records = [item for item in gathered if item is not None]
                summary = validate_topology_records(
                    records,
                    expected_nodes=args.expected_nodes,
                    expected_local_world_size=args.expected_local_world_size,
                    expected_world_size=args.expected_world_size,
                    require_cuda=True,
                )
                summary_payload = summary.to_dict()
                _atomic_json(
                    args.output,
                    {
                        "schema_version": "distillschedule.topology_probe.v1",
                        "status": "PASS",
                        "summary": summary_payload,
                        "records": records,
                    },
                )
            except Exception as exc:  # validation error is broadcast to every rank
                error = str(exc)
                _atomic_json(
                    args.output,
                    {
                        "schema_version": "distillschedule.topology_probe.v1",
                        "status": "FAIL",
                        "error": error,
                        "records": [item for item in gathered if item is not None],
                    },
                )
        payload: list[object | None] = [error]
        dist.broadcast_object_list(payload, src=0)
        dist.barrier()
        if payload[0] is not None:
            print(f"topology_probe_failed={payload[0]}", file=sys.stderr)
            return 2
        if rank == 0:
            print(json.dumps({"status": "PASS", "summary": summary_payload}, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"topology_probe_error={exc}", file=sys.stderr)
        return 2
    finally:
        if initialized:
            try:
                import torch.distributed as dist

                dist.destroy_process_group()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())

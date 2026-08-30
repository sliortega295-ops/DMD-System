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


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()

    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != WORLD_SIZE:
        raise RuntimeError("transport probe requires world16")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=args.timeout_seconds))
    generator_group = dist.new_group(ranks=list(GENERATOR_RANKS), backend="nccl")
    score_group = dist.new_group(ranks=list(SCORE_RANKS), backend="nccl")

    # Collectively initialize default world before asymmetric default-group P2P.
    dist.barrier()
    subgroup = generator_group if rank < 8 else score_group
    subgroup_token = torch.tensor([rank + 1], dtype=torch.int64, device=device)
    dist.all_reduce(subgroup_token, group=subgroup)
    expected_subgroup = sum(value + 1 for value in (GENERATOR_RANKS if rank < 8 else SCORE_RANKS))
    if int(subgroup_token.cpu()) != expected_subgroup:
        raise RuntimeError("role subgroup all-reduce mismatch")

    peer = rank + 8 if rank < 8 else rank - 8
    observed: list[int] = []
    observed_gradients: list[int] = []
    retained: list[torch.Tensor] = []
    if rank < 8:
        for wave in range(2):
            header = torch.tensor([rank, peer, wave, rank + 100 * wave], device=device)
            payload = torch.full((256,), float(rank + 100 * wave), dtype=torch.bfloat16, device=device)
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
        if observed_gradients != [peer + 1000, peer + 1001]:
            raise RuntimeError("P2P returned-gradient identity mismatch")
    else:
        for wave in range(2):
            header = torch.empty(4, dtype=torch.int64, device=device)
            payload = torch.empty(256, dtype=torch.bfloat16, device=device)
            dist.recv(header, src=peer)
            dist.recv(payload, src=peer)
            expected = peer + 100 * wave
            if [int(value) for value in header.cpu()] != [peer, rank, wave, expected]:
                raise RuntimeError("P2P identity header mismatch")
            if not torch.equal(payload, torch.full_like(payload, float(expected))):
                raise RuntimeError("P2P payload mismatch")
            observed.append(expected)
            gradient = torch.tensor([rank + 1000 + wave], dtype=torch.int64, device=device)
            dist.send(gradient, dst=peer)

    torch.cuda.synchronize(device)
    dist.barrier()
    receipt = {
        "status": "PASS",
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "peer": peer,
        "waves": 2,
        "subgroup_size": 8,
        "default_world_initialized_collectively": True,
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
        atomic_json(
            args.output_dir / "summary.json",
            {
                "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
                "rank_count": len(rows),
                "generator_ranks": list(GENERATOR_RANKS),
                "score_ranks": list(SCORE_RANKS),
                "waves": 2,
            },
        )
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

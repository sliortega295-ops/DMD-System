"""Two-node A100 topology helpers with world16 data semantics.

HSDP changes only physical parameter placement. It must not collapse the data
parallel contract: all 16 global ranks still consume independent B1 samples and
independent RNG streams, matching the world16 FSDP baseline.
"""

from __future__ import annotations

import dataclasses
import os
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from loguru import logger
from torch.distributed.device_mesh import init_device_mesh

_SHARD_RANK = 0
_SHARD_SIZE = 1


@dataclasses.dataclass(frozen=True)
class HybridTopology:
    world_size: int
    shard_size: int
    replicate_size: int
    replicate_rank: int
    shard_rank: int


def resolve_hybrid_topology(world_size: int, rank: int, shard_size: int) -> HybridTopology:
    world_size = int(world_size)
    rank = int(rank)
    shard_size = int(shard_size)
    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}.")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank={rank} is outside world_size={world_size}.")
    if shard_size < 1 or world_size % shard_size:
        raise ValueError(
            f"hybrid shard_size={shard_size} must be positive and divide WORLD_SIZE={world_size}."
        )
    replicate_size = world_size // shard_size
    return HybridTopology(
        world_size=world_size,
        shard_size=shard_size,
        replicate_size=replicate_size,
        replicate_rank=rank // shard_size,
        shard_rank=rank % shard_size,
    )


def _hybrid_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("distributed", {}).get("hybrid_shard", {})


def hybrid_enabled(config: dict[str, Any]) -> bool:
    return bool(_hybrid_config(config).get("enabled", False))


def init_distributed_a100(config: dict[str, Any] | None = None) -> None:
    """Initialize ordinary world16 FSDP or optional 2x8 HSDP.

    For HSDP, the 2-D mesh is used only by FSDP2. LightX2V's dataset/data-rank
    globals remain world16 so every global rank sees a distinct sample. Model
    inputs and RNG are never broadcast across the 8-way parameter-shard group.
    """

    config = config or {}
    if not hybrid_enabled(config):
        from lightx2v_train.runtime.distributed import init_distributed

        init_distributed(config)
        return
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("hybrid_shard.enabled=true requires torchrun RANK/WORLD_SIZE")

    import lightx2v_train.runtime.distributed as runtime_dist

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist_config = config.get("distributed", {})
    if not dist.is_initialized():
        dist.init_process_group(
            backend=dist_config.get("backend", "nccl"),
            timeout=timedelta(minutes=int(dist_config.get("timeout_minutes", 120))),
        )

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    shard_size = int(
        _hybrid_config(config).get("shard_size", int(os.environ.get("LOCAL_WORLD_SIZE", 8)))
    )
    topology = resolve_hybrid_topology(world_size, rank, shard_size)
    mesh = init_device_mesh(
        "cuda",
        (topology.replicate_size, topology.shard_size),
        mesh_dim_names=("replicate", "shard"),
    )

    global _SHARD_RANK, _SHARD_SIZE
    _SHARD_RANK = topology.shard_rank
    _SHARD_SIZE = topology.shard_size

    runtime_dist._DEVICE_MESH = mesh
    runtime_dist._FSDP_DEVICE_MESH = mesh
    # Critical control-variable rule: HSDP is placement, not a batch change.
    runtime_dist._DP_GROUP = dist.group.WORLD
    runtime_dist._SP_GROUP = None
    runtime_dist._DP_RANK = rank
    runtime_dist._DP_WORLD_SIZE = world_size
    runtime_dist._SP_RANK = 0
    runtime_dist._SP_WORLD_SIZE = 1

    _patch_fsdp_enabled()
    logger.info(
        "[h3-a100] HSDP initialized world={} replicate={} shard={} "
        "replicate_rank={} shard_rank={} data_parallel_world={}",
        topology.world_size,
        topology.replicate_size,
        topology.shard_size,
        topology.replicate_rank,
        topology.shard_rank,
        world_size,
    )


def _patch_fsdp_enabled() -> None:
    """Recognize both 1-D FSDP2 and a 1x8/2x8 HSDP mesh."""

    import lightx2v_train.runtime.fsdp as fsdp_runtime

    def enabled(config: dict[str, Any]) -> bool:
        fsdp_config = config.get("distributed", {}).get("fsdp2", {})
        return (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
            and bool(fsdp_config.get("enabled", False))
        )

    fsdp_runtime.fsdp2_enabled = enabled
    try:
        import lightx2v_train.runtime.parallel as parallel_runtime

        parallel_runtime.fsdp2_enabled = enabled
    except ImportError:
        pass


def shard_group_size() -> int:
    return _SHARD_SIZE


def shard_group_rank() -> int:
    return _SHARD_RANK

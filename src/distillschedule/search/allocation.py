"""Physical devices and logical samples are independent planning contracts.

The existing world16 adapters retain their own contracts. Hardware adapters use
this explicit mapping instead of changing a logical batch when the number of
workers changes. No tensors, distributed processes or CUDA contexts are created.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Mapping

from distillschedule.workloads.best_known.registry import fingerprint
from .contracts import require

SCHEMA = "distillschedule.execution_allocation.v1"


@dataclass(frozen=True)
class PhysicalDevice:
    rank: int
    node: int
    local_rank: int
    uuid: str
    architecture: str
    capacity_bytes: int

    def __post_init__(self) -> None:
        for name in ("rank", "node", "local_rank"):
            require(type(getattr(self, name)) is int and getattr(self, name) >= 0,
                    f"invalid physical device {name}")
        require(isinstance(self.uuid, str) and self.uuid.startswith("GPU-"),
                "observed GPU UUID required")
        require(self.architecture in ("sm80", "sm90"), "unsupported GPU architecture")
        require(type(self.capacity_bytes) is int and self.capacity_bytes > 0,
                "positive observed GPU capacity required")


@dataclass(frozen=True)
class ExecutionAllocation:
    devices: tuple[PhysicalDevice, ...]
    logical_global_batch: int
    # Each role has one row per physical rank. Empty rows mean that the role
    # does not execute on that rank; all logical samples still appear once.
    role_slots: Mapping[str, tuple[tuple[int, ...], ...]]
    physical_microbatch: int = 1
    schema_version: str = SCHEMA

    def __post_init__(self) -> None:
        require(self.schema_version == SCHEMA, "allocation schema mismatch")
        require(isinstance(self.devices, tuple) and self.devices
                and all(isinstance(d, PhysicalDevice) for d in self.devices),
                "explicit physical device inventory required")
        require(tuple(d.rank for d in self.devices) == tuple(range(len(self.devices))),
                "physical ranks must be ordered and contiguous")
        require(len({d.uuid for d in self.devices}) == len(self.devices), "duplicate GPU UUID")
        require(len({(d.node, d.local_rank) for d in self.devices}) == len(self.devices),
                "duplicate node/local-rank placement")
        nodes = sorted({d.node for d in self.devices})
        require(nodes == list(range(len(nodes))), "node ids must be contiguous")
        for node in nodes:
            local = sorted(d.local_rank for d in self.devices if d.node == node)
            require(local == list(range(len(local))), "local ranks must be contiguous")
        require(type(self.logical_global_batch) is int and self.logical_global_batch > 0,
                "positive logical global batch required")
        require(type(self.physical_microbatch) is int and self.physical_microbatch == 1,
                "native hardware adapters currently preserve independent B1 calls")
        require(isinstance(self.role_slots, Mapping) and self.role_slots,
                "explicit logical sample mapping required")
        normalized = {}
        for role, rows in self.role_slots.items():
            require(isinstance(role, str) and bool(role), "invalid role name")
            require(isinstance(rows, (tuple, list)) and len(rows) == self.world_size,
                    f"{role}: expected one slot row per physical rank")
            require(all(isinstance(row, (tuple, list)) for row in rows),
                    f"{role}: slots must be explicit sequences")
            flat = [slot for row in rows for slot in row]
            require(all(type(slot) is int for slot in flat)
                    and sorted(flat) == list(range(self.logical_global_batch)),
                    f"{role}: independent logical slots must cover the batch exactly once")
            normalized[role] = tuple(tuple(row) for row in rows)
        object.__setattr__(self, "role_slots", MappingProxyType(normalized))

    @property
    def world_size(self) -> int:
        return len(self.devices)

    @property
    def node_count(self) -> int:
        return len({d.node for d in self.devices})

    @property
    def memory_limits(self) -> dict[int, int]:
        return {d.rank: d.capacity_bytes for d in self.devices}

    def role_ranks(self, role: str) -> tuple[int, ...]:
        require(role in self.role_slots, "unknown logical role")
        return tuple(rank for rank, slots in enumerate(self.role_slots[role]) if slots)

    def loss_scale_before_rank_mean(self, role: str) -> float:
        """Scale EACH B1 loss before the role group's rank-mean reduction.

        world8/global16 is 1/2 per loss, followed by rank mean over 8. This
        remains correct for unequal slot counts; using 1/local_count would not.
        The native adapter must prove it actually uses this convention.
        """
        return len(self.role_ranks(role)) / self.logical_global_batch

    def validate_worker(self, *, rank: int, world_size: int, node: int,
                        local_rank: int, device_uuid: str) -> None:
        require(type(rank) is int and 0 <= rank < self.world_size,
                "unexpected physical worker rank")
        require(all(type(value) is int and value >= 0
                    for value in (world_size, node, local_rank)),
                "worker placement requires integer indices")
        device = self.devices[rank]
        require((world_size, node, local_rank, device_uuid) ==
                (self.world_size, device.node, device.local_rank, device.uuid),
                "actual worker placement differs from selected allocation")

    def validate_slot_receipts(self, receipts: list[dict[str, Any]]) -> None:
        require(isinstance(receipts, list) and len(receipts) == self.world_size
                and all(isinstance(r, Mapping) and type(r.get("rank")) is int
                        for r in receipts)
                and {r.get("rank") for r in receipts} == set(range(self.world_size)),
                "physical worker receipt coverage differs from allocation")
        for row in receipts:
            rank = row["rank"]
            require(row.get("allocation_sha256") == self.digest,
                    "worker allocation binding missing or changed")
            actual = row.get("logical_slots")
            require(isinstance(actual, Mapping) and set(actual) == set(self.role_slots),
                    "worker logical roles missing or changed")
            require(all(isinstance(values, list) and all(type(slot) is int for slot in values)
                        for values in actual.values()), "logical slots require integer indices")
            require(all(actual[role] == list(slots[rank])
                        for role, slots in self.role_slots.items()),
                    "worker logical samples differ from selected slot mapping")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version,
                "devices": [asdict(d) for d in self.devices],
                "logical_global_batch": self.logical_global_batch,
                "physical_microbatch": self.physical_microbatch,
                "role_slots": {role: [list(row) for row in rows]
                               for role, rows in self.role_slots.items()}}

    @property
    def digest(self) -> str:
        return fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionAllocation":
        require(isinstance(value, Mapping) and set(value) ==
                {"schema_version", "devices", "logical_global_batch",
                 "physical_microbatch", "role_slots"}, "unknown/missing allocation fields")
        require(isinstance(value["devices"], list), "device inventory must be a list")
        return cls(tuple(PhysicalDevice(**d) for d in value["devices"]),
                   value["logical_global_batch"], value["role_slots"],
                   value["physical_microbatch"], value["schema_version"])

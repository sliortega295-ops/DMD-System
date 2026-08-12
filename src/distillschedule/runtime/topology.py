"""Pure, testable validation for multi-node process-group topology evidence."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from distillschedule.core.contracts import ContractViolation


@dataclass(frozen=True)
class TopologyRecord:
    rank: int
    local_rank: int
    node_rank: int
    world_size: int
    local_world_size: int
    hostname: str
    cuda_device_index: int
    cuda_device_count: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TopologyRecord":
        def integer(key: str) -> int:
            item = value.get(key)
            if isinstance(item, bool) or not isinstance(item, int):
                raise ContractViolation(
                    f"{key} must be an integer", path="runtime.topology"
                )
            return item

        hostname = value.get("hostname")
        if not isinstance(hostname, str) or not hostname.strip():
            raise ContractViolation(
                "hostname must be a non-empty string", path="runtime.topology"
            )
        return cls(
            rank=integer("rank"),
            local_rank=integer("local_rank"),
            node_rank=integer("node_rank"),
            world_size=integer("world_size"),
            local_world_size=integer("local_world_size"),
            hostname=hostname.strip(),
            cuda_device_index=integer("cuda_device_index"),
            cuda_device_count=integer("cuda_device_count"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "local_rank": self.local_rank,
            "node_rank": self.node_rank,
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "hostname": self.hostname,
            "cuda_device_index": self.cuda_device_index,
            "cuda_device_count": self.cuda_device_count,
        }


@dataclass(frozen=True)
class TopologySummary:
    world_size: int
    nodes: int
    local_world_size: int
    hosts_by_node: Mapping[int, str]
    ranks_by_node: Mapping[int, tuple[int, ...]]

    def to_dict(self) -> dict[str, object]:
        return {
            "world_size": self.world_size,
            "nodes": self.nodes,
            "local_world_size": self.local_world_size,
            "hosts_by_node": {str(key): value for key, value in self.hosts_by_node.items()},
            "ranks_by_node": {
                str(key): list(value) for key, value in self.ranks_by_node.items()
            },
        }


def validate_topology_records(
    records: Iterable[TopologyRecord | Mapping[str, Any]],
    *,
    expected_nodes: int,
    expected_local_world_size: int,
    expected_world_size: int,
    require_cuda: bool = True,
) -> TopologySummary:
    if expected_nodes <= 0 or expected_local_world_size <= 0 or expected_world_size <= 0:
        raise ContractViolation(
            "expected topology dimensions must be positive", path="runtime.topology"
        )
    if expected_nodes * expected_local_world_size != expected_world_size:
        raise ContractViolation(
            "expected_nodes * expected_local_world_size must equal expected_world_size",
            path="runtime.topology",
        )
    normalized = tuple(
        item if isinstance(item, TopologyRecord) else TopologyRecord.from_mapping(item)
        for item in records
    )
    if len(normalized) != expected_world_size:
        raise ContractViolation(
            f"received {len(normalized)} records, expected {expected_world_size}",
            path="runtime.topology",
        )
    expected_ranks = set(range(expected_world_size))
    observed_ranks = {item.rank for item in normalized}
    if observed_ranks != expected_ranks:
        missing = sorted(expected_ranks - observed_ranks)
        extra = sorted(observed_ranks - expected_ranks)
        raise ContractViolation(
            f"global ranks mismatch; missing={missing}, extra={extra}",
            path="runtime.topology.rank",
        )
    if len(observed_ranks) != len(normalized):
        raise ContractViolation("global ranks are not unique", path="runtime.topology.rank")
    expected_node_ranks = set(range(expected_nodes))
    observed_node_ranks = {item.node_rank for item in normalized}
    if observed_node_ranks != expected_node_ranks:
        raise ContractViolation(
            f"node ranks mismatch; observed={sorted(observed_node_ranks)}",
            path="runtime.topology.node_rank",
        )
    by_node: dict[int, list[TopologyRecord]] = defaultdict(list)
    for item in normalized:
        if item.world_size != expected_world_size:
            raise ContractViolation(
                f"rank {item.rank} reports WORLD_SIZE={item.world_size}",
                path="runtime.topology.world_size",
            )
        if item.local_world_size != expected_local_world_size:
            raise ContractViolation(
                f"rank {item.rank} reports LOCAL_WORLD_SIZE={item.local_world_size}",
                path="runtime.topology.local_world_size",
            )
        by_node[item.node_rank].append(item)
    hosts_by_node: dict[int, str] = {}
    ranks_by_node: dict[int, tuple[int, ...]] = {}
    observed_hosts: set[str] = set()
    expected_local_ranks = set(range(expected_local_world_size))
    for node_rank in range(expected_nodes):
        node_records = by_node[node_rank]
        if len(node_records) != expected_local_world_size:
            raise ContractViolation(
                f"node {node_rank} has {len(node_records)} ranks, expected {expected_local_world_size}",
                path="runtime.topology.node_rank",
            )
        hosts = {item.hostname for item in node_records}
        if len(hosts) != 1:
            raise ContractViolation(
                f"node rank {node_rank} maps to multiple hostnames: {sorted(hosts)}",
                path="runtime.topology.hostname",
            )
        host = next(iter(hosts))
        if host in observed_hosts:
            raise ContractViolation(
                f"hostname {host!r} is reused by multiple node ranks",
                path="runtime.topology.hostname",
            )
        observed_hosts.add(host)
        hosts_by_node[node_rank] = host
        local_ranks = {item.local_rank for item in node_records}
        if local_ranks != expected_local_ranks or len(local_ranks) != len(node_records):
            raise ContractViolation(
                f"node {node_rank} local ranks mismatch; observed={sorted(local_ranks)}",
                path="runtime.topology.local_rank",
            )
        if require_cuda:
            for item in node_records:
                if item.cuda_device_count < expected_local_world_size:
                    raise ContractViolation(
                        f"rank {item.rank} sees only {item.cuda_device_count} CUDA devices",
                        path="runtime.topology.cuda_device_count",
                    )
                if item.cuda_device_index != item.local_rank:
                    raise ContractViolation(
                        f"rank {item.rank} is bound to CUDA device {item.cuda_device_index}, expected {item.local_rank}",
                        path="runtime.topology.cuda_device_index",
                    )
        ranks_by_node[node_rank] = tuple(sorted(item.rank for item in node_records))
    return TopologySummary(
        world_size=expected_world_size,
        nodes=expected_nodes,
        local_world_size=expected_local_world_size,
        hosts_by_node=hosts_by_node,
        ranks_by_node=ranks_by_node,
    )


def validate_topology_probe_payload(
    value: Mapping[str, Any],
    *,
    expected_nodes: int,
    expected_local_world_size: int,
    expected_world_size: int,
    require_cuda: bool = True,
) -> TopologySummary:
    """Validate the serialized payload emitted by ``probe_world16_topology.py``.

    Keeping this check beside the record validator prevents the launcher from
    inventing a second, incompatible interpretation of the probe schema.
    """

    if value.get("schema_version") != "distillschedule.topology_probe.v1":
        raise ContractViolation(
            "unexpected topology probe schema", path="runtime.topology.schema_version"
        )
    if value.get("status") != "PASS":
        raise ContractViolation(
            "topology probe status is not PASS", path="runtime.topology.status"
        )
    records = value.get("records")
    if not isinstance(records, list):
        raise ContractViolation(
            "topology probe records must be a list", path="runtime.topology.records"
        )
    summary = validate_topology_records(
        records,
        expected_nodes=expected_nodes,
        expected_local_world_size=expected_local_world_size,
        expected_world_size=expected_world_size,
        require_cuda=require_cuda,
    )
    serialized_summary = value.get("summary")
    if not isinstance(serialized_summary, Mapping):
        raise ContractViolation(
            "topology probe summary must be an object", path="runtime.topology.summary"
        )
    if dict(serialized_summary) != summary.to_dict():
        raise ContractViolation(
            "topology probe summary does not match its records",
            path="runtime.topology.summary",
        )
    return summary

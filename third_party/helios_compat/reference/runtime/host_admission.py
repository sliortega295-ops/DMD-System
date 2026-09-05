"""Conservative host-memory admission for real-14B lifecycle experiments.

The check deliberately rejects a process-wide strict NUMA membind.  Small
runtime-owned staging arenas may later use NUMA-local allocation, but binding
the entire CPU-offloaded process can exhaust one 126-GiB NUMA node while the
rest of the host remains idle.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

GIB = 1 << 30


def _read_limit(path: Path) -> int | None:
    try:
        value = path.read_text().strip()
    except OSError:
        return None
    if not value or value == "max":
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _read_meminfo_bytes(path: Path, key: str) -> int | None:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    prefix = f"{key}:"
    for line in lines:
        if not line.startswith(prefix):
            continue
        fields = line.split()
        if len(fields) < 2:
            return None
        scale = 1024 if len(fields) < 3 or fields[2].lower() == "kb" else 1
        try:
            return int(fields[1]) * scale
        except ValueError:
            return None
    return None


def strict_numa_policies(lines: Iterable[str]) -> tuple[str, ...]:
    policies: set[str] = set()
    for line in lines:
        fields = line.split()
        if len(fields) < 2:
            continue
        policy = fields[1]
        if policy.startswith("bind:"):
            policies.add(policy)
    return tuple(sorted(policies))


@dataclass(frozen=True)
class SharedBlockCandidate:
    component: str
    block_index: int
    bytes_per_rank: int
    expected_materializations: int = 0


@dataclass(frozen=True)
class MinimumSharePlan:
    schema_version: str
    status: str
    rejection_reason: str | None
    rank_count: int
    base_peak_bytes: int
    unaffected_peak_bytes: int
    effective_cap_bytes: int
    reserve_bytes: int
    admission_limit_bytes: int
    deficit_bytes: int
    selected: tuple[SharedBlockCandidate, ...]
    projected_savings_bytes: int
    projected_affected_peak_bytes: int
    projected_peak_bytes: int
    projected_headroom_bytes: int
    assumption: str

    @property
    def admitted(self) -> bool:
        return self.status == "ADMITTED"


def plan_minimum_shared_blocks(
    *,
    base_peak_bytes: int,
    effective_cap_bytes: int,
    reserve_bytes: int,
    rank_count: int,
    candidates: Sequence[SharedBlockCandidate],
    uncertainty_bytes: int = 0,
    unaffected_peak_bytes: int = 0,
) -> MinimumSharePlan:
    """Greedily select a minimum-size immutable sharing set for node admission.

    The projection assumes every selected file-backed immutable page is physically
    resident once per node rather than once per rank. Actual cgroup admission remains
    authoritative. Ties favor fewer expected materializations to reduce the measured
    shared-backing hot-path penalty.
    """
    values = (
        base_peak_bytes,
        unaffected_peak_bytes,
        effective_cap_bytes,
        reserve_bytes,
        uncertainty_bytes,
    )
    if any(value < 0 for value in values):
        raise ValueError("memory values must be non-negative")
    if rank_count < 2:
        raise ValueError("minimum-share planning requires at least two ranks")
    if len({(candidate.component, candidate.block_index) for candidate in candidates}) != len(candidates):
        raise ValueError("shared-block candidates must have unique component/index keys")
    if any(candidate.bytes_per_rank <= 0 for candidate in candidates):
        raise ValueError("candidate bytes_per_rank must be positive")
    admission_limit = max(0, effective_cap_bytes - reserve_bytes)
    affected_peak = base_peak_bytes + uncertainty_bytes
    fixed_peak = unaffected_peak_bytes + uncertainty_bytes
    initial_peak = max(affected_peak, fixed_peak)
    deficit = max(0, initial_peak - admission_limit)
    if fixed_peak > admission_limit:
        return MinimumSharePlan(
            schema_version="distillschedule.minimum_share_plan.v2",
            status="REJECTED",
            rejection_reason=(
                "an ownership-unaffected lifecycle phase exceeds the reserve-qualified "
                "limit; use a phase-specific admission wave before selecting shared blocks"
            ),
            rank_count=rank_count,
            base_peak_bytes=base_peak_bytes,
            unaffected_peak_bytes=unaffected_peak_bytes,
            effective_cap_bytes=effective_cap_bytes,
            reserve_bytes=reserve_bytes,
            admission_limit_bytes=admission_limit,
            deficit_bytes=deficit,
            selected=(),
            projected_savings_bytes=0,
            projected_affected_peak_bytes=affected_peak,
            projected_peak_bytes=initial_peak,
            projected_headroom_bytes=admission_limit - initial_peak,
            assumption=(
                "selected immutable file-backed pages affect only the ownership-active "
                "phase; actual cgroup measurement overrides this projection"
            ),
        )
    affected_deficit = max(0, affected_peak - admission_limit)
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.bytes_per_rank,
            candidate.expected_materializations,
            candidate.component,
            candidate.block_index,
        ),
    )
    selected: list[SharedBlockCandidate] = []
    savings = 0
    for candidate in ordered:
        if savings >= affected_deficit:
            break
        selected.append(candidate)
        savings += (rank_count - 1) * candidate.bytes_per_rank
    admitted = savings >= affected_deficit
    projected_affected_peak = max(0, affected_peak - savings)
    projected_peak = max(fixed_peak, projected_affected_peak)
    reason = None if admitted else "candidate immutable sharing set cannot close the reserve-qualified deficit"
    return MinimumSharePlan(
        schema_version="distillschedule.minimum_share_plan.v2",
        status="ADMITTED" if admitted else "REJECTED",
        rejection_reason=reason,
        rank_count=rank_count,
        base_peak_bytes=base_peak_bytes,
        unaffected_peak_bytes=unaffected_peak_bytes,
        effective_cap_bytes=effective_cap_bytes,
        reserve_bytes=reserve_bytes,
        admission_limit_bytes=admission_limit,
        deficit_bytes=deficit,
        selected=tuple(selected),
        projected_savings_bytes=savings,
        projected_affected_peak_bytes=projected_affected_peak,
        projected_peak_bytes=projected_peak,
        projected_headroom_bytes=admission_limit - projected_peak,
        assumption=(
            "selected immutable file-backed pages affect only the ownership-active "
            "phase and occupy one physical node copy; actual cgroup measurement "
            "overrides this projection"
        ),
    )


@dataclass(frozen=True)
class HostAdmission:
    schema_version: str
    status: str
    rejection_reasons: tuple[str, ...]
    projected_peak_bytes: int
    reserve_bytes: int
    required_bytes: int
    additional_bytes_needed: int
    physical_required_bytes: int
    physical_total_bytes: int | None
    physical_available_bytes: int | None
    cgroup_max_bytes: int | None
    cgroup_current_bytes: int | None
    effective_cap_bytes: int | None
    strict_numa_policies: tuple[str, ...]

    @property
    def admitted(self) -> bool:
        return self.status == "ADMITTED"


def evaluate_host_admission(
    *,
    projected_peak_bytes: int,
    reserve_bytes: int,
    physical_total_bytes: int | None,
    physical_available_bytes: int | None,
    cgroup_max_bytes: int | None,
    cgroup_current_bytes: int | None,
    numa_policies: Iterable[str] = (),
) -> HostAdmission:
    if projected_peak_bytes < 0 or reserve_bytes < 0:
        raise ValueError("projected peak and reserve must be non-negative")
    caps = [value for value in (physical_total_bytes, cgroup_max_bytes) if value]
    effective_cap = min(caps) if caps else None
    required = projected_peak_bytes + reserve_bytes
    current = cgroup_current_bytes or 0
    additional = max(0, projected_peak_bytes - current)
    physical_required = additional + reserve_bytes
    strict = tuple(sorted(set(numa_policies)))
    reasons: list[str] = []
    if strict:
        reasons.append(
            "process-wide strict NUMA membind is forbidden for CPU-offloaded 14B runs: "
            + ",".join(strict)
        )
    if effective_cap is None:
        reasons.append("no physical or cgroup host-memory cap could be read")
    elif required > effective_cap:
        reasons.append(
            f"projected peak plus reserve ({required}) exceeds effective cap ({effective_cap})"
        )
    if physical_available_bytes is None:
        reasons.append("physical MemAvailable could not be read")
    elif physical_required > physical_available_bytes:
        reasons.append(
            f"additional peak plus reserve ({physical_required}) exceeds physical available "
            f"memory ({physical_available_bytes})"
        )
    return HostAdmission(
        schema_version="distillschedule.host_admission.v1",
        status="REJECTED" if reasons else "ADMITTED",
        rejection_reasons=tuple(reasons),
        projected_peak_bytes=projected_peak_bytes,
        reserve_bytes=reserve_bytes,
        required_bytes=required,
        additional_bytes_needed=additional,
        physical_required_bytes=physical_required,
        physical_total_bytes=physical_total_bytes,
        physical_available_bytes=physical_available_bytes,
        cgroup_max_bytes=cgroup_max_bytes,
        cgroup_current_bytes=cgroup_current_bytes,
        effective_cap_bytes=effective_cap,
        strict_numa_policies=strict,
    )


def inspect_host_admission(
    *,
    projected_peak_gib: float,
    reserve_gib: float,
    meminfo_path: Path = Path("/proc/meminfo"),
    cgroup_max_path: Path = Path("/sys/fs/cgroup/memory.max"),
    cgroup_current_path: Path = Path("/sys/fs/cgroup/memory.current"),
    numa_maps_path: Path = Path("/proc/self/numa_maps"),
) -> HostAdmission:
    try:
        policies = strict_numa_policies(numa_maps_path.read_text().splitlines())
    except OSError:
        policies = ()
    return evaluate_host_admission(
        projected_peak_bytes=int(projected_peak_gib * GIB),
        reserve_bytes=int(reserve_gib * GIB),
        physical_total_bytes=_read_meminfo_bytes(meminfo_path, "MemTotal"),
        physical_available_bytes=_read_meminfo_bytes(meminfo_path, "MemAvailable"),
        cgroup_max_bytes=_read_limit(cgroup_max_path),
        cgroup_current_bytes=_read_limit(cgroup_current_path),
        numa_policies=policies,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projected-peak-gib", type=float, required=True)
    parser.add_argument("--reserve-gib", type=float, default=64.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect_host_admission(
        projected_peak_gib=args.projected_peak_gib,
        reserve_gib=args.reserve_gib,
    )
    payload = json.dumps(asdict(result), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    return 0 if result.admitted else 2


if __name__ == "__main__":
    raise SystemExit(main())

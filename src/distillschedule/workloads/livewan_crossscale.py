"""Contracts for the Wan1.3B/14B cross-scale SF-DMD workload.

The module deliberately contains no model code.  It owns the immutable logical
batch, rank routing, version, and receipt rules shared by the sequential FSDP2
reference and the persistent role-pool runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


LIVEWAN_COMMIT = "1eabf43c777bc5b2816926ca998cd4e4c2dfa3d5"
WAN21_COMMIT = "9737cba9c1c3c4d04b33fcad41c111989865d315"
WORLD_SIZE = 16
GLOBAL_BATCH = 16
FULL5_STEPS = 5
GENERATOR_UPDATES = 1
CRITIC_UPDATES = 5
GENERATOR_SIZE = "1.3B"
CRITIC_SIZE = "1.3B"
TEACHER_SIZE = "14B"
LORA_RANK = 32


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class SlotRoute:
    logical_step: int
    logical_slot: int
    generator_rank: int
    score_rank: int
    wave: int
    generator_version: int
    critic_version: int

    @property
    def operation_key(self) -> str:
        return (
            f"step={self.logical_step}/slot={self.logical_slot}/wave={self.wave}/"
            f"g={self.generator_rank}/s={self.score_rank}/"
            f"G{self.generator_version}/C{self.critic_version}"
        )


@dataclass(frozen=True)
class PoolLayout:
    """One physical role placement; model residency may be a superset."""

    name: str
    generator_ranks: tuple[int, ...]
    score_ranks: tuple[int, ...]
    teacher_ranks: tuple[int, ...]
    overlap: bool = True

    def validate(self, *, world_size: int = WORLD_SIZE) -> None:
        all_ranks = set(range(world_size))
        for label, ranks in (
            ("generator", self.generator_ranks),
            ("score", self.score_ranks),
            ("teacher", self.teacher_ranks),
        ):
            _require(bool(ranks), f"{label} pool must not be empty")
            _require(len(ranks) == len(set(ranks)), f"{label} ranks repeat")
            _require(set(ranks) <= all_ranks, f"{label} rank outside world{world_size}")
        _require(
            set(self.teacher_ranks) <= set(self.score_ranks),
            "Teacher FSDP ranks must be score-capable ranks",
        )
        if not self.overlap:
            _require(
                not (set(self.generator_ranks) & set(self.score_ranks)),
                "exclusive layout has overlapping G/Score ranks",
            )
        _require(
            set(self.generator_ranks) | set(self.score_ranks) == all_ranks,
            "G and Score active pools must cover world16",
        )

    @property
    def generator_pool_size(self) -> int:
        return len(self.generator_ranks)

    @property
    def score_pool_size(self) -> int:
        return len(self.score_ranks)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


BASELINE_LAYOUT = PoolLayout(
    name="world16_fsdp2_sequential",
    generator_ranks=tuple(range(16)),
    score_ranks=tuple(range(16)),
    teacher_ranks=tuple(range(16)),
    overlap=True,
)

PIPELINE_8G8S_LAYOUT = PoolLayout(
    name="persistent_8g_8score_teacher_fsdp8",
    generator_ranks=tuple(range(8)),
    score_ranks=tuple(range(8, 16)),
    teacher_ranks=tuple(range(8, 16)),
    overlap=False,
)

# C10 changes only which two immutable trace rows each already-existing
# Generator/Score rank pair executes.  The rank-pair communication graph and
# two B1 samples per lane remain unchanged.  This fixed permutation was
# preregistered from the immutable trace's public ``world_frames`` field; it
# reduces the per-step maximum generated-block load from [12,11,10,12,11] to
# [11,10,10,11,11] without changing total work or any trace row.
TRACE_BALANCED_8G8S_SLOT_PAIRS = (
    (0, 4),
    (1, 9),
    (2, 8),
    (3, 10),
    (5, 12),
    (6, 11),
    (7, 15),
    (13, 14),
)


def validate_slot_groups(
    groups: Sequence[Sequence[int]],
    *,
    lanes: int = 8,
    batch: int = GLOBAL_BATCH,
    slots_per_lane: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Return one deterministic global-batch slot assignment per fixed lane."""

    expected = slots_per_lane if slots_per_lane is not None else batch // lanes
    _require(batch % lanes == 0, "global batch is not divisible by fixed lanes")
    _require(expected == batch // lanes, "slot-group width differs from global batch")
    normalized = tuple(tuple(int(slot) for slot in group) for group in groups)
    _require(len(normalized) == lanes, "slot-group policy must cover every lane")
    _require(
        all(len(group) == expected for group in normalized),
        "fixed lane owns the wrong number of logical slots",
    )
    routed = [slot for group in normalized for slot in group]
    _require(
        sorted(routed) == list(range(batch)),
        "slot-group policy is not a global-batch bijection",
    )
    return normalized


def validate_slot_pairs(
    pairs: Sequence[Sequence[int]], *, lanes: int = 8, batch: int = GLOBAL_BATCH
) -> tuple[tuple[int, int], ...]:
    """Return the legacy deterministic two-slot assignment for every lane."""

    normalized = validate_slot_groups(
        pairs, lanes=lanes, batch=batch, slots_per_lane=2
    )
    return tuple((group[0], group[1]) for group in normalized)


validate_slot_pairs(TRACE_BALANCED_8G8S_SLOT_PAIRS)

PIPELINE_6G10S_LAYOUT = PoolLayout(
    name="persistent_6g_10score_teacher_fsdp10",
    generator_ranks=tuple(range(6)),
    score_ranks=tuple(range(6, 16)),
    teacher_ranks=tuple(range(6, 16)),
    overlap=False,
)

# The logical 6G+10Score layout is unchanged, but the current two-node
# substrate cannot execute large collectives on the asymmetric cross-node
# ranks[6:16] group.  This placement keeps Teacher materialization and the
# first/last legs of Critic gradient synchronization node-local.  Only the two
# leaders exchange the already reduced Critic gradient buckets.
PIPELINE_6G10S_NODE_LOCAL_LAYOUT = PoolLayout(
    name="persistent_6g_10score_teacher_node_local_hierarchical",
    generator_ranks=tuple(range(6)),
    score_ranks=tuple(range(6, 16)),
    teacher_ranks=tuple(range(6, 16)),
    overlap=False,
)
PIPELINE_6G10S_SCORE_NODE_GROUPS = (
    (6, 7),
    tuple(range(8, 16)),
)
PIPELINE_6G10S_SCORE_LEADERS = (6, 8)


def balanced_slots(ranks: Sequence[int], *, batch: int = GLOBAL_BATCH) -> dict[int, tuple[int, ...]]:
    """Assign every logical sample once with rank-local counts differing by <=1."""

    _require(bool(ranks), "cannot route a batch to an empty pool")
    out: dict[int, list[int]] = {int(rank): [] for rank in ranks}
    for slot in range(batch):
        out[int(ranks[slot % len(ranks)])].append(slot)
    routed = [slot for slots in out.values() for slot in slots]
    _require(sorted(routed) == list(range(batch)), "slot routing is not bijective")
    counts = [len(slots) for slots in out.values()]
    _require(max(counts) - min(counts) <= 1, "slot routing is not balanced")
    return {rank: tuple(slots) for rank, slots in out.items()}


def paired_pipeline_routes(
    logical_step: int,
    *,
    generator_version: int,
    critic_version: int,
    layout: PoolLayout = PIPELINE_8G8S_LAYOUT,
) -> tuple[SlotRoute, ...]:
    """Compile deterministic slot-to-rank routes for the paired role pipeline.

    The production 8G+8Score path has two B1 waves.  General pool layouts use
    the same round-robin mapping and may carry more than one local sample; the
    runtime must preserve the operation keys and global-mean loss weighting.
    """

    layout.validate()
    routes: list[SlotRoute] = []
    for slot in range(GLOBAL_BATCH):
        gi = slot % len(layout.generator_ranks)
        si = slot % len(layout.score_ranks)
        routes.append(
            SlotRoute(
                logical_step=logical_step,
                logical_slot=slot,
                generator_rank=layout.generator_ranks[gi],
                score_rank=layout.score_ranks[si],
                wave=max(slot // len(layout.generator_ranks), slot // len(layout.score_ranks)),
                generator_version=generator_version,
                critic_version=critic_version,
            )
        )
    _require({route.logical_slot for route in routes} == set(range(GLOBAL_BATCH)), "missing slot")
    return tuple(routes)


def full5_routes(layout: PoolLayout = PIPELINE_8G8S_LAYOUT) -> tuple[SlotRoute, ...]:
    routes: list[SlotRoute] = []
    generator_version = 0
    critic_version = 0
    for step in range(FULL5_STEPS):
        routes.extend(
            paired_pipeline_routes(
                step,
                generator_version=generator_version,
                critic_version=critic_version,
                layout=layout,
            )
        )
        if step == 0:
            generator_version += 1
        critic_version += 1
    return tuple(routes)


@dataclass(frozen=True)
class TraceIdentity:
    logical_step: int
    logical_slot: int
    dataset_index: int
    seed: int
    operation_key: str


def compile_trace_identities(
    *,
    sample_count: int,
    seed: int,
    layout: PoolLayout = PIPELINE_8G8S_LAYOUT,
) -> tuple[TraceIdentity, ...]:
    _require(sample_count >= GLOBAL_BATCH, "trace source needs at least 16 samples")
    rows: list[TraceIdentity] = []
    for route in full5_routes(layout):
        index = (route.logical_step * GLOBAL_BATCH + route.logical_slot) % sample_count
        sample_seed = seed + route.logical_step * 100_003 + route.logical_slot
        rows.append(
            TraceIdentity(
                logical_step=route.logical_step,
                logical_slot=route.logical_slot,
                dataset_index=index,
                seed=sample_seed,
                operation_key=route.operation_key,
            )
        )
    return tuple(rows)


def trace_sha256(rows: Iterable[TraceIdentity]) -> str:
    payload = [asdict(row) for row in rows]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_full5_rank_receipts(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fail closed on the logical contract before a timing is claimable."""

    ranks = sorted(int(row["rank"]) for row in receipts)
    _require(ranks == list(range(WORLD_SIZE)), f"expected ranks 0..15, got {ranks}")
    trace_hashes = {str(row["trace_sha256"]) for row in receipts}
    _require(len(trace_hashes) == 1, "rank trace hashes differ")
    for row in receipts:
        _require(
            int(row["logical_generator_updates"]) == GENERATOR_UPDATES,
            "G update count mismatch",
        )
        _require(
            int(row["logical_critic_updates"]) == CRITIC_UPDATES,
            "C update count mismatch",
        )
        _require(bool(row.get("finite", False)), "non-finite result")
        _require(not bool(row.get("cuda_oom", False)), "CUDA OOM recorded")
        _require(not bool(row.get("host_oom", False)), "host OOM recorded")
    return {
        "status": "PASS",
        "rank_count": len(receipts),
        "trace_sha256": next(iter(trace_hashes)),
        "generator_updates": GENERATOR_UPDATES,
        "critic_updates": CRITIC_UPDATES,
        "global_batch": GLOBAL_BATCH,
        "steps": FULL5_STEPS,
    }


def speedup_receipt(*, baseline_seconds: float, pipeline_seconds: float) -> dict[str, Any]:
    _require(baseline_seconds > 0 and pipeline_seconds > 0, "timings must be positive")
    speedup = baseline_seconds / pipeline_seconds
    return {
        "baseline_seconds": float(baseline_seconds),
        "pipeline_seconds": float(pipeline_seconds),
        "speedup": speedup,
        "target_speedup": 1.30,
        "target_pipeline_seconds": baseline_seconds / 1.30,
        "status": "PASS" if speedup >= 1.30 else "BELOW_TARGET",
    }

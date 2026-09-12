"""Generate legal frozen-backbone sharing partitions from state identities.

Model family names and equal parameter counts are deliberately not evidence
that two roles can share. The adapter supplies checkpoint, layout and scoped
recompute capabilities; actual storage/memory costs are calibrated separately.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

from .contracts import require


@dataclass(frozen=True)
class RoleState:
    role: str
    architecture_sha256: str
    frozen_base_sha256: str
    base_dtype: str
    placement_fingerprint: str
    base_is_frozen: bool
    # Parameter *storage* identities, not their names (both adapters may use
    # "lora_A.weight"). Optimizers must own disjoint storage allocations.
    trainable_storage_ids: tuple[str, ...]
    adapter_scope: str
    scoped_checkpoint_replay: bool

    def __post_init__(self) -> None:
        require(bool(self.role), "role identity missing")
        for sha in (self.architecture_sha256, self.frozen_base_sha256, self.placement_fingerprint):
            require(len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), "incomplete sharing identity")
        require(type(self.base_is_frozen) is bool and type(self.scoped_checkpoint_replay) is bool,
                "sharing capabilities must be booleans")
        require(self.base_dtype in {"bfloat16", "float32", "float16"}, "unknown backbone dtype")
        require(all(self.trainable_storage_ids) and len(set(self.trainable_storage_ids)) == len(self.trainable_storage_ids),
                "duplicate/invalid trainable storage identity")


def sharing_compatibility(left: RoleState, right: RoleState) -> dict[str, Any]:
    reasons = []
    if not left.base_is_frozen or not right.base_is_frozen:
        reasons.append("MUTABLE_BASE")
    for key in ("architecture_sha256", "frozen_base_sha256", "base_dtype", "placement_fingerprint"):
        if getattr(left, key) != getattr(right, key):
            reasons.append("DIFFERENT_" + key.upper())
    if set(left.trainable_storage_ids) & set(right.trainable_storage_ids):
        reasons.append("TRAINABLE_STORAGE_ALIAS")
    if not (left.adapter_scope and right.adapter_scope and
            left.scoped_checkpoint_replay and right.scoped_checkpoint_replay):
        reasons.append("UNSCOPED_ADAPTER_OR_RECOMPUTE")
    if left.trainable_storage_ids and right.trainable_storage_ids and left.adapter_scope == right.adapter_scope:
        reasons.append("TRAINABLE_ADAPTER_SCOPE_COLLISION")
    return {"roles": [left.role, right.role], "status": "COMPATIBLE" if not reasons else "REJECTED",
            "reasons": reasons, "gpu_validation": "NOT_RUN",
            "requires_serialized_adapter_scope": not reasons}


def enumerate_sharing_partitions(roles: Sequence[RoleState]) -> list[tuple[tuple[str, ...], ...]]:
    """Bounded Bell-partition enumeration for the small DMD role set.

This generates choices; it does not assume maximal sharing is the winner.
Sharing can remove inter-role overlap and incur role handoff costs.
"""
    require(1 <= len(roles) <= 6, "sharing search limited to six physical roles")
    require(len({r.role for r in roles}) == len(roles), "duplicate sharing role")
    ordered = sorted(roles, key=lambda r: r.role)
    choices: list[list[list[RoleState]]] = [[]]
    for role in ordered:
        next_choices = []
        for groups in choices:
            next_choices.append([*groups, [role]])
            for index, group in enumerate(groups):
                if all(sharing_compatibility(other, role)["status"] == "COMPATIBLE" for other in group):
                    next_choices.append([list(g) + ([role] if i == index else []) for i, g in enumerate(groups)])
        choices = next_choices
    return [tuple(tuple(r.role for r in group) for group in groups) for groups in choices]


def sharing_report(roles: Sequence[RoleState]) -> Mapping[str, Any]:
    return {"schema_version": "distillschedule.sharing_capability.v1",
            "pair_checks": [sharing_compatibility(a, b) for a, b in combinations(roles, 2)],
            "legal_partitions": enumerate_sharing_partitions(roles),
            "selection": "REQUIRES_CALIBRATED_MEMORY_HANDOFF_AND_OVERLAP_COSTS",
            "gpu_execution": "NOT_RUN"}

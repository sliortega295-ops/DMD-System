"""Fail-closed fairness contracts for matched GAN baseline/pipeline timing.

The timing comparison is only meaningful when both executions consume the same
logical 16 samples and the same stochastic model inputs.  A shared seed is not
sufficient once world16 execution is rescheduled into two eight-rank waves,
because RNG call order changes.  Formal runs therefore record *consumed trace
identities* per logical sample rather than comparing seeds.

This module deliberately does not sample random values.  It validates receipts
emitted by the real baseline/pipeline workers after they have looked up their
operation-keyed tensors and control decisions from the immutable GAN trace.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "distillschedule.gan_fairness_receipt.v1"
FULL5_STEPS = (1000, 1001, 1002, 1003, 1004)
LOGICAL_SAMPLE_SLOTS = tuple(range(16))
GENERATOR_UPDATE_STEPS = (1000,)

# These are semantic *consumption* identities.  One role may represent a packed
# ordered tensor sequence (for example every re-noise tensor in a rollout).
# Hashing the packed sequence keeps the contract independent of the number of
# physical P2P messages while still detecting a changed random path.
GENERATOR_RANDOM_ROLES = (
    "source_primary",
    "source_secondary",
    "text_embedding",
    "processed_generator_history",
    "generator_rollout_entry_noise",
    "generator_rollout_renoise_sequence",
    "generator_block_noise_stage1",
    "generator_block_noise_stage2",
    "generator_dmd_timestep",
    "generator_dmd_noise",
    "generator_gan_g_noise",
    "generator_gan_g_crop",
)

CRITIC_RANDOM_ROLES = (
    "source_primary",
    "source_secondary",
    "text_embedding",
    "processed_critic_history",
    "critic_rollout_entry_noise",
    "critic_rollout_renoise_sequence",
    "critic_block_noise_stage1",
    "critic_block_noise_stage2",
    "critic_timestep",
    "critic_noise",
    "gan_d_fake_crop",
    "gan_d_real_crop",
    "gan_d_r1_crop",
    "r1_epsilon",
)

# R2 is inactive in the frozen official GAN contract (r2_weight=0).  If that
# changes, receipt validation must explicitly opt in rather than silently
# accepting a different stochastic workload.
OPTIONAL_R2_ROLES = ("gan_d_r2_crop", "r2_epsilon")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def object_digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{path} must be a 64-character SHA256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{path} is not hexadecimal") from exc
    return value.lower()


@dataclass(frozen=True)
class GanLogicalSampleAddress:
    """Canonical mapping between world16 baseline and persistent 8G+8Score."""

    logical_sample_slot: int
    baseline_rank: int
    pipeline_wave: int
    pipeline_generator_rank: int
    pipeline_score_rank: int

    def __post_init__(self) -> None:
        slot = self.logical_sample_slot
        if slot not in LOGICAL_SAMPLE_SLOTS:
            raise ValueError("logical sample slot must be in [0, 16)")
        expected = (
            slot,
            slot // 8,
            slot % 8,
            8 + (slot % 8),
        )
        observed = (
            self.baseline_rank,
            self.pipeline_wave,
            self.pipeline_generator_rank,
            self.pipeline_score_rank,
        )
        if observed != expected:
            raise ValueError(
                "physical sample mapping drift: "
                f"slot={slot}, observed={observed}, expected={expected}"
            )

    @classmethod
    def for_slot(cls, slot: int) -> "GanLogicalSampleAddress":
        return cls(
            logical_sample_slot=slot,
            baseline_rank=slot,
            pipeline_wave=slot // 8,
            pipeline_generator_rank=slot % 8,
            pipeline_score_rank=8 + (slot % 8),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "logical_sample_slot": self.logical_sample_slot,
            "baseline_rank": self.baseline_rank,
            "pipeline_wave": self.pipeline_wave,
            "pipeline_generator_rank": self.pipeline_generator_rank,
            "pipeline_score_rank": self.pipeline_score_rank,
        }


def canonical_sample_mapping() -> tuple[GanLogicalSampleAddress, ...]:
    return tuple(GanLogicalSampleAddress.for_slot(slot) for slot in LOGICAL_SAMPLE_SLOTS)


def _normalize_dit(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    scope = value.get("stage_schedule_scope")
    if scope != "global_rank0_broadcast_equivalent":
        raise ValueError(
            f"{path}.stage_schedule_scope={scope!r}; formal GAN timing forbids rank-local schedules"
        )
    exits_raw = value.get("stage_exits")
    if not isinstance(exits_raw, Sequence) or isinstance(exits_raw, (str, bytes)):
        raise ValueError(f"{path}.stage_exits must be a sequence")
    exits = tuple(int(item) for item in exits_raw)
    if len(exits) != 3 or any(item < 0 for item in exits):
        raise ValueError(f"{path}.stage_exits must contain three non-negative exits")

    # For the Helios denoising loop, an exit at index e executes transformer
    # calls for indices 0..e, hence e+1 complete DiT forwards in that stage.
    derived_counts = tuple(item + 1 for item in exits)
    counts_raw = value.get("stage_call_counts")
    if counts_raw is not None:
        counts = tuple(int(item) for item in counts_raw)
        if counts != derived_counts:
            raise ValueError(
                f"{path}.stage_call_counts={counts} disagrees with exits {exits}"
            )
    total = int(value.get("total_dit_forwards", sum(derived_counts)))
    if total != sum(derived_counts):
        raise ValueError(f"{path}.total_dit_forwards does not match stage exits")
    return {
        "stage_schedule_scope": "global_rank0_broadcast_equivalent",
        "stage_exits": list(exits),
        "stage_call_counts": list(derived_counts),
        "total_dit_forwards": total,
    }


def _normalize_role_hashes(
    value: Any,
    *,
    required: Sequence[str],
    allow_r2: bool,
    path: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    expected = set(required)
    if allow_r2:
        expected.update(OPTIONAL_R2_ROLES)
    observed = set(str(key) for key in value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{path} role inventory drift: missing={missing}, extra={extra}")
    return {
        role: _sha256(value[role], path=f"{path}.{role}")
        for role in sorted(expected)
    }


def _normalize_phase(
    value: Any,
    *,
    required_roles: Sequence[str],
    allow_r2: bool,
    path: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    trace_operation = _sha256(
        value.get("trace_operation_sha256"),
        path=f"{path}.trace_operation_sha256",
    )
    return {
        "trace_operation_sha256": trace_operation,
        "dit": _normalize_dit(value.get("dit"), path=f"{path}.dit"),
        "consumed": _normalize_role_hashes(
            value.get("consumed"),
            required=required_roles,
            allow_r2=allow_r2,
            path=f"{path}.consumed",
        ),
    }


def normalize_gan_fairness_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize one actual baseline/pipeline consumption receipt."""

    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported GAN fairness receipt schema")
    family = str(value.get("family", ""))
    if family not in {"baseline", "pipeline"}:
        raise ValueError("GAN fairness receipt family must be baseline or pipeline")
    if value.get("status") != "PASS":
        raise ValueError("GAN fairness receipt did not PASS")
    trace_bundle_hash = _sha256(
        value.get("trace_bundle_hash"), path="trace_bundle_hash"
    )
    allow_r2 = bool(value.get("r2_enabled", False))
    # The frozen official contract has R2 disabled.  A true value must therefore
    # be a different workload contract, not a timing comparison against PR #4.
    if allow_r2:
        raise ValueError("formal frozen GAN comparison requires r2_enabled=false")

    entries_raw = value.get("entries")
    if not isinstance(entries_raw, list):
        raise ValueError("GAN fairness receipt entries must be a list")
    expected_pairs = {
        (step, slot) for step in FULL5_STEPS for slot in LOGICAL_SAMPLE_SLOTS
    }
    observed_pairs: set[tuple[int, int]] = set()
    normalized_entries: list[dict[str, Any]] = []
    for index, raw in enumerate(entries_raw):
        path = f"entries[{index}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path} must be a mapping")
        step = int(raw.get("logical_step", -1))
        slot = int(raw.get("logical_sample_slot", -1))
        pair = (step, slot)
        if pair not in expected_pairs:
            raise ValueError(f"{path} has out-of-contract step/slot {pair}")
        if pair in observed_pairs:
            raise ValueError(f"duplicate GAN fairness entry {pair}")
        observed_pairs.add(pair)
        canonical_address = GanLogicalSampleAddress.for_slot(slot)
        physical = raw.get("physical")
        if not isinstance(physical, Mapping):
            raise ValueError(f"{path}.physical must be a mapping")
        if family == "baseline":
            if int(physical.get("rank", -1)) != canonical_address.baseline_rank:
                raise ValueError(f"{path} baseline rank does not match logical slot")
        else:
            observed_location = (
                int(physical.get("wave", -1)),
                int(physical.get("generator_rank", -1)),
                int(physical.get("score_rank", -1)),
            )
            expected_location = (
                canonical_address.pipeline_wave,
                canonical_address.pipeline_generator_rank,
                canonical_address.pipeline_score_rank,
            )
            if observed_location != expected_location:
                raise ValueError(
                    f"{path} pipeline sample mapping={observed_location}, "
                    f"expected={expected_location}"
                )

        generator_raw = raw.get("generator")
        if step in GENERATOR_UPDATE_STEPS:
            generator = _normalize_phase(
                generator_raw,
                required_roles=GENERATOR_RANDOM_ROLES,
                allow_r2=False,
                path=f"{path}.generator",
            )
        else:
            if generator_raw not in (None, {}):
                raise ValueError(
                    f"{path}.generator must be absent on Critic-only step {step}"
                )
            generator = None
        critic = _normalize_phase(
            raw.get("critic"),
            required_roles=CRITIC_RANDOM_ROLES,
            allow_r2=False,
            path=f"{path}.critic",
        )
        normalized_entries.append(
            {
                "logical_step": step,
                "logical_sample_slot": slot,
                # Physical placement is deliberately omitted from canonical
                # comparison after it has been validated above.
                "generator": generator,
                "critic": critic,
            }
        )

    if observed_pairs != expected_pairs:
        missing = sorted(expected_pairs - observed_pairs)
        raise ValueError(f"GAN fairness receipt misses logical work: {missing}")
    normalized_entries.sort(
        key=lambda item: (item["logical_step"], item["logical_sample_slot"])
    )
    canonical = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "trace_bundle_hash": trace_bundle_hash,
        "r2_enabled": False,
        "logical_steps": list(FULL5_STEPS),
        "logical_sample_slots": list(LOGICAL_SAMPLE_SLOTS),
        "entries": normalized_entries,
    }
    canonical["fairness_contract_digest"] = object_digest(canonical)
    return canonical


def compare_gan_fairness_receipts(
    baseline: Mapping[str, Any], pipeline: Mapping[str, Any]
) -> dict[str, Any]:
    """Require exact stochastic/DiT equality after canonical sample remapping."""

    a = normalize_gan_fairness_receipt(baseline)
    b = normalize_gan_fairness_receipt(pipeline)
    if a != b:
        # Produce a compact first mismatch for logs rather than dumping 80 large
        # entries.  Full receipts remain immutable artifacts.
        if a["trace_bundle_hash"] != b["trace_bundle_hash"]:
            reason = "trace_bundle_hash"
        else:
            reason = "consumed stochastic tensor or DiT schedule"
        raise ValueError(f"GAN baseline/pipeline fairness mismatch: {reason}")
    return {
        "schema_version": "distillschedule.gan_fairness_pair.v1",
        "status": "PASS",
        "logical_window": [FULL5_STEPS[0], FULL5_STEPS[-1]],
        "logical_samples_per_step": len(LOGICAL_SAMPLE_SLOTS),
        "fairness_contract_digest": a["fairness_contract_digest"],
        "trace_bundle_hash": a["trace_bundle_hash"],
        "sample_mapping_digest": object_digest(
            [item.to_dict() for item in canonical_sample_mapping()]
        ),
        "guarantee": (
            "same logical source/random tensors and same complete DiT forward "
            "counts; physical scheduling may differ"
        ),
    }


__all__ = [
    "CRITIC_RANDOM_ROLES",
    "FULL5_STEPS",
    "GENERATOR_RANDOM_ROLES",
    "GanLogicalSampleAddress",
    "LOGICAL_SAMPLE_SLOTS",
    "SCHEMA_VERSION",
    "canonical_sample_mapping",
    "compare_gan_fairness_receipts",
    "normalize_gan_fairness_receipt",
    "object_digest",
]

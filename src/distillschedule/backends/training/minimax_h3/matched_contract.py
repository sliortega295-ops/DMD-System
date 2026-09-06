"""Fail-closed matched-compute contract for controlled MiniMax-H3 DMD timing.

The contract mirrors the DMD-System MiniMax matched benchmark: every outer DMD
cycle uses six independent global batches (one Student + five Fake), every
back-simulation runs all four Student model evaluations, and application-level
DiT forward/backward counts are fixed. Activation-checkpoint recomputation is
not counted as an application forward.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

STUDENT_ROLE = "student"
FAKE_ROLE = "fake"
TEACHER_ROLE = "teacher"

FIXED_END_STEP_IDX = 3
EXPECTED_FORWARD_COUNTS = {STUDENT_ROLE: 24, FAKE_ROLE: 6, TEACHER_ROLE: 1}
EXPECTED_GRAD_FORWARD_COUNTS = {STUDENT_ROLE: 1, FAKE_ROLE: 5, TEACHER_ROLE: 0}
EXPECTED_BACKWARD_COUNTS = {STUDENT_ROLE: 1, FAKE_ROLE: 5}
EXPECTED_SAMPLE_STAGES = ["student"] + [f"fake_{index}" for index in range(5)]


def sample_identity(sample: Any) -> str:
    """Return a stable dataset identity comparable across ranks and schedules."""
    meta = sample.get("meta", {}) if isinstance(sample, dict) else {}
    if not isinstance(meta, dict):
        raise RuntimeError(f"Matched MiniMax sample meta must be a dict, got {type(meta)!r}")
    keys = (
        "id",
        "row_index",
        "condition_path",
        "prompt_path",
        "lmdb_path",
        "video_latent_path",
        "audio_latent_path",
    )
    identity = {key: str(meta[key]) for key in keys if key in meta and meta[key] is not None}
    if not identity:
        raise RuntimeError(
            "Matched MiniMax timing requires rank-qualified sample identity in meta "
            "(id/row_index/condition_path/lmdb_path/etc.)."
        )
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


@dataclass
class MatchedCycleCensus:
    """Per-rank application-level compute and sample census for one DMD cycle."""

    enabled: bool = True
    fixed_end_step_idx: int = FIXED_END_STEP_IDX
    expected_world_size: int = 16
    require_unique_samples: bool = True
    forward_counts: dict[str, int] = field(default_factory=dict)
    grad_forward_counts: dict[str, int] = field(default_factory=dict)
    backward_counts: dict[str, int] = field(default_factory=dict)
    samples: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.forward_counts = {role: 0 for role in EXPECTED_FORWARD_COUNTS}
        self.grad_forward_counts = {role: 0 for role in EXPECTED_GRAD_FORWARD_COUNTS}
        self.backward_counts = {role: 0 for role in EXPECTED_BACKWARD_COUNTS}
        self.samples = []

    def note_forward(self, role: str, *, grad_enabled: bool) -> None:
        if not self.enabled:
            return
        if role not in self.forward_counts:
            raise RuntimeError(f"Unknown matched MiniMax DiT role: {role!r}")
        self.forward_counts[role] += 1
        if grad_enabled:
            self.grad_forward_counts[role] += 1

    def note_backward(self, role: str) -> None:
        if not self.enabled:
            return
        if role not in self.backward_counts:
            raise RuntimeError(f"Unknown matched MiniMax backward role: {role!r}")
        self.backward_counts[role] += 1

    def note_sample(self, stage: str, sample: Any) -> None:
        if not self.enabled:
            return
        self.samples.append({"stage": str(stage), "identity": sample_identity(sample)})

    def snapshot(self) -> dict[str, Any]:
        return {
            "fixed_end_step_idx": int(self.fixed_end_step_idx),
            "forward_counts": dict(self.forward_counts),
            "grad_forward_counts": dict(self.grad_forward_counts),
            "backward_counts": dict(self.backward_counts),
            "samples": list(self.samples),
        }

    def validate_local(self) -> list[str]:
        if not self.enabled:
            return []
        errors: list[str] = []
        if self.fixed_end_step_idx != FIXED_END_STEP_IDX:
            errors.append(
                f"fixed_end_step_idx={self.fixed_end_step_idx} != matched baseline {FIXED_END_STEP_IDX}"
            )
        if self.forward_counts != EXPECTED_FORWARD_COUNTS:
            errors.append(f"forward_counts={self.forward_counts} != {EXPECTED_FORWARD_COUNTS}")
        if self.grad_forward_counts != EXPECTED_GRAD_FORWARD_COUNTS:
            errors.append(
                f"grad_forward_counts={self.grad_forward_counts} != {EXPECTED_GRAD_FORWARD_COUNTS}"
            )
        if self.backward_counts != EXPECTED_BACKWARD_COUNTS:
            errors.append(f"backward_counts={self.backward_counts} != {EXPECTED_BACKWARD_COUNTS}")
        stages = [item["stage"] for item in self.samples]
        if stages != EXPECTED_SAMPLE_STAGES:
            errors.append(f"sample_stages={stages} != {EXPECTED_SAMPLE_STAGES}")
        identities = [item["identity"] for item in self.samples]
        if self.require_unique_samples and len(set(identities)) != len(identities):
            errors.append("same rank reused a sample inside one Student1/Fake5 cycle")
        return errors


def validate_global_snapshots(
    snapshots: list[dict[str, Any]],
    *,
    expected_world_size: int,
    require_unique_samples: bool,
) -> list[str]:
    """Validate world-wide batch semantics without changing the timed compute graph."""
    errors: list[str] = []
    if len(snapshots) != expected_world_size:
        errors.append(f"world snapshots={len(snapshots)} != expected {expected_world_size}")
        return errors

    all_identities: list[str] = []
    for rank, snapshot in enumerate(snapshots):
        if snapshot.get("forward_counts") != EXPECTED_FORWARD_COUNTS:
            errors.append(f"rank {rank} forward census mismatch")
        if snapshot.get("grad_forward_counts") != EXPECTED_GRAD_FORWARD_COUNTS:
            errors.append(f"rank {rank} grad-forward census mismatch")
        if snapshot.get("backward_counts") != EXPECTED_BACKWARD_COUNTS:
            errors.append(f"rank {rank} backward census mismatch")
        samples = snapshot.get("samples", [])
        stages = [item.get("stage") for item in samples]
        if stages != EXPECTED_SAMPLE_STAGES:
            errors.append(f"rank {rank} sample stage sequence mismatch: {stages}")
        all_identities.extend(str(item.get("identity")) for item in samples)

    expected_samples = expected_world_size * len(EXPECTED_SAMPLE_STAGES)
    if len(all_identities) != expected_samples:
        errors.append(f"sample identities={len(all_identities)} != expected {expected_samples}")
    if require_unique_samples and len(set(all_identities)) != len(all_identities):
        errors.append(
            f"global sample reuse detected: unique={len(set(all_identities))} total={len(all_identities)}"
        )
    return errors

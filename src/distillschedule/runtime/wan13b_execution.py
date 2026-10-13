"""Explicit distributed admission for the B1 Wan weak-scaling experiment.

Canonical callers continue to use world16. World8 is not inferred from the
live launcher: it must be selected by a versioned weak-scaling manifest.
No model, update ratio, trace, seed, or optimizer hyperparameter is changed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class WanDistributedExecution:
    world_size: int = 16
    batch_per_rank: int = 1
    experiment: str = "canonical_world16"

    def __post_init__(self):
        if isinstance(self.world_size, bool) or not isinstance(self.world_size, int):
            raise TypeError("world size must be an integer, not a boolean")
        if self.world_size not in (8, 16):
            raise ValueError("only the admitted world8/world16 experiment is supported")
        if self.batch_per_rank != 1 or isinstance(self.batch_per_rank, bool):
            raise ValueError("weak scaling keeps B1 per global rank")
        if self.experiment not in ("canonical_world16", "weak_scaling"):
            raise ValueError("unrecognized distributed experiment")
        if self.world_size == 8 and self.experiment != "weak_scaling":
            raise ValueError("world8 requires explicit weak_scaling opt-in")

    @property
    def global_batch(self):
        return self.world_size * self.batch_per_rank

    @property
    def sample_count_per_cycle(self):
        return 6 * self.global_batch  # one G sample and five C samples/rank

    @property
    def gradient_divisor(self):
        return float(self.world_size)

    def validate_live_world(self, observed):
        if observed != self.world_size:
            raise ValueError(f"live world {observed} differs from admitted {self.world_size}")

    def receipt(self):
        return dict(asdict(self), global_batch=self.global_batch,
                    gradient_divisor=self.gradient_divisor,
                    sample_count_per_cycle=self.sample_count_per_cycle,
                    logical_updates="G1_C5_EMA5", rollout_batch_size=1,
                    score_packing=False)


def admit_execution(value=None):
    if value is None:
        return WanDistributedExecution()
    if not isinstance(value, WanDistributedExecution):
        raise TypeError("distributed execution must be the typed Wan contract")
    return value

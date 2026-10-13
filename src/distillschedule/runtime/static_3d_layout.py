"""Static TP4/DP2/PP2 baseline layout, not a timing simulator or executor.

Keeps logical samples distinct from collaborating physical ranks. A TP/PP
replica consumes eight B1 samples per update, so DP2 still means global B16.
Native model/LoRA/optimizer bindings must be supplied before GPU admission.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Static3DLayout:
    pp: int = 2
    dp: int = 2
    tp: int = 4
    microbatches: int = 8

    def __post_init__(self):
        if (self.pp, self.dp, self.tp, self.microbatches) != (2, 2, 4, 8):
            raise ValueError("the static paper baseline is TP4/DP2/PP2, global B16")

    def physical_rank(self, pp: int, dp: int, tp: int) -> int:
        if not (0 <= pp < self.pp and 0 <= dp < self.dp and 0 <= tp < self.tp):
            raise ValueError("out-of-range 3D coordinate")
        return (pp * self.dp + dp) * self.tp + tp

    def coordinate(self, rank: int) -> tuple[int, int, int]:
        if not 0 <= rank < 16:
            raise ValueError("static layout requires physical ranks 0..15")
        return rank // 8, (rank % 8) // 4, rank % 4

    def logical_sample_rank(self, dp: int, microbatch: int) -> int:
        if not (0 <= dp < self.dp and 0 <= microbatch < self.microbatches):
            raise ValueError("invalid logical sample coordinate")
        return dp * self.microbatches + microbatch

    def groups(self) -> dict[str, list[list[int]]]:
        return {
            "tp": [[self.physical_rank(p,d,t) for t in range(self.tp)]
                   for p in range(self.pp) for d in range(self.dp)],
            "dp": [[self.physical_rank(p,d,t) for d in range(self.dp)]
                   for p in range(self.pp) for t in range(self.tp)],
            "pp": [[self.physical_rank(p,d,t) for p in range(self.pp)]
                   for d in range(self.dp) for t in range(self.tp)],
        }

    def block_ranges(self, blocks: int) -> tuple[tuple[int,int], tuple[int,int]]:
        if blocks <= 0 or blocks % self.pp:
            raise ValueError("two equal PP stages require an even block count")
        middle = blocks // self.pp
        return (0,middle),(middle,blocks)

    def receipt(self) -> dict:
        return dict(layout="TP4_DP2_PP2", world_size=16,
                    groups=self.groups(), global_batch=self.dp*self.microbatches,
                    microbatch_size=1, microbatches_per_dp_replica=self.microbatches,
                    sample_mapping=[[self.logical_sample_rank(d,m)
                                     for m in range(self.microbatches)] for d in range(self.dp)],
                    application_census_scope="PER_LOGICAL_SOURCE_SAMPLE_NOT_PER_PARTIAL_TP_PP_STAGE",
                    timing="NOT_RUN_NO_NATIVE_EXECUTOR_IMPLIED")


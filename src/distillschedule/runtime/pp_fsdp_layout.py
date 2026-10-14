"""World16 PP2 plus real data-parallel FSDP lanes, never tensor parallelism.

This is an opt-in placement contract, not an executor or a GPU PASS receipt.
PP peers consume the same sample; shard/replica coordinates consume different
samples. Two physical B1 waves retain logical global batch16 in both capacity
configurations. A CUDA OOM may change shard4/replica2 to shard8/replica1 only.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PPFSDPLayout:
    shard: int = 4
    pp: int = 2
    microbatches: int = 2

    def __post_init__(self):
        if (type(self.shard) is not int or self.shard not in (4, 8)
                or self.pp != 2 or self.microbatches != 2):
            raise ValueError('admitted world16 ladder is PP2/shard4/replica2 then PP2/shard8/replica1')

    @property
    def replicate(self) -> int:
        return 8 // self.shard

    def coordinate(self, rank: int) -> tuple[int, int, int]:
        if type(rank) is not int or not 0 <= rank < 16:
            raise ValueError('physical ranks must be 0..15')
        return rank // 8, (rank % 8) // self.shard, rank % self.shard

    def physical_rank(self, pp: int, replica: int, shard: int) -> int:
        if not (0 <= pp < 2 and 0 <= replica < self.replicate and 0 <= shard < self.shard):
            raise ValueError('invalid PP/FSDP/replica coordinate')
        return pp * 8 + replica * self.shard + shard

    def source_rank(self, physical_rank: int, wave: int) -> int:
        self.coordinate(physical_rank)
        if type(wave) is not int or not 0 <= wave < 2:
            raise ValueError('exactly two independent B1 waves form global batch16')
        return (physical_rank % 8) + 8 * wave

    def block_ranges(self, count: int) -> tuple[tuple[int, int], tuple[int, int]]:
        if type(count) is not int or count <= 0 or count % 2:
            raise ValueError('native PP boundary requires an admitted even block count')
        return (0, count // 2), (count // 2, count)

    def groups(self) -> dict[str, list[list[int]]]:
        return {
            'fsdp': [[self.physical_rank(p, r, s) for s in range(self.shard)]
                     for p in range(2) for r in range(self.replicate)],
            'replica': [[self.physical_rank(p, r, s) for r in range(self.replicate)]
                        for p in range(2) for s in range(self.shard)],
            'pp': [[i, i + 8] for i in range(8)],
        }

    def receipt(self) -> dict:
        return dict(
            topology=f'PP2_FSDP{self.shard}_REPLICA{self.replicate}', world_size=16,
            TP=1, PP=2, fsdp_shard=self.shard, dp_replicate=self.replicate,
            independent_data_lanes=8, microbatch_size=1, microbatches_per_lane=2,
            global_batch=16, groups=self.groups(),
            sample_mapping=[[self.source_rank(i, m) for m in range(2)] for i in range(8)],
            update_rule='MEAN2_MICROBATCHES_THEN_MEAN8_DATA_LANES_ONE_COMMIT',
            count_scope='PER_LOGICAL_SOURCE_SAMPLE; PP_PEERS_ARE_NOT_EXTRA_SAMPLES',
            numerical='NOT_RUN', runtime='NOT_RUN_CONTRACT_ONLY',
        )

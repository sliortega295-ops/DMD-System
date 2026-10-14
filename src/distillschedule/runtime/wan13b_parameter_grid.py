"""Explicit workload changes for the main-experiment grid, not plan knobs.

Canonical StaticWorld16Trace is untouched. Every point requires a fresh
source-generated trace and its own matching physical baseline. This module
does not imply an implemented native worker or a measured grid point.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from .wan13b_execution import WanDistributedExecution


@dataclass(frozen=True)
class WanGridDistributedExecution(WanDistributedExecution):
    experiment: str = 'parameter_grid'
    fake_updates: int = 5

    def __post_init__(self):
        if type(self.world_size) is not int or self.world_size != 16:
            raise ValueError('this workload grid requires explicit world16')
        if type(self.batch_per_rank) is not int or self.batch_per_rank != 1:
            raise ValueError('grid preserves independent B1/global16')
        if self.experiment != 'parameter_grid' or type(self.fake_updates) is not int or self.fake_updates not in (1, 3, 5):
            raise ValueError('explicit native update-ratio admission required')

    @property
    def sample_count_per_cycle(self):
        return (1 + self.fake_updates) * self.global_batch

    def receipt(self):
        return dict(world_size=self.world_size, batch_per_rank=1,
                    experiment=self.experiment, fake_updates=self.fake_updates,
                    global_batch=16, gradient_divisor=16.0,
                    sample_count_per_cycle=self.sample_count_per_cycle,
                    logical_updates=f'G1_C{self.fake_updates}_EMA{self.fake_updates}',
                    rollout_batch_size=1, score_packing=False)


@dataclass(frozen=True)
class WanParameterGridSpec:
    height: int
    width: int
    fake_updates: int
    pyramid_budget: tuple[int, int, int]
    seed: int = 20260903
    start_step: int = 30

    def __post_init__(self):
        ints = (self.height, self.width, self.fake_updates, self.seed, self.start_step)
        if any(type(v) is not int for v in ints):
            raise TypeError('integer workload fields must not be boolean/coerced')
        if (self.height, self.width) not in ((384, 640), (448, 832), (704, 1280)):
            raise ValueError('unregistered resolution')
        if self.height % 64 or self.width % 64:
            raise ValueError('three-stage pyramid needs 64-aligned dimensions')
        if self.fake_updates not in (1, 3, 5):
            raise ValueError('unregistered Student/Fake update ratio')
        if not isinstance(self.pyramid_budget, tuple) or self.pyramid_budget not in ((1, 1, 1), (2, 2, 2)):
            raise ValueError('unregistered explicit pyramid budget')
        if any(type(v) is not int for v in self.pyramid_budget):
            raise TypeError('budgets must be real integers')
        if self.seed < 0 or self.start_step != 30 or self.start_step % self.fake_updates:
            raise ValueError('native modulo schedule requires the common aligned first G step30')

    @property
    def logical_steps(self):
        return tuple(range(self.start_step, self.start_step + self.fake_updates))

    @property
    def spatial_shapes(self):
        return tuple((self.height // d, self.width // d) for d in (32, 16, 8))

    def receipt(self):
        row = dict(schema='dmd.wan_parameter_grid_workload.v1', **asdict(self),
                   logical_steps=list(self.logical_steps),
                   stage_spatial_shapes=[list(s) for s in self.spatial_shapes],
                   world_size=16, batch_per_rank=1, global_batch=16,
                   generator_updates=1, critic_updates=self.fake_updates,
                   ema_updates=self.fake_updates, rollout_batch_size=1,
                   score_packing=False, dtype='bfloat16',
                   frozen_base='BF16', lora_compute='FP32', gradient_reduce='FP32',
                   input_policy='FRESH_SOURCE_GENERATED_RANK16_TRACE_NOT_TRUNCATED_F5')
        row['fingerprint'] = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        return row

    def validate_rank_manifest(self, manifest):
        contract = manifest['contract']
        required = dict(resolution=[self.height, self.width],
                        stage_spatial_shapes=[list(s) for s in self.spatial_shapes],
                        full_latent_shape=[1, 16, 9, self.height // 8, self.width // 8],
                        dtype='bfloat16', dfake_gen_update_ratio=self.fake_updates)
        for key, value in required.items():
            if contract.get(key) != value:
                raise ValueError('fresh grid trace contract mismatch: ' + key)
        operations = {row['key']: row for row in manifest['operations']}
        actual_branches = sorted(int(key.split('/')[1]) for key in operations if key.endswith('/branch'))
        if actual_branches != list(self.logical_steps):
            raise ValueError('old/cohort-truncated or mismatched native step interval')
        for step in self.logical_steps:
            branch = operations[f'step/{step}/branch']['control']
            required = dict(global_step=step, dfake_gen_update_ratio=self.fake_updates,
                            train_generator=step == self.start_step, train_critic=True,
                            gan=False, reward=False, vae_decode=False)
            if any(branch.get(k) != v for k, v in required.items()):
                raise ValueError('native branch/update order mismatch')
            roles = ('generator', 'critic') if step == self.start_step else ('critic',)
            for role in roles:
                prefix = f'step/{step}/{role}'
                control = operations[prefix + '/rollout']['control']
                stages = control['pyramid_stage_call_sequence']
                if len(stages) != 3:
                    raise ValueError('missing native pyramid stage')
                # The native amplified-history branch may expand a budget.
                # The generator must preserve and disclose this, not silently
                # switch that branch off or prescribe convenient exits.
                for index, stage in enumerate(stages):
                    budget = stage['denoising_step_budget']
                    if type(budget) is not int:
                        raise TypeError('native budgets must not be coerced floats/booleans')
                    if budget not in (self.pyramid_budget[index], 2 * self.pyramid_budget[index]):
                        raise ValueError('native budget does not match configured workload')
                    if list(stage['latent_shape']) != [1, 16, 9, *self.spatial_shapes[index]]:
                        raise ValueError('stage shape mismatch')
                operations[prefix + ('/dmd' if role == 'generator' else '/train')]


def enumerate_main_grid():
    return tuple(WanParameterGridSpec(h, w, ratio, budget)
                 for h, w in ((384, 640), (448, 832), (704, 1280))
                 for ratio in (1, 3, 5)
                 for budget in ((1, 1, 1), (2, 2, 2)))

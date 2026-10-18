"""Expose EXISTING Wan placement/reduction/chunk/epilogue paths to new tasks."""
from __future__ import annotations
from dataclasses import dataclass
from contextlib import contextmanager
import os
from typing import Mapping
from distillschedule.core.contracts import require
from distillschedule.search.wan_implementation import WanImplementation, PRECISION
from .wan_task import OriginalWanTask
from distillschedule.search.wan_candidate_bundle import WanCandidateBundle, OFF, INCUMBENT
from distillschedule.workloads.best_known.registry import fingerprint


@dataclass(frozen=True)
class WanStrategy:
    placement: str
    chunk_tokens: int
    lora_epilogue: bool
    rollout_window: int = 1
    candidate_bundle: WanCandidateBundle = OFF
    retained_critic_blocks: tuple[int, ...] = ()
    retention_policy: str = 'retain4.v1'
    rollout_graph_pool_policy: str = 'isolated.v1'
    retention_execution: str = 'manual-only.v1'

    def __post_init__(self):
        # Original native admission remains unchanged. In particular, FSDP only
        # supports its control kernel configuration until another arm is validated.
        WanImplementation(self.placement, self.chunk_tokens, self.lora_epilogue, candidate_bundle=self.candidate_bundle)
        require(self.rollout_graph_pool_policy in ('isolated.v1', 'shared-scratch.v1'),
                'unknown rollout graph pool policy', path='wan.strategy')
        require(self.rollout_graph_pool_policy == 'isolated.v1' or
                (self.candidate_bundle.enabled and self.placement == 'resident'
                 and self.retained_critic_blocks == (26,27,28,29)
                 and self.retention_policy == 'retain4.v1'),
                'shared scratch experiment requires exact incumbent retainC26-29', path='wan.strategy')
        require(type(self.rollout_window) is int and 1 <= self.rollout_window <= 64,
                'invalid parameterized rollout window', path='wan.strategy')
        require(isinstance(self.retained_critic_blocks, tuple)
                and all(type(i) is int and 0 <= i < 30 for i in self.retained_critic_blocks)
                and tuple(sorted(set(self.retained_critic_blocks))) == self.retained_critic_blocks,
                'retained Critic indices must be sorted unique block integers', path='wan.strategy')
        require(self.retention_policy in ('retain4.v1', 'tail8.v1', 'tail6.v1'),
                'unknown experimental retention policy', path='wan.strategy')
        require(self.retention_policy != 'tail8.v1' or self.retained_critic_blocks == tuple(range(22,30)),
                'tail8.v1 requires the exact eight final Critic blocks', path='wan.strategy')
        require(self.retention_policy != 'tail6.v1' or self.retained_critic_blocks == tuple(range(24,30)),
                'tail6.v1 requires the exact six final Critic blocks', path='wan.strategy')
        require(not self.retained_critic_blocks or
                (self.placement == 'resident' and self.candidate_bundle.enabled
                 and (len(self.retained_critic_blocks) <= 4 or self.retention_policy in ('tail8.v1', 'tail6.v1'))),
                'experimental retention requires the incumbent Resident bundle and at most4 blocks', path='wan.strategy')
        require(self.retention_execution in ('manual-only.v1', 'planner-retain4.v1')
                and (self.retention_execution == 'manual-only.v1'
                     or (self.retained_critic_blocks == (26, 27, 28, 29)
                         and self.retention_policy == 'retain4.v1')),
                'planner retention requires explicit versioned retainC26-29 policy', path='wan.strategy')

    @property
    def backend(self):
        return WanImplementation(self.placement, self.chunk_tokens, self.lora_epilogue, candidate_bundle=self.candidate_bundle)

    @property
    def id(self):
        suffix = '-retainC' + '_'.join(map(str,self.retained_critic_blocks)) if self.retained_critic_blocks else ''
        if self.retention_policy != 'retain4.v1': suffix += '-' + self.retention_policy.replace('.', '')
        if self.rollout_graph_pool_policy != 'isolated.v1': suffix += '-rollout-shared-scratch-v1'
        if self.retention_execution != 'manual-only.v1': suffix += '-planner-retain4-v1'
        return f'{self.backend.id}-original-w{self.rollout_window}{suffix}'

    @property
    def assignment(self):
        return {**self.backend.assignment, 'rollout_window': str(self.rollout_window),
                **({'state_policy_sha256': fingerprint(self.compute_policy)} if self.compute_policy else {})}

    @property
    def compute_policy(self):
        """State choices affecting measured compute/workspace, not DAG order.

        Empty preserves the legacy component identity byte-for-byte. Retention
        and CUDA Graph scratch allocation are not captured by backend.id; they
        must not borrow each other's measurements or runtime evidence.
        """
        if (not self.retained_critic_blocks and self.retention_policy == 'retain4.v1'
                and self.rollout_graph_pool_policy == 'isolated.v1'):
            return {}
        return dict(schema_version='distillschedule.wan_compute_state_policy.v1',
                    retained_critic_blocks=list(self.retained_critic_blocks),
                    retention_policy=self.retention_policy,
                    rollout_graph_pool_policy=self.rollout_graph_pool_policy,
                    retention_execution=self.retention_execution)

    @property
    def compute_key(self):
        return (self.backend.id + '-state-' + fingerprint(self.compute_policy)
                if self.compute_policy else self.backend.id)

    def to_dict(self):
        return dict(**({"candidate_bundle": self.candidate_bundle.to_dict()} if self.candidate_bundle.enabled else {}),
                    **({'retention_execution': self.retention_execution} if self.retention_execution != 'manual-only.v1' else {}),
                    **({'retention_policy':self.retention_policy} if self.retention_policy != 'retain4.v1' else {}),
                    **({'rollout_graph_pool_policy':self.rollout_graph_pool_policy} if self.rollout_graph_pool_policy != 'isolated.v1' else {}),
                    **({'retained_critic_blocks': list(self.retained_critic_blocks)} if self.retained_critic_blocks else {}),
                    placement=self.placement,chunk_tokens=self.chunk_tokens,
                    lora_epilogue=self.lora_epilogue,rollout_window=self.rollout_window)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        bundle = WanCandidateBundle.from_dict(value.pop('candidate_bundle')) if 'candidate_bundle' in value else OFF
        retained = value.pop('retained_critic_blocks', [])
        retention_policy = value.pop('retention_policy', 'retain4.v1')
        pool_policy = value.pop('rollout_graph_pool_policy', 'isolated.v1')
        execution = value.pop('retention_execution', 'manual-only.v1')
        require(isinstance(retained,list), 'retained_critic_blocks must be a JSON list', path='wan.strategy')
        require(set(value)=={'placement','chunk_tokens','lora_epilogue','rollout_window'},
                'strategy has unknown or missing fields', path='wan.strategy')
        return cls(**value, candidate_bundle=bundle, retained_critic_blocks=tuple(retained), retention_policy=retention_policy,
                   rollout_graph_pool_policy=pool_policy, retention_execution=execution)

    def validate_for(self, task):
        self.candidate_bundle.validate_task(task)
        require(isinstance(task,OriginalWanTask) and self.rollout_window <= task.fake_updates,
                'window exceeds actual update count', path='wan.strategy')

    def environment(self):
        return {**self.backend.environment(),
            'DISTILLSCHEDULE_WAN13B_GENERATOR_GRAPH_POOL': self.rollout_graph_pool_policy,
            'DISTILLSCHEDULE_WAN13B_RETAIN_CRITIC_BLOCKS': ','.join(map(str,self.retained_critic_blocks)),
            'DISTILLSCHEDULE_SEARCH_CANDIDATE_ID':self.backend.id,
            # Legacy generated-window dispatch is not used. The task-bound v2
            # order is passed directly to the original prepared action handlers.
            'DISTILLSCHEDULE_WAN_ROLLOUT_WINDOW':'',
            'NCCL_ALGO':'Tree' if self.placement=='resident' else '',
            'DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE':str(int(self.lora_epilogue)),
            'DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_CAST_CACHE':str(int(self.candidate_bundle.generator_nograd_lora_cast_cache)),
            'DISTILLSCHEDULE_WAN13B_EMA_IMPLEMENTATION':'reference',
            'DISTILLSCHEDULE_WAN13B_LORA_OPERATOR_ATTRIBUTION':'0',
            'DISTILLSCHEDULE_NSYS_EXACT_RANGE':'0'}


def implementations(task: OriginalWanTask, *, triton_available: bool, incumbent_bundle_available: bool = False,
                    critic_retention_available: bool = False):
    require(type(triton_available) is bool,'explicit capability required',path='wan.strategy')
    result=[WanStrategy('fsdp',512,False)]
    for chunk in (512,1024,2048):
        for fused in ((False,True) if triton_available else (False,)):
            result.append(WanStrategy('resident',chunk,fused))
    require(type(incumbent_bundle_available) is bool, "explicit bundle capability required", path="wan.strategy")
    require(type(critic_retention_available) is bool
            and (not critic_retention_available or incumbent_bundle_available),
            "Critic retention requires explicit incumbent bundle capability", path="wan.strategy")
    if incumbent_bundle_available:
        require(triton_available, "incumbent bundle requires Triton", path="wan.strategy")
        INCUMBENT.validate_task(task)
        result.append(WanStrategy("resident", 2048, True, candidate_bundle=INCUMBENT))
        if critic_retention_available:
            # Only the already-executed bounded four-block mechanism enters
            # inventory. Failed tail6/tail8 and unqualified SAC are not added.
            result.append(WanStrategy("resident", 2048, True, candidate_bundle=INCUMBENT,
                                      retained_critic_blocks=(26, 27, 28, 29), retention_execution='planner-retain4.v1'))
    return tuple(result)


def capability_report(task: OriginalWanTask, *, triton_available: bool, incumbent_bundle_available: bool = False,
                      critic_retention_available: bool = False):
    return {'schema_version':'distillschedule.original_wan_capabilities.v1',
        'task':task.to_dict(),'latent_shape':list(task.latent_shape),
        'implementations':[c.to_dict() for c in implementations(task,triton_available=triton_available,
            incumbent_bundle_available=incumbent_bundle_available, critic_retention_available=critic_retention_available)],
        'opt_in_coupled_bundles':[dict(bundle=INCUMBENT.to_dict(), task_supported=task==OriginalWanTask(),
            admission='EXACT_ORIGINAL_TASK_AND_IMMUTABLE_ASSETS_ONLY', gpu_execution='NOT_RUN_ON_CURRENT_SOURCE',
            timing='HISTORICAL_PROVENANCE_NOT_CURRENT_CALIBRATION', precision='CURRENT_GENERATED_POLICY_NO_DRIVER_OVERRIDE')],
        'rollout_windows':list(range(1,task.fake_updates+1)),
        'backend_routes':{'fsdp':'apply_wan13b_fair_fsdp2_policy + build_fair_engine',
                          'resident':'build_fair_engine / GradReadyAsyncResidentEngine / 128 MiB / NCCL Tree'},
        'kernel_routes':['configure_modulation_chunk_tokens','install_generator_nograd_lora_fused_epilogue'],
        'protocol':'ORIGINAL_WAN_HELIOS_PYRAMID_NOGAN',
        'status':'SOURCE_CAPABILITIES_REQUIRE_MATCHED_CALIBRATION',
        'gpu_execution':'NOT_RUN','claimable_speedup':False,
        **({'opt_in_compute_state_policy': dict(version='planner-retain4.v1',
            retained_critic_blocks=[26,27,28,29], runtime='OriginalWanSession/checkpoint_block_retention.install',
            admission='FRESH_SOURCE_BOUND_CALIBRATION_AND_MATCHED_NUMERICAL_REVIEW_REQUIRED',
            gpu_execution='NOT_RUN_ON_CURRENT_SOURCE')} if critic_retention_available else {}),
        'limitations':['world16/B1','one model family','no new backbone sharing',
                       'no arbitrary pool topology','task-matching native trace/bootstrap required']}


@contextmanager
def strategy_environment(strategy: WanStrategy):
    """Before NCCL/model construction, restored on exit; no package/global patches."""
    values=strategy.environment();old={k:os.environ.get(k) for k in values}
    # An unrelated old search choice must not leak into new parameterized work.
    prior=os.environ.get('DISTILLSCHEDULE_SEARCH_CANDIDATE_ID')
    require(not prior or prior==strategy.backend.id,'conflicting externally selected implementation',path='wan.strategy')
    try:
        for k,v in values.items():
            if v:os.environ[k]=v
            else:os.environ.pop(k,None)
        yield
    finally:
        for k,v in old.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v

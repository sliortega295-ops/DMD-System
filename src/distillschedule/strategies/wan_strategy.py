"""Expose EXISTING Wan placement/reduction/chunk/epilogue paths to new tasks."""
from __future__ import annotations
from dataclasses import dataclass
from contextlib import contextmanager
import os
from typing import Mapping
from distillschedule.core.contracts import require
from distillschedule.search.wan_implementation import WanImplementation, PRECISION
from .wan_task import OriginalWanTask


@dataclass(frozen=True)
class WanStrategy:
    placement: str
    chunk_tokens: int
    lora_epilogue: bool
    rollout_window: int = 1

    def __post_init__(self):
        # Original native admission remains unchanged. In particular, FSDP only
        # supports its control kernel configuration until another arm is validated.
        WanImplementation(self.placement, self.chunk_tokens, self.lora_epilogue)
        require(type(self.rollout_window) is int and 1 <= self.rollout_window <= 64,
                'invalid parameterized rollout window', path='wan.strategy')

    @property
    def backend(self):
        return WanImplementation(self.placement, self.chunk_tokens, self.lora_epilogue)

    @property
    def id(self):
        return f'{self.backend.id}-original-w{self.rollout_window}'

    @property
    def assignment(self):
        return {**self.backend.assignment, 'rollout_window': str(self.rollout_window)}

    def to_dict(self):
        return dict(placement=self.placement,chunk_tokens=self.chunk_tokens,
                    lora_epilogue=self.lora_epilogue,rollout_window=self.rollout_window)

    @classmethod
    def from_dict(cls, value):
        require(set(value)=={'placement','chunk_tokens','lora_epilogue','rollout_window'},
                'strategy has unknown or missing fields', path='wan.strategy')
        return cls(**value)

    def validate_for(self, task):
        require(isinstance(task,OriginalWanTask) and self.rollout_window <= task.fake_updates,
                'window exceeds actual update count', path='wan.strategy')

    def environment(self):
        return {**self.backend.environment(),
            'DISTILLSCHEDULE_SEARCH_CANDIDATE_ID':self.backend.id,
            # Legacy generated-window dispatch is not used. The task-bound v2
            # order is passed directly to the original prepared action handlers.
            'DISTILLSCHEDULE_WAN_ROLLOUT_WINDOW':'',
            'NCCL_ALGO':'Tree' if self.placement=='resident' else '',
            'DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE':str(int(self.lora_epilogue)),
            'DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_CAST_CACHE':'0',
            'DISTILLSCHEDULE_WAN13B_EMA_IMPLEMENTATION':'reference',
            'DISTILLSCHEDULE_WAN13B_LORA_OPERATOR_ATTRIBUTION':'0',
            'DISTILLSCHEDULE_NSYS_EXACT_RANGE':'0'}


def implementations(task: OriginalWanTask, *, triton_available: bool):
    require(type(triton_available) is bool,'explicit capability required',path='wan.strategy')
    result=[WanStrategy('fsdp',512,False)]
    for chunk in (512,1024,2048):
        for fused in ((False,True) if triton_available else (False,)):
            result.append(WanStrategy('resident',chunk,fused))
    return tuple(result)


def capability_report(task: OriginalWanTask, *, triton_available: bool):
    return {'schema_version':'distillschedule.original_wan_capabilities.v1',
        'task':task.to_dict(),'latent_shape':list(task.latent_shape),
        'implementations':[c.to_dict() for c in implementations(task,triton_available=triton_available)],
        'rollout_windows':list(range(1,task.fake_updates+1)),
        'backend_routes':{'fsdp':'apply_wan13b_fair_fsdp2_policy + build_fair_engine',
                          'resident':'build_fair_engine / GradReadyAsyncResidentEngine / 128 MiB / NCCL Tree'},
        'kernel_routes':['configure_modulation_chunk_tokens','install_generator_nograd_lora_fused_epilogue'],
        'protocol':'ORIGINAL_WAN_HELIOS_PYRAMID_NOGAN',
        'status':'SOURCE_CAPABILITIES_REQUIRE_MATCHED_CALIBRATION',
        'gpu_execution':'NOT_RUN','claimable_speedup':False,
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

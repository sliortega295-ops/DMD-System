"""Single original-model bridge, shared by calibration and selected execution.

Importing this module does not import Torch, load weights, launch ranks, or pick
an already-tuned recipe. Tests can inject an existing-helper facade explicitly.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from distillschedule.core.contracts import require
from distillschedule.runtime.wan_cycle_plan import prepare_parameterized_wan_cycle
from .wan_task import OriginalWanTask
from .wan_strategy import WanStrategy


def load_original_backend(root: Path):
    path=root/'scripts/distillschedule/probe_wan13b_fair_world16.py'
    name='_distillschedule_original_wan_backend'
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path)
    require(spec is not None and spec.loader is not None,'missing original Wan backend',path='wan.backend')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    try:spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name,None);raise
    return module


class OriginalWanSession:
    def __init__(self, *, task, strategy, fair, bootstrap, model_root, device, rank,
                 trace, negative, diagnostics, asset_identity=None):
        strategy.validate_for(task)
        require(trace.variant=='no-gan' and tuple(trace.logical_steps)==task.source_steps,
                'backend trace does not match task',path='wan.session')
        require(int(bootstrap.manifest['lora_rank'])==task.lora_rank,
                'bootstrap does not match task rank',path='wan.session')
        if strategy.candidate_bundle.enabled:
            require(asset_identity is not None, 'bundle requires validated semantic asset identities', path='wan.session')
            strategy.candidate_bundle.validate_assets(asset_identity)
        self.task,self.strategy,self.fair=task,strategy,fair
        self.rank,self.device,self.trace=rank,device,trace
        self.negative,self.diagnostics=negative,diagnostics
        args=SimpleNamespace(model_root=Path(model_root),placement=strategy.backend.native_placement,
            original_wan_task=task,timing_mode=True,profile_mode=False,trajectory_mode=False,validate_only=False)
        # The caller installs the strategy environment before NCCL. Verify that
        # metadata and physical construction cannot silently use different arms.
        from distillschedule.search.wan_implementation import current_implementation
        require(current_implementation()==strategy.backend,'strategy environment not applied before construction',path='wan.session')
        self.g,self.c,self.ema,self.build=fair.base._build_engines(args=args,bootstrap=bootstrap,device=device)
        from distillschedule.runtime.fsdp2_p0 import StaticFSDP2P0Policy
        self.policy=StaticFSDP2P0Policy(mesh_policy='world16-fsdp',rollout_batch_size=1,pack_real_score_pair=False)
        self.clean=strategy.placement=='resident'
        self.runtime_cache={} if self.clean else None
        self.finite=[]

    def versions(self):
        return dict(generator=int(self.g.global_steps),critic=int(self.c.global_steps),ema=int(self.ema.optimization_step))

    def prepare(self, *, operation_scope=None, operation_order=None):
        if operation_order is not None:
            self.strategy.candidate_bundle.validate_order(operation_order)
        return prepare_parameterized_wan_cycle(p0=self.fair.base.p0,
            source_steps=self.task.source_steps,ema_after=self.task.ema_after,
            trace_contract=self.task.trace_contract(),max_live_rollouts=self.strategy.rollout_window,
            rank=self.rank,device=self.device,generator_engine=self.g,critic_engine=self.c,ema=self.ema,
            trace=self.trace,negative=self.negative,diagnostics=self.diagnostics,policy=self.policy,
            clean_runtime=self.clean,runtime_cache=self.runtime_cache,deferred_finite_checks=self.finite,
            operation_scope=operation_scope,operation_order=operation_order)

"""Bind parameterized native tensors to the ORIGINAL rank-qualified trace API."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from .wan import OriginalWanTask, TRACE_KEY
from distillschedule.search.contracts import require
from distillschedule.workloads.best_known.registry import fingerprint


def validate_manifest(manifest, task: OriginalWanTask, *, allow_legacy=False):
    contract=manifest.get('contract',{})
    legacy=TRACE_KEY not in contract
    require(not legacy or (allow_legacy and task==OriginalWanTask()),
            'new parameters require a matching native trace; never truncate/resize an old Full5 trace')
    expected=task.trace_contract()
    if legacy: expected.pop(TRACE_KEY)
    require(all(contract.get(k)==v for k,v in expected.items()), 'trace training contract differs from task')
    operations=manifest.get('operations',[])
    by_key={op['key']:op for op in operations}
    require(len(by_key)==len(operations),'duplicate operation keys')
    rollouts={k for k in by_key if k.endswith('/rollout')}
    expected_rollouts={f'step/{s}/critic/rollout' for s in task.source_steps}|{'step/10/generator/rollout'}
    require(rollouts==expected_rollouts,'trace rollout inventory differs from declared update count')
    branch_keys={k for k in by_key if k.endswith('/branch')}
    require(branch_keys=={f'step/{s}/branch' for s in task.source_steps},'trace branch inventory differs')
    require('step/10/generator/dmd' in by_key,'generator DMD operation missing')
    for i,s in enumerate(task.source_steps):
        branch=by_key[f'step/{s}/branch']['control']
        required={'global_step':s,'dfake_gen_update_ratio':task.fake_updates,'train_generator':i==0,
                  'train_critic':True,'gan':False,'reward':False,'vae_decode':False}
        require(all(branch.get(k)==v for k,v in required.items()),'native branch/update rule differs')
        require(f'step/{s}/critic/train' in by_key,'critic train operation missing')
    tensors=manifest.get('tensors',{})
    for key in sorted(rollouts):
        prefix=key.removesuffix('/rollout')
        control=by_key[key]['control']; schedule=control.get('pyramid_stage_call_sequence')
        require(isinstance(schedule,list) and len(schedule)==3,'native pyramid needs exactly three stages')
        for stage,row in enumerate(schedule):
            require(type(row.get('stage')) is int and row['stage']==stage,'stage order changed')
            require(type(row.get('call_count')) is int and 1<=row['call_count']<=2*task.stage2_steps[stage],
                    'stage calls exceed declared native (including amplified) budget')
            require(row.get('denoising_exit')==row['call_count']-1,'stage exit/count mismatch')
            require(tuple(row.get('latent_shape',()))==(*task.latent_shape[:3],*task.stage_shapes[stage]),
                    'stage latent shape differs from task')
        require(control.get('call_count')==sum(x['call_count'] for x in schedule),'total call count mismatch')
        for name in ('entry_noise','rollout_noise','dmd_noise' if '/generator' in prefix else 'critic_noise'):
            descriptor=tensors.get(prefix+'/'+name,{})
            require(tuple(descriptor.get('shape',()))==task.latent_shape and descriptor.get('dtype')=='bfloat16',
                    f'native trace tensor geometry/dtype changed: {prefix}/{name}')
    return {'task_sha256':task.digest,'rollout_operations':sorted(rollouts),'legacy_exact_contract':legacy}


def open_trace(root, task: OriginalWanTask, *, expected_hash: str, allow_legacy=False):
    """Reuse original payload loading and entry identities; no data synthesis."""
    from runtime.static_world16_trace import StaticWorld16Trace
    from runtime.operation_trace import (OperationTrace,load_operation_trace_bundle_manifest,
        resolve_rank_trace_root,validate_official_rank_stage_schedule_semantics)
    class BoundTrace(StaticWorld16Trace):
        def __init__(self):
            self.root=Path(root).resolve(); self.variant='no-gan'; self.gan=False
            self.logical_steps=task.source_steps
            bundle=load_operation_trace_bundle_manifest(self.root,verify=True)
            require(bundle is not None and len(bundle.get('ranks',()))==16,'original Wan requires rank16 trace bundle')
            require(bundle['bundle_hash']==expected_hash,'trace bundle identity differs')
            self.bundle_hash=bundle['bundle_hash'];self._control={};self._loaded={}
            for rank in range(16):
                path=resolve_rank_trace_root(self.root,rank)
                manifest=json.loads((path/'trace.json').read_text())
                validate_manifest(manifest,task,allow_legacy=allow_legacy)
                self._control[rank]=OperationTrace(root=path,manifest=manifest,tensors={})
            self.rank_control_validation=validate_official_rank_stage_schedule_semantics(self._control)
            self.binding_hash=fingerprint({'schema':'distillschedule.original_wan_trace.v1',
                                          'task':task.digest,'bundle_hash':self.bundle_hash})
            self.task=task
    return BoundTrace()

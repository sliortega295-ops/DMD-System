"""Task-derived checks using the original rollout census and backend receipts."""
from __future__ import annotations
from distillschedule.core.contracts import require


def validate_build(strategy, build, *, executed=False, cycles=0):
    from distillschedule.runtime.wan_incumbent_evidence import validate_bundle_build
    bundle_effects = validate_bundle_build(strategy.candidate_bundle, build, executed=executed, cycles=cycles)
    expected_sync = ('WORLD16_GRAD_READY_ASYNC_SUM_THEN_DIVIDE_ALL_REDUCE'
                     if strategy.placement == 'resident' else 'FSDP2_SUM_THEN_DIVIDE_REDUCE_SCATTER')
    for role in ('generator','critic'):
        value = build[role]
        require(value['engine']['gradient_sync'] == expected_sync,
                'selected communication backend not applied', path='wan.actual')
        placement = value['placement']
        require(placement['param_compute_dtype_policy'] == 'PRESERVE_ORIGINAL_PER_PARAMETER'
                and placement['trainable_parameter_dtypes'] == ['torch.float32']
                and 'torch.bfloat16' in placement['frozen_parameter_dtypes'],
                'native precision changed', path='wan.actual')
        chunks = value['metadata']['modulation_chunk_runtime']['effective_tokens']
        require(chunks == [strategy.chunk_tokens], 'selected block chunk not applied', path='wan.actual')
        fusion = value['metadata']['generator_nograd_lora_fused_epilogue']
        require(fusion['requested'] == strategy.lora_epilogue and
                fusion['enabled'] == (strategy.lora_epilogue and role == 'generator'),
                'selected epilogue not installed', path='wan.actual')
        if executed and strategy.lora_epilogue and role == 'generator':
            require(fusion.get('fused_invocation_count',0)>0 and fusion.get('fused_element_count',0)>0,
                    'requested fusion executed no elements', path='wan.actual')
    return {'status':'PASS_BACKEND_EFFECTS','gradient_sync':expected_sync,
            'chunk_tokens':strategy.chunk_tokens,'lora_epilogue':strategy.lora_epilogue,
            'actual_compute_observed':executed, 'candidate_bundle_effects': bundle_effects}


def validate_cycle(task, trace, result):
    from distillschedule.runtime.fsdp2_p0 import expected_trace_rollout_call_inventory,validate_exact_rollout_call_inventory
    g = result['generator'];rollouts = result['rollouts'];critics = result['critic']
    require(len(rollouts)==len(critics)==task.fake_updates
            and all(int(r.get('batch_size',1))==1 for r in rollouts),
            'original rollout/update inventory changed',path='wan.census')
    require([r['logical_step'] for r in critics] == list(task.source_steps)
            and all(r['score_forwards']==1 for r in critics) and g['physical_score_forwards']==3,
            'score forward or source-step count changed',path='wan.census')
    keys = [f'step/{task.first_step}/generator/rollout',*(f'step/{s}/critic/rollout' for s in task.source_steps)]
    normalized=[]
    for key,row in zip(keys,[g['rollout'],*rollouts],strict=True):
        actual=validate_exact_rollout_call_inventory(row,label=key)
        expected=expected_trace_rollout_call_inventory(trace.rank_control_validation,
                                                        operation=key,expected_rank_count=16)
        require(actual==expected,'executed sampler differs from current trace',path='wan.census')
        normalized.append(actual)
    return {'status':'PASS_TRACE_DERIVED_CALL_CENSUS','operations':keys,'inventories':normalized,
            'g_updates':1,'critic_updates':task.fake_updates,'ema_updates':len(task.ema_after)}


def compare_runs(reference_root,candidate_root,*,rtol=1e-5,atol=1e-7):
    """Compare matched full trainable/Adam/EMA evidence, retaining REVIEW."""
    import json
    import math
    from pathlib import Path
    import torch
    from safetensors import safe_open
    from distillschedule.workloads.best_known.registry import file_sha256
    require(all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in (rtol,atol)),
            'invalid comparison tolerance',path='wan.compare')
    left=[json.loads((Path(reference_root)/f'rank_{r:02d}.json').read_text()) for r in range(16)]
    right=[json.loads((Path(candidate_root)/f'rank_{r:02d}.json').read_text()) for r in range(16)]
    for rank,(a,b) in enumerate(zip(left,right,strict=True)):
        require(a['rank']==b['rank']==rank and a['status']==b['status']=='COMPLETED_ORIGINAL_WAN_WORK'
                and a['identity']==b['identity'] and a['task']==b['task']
                and a['initial_versions']==b['initial_versions'] and a['final_versions']==b['final_versions'],
                'cannot compare unmatched or incomplete native executions',path='wan.compare')
        require(a.get('stream_sha256') == b.get('stream_sha256')
                and a.get('measurement_scope') == b.get('measurement_scope'),
                'cannot compare different cohort streams or execution scopes', path='wan.compare')
        if a.get('stream_sha256') is not None:
            require(all(x.get('cohort') == y.get('cohort') for x,y in zip(a['cycles'],b['cycles'])),
                    'cohort identities or absolute source steps differ', path='wan.compare')
        require(len(a['cycles'])==len(b['cycles']) and all(
            x['call_census']==y['call_census'] for x,y in zip(a['cycles'],b['cycles'],strict=True)),
            'compared executions changed logical model work',path='wan.compare')
    rows=[]
    for name in ('generator','critic','generator_optimizer','critic_optimizer','ema'):
        a=left[0]['state_evidence'][name];b=right[0]['state_evidence'][name]
        for value in (a,b):
            require(file_sha256(Path(value['path']))==value['sha256'],'state artifact changed',path='wan.compare')
        with safe_open(a['path'],framework='pt',device='cpu') as x, safe_open(b['path'],framework='pt',device='cpu') as y:
            require(set(x.keys())==set(y.keys()),'state tensor inventories differ',path='wan.compare')
            for key in x.keys():
                v,w=x.get_tensor(key),y.get_tensor(key)
                require(v.shape==w.shape and v.dtype==w.dtype,'state shape/dtype changed',path='wan.compare')
                finite=bool(torch.isfinite(v).all() and torch.isfinite(w).all())
                close=finite and torch.allclose(v,w,rtol=rtol,atol=atol)
                difference=(v.float()-w.float()).abs()
                rows.append({'state':name,'tensor':key,'pass':bool(close),'finite':finite,
                             'max_abs':float(difference.max()) if finite else None,
                             'mean_abs':float(difference.mean()) if finite else None})
    require(rows,'no numerical state evidence',path='wan.compare')
    return {'status':'PASS' if all(r['pass'] for r in rows) else 'REVIEW',
            'rtol':rtol,'atol':atol,'tensor_count':len(rows),'failed_tensors':[r for r in rows if not r['pass']],
            'max_abs':max((r['max_abs'] for r in rows if r['max_abs'] is not None),default=None),
            'scope':('FULL_TRAINABLE_ADAM_EMA_AFTER_MATCHED_FRESH_COHORTS' if left[0].get('stream_sha256')
                     else 'FULL_TRAINABLE_ADAM_EMA_AFTER_MATCHED_FIXED_TRACE'), 'quality_validation':'NOT_RUN',
            'claimable_speedup':False}

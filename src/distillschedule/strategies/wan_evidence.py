"""Task-derived checks using the original rollout census and backend receipts."""
from __future__ import annotations
from distillschedule.core.contracts import require


def validate_build(strategy, build, *, executed=False):
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
            'actual_compute_observed':executed}


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

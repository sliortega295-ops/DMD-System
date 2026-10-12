"""Original helper dispatch tests; no GPU or original-model parity claim."""
from types import SimpleNamespace as NS
import pytest
from distillschedule.runtime.wan_cycle_plan import prepare_parameterized_wan_cycle, prepare_wan_cycle


def fixture(n=3, window=1, ema_after=None):
    steps = tuple(range(10, 10+n))
    ema_after = tuple(range(n)) if ema_after is None else ema_after
    g, c, e = NS(global_steps=2), NS(global_steps=10), NS(optimization_step=10)
    calls = []
    def gupdate(**kwargs):
        calls.append(('g', g.global_steps, c.global_steps))
        g.global_steps += 1
        return {'physical_score_forwards': 3}
    def rollout(**kwargs):
        assert len(kwargs['steps']) == 1
        step = kwargs['steps'][0]
        calls.append(('r', step, g.global_steps))
        kwargs['entry_outputs'][step] = NS(tensors={'source': step})
        return {step: ('latent', step, g.global_steps)}, [{'batch_size': 1}], {}
    def update(**kwargs):
        step = kwargs['step']
        assert kwargs['entry'].tensors['source'] == step
        assert kwargs['x0'] == ('latent', step, 3)
        calls.append(('c', step, c.global_steps))
        c.global_steps += 1
        return {'logical_step': step, 'score_forwards': 1}
    def ema():
        calls.append(('e', e.optimization_step))
        e.optimization_step += 1
    e.step = ema
    trace = NS(logical_steps=steps, variant='no-gan')
    contract = dict(schema_version='distillschedule.wan_prepared_trace_contract.v1',
                    protocol='wan_native_nogan_post_g_independent_rollouts',
                    source_steps=list(steps), generator_source_step=10, ema_after=list(ema_after),
                    physical_batch=1, world_size=16, input_binding='validated_operation_qualified_trace')
    kwargs = dict(p0=NS(_run_nogan_generator_update_p0=gupdate, _precompute_critic_rollouts=rollout,
                       _run_nogan_critic_precomputed=update), source_steps=steps, max_live_rollouts=window,
                  rank=0, device='CPU_HANDLER_FIXTURE', generator_engine=g, critic_engine=c,
                  ema=e, trace=trace, negative=object(), diagnostics=object(),
                  policy=NS(rollout_batch_size=1, generator_nograd_persistent_unshard=False),
                  clean_runtime=True, runtime_cache={}, deferred_finite_checks=[])
    return kwargs, contract, calls, ema_after


@pytest.mark.parametrize('n', [1, 2, 3, 5, 8])
@pytest.mark.parametrize('window_mode', ['one', 'all'])
@pytest.mark.parametrize('ema_mode', ['all', 'none', 'last'])
def test_original_helper_binding_variable_updates(n, window_mode, ema_mode):
    window = 1 if window_mode == 'one' else n
    ema = tuple(range(n)) if ema_mode == 'all' else (() if ema_mode == 'none' else (n-1,))
    kwargs, contract, calls, ema = fixture(n, window, ema)
    program = prepare_parameterized_wan_cycle(**kwargs, trace_contract=contract, ema_after=ema)
    assert calls == []  # no model execution during preparation
    result = program.run()
    assert len(result['critic']) == len(result['rollouts']) == n
    assert [r['logical_step'] for r in result['critic']] == list(range(10, 10+n))
    assert [r['batch_index'] for r in result['rollouts']] == list(range(n))
    assert next(i for i, row in enumerate(calls) if row[0] == 'c') == window+1
    assert sum(row[0] == 'e' for row in calls) == len(ema)
    assert program.receipt()['native_versions'] == dict(generator=3, critic=10+n, ema=10+len(ema))
    assert program.receipt()['status'] == 'PASS'


@pytest.mark.parametrize('window', [1, 3, 5])
def test_full5_old_and_new_dispatch_identical(window):
    old_kw, _, old_calls, _ = fixture(5, window)
    new_kw, contract, new_calls, ema = fixture(5, window)
    old = prepare_wan_cycle(**old_kw)
    new = prepare_parameterized_wan_cycle(**new_kw, trace_contract=contract, ema_after=ema)
    assert old.run() == new.run()
    assert old_calls == new_calls
    assert old.receipt() == new.receipt()


@pytest.mark.parametrize('field,value', [
    ('schema_version', 'unknown'), ('protocol', 'euler'), ('world_size', 8),
    ('world_size', True), ('physical_batch', 2), ('physical_batch', True),
    ('generator_source_step', 11), ('input_binding', 'same_global_seed'),
    ('source_steps', [10]), ('ema_after', []), ('extra_ignored_flag', True),
])
def test_bad_contract_rejected_before_any_native_work(field, value):
    kwargs, contract, calls, ema = fixture()
    contract[field] = value
    with pytest.raises(ValueError):
        prepare_parameterized_wan_cycle(**kwargs, trace_contract=contract, ema_after=ema)
    assert calls == []


@pytest.mark.parametrize('kind', ['truncated_trace', 'gan', 'persistent', 'batched', 'window', 'rank'])
def test_unadapted_trace_or_backend_is_not_silently_admitted(kind):
    kwargs, contract, calls, ema = fixture()
    if kind == 'truncated_trace': kwargs['trace'].logical_steps = tuple(range(10,15))
    if kind == 'gan': kwargs['trace'].variant = 'gan'
    if kind == 'persistent': kwargs['policy'].generator_nograd_persistent_unshard = True
    if kind == 'batched': kwargs['policy'].rollout_batch_size = 2
    if kind == 'window': kwargs['max_live_rollouts'] = 4
    if kind == 'rank': kwargs['rank'] = 16
    with pytest.raises(ValueError):
        prepare_parameterized_wan_cycle(**kwargs, trace_contract=contract, ema_after=ema)
    assert calls == []


def test_native_commit_mismatch_still_fails():
    kwargs, contract, calls, ema = fixture()
    # A handler returning successfully without a real optimizer commit must fail.
    kwargs['p0']._run_nogan_critic_precomputed = lambda **kw: {'logical_step': kw['step']}
    program = prepare_parameterized_wan_cycle(**kwargs, trace_contract=contract, ema_after=ema)
    with pytest.raises(ValueError, match='native engine commits'):
        program.run()
    assert program.receipt()['status'] == 'FAIL'


def test_legacy_api_still_rejects_non_full5():
    kwargs, _, calls, _ = fixture(1)
    with pytest.raises(ValueError, match='exactly five'):
        prepare_wan_cycle(**kwargs)
    assert calls == []

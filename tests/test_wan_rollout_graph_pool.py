"""CPU policy/source tests; fake CUDA API is SYNTHETIC_TEST_ONLY, not qualification."""
from types import SimpleNamespace
import os
from pathlib import Path

import pytest

from distillschedule.runtime.wan13b_rollout_graph_pool import (
    ENV, ISOLATED, SHARED, IndependentRolloutPool, requested_policy,
)
from distillschedule.search.wan_candidate_bundle import INCUMBENT
from distillschedule.strategies.wan_strategy import WanStrategy, strategy_environment


def fake_torch():
    state = dict(grad=False, capturing=False, stream=1, stream_creations=0)
    def new_stream():
        state['stream_creations'] += 1
        return object()
    cuda = SimpleNamespace(
        current_stream=lambda: SimpleNamespace(device='cuda:0', cuda_stream=state['stream']),
        is_current_stream_capturing=lambda: state['capturing'],
        graph_pool_handle=lambda: ('SYNTHETIC_TEST_ONLY', 7), Stream=new_stream)
    return SimpleNamespace(cuda=cuda, is_grad_enabled=lambda: state['grad']), state


def strategy(**kwargs):
    return WanStrategy('resident', 2048, True, candidate_bundle=INCUMBENT,
                       retained_critic_blocks=(26,27,28,29), **kwargs)


def test_default_unchanged_no_cuda_api(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    p = IndependentRolloutPool()
    assert p.policy == ISOLATED
    p.before_call(object())
    assert p.capture_options(object()) == {}
    p.verify_capture(object())
    assert p.receipt['verified_capture_count'] == 0
    assert 'rollout_graph_pool_policy' not in strategy().to_dict()


@pytest.mark.parametrize('value', ['', '1', 'shared', 'critic-shared'])
def test_unknown_policy_refused(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    with pytest.raises(ValueError): requested_policy()


def test_one_pool_one_capture_stream_actual_verification(monkeypatch):
    monkeypatch.setenv(ENV, SHARED)
    p = IndependentRolloutPool()
    t, state = fake_torch()
    with pytest.raises(ValueError): p.capture_options(t)
    p.before_call(t)
    a = p.capture_options(t)
    for _ in range(3):
        p.before_call(t)
        b = p.capture_options(t)
        assert a['stream'] is b['stream'] and a['pool'] == b['pool']
        p.verify_capture(SimpleNamespace(pool=lambda: a['pool']))
    assert state['stream_creations'] == 1
    assert p.receipt['verified_capture_count'] == 3
    with pytest.raises(ValueError): p.verify_capture(SimpleNamespace(pool=lambda: 'WRONG'))


@pytest.mark.parametrize('field,value', [('grad', True), ('capturing', True), ('stream', 2)])
def test_unsafe_replay_refused(monkeypatch, field, value):
    monkeypatch.setenv(ENV, SHARED)
    p = IndependentRolloutPool()
    t, state = fake_torch()
    p.before_call(t)
    state[field] = value
    with pytest.raises(ValueError): p.before_call(t)


def test_changed_host_thread_refused(monkeypatch):
    monkeypatch.setenv(ENV, SHARED)
    p = IndependentRolloutPool()
    t, _ = fake_torch()
    p.before_call(t)
    monkeypatch.setattr('distillschedule.runtime.wan13b_rollout_graph_pool.threading.get_ident', lambda: -1)
    with pytest.raises(ValueError): p.before_call(t)


def test_typed_optin_and_environment_restoration(monkeypatch):
    child = strategy(rollout_graph_pool_policy=SHARED)
    assert WanStrategy.from_dict(child.to_dict()) == child
    assert child.id != strategy().id
    monkeypatch.delenv(ENV, raising=False)
    with strategy_environment(child): assert os.environ[ENV] == SHARED
    assert ENV not in os.environ
    with strategy_environment(strategy()): assert os.environ[ENV] == ISOLATED
    assert ENV not in os.environ
    assert child.backend == strategy().backend
    assert child.retained_critic_blocks == strategy().retained_critic_blocks


@pytest.mark.parametrize('kwargs', [dict(rollout_graph_pool_policy='bad'),
    dict(rollout_graph_pool_policy=SHARED, retained_critic_blocks=()),
    dict(rollout_graph_pool_policy=SHARED, retained_critic_blocks=tuple(range(24,30)), retention_policy='tail6.v1')])
def test_candidate_cannot_retry_old_oom_or_run_unbound(kwargs):
    defaults=dict(placement='resident', chunk_tokens=2048, lora_epilogue=True,
                  candidate_bundle=INCUMBENT, retained_critic_blocks=(26,27,28,29))
    with pytest.raises(Exception): WanStrategy(**{**defaults, **kwargs})


def test_source_keeps_output_clones_and_gradient_pools_separate():
    root=Path(__file__).resolve().parents[1]/'src/distillschedule/runtime'
    generator=(root/'wan13b_rollout_cuda_graph.py').read_text()
    critic=(root/'wan13b_critic_grad_graph.py').read_text()
    assert 'return clone_tree(output)' in generator
    assert 'self.graph_pool.before_call(torch)' in generator
    assert 'self.graph_pool.verify_capture(graph)' in generator
    assert 'IndependentRolloutPool' not in critic
    assert 'pool=' not in critic


def test_selected_policy_requires_real_pool_receipt_and_capture_census(monkeypatch):
    from distillschedule.strategies.wan_evidence import validate_build
    from test_strategy_wan_effects import build
    # Synthetic other-module receipts only; exercise the actual new gate.
    monkeypatch.setattr('distillschedule.runtime.wan_incumbent_evidence.validate_bundle_build',
                        lambda *a, **k: {'SYNTHETIC_TEST_ONLY': True})
    child = strategy(rollout_graph_pool_policy=SHARED)
    receipt = build(child)
    receipt['generator']['metadata']['checkpoint_block_retention'] = dict(enabled=False)
    receipt['critic']['metadata']['checkpoint_block_retention'] = dict(
        enabled=True, block_indices=[26,27,28,29], weights_merged=False,
        parameter_dtype_changed=False, retained_forward_calls=16, checkpoint_forward_calls=120,
        block_forward_counts={str(i):4 for i in (26,27,28,29)})
    with pytest.raises(ValueError, match='pool differs'):
        validate_build(child, receipt, executed=True, cycles=2)
    pool = IndependentRolloutPool()
    monkeypatch.setenv(ENV, SHARED)
    pool = IndependentRolloutPool()
    t, _ = fake_torch()
    pool.before_call(t)
    opts = pool.capture_options(t)
    graph = dict(cache_entry_count=3, scratch_pool=pool.receipt)
    receipt['generator']['metadata']['generator_rollout_cuda_graph'] = graph
    with pytest.raises(ValueError, match='verified captures'):
        validate_build(child, receipt, executed=True, cycles=2)
    for _ in range(3): pool.verify_capture(SimpleNamespace(pool=lambda: opts['pool']))
    assert validate_build(child, receipt, executed=True, cycles=2)['status'] == 'PASS_BACKEND_EFFECTS'
    pool.receipt['critic_grad_pool_shared'] = True
    with pytest.raises(ValueError, match='leaked'):
        validate_build(child, receipt, executed=True, cycles=2)

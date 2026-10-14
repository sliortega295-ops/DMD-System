import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from distillschedule.runtime.wan13b_parameter_grid import (
    WanParameterGridSpec, WanGridDistributedExecution, enumerate_main_grid,
)
from distillschedule.runtime.wan13b_execution import admit_execution, WanDistributedExecution


def test_main_grid_has_18_distinct_workload_fingerprints():
    grid = enumerate_main_grid()
    assert len(grid) == 18
    assert len({p.receipt()['fingerprint'] for p in grid}) == 18
    assert all(p.receipt()['global_batch'] == 16 for p in grid)


@pytest.mark.parametrize('ratio', [1, 3, 5])
def test_native_modulo_interval_starts_with_one_generator(ratio):
    p = WanParameterGridSpec(384, 640, ratio, (2, 2, 2))
    assert len(p.logical_steps) == ratio
    assert sum(step % ratio == 0 for step in p.logical_steps) == 1
    assert p.receipt()['critic_updates'] == p.receipt()['ema_updates'] == ratio
    e = WanGridDistributedExecution(fake_updates=ratio)
    assert admit_execution(e) is e
    assert e.sample_count_per_cycle == 16 * (1 + ratio)
    assert e.receipt()['logical_updates'] == f'G1_C{ratio}_EMA{ratio}'
    assert e.gradient_divisor == 16


def test_canonical_execution_receipt_unchanged():
    e = WanDistributedExecution()
    assert e.receipt() == dict(world_size=16, batch_per_rank=1, experiment='canonical_world16',
                             global_batch=16, gradient_divisor=16.0,
                             sample_count_per_cycle=96, logical_updates='G1_C5_EMA5',
                             rollout_batch_size=1, score_packing=False)


@pytest.mark.parametrize('kw', [dict(height=480), dict(fake_updates=True),
                                dict(fake_updates=2), dict(start_step=10),
                                dict(pyramid_budget=(8, 8, 8)), dict(pyramid_budget=(True, True, True))])
def test_unregistered_or_old_trace_bindings_rejected(kw):
    args = dict(height=384, width=640, fake_updates=3, pyramid_budget=(2, 2, 2))
    with pytest.raises((ValueError, TypeError)):
        WanParameterGridSpec(**dict(args, **kw))


def grid_worker():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('tested_native_grid_worker', root / 'scripts/distillschedule/probe_wan13b_parameter_grid_c2.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('ratio', [1, 3, 5])
def test_driver_runs_each_fake_update_and_ema_independently(ratio):
    """CPU driver integration only, not native model/GPU equivalence proof."""
    import torch
    module = grid_worker()
    spec = WanParameterGridSpec(384, 640, ratio, (2, 2, 2))
    calls = []
    parameter = torch.nn.Parameter(torch.tensor(0.25))
    optimizer = torch.optim.SGD([parameter], lr=0.1)

    def generator_update(**kw):
        calls.append('G')
        return {'physical_score_forwards': 3, 'rollout': {'count': 1}}

    def prepare(**kw):
        assert kw['steps'] == spec.logical_steps
        calls.append('rollouts')
        kw['entry_outputs'].update({s: {'step': s} for s in spec.logical_steps})
        return {s: torch.tensor(float(s)) for s in spec.logical_steps}, [{} for s in spec.logical_steps]

    def critic_update(**kw):
        calls.append(('C', kw['step']))
        assert kw['entry']['step'] == kw['step']
        optimizer.zero_grad(); (parameter.square() + kw['x0']).backward(); optimizer.step()
        return {'logical_step': kw['step'], 'score_forwards': 1}

    ema = SimpleNamespace(step=lambda: calls.append('EMA'))
    p0 = SimpleNamespace(_run_nogan_generator_update_p0=generator_update,
                         _precompute_critic_rollouts=prepare,
                         _run_nogan_critic_precomputed=critic_update)
    module.execute_cycle(SimpleNamespace(p0=p0), spec, rank=0, device='cpu',
                         generator=None, critic=None, ema=ema, trace=None,
                         negative=None, diagnostics=None, policy=None, clean=True,
                         cache={}, finite=[])
    expected = ['G', 'rollouts']
    for s in spec.logical_steps:expected += [('C', s), 'EMA']
    assert calls == expected
    assert parameter.item() == pytest.approx(0.25 * 0.8 ** ratio)


@pytest.mark.parametrize('ratio', [1, 3, 5])
def test_critic_graph_validator_uses_real_point_update_count(monkeypatch, ratio):
    from distillschedule.runtime.wan13b_critic_grad_graph import validate_build
    monkeypatch.setenv('DISTILLSCHEDULE_WAN13B_CRITIC_GRAD_CUDA_GRAPH', '1')
    c = dict(enabled=True, replay_calls=2*ratio, backward_replay_calls=2*ratio,
             reference_calls=0, capture_entries=1,
             cold_original_forward_calls=4, cold_original_backward_calls=4)
    b = dict(generator=dict(metadata={}), critic=dict(metadata=dict(critic_grad_cuda_graph=c)))
    validate_build(b, cycles=2, reference_checks=False, updates_per_cycle=ratio)
    c['replay_calls'] += 1
    with pytest.raises(AssertionError):
        validate_build(b, cycles=2, reference_checks=False, updates_per_cycle=ratio)

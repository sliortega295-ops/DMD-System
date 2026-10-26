"""Actual native Fake-epsilon activation retention through the common Program.

These are random tiny upstream UNets, not pretrained quality or GPU speed.
"""
import json
from pathlib import Path

import pytest
import torch

from distillschedule.open_workload.checkpoint_policy import native_effect_evidence
from distillschedule.open_workload.torch_backend import Session, candidates
from distillschedule.training.checkpoint import cpu_tree, rng_state
from distillschedule.training.dmd2 import build_dmd2
from test_training_dmd2_cpu import model_and_feeds, source_root
from test_training_dmd2_owner_execution import native_assets, physical, next_cycle
from test_training_end_to_end import equal, run_cli


def component_config(native_assets):
    path = Path(native_assets['config'])
    value = json.loads(path.read_text())
    value['model']['parameters']['native_checkpoint_policy'] = 'component-v1'
    value['model']['parameters']['settings']['gradient_checkpointing'] = True
    path.write_text(json.dumps(value))
    return path


def retention_option():
    return dict(backend='torch-callable-resident-v1', share_immutable=False, checkpoint_bindings=[],
        owner_execution={'native_cycle': dict(generator_checkpointing=True, fake_checkpointing=False)})


def test_actual_component_policy_changes_only_fake_replay_and_restores(native_assets):
    component_config(native_assets)
    _, components, feeds = model_and_feeds(native_assets)
    owner = components.state['native_cycle']
    flags, outer = physical(owner), rng_state()
    descriptor = owner.execution_options()
    assert descriptor['baseline'] == dict(generator_checkpointing=True, fake_checkpointing=True)
    assert descriptor['effect']['kind'] == 'native_component_checkpoint'
    work = build_dmd2(components, feeds, {}, 0, 17)
    assert candidates(work)['candidates'][1] == retention_option()
    reference = Session(work, candidates(work)['candidates'][0], persistent=True)
    selected = Session(work, retention_option(), persistent=True)
    for cycle in range(3):
        observed, expected = [], []
        reference.state['native_cycle'].observer = lambda event, value: expected.append((event, cpu_tree(value)))
        selected.state['native_cycle'].observer = lambda event, value: observed.append((event, cpu_tree(value)))
        a, b = reference.run(), selected.run()
        assert equal(a, b) and equal(observed, expected)
        assert equal(reference.state['native_cycle'].state_dict(), selected.state['native_cycle'].state_dict())
        assert native_effect_evidence(retention_option(), selected.effects)['active']
        counts = selected.effects['owner_execution']['native_cycle']['components']
        assert counts['generator']['checkpoint_requests'] > 0
        assert counts['generator']['recompute_calls'] > 0
        assert counts['fake']['checkpoint_requests'] == counts['fake']['recompute_calls'] == 0
        assert counts['fake']['forward_calls'] == counts['fake']['retained_calls'] == 2
        assert counts['fake']['disabled_verified']
        assert physical(selected.state['native_cycle']) == flags
        assert not selected.state['native_cycle'].model.guidance_model.fake_unet._forward_pre_hooks
        if cycle < 2:
            for session in (reference, selected):
                session.bind_cycle(next_cycle(work, session, cycle + 1))
    assert equal(outer, rng_state()) and physical(owner) == flags


def test_component_policy_exception_clears_hooks_and_original_flags(native_assets):
    component_config(native_assets)
    _, components, feeds = model_and_feeds(native_assets)
    owner = components.state['native_cycle']
    before = physical(owner)
    session = Session(build_dmd2(components, feeds, {}, 0, 17), retention_option())
    def fail(event, value):
        if event == 'guidance_gradients':
            raise RuntimeError('test failure after retained epsilon forward')
    session.state['native_cycle'].observer = fail
    with pytest.raises(RuntimeError, match='retained epsilon'):
        session.run()
    active = session.state['native_cycle']
    assert physical(active) == before
    assert not active.model.guidance_model.fake_unet._forward_pre_hooks


def test_component_public_manual_checkpoint_fresh_process_resume(native_assets, tmp_path):
    path = component_config(native_assets)
    value = json.loads(path.read_text())
    value['execution'] = dict(adapter='program', mode='manual', implementation=retention_option())
    path.write_text(json.dumps(value))
    first = run_cli('train', path, '--cycles', 1, '--output', tmp_path / 'first')
    resumed = run_cli('train', path, '--cycles', 3, '--resume', first['checkpoint'], '--output', tmp_path / 'resume')
    full = run_cli('train', path, '--cycles', 3, '--output', tmp_path / 'full')
    assert equal(torch.load(Path(resumed['checkpoint']) / 'state.pt', weights_only=True),
                 torch.load(Path(full['checkpoint']) / 'state.pt', weights_only=True))
    metrics = [json.loads(row) for row in (tmp_path / 'resume/metrics.jsonl').read_text().splitlines()]
    assert all(native_effect_evidence(retention_option(), row['execution']['effects'])['active'] for row in metrics)
    assert all(row['finite'] for row in metrics)


@pytest.mark.parametrize('policy', ['unknown', True, None, {'fake': False}])
def test_unknown_component_policy_rejected_before_native_construction(native_assets, policy):
    from distillschedule.training.sd import _arguments
    value = json.loads(Path(native_assets['config']).read_text())['model']['parameters']
    value['native_checkpoint_policy'] = policy
    with pytest.raises(ValueError, match='checkpoint policy'):
        _arguments(value, sdxl='sdxl' in native_assets['assets'])

"""Owned flow generation and local-shard closure, CPU evidence only.

Matrix fixtures test routing/lifecycle, not native model quality. Optional native
cases reuse the existing pinned tiny FLUX/LTX classes without changing admission.
"""
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distillschedule.training import dmd
from distillschedule.training.checkpoint import pack_state, rng_state
from test_training_end_to_end import equal
from test_training_flow_model_methods import native_config, _native_owned
from test_open_workload_parameter_shadows import CPU_FSDP, implementation, flow_workload, flow_settings, matrix_feeds
from test_open_workload_shard_boundaries import _components, _init


class ImageProbe(dmd.FlowDMDMethods, torch.nn.Module):
    generation_sigmas = (1., .5, 0.)
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4))
        self.entries = 0

    def validate_feeds(self, feeds):
        assert feeds['latent_shape'].tolist() == [1, 1, 2, 2]

    def rollout(self, feeds, schedule, gradient_policy):
        self.validate_feeds(feeds)
        assert gradient_policy == 'all_steps'
        value = torch.randn(1, 4)
        for start, stop in zip(schedule, schedule[1:]):
            value = value + (stop - start) * value * self.weight
        return {'image': value}

    def dmd_forward(self, *args, **kwargs):
        self.entries += 1
        return super().dmd_forward(*args, **kwargs)


def test_image_generation_uses_current_owned_entry_and_preserves_reference_rng():
    from distillschedule.training.flow_image import generate
    role = ImageProbe()
    state = {'student': role, 'fake': ImageProbe()}
    feeds = {'latent_shape': torch.tensor([1, 1, 2, 2])}
    api = SimpleNamespace(unpack=lambda tokens, shape: tokens.reshape(shape))
    torch.manual_seed(202)
    before = rng_state()
    expected = role.rollout(feeds, role.generation_sigmas, 'all_steps')['image'].reshape(1, 1, 2, 2)
    after = rng_state()
    torch.set_rng_state(before['torch'])
    actual = generate(api, state, feeds, {})['image_latents']
    assert torch.equal(actual, expected) and equal(after, rng_state())
    assert role.entries == 1 and state['fake'].entries == 0
    assert role.weight.grad is None and role.weight.requires_grad
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('fault', ['input', 'schedule'])
def test_pre_forward_input_refusal_keeps_owner_rng_and_reuse(fault):
    role = ImageProbe()
    feeds = {'latent_shape': torch.tensor([1, 1, 2, 2])}
    bad = {'latent_shape': torch.tensor([1, 2, 2, 2])} if fault == 'input' else feeds
    schedule = (1., .5, 0.) if fault == 'input' else (1., .5, .6, 0.)
    before = rng_state()
    with pytest.raises((AssertionError, ValueError)):
        dmd.generation_rollout(role, bad, {'rollout_sigmas': schedule})
    assert equal(before, rng_state()) and role.entries == 0
    assert not getattr(role, '_dmd_generation_failed', False)
    role.validate_checkpoint_boundary()
    dmd.generation_rollout(role, feeds, {'rollout_sigmas': (1., .5, 0.)})
    assert role.entries == 1


def test_failed_forward_preserves_original_error_and_rejects_future_math(monkeypatch):
    import distillschedule.open_workload.sharded as sharded
    role = ImageProbe()
    feeds = {'latent_shape': torch.tensor([1, 1, 2, 2])}
    def fail(*args): raise RuntimeError('original model forward error')
    def cleanup(*args): raise OSError('best effort cleanup failed')
    role.rollout = fail
    monkeypatch.setattr(sharded, 'close_sharded_forward', cleanup)
    with pytest.raises(RuntimeError, match='original model forward error'):
        dmd.generation_rollout(role, feeds, {'rollout_sigmas': (1., .5, 0.)})
    before = rng_state()
    with pytest.raises(ValueError, match='failed flow generation owner'):
        dmd.generation_rollout(role, feeds, {'rollout_sigmas': (1., .5, 0.)})
    with pytest.raises(ValueError, match='failed flow generation owner'):
        role.dmd_forward({'feeds': feeds}, flow_role='student', phase='student', cycle=0,
            fake_index=0, flow_policy={'rollout_sigmas': (1., .5, 0.), 'gradient_policy': 'all_steps'})
    assert equal(before, rng_state()) and role.weight.grad is None


def _matrix_generate(state, feeds, parameters):
    sample = dmd.generation_rollout(state['student'], feeds, {'rollout_sigmas': (1., .5, 0.)})['payload']
    return {'payload': sample, 'latent_schema': {'payload': {
        'layout': 'BC', 'shape': list(sample.shape), 'dtype': str(sample.dtype)}}}


def _model(native, rank):
    from distillschedule.training.adapters import ModelComponents
    if native is None:
        model, feeds = _components(), matrix_feeds(rank)
        model.generate = _matrix_generate
        settings = flow_settings()
    else:
        owner, feeds = _native_owned(native)
        model = ModelComponents(state=owner.state, immutable_modules=owner.immutable_modules,
            prediction=owner.prediction, generate=owner.generate, decode=owner.decode,
            capabilities=(*owner.capabilities, 'dmd_flow_private_replicated_b1_v1'))
        settings = dict(native[1].algorithm['parameters'])
    # Test-only scope: the existing independent sequential B1 oracle supports
    # CFG1. Retain each production recipe/default; both test arms use this scope.
    settings.update(teacher_cfg=1., fake_cfg=1.)
    for value in model.state.values():
        if isinstance(value, torch.nn.Module): value.eval()
    return model, feeds, settings


def _workload(model, feeds, settings, native, *, cycle=0, versions=None):
    w = flow_workload(model, feeds, settings, cycle, versions=versions)
    w.replica_semantics = replace(w.replica_semantics,
        frozen_module_bindings={'teacher_forward': 'teacher'},
        forward_methods={**w.replica_semantics.forward_methods, 'teacher_forward': 'dmd_forward'},
        # Partial cuts leave both child and root-owned parameters live. Native
        # fixtures qualify their real existing root; the matrix proves cuts.
        shard_boundaries={} if native else {'student': ('blocks.0', 'blocks.2')})
    return w.validate()


def _generation_worker(rank, directory, mode, stage, native=None):
    from distillschedule.open_workload.execution import ExecutionBackend
    from distillschedule.training.checkpoint import cpu_tree, restore_checkpoint, save_checkpoint
    from distillschedule.training.session import _generate_values
    from test_open_workload_parameter_shadows import _assert_close, _snapshot
    from torch.distributed.tensor import DTensor
    from flow_replica_oracle import sequential_b1_oracle
    root = Path(directory)
    torch.set_num_threads(1)
    # Actual native providers keep their original distributed construction gate.
    # As in R69/R73, tiny owners are constructed before the private CPU group.
    model, feeds, settings = _model(native, rank)
    if stage == 'write':
        reference, _, _ = _model(native, rank)
        inputs = [matrix_feeds(r) for r in range(2)] if native is None else [feeds, feeds]
    config = SimpleNamespace(trainer={'device': 'cpu', 'seed': 715})
    _init(rank, str(root / ('rdzv-' + stage)))
    try:
        w = _workload(model, feeds, settings, native)
        session = ExecutionBackend(w).session(implementation(mode), copy_state=False, persistent=True)
        model.state = session.state
        identity = {'fixture': 'owned-generation-cpu2', 'mode': mode, 'family': native[0] if native else None}
        if stage == 'write':
            sequential_b1_oracle(reference.state, inputs, settings, 53)
            session.run()
            versions = session.receipt['dispatcher']['final_versions']
            _assert_close(_snapshot(reference.state), _snapshot(session.state))
            save_checkpoint(root / 'checkpoint', session.state, identity=identity, cycle=1, cursor=1,
                            versions=versions)
        else:
            restored = restore_checkpoint(root / 'checkpoint', session.state, identity=identity)
            assert restored['cycle'] == restored['cursor'] == 1
            model.state = session.state
            versions = restored['versions']
        current = _workload(model, feeds, settings, native, cycle=1, versions=versions)
        session.bind_cycle(current, graph=ExecutionBackend(current).graph(implementation(mode)))
        state = pack_state(session.state)
        parameter_ids = {role: tuple(id(p) for p in session.state[role].parameters())
                         for role in ('student', 'fake', 'teacher')}
        student, fake = session.state['student'], session.state['fake']
        original, unrelated = student.dmd_forward, fake.reshard
        entries = []
        def observe(*args, **kwargs):
            entries.append((id(student), kwargs['flow_role'], kwargs['phase']))
            return original(*args, **kwargs)
        def wrong_owner():
            raise AssertionError('generation closed an unrelated Fake owner')
        student.dmd_forward, fake.reshard = observe, wrong_owner
        try:
            values = _generate_values(config, model, feeds, {}, latents_required=True)
        finally:
            student.dmd_forward, fake.reshard = original, unrelated
        assert entries == [(id(student), 'student', 'generation')]
        assert equal(state, pack_state(session.state)), 'generation changed checkpointed training state'
        assert parameter_ids == {role: tuple(id(p) for p in session.state[role].parameters())
                                 for role in parameter_ids}
        assert all(isinstance(p, DTensor) for p in student.parameters()), 'output must return all groups to shards'
        actual_rng = rng_state()
        if stage == 'write':
            # Direct legacy rollout is the independent numeric/RNG reference.
            generate = reference.generate
            if native is None:
                def legacy(state, feeds, parameters):
                    sample = state['student'].rollout(feeds, (1., .5, 0.), 'all_steps')['payload']
                    return {'payload': sample, 'latent_schema': {'payload': {
                        'layout': 'BC', 'shape': list(sample.shape), 'dtype': str(sample.dtype)}}}
            else:
                # Same original adapter packing/decoder with only the new
                # entry replaced by its former direct rollout call.
                from unittest.mock import patch
                def direct(owner, current_feeds, parameters):
                    schedule = parameters.get('rollout_sigmas', owner.generation_sigmas)
                    return owner.rollout(current_feeds, schedule, 'all_steps')
                with patch.object(dmd, 'generation_rollout', direct):
                    expected = _generate_values(config, reference, feeds, {}, latents_required=True)
            if native is None:
                reference.generate = legacy
                expected = _generate_values(config, reference, feeds, {}, latents_required=True)
                reference.generate = generate
            _assert_close(cpu_tree(expected), cpu_tree(values))
            assert equal(actual_rng, rng_state())
            sequential_b1_oracle(reference.state, inputs, settings, 54)
            outputs = session.run()
            _assert_close(_snapshot(reference.state), _snapshot(session.state))
            torch.save({'values': cpu_tree(values), 'state': pack_state(session.state),
                        'outputs': cpu_tree(outputs), 'rng': rng_state()}, root / f'expected-{rank}.pt')
        else:
            expected = torch.load(root / f'expected-{rank}.pt', weights_only=True)
            assert equal(cpu_tree(values), expected['values'])
            outputs = session.run()
            assert equal(expected['state'], pack_state(session.state))
            assert equal(expected['outputs'], cpu_tree(outputs)) and equal(expected['rng'], rng_state())
        # A failed output ends ownership; cleanup uses public native APIs only.
        failing = student.blocks[2] if native is None else student.native
        def fail_forward(*args): raise RuntimeError('injected native output failure')
        handle = failing.register_forward_pre_hook(fail_forward)
        try:
            with pytest.raises(RuntimeError, match='injected native output failure'):
                _generate_values(config, model, feeds, {}, latents_required=True)
        finally:
            handle.remove()
        # Torch's registered-method wrapper lacks a finally block. In
        # keep-forward mode public reshard can remain a no-op after an error;
        # no private state is rewritten to claim the failed owner reusable.
        if mode == 'fsdp2_reshard':
            assert all(isinstance(p, DTensor) for p in student.parameters())
        refused_rng = rng_state()
        refused_counters = session.state['dmd_counters'].tolist()
        with pytest.raises(ValueError, match='failed flow generation owner'):
            dmd.generation_rollout(student, feeds, {'rollout_sigmas': settings['rollout_sigmas']})
        with pytest.raises(ValueError, match='failed flow generation owner'):
            student.validate_checkpoint_boundary()
        with pytest.raises(ValueError, match='failed flow generation owner'):
            save_checkpoint(root / ('failed-' + stage), session.state, identity=identity,
                cycle=2, cursor=2, versions=session.receipt['dispatcher']['final_versions'])
        assert not list((root / ('failed-' + stage)).glob('_SUCCESS'))
        assert equal(refused_rng, rng_state()) and refused_counters == session.state['dmd_counters'].tolist()
        assert not torch.cuda.is_initialized()
        (root / f'{stage}-{rank}.json').write_text(json.dumps(dict(native=native[0] if native else None,
            outer_calls=1, state_unchanged=True, partial_cuts=native is None,
            restored=stage == 'restore', failure_reuse_rejected=True,
            exception_all_groups_closed=mode == 'fsdp2_reshard', cuda_initialized=False)))
    finally:
        dist.destroy_process_group()


@CPU_FSDP
@pytest.mark.parametrize('mode', ['fsdp2_reshard', 'fsdp2_keep_forward'])
def test_owned_generation_partial_groups_and_restored_continuation(tmp_path, mode):
    for stage in ('write', 'restore'):
        mp.spawn(_generation_worker, args=(str(tmp_path), mode, stage), nprocs=2, join=True)


@CPU_FSDP
@pytest.mark.parametrize('mode', ['fsdp2_reshard', 'fsdp2_keep_forward'])
def test_native_owned_generation_restored_current_student_and_closed_groups(tmp_path, native_config, mode):
    for stage in ('write', 'restore'):
        mp.spawn(_generation_worker, args=(str(tmp_path), mode, stage, native_config), nprocs=2, join=True)



def _preflight_worker(rank, directory, fault, adapter):
    from distillschedule.open_workload.execution import ExecutionBackend
    from distillschedule.open_workload.replicated import control_plane
    from distillschedule.training import distributed
    from torch.distributed.tensor import DTensor
    import os
    root = Path(directory)
    torch.set_num_threads(1)
    model, feeds, settings = _model(None, rank)
    os.environ.update(WORLD_SIZE='2', RANK=str(rank), LOCAL_RANK=str(rank),
                      LOCAL_WORLD_SIZE='2', GROUP_RANK='0')
    dist.init_process_group('gloo', init_method=(root / 'preflight-rdzv').as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=10))
    metadata = None
    try:
        metadata = dist.new_group(backend='gloo', timeout=timedelta(seconds=30))
        with control_plane(metadata):
            w = _workload(model, feeds, settings, None)
            session = ExecutionBackend(w).session(implementation('fsdp2_keep_forward'),
                                                 copy_state=False, persistent=True)
            student = session.state['student']
            inputs, schedule = dict(feeds), (1., .5, 0.)
            if rank == 0:
                if fault == 'input': inputs.pop('null_condition')
                elif fault == 'poisoned': student._dmd_generation_failed = True
                elif fault == 'invalid-schedule': schedule = (1., .25, .5, 0.)
            if fault == 'different-schedule' and rank == 1:
                schedule = (1., .25, 0.)
            parameters = {'rollout_sigmas': schedule}
            if fault == 'unknown-option' and rank == 0:
                parameters['unsupported'] = True
            if adapter == 'image':
                from distillschedule.training.flow_image import generate
                callback = lambda: generate(None, {'student': student}, inputs, parameters)
            else:
                from distillschedule.training.ltx import generate
                callback = lambda: generate({'student': student}, inputs, parameters)
            before, before_rng = pack_state(session.state), rng_state()
            original, entries = student.dmd_forward, []
            def observe(*args, **kwargs):
                entries.append(1)
                return original(*args, **kwargs)
            student.dmd_forward = observe
            try:
                with pytest.raises(ValueError, match='flow generation (admission|schedule)') as error:
                    # Match the public evaluator's outer checked boundary too.
                    distributed.checked(callback, 'checkpoint generation')
                messages = distributed.gather(str(error.value))
                assert all(message == messages[0] for message in messages)
                assert entries == [], 'a rank entered native FSDP before collective admission'
                assert equal(before, pack_state(session.state)) and equal(before_rng, rng_state())
                assert all(isinstance(parameter, DTensor) for parameter in student.parameters())
                assert bool(getattr(student, '_dmd_generation_failed', False)) == (fault == 'poisoned' and rank == 0)
                if fault != 'poisoned':
                    # Different healthy rank-local samples remain legal after
                    # fixing the request; they are never compared for equality.
                    dmd.generation_rollout(student, feeds, {'rollout_sigmas': (1., .5, 0.)})
                    assert entries == [1]
                    assert all(isinstance(parameter, DTensor) for parameter in student.parameters())
            finally:
                student.dmd_forward = original
            # Even inside a real two-rank group the unsharded reference keeps
            # its local path; it does not acquire a metadata collective.
            checked, gather = distributed.checked, distributed.gather
            def forbidden(*args, **kwargs):
                raise AssertionError('unsharded reference entered metadata admission')
            distributed.checked = distributed.gather = forbidden
            try:
                probe = ImageProbe()
                dmd.generation_rollout(probe, {'latent_shape': torch.tensor([1, 1, 2, 2])}, {'rollout_sigmas': (1., .5, 0.)})
                assert probe.entries == 1
            finally:
                distributed.checked, distributed.gather = checked, gather
            (root / f'preflight-{rank}.json').write_text(json.dumps(dict(fault=fault, adapter=adapter,
                native_forwards_before_refusal=0, rng_unchanged=True, healthy_owner_preserved=True,
                input_comparison=False, unsharded_metadata_collectives=0,
                error=messages[0], cuda_initialized=torch.cuda.is_initialized())))
    finally:
        if metadata is not None: dist.destroy_process_group(metadata)
        dist.destroy_process_group()


@CPU_FSDP
@pytest.mark.parametrize('adapter', ['image', 'ltx'])
@pytest.mark.parametrize('fault', ['input', 'poisoned', 'unknown-option', 'invalid-schedule', 'different-schedule'])
def test_owned_generation_collectively_refuses_before_native_forward(tmp_path, fault, adapter):
    mp.spawn(_preflight_worker, args=(str(tmp_path), fault, adapter), nprocs=2, join=True)
    rows = [json.loads((tmp_path / f'preflight-{rank}.json').read_text()) for rank in range(2)]
    assert rows[0] == rows[1] and not rows[0]['cuda_initialized']

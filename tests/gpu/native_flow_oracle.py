"""One direct native-role flow-DMD oracle shared by AV, image and video GPU tests."""
from copy import deepcopy
from dataclasses import replace
import gc
import json
import os
from pathlib import Path

import pytest
import torch

from distillschedule.training.checkpoint import cpu_tree
from test_training_flow_image_device import CUDA_MANUAL, CUDA_UNSUPPORTED


manual_sharing_only = pytest.mark.skipif(os.environ.get('DMD_CUDA_IMMUTABLE_SHARING') != '1',
    reason='opt in to actual native SYNTHETIC_TEST_ONLY CUDA immutable-sharing qualification')


def manual_config(config, mode):
    """Persist the real public manual request for fresh CLI processes."""
    from distillschedule.training.config import load_training_config
    if mode == 'reference':
        return config
    assert mode == 'manual'
    value = config.to_dict(); value['execution'] = deepcopy(CUDA_MANUAL)
    config.source.write_text(json.dumps(value))
    return load_training_config(config.source)


def assert_manual_cycles(config, root, monkeypatch):
    """Sequential fresh native owners; no paired CUDA clone or fake-model proof."""
    from distillschedule.open_workload import planning
    from distillschedule.open_workload.torch_backend import resolve_module, sharing_inventory, storage_bytes, storage_key
    from distillschedule.training.checkpoint import pack_state, rng_state
    from distillschedule.training.policy import CyclePolicy
    from distillschedule.training.session import construct, build_cycle
    from test_training_end_to_end import equal

    def cycles(execution):
        request = replace(config, execution=execution)
        components, dataset, algorithm, _ = construct(request)
        assert components.metadata['evidence_kind'] in (
            'SYNTHETIC_TEST_ONLY', 'RANDOM_TINY_ACTUAL_NATIVE_CLASSES_NOT_PRETRAINED')
        state = components.state; device = state['student'].device
        assert device.type == 'cuda' and torch.are_deterministic_algorithms_enabled()
        initial = pack_state(state); initial_rng = rng_state(device=device)
        gradients = []; handles = []
        def observe(role):
            def before_step(optimizer, args, kwargs):
                gradients.append((role, cpu_tree({name: p.grad for name, p in state[role].named_parameters()
                                                 if p.requires_grad})))
            return before_step
        owners = {role: {id(p) for p in state[role].parameters() if p.requires_grad} for role in ('student', 'fake')}
        for role in owners:
            handles.append(state[role + '_optimizer'].register_step_pre_hook(observe(role)))
        policy = CyclePolicy(execution); versions = None; rows = []; first_session = None
        try:
            for cycle in range(3):
                workload = build_cycle(request, components, dataset, algorithm, cycle=cycle, cursor=cycle, versions=versions)
                if cycle == 0:
                    pairs = sharing_inventory(workload)
                    assert pairs, 'real equal immutable native parameters must exist'
                    before_storage = storage_bytes((state, workload.feeds))
                gradients.clear(); outer = rng_state(device=device)
                session, outputs = policy.run(workload, root / execution['mode'])
                assert equal(outer, rng_state(device=device))
                if first_session is None:
                    first_session = session
                assert session is first_session and all(session.state[key] is value for key, value in state.items())
                assert not session.effects['checkpoint_calls'] and not session.effects['recompute_calls']
                assert session.receipt['dispatcher']['executed_operations'] == [
                    *[call.id for call in workload.program.calls], '__open_finish__']
                versions = session.receipt['dispatcher']['final_versions']
                trainable_storage = {}
                for role, count in (('student', cycle + 1), ('fake', 5 * (cycle + 1))):
                    optimizer = state[role + '_optimizer']
                    assert owners[role] == {id(p) for p in state[role].parameters() if p.requires_grad}
                    assert owners[role] == {id(p) for group in optimizer.param_groups for p in group['params']}
                    trainable_storage[role] = {storage_key(p) for p in state[role].parameters() if p.requires_grad}
                    assert all(p.grad is None for p in state[role].parameters())
                    assert state[role + '_scheduler'].last_epoch == count
                    assert all(value['step'].device == device and int(value['step']) == count for value in optimizer.state.values())
                assert not owners['student'] & owners['fake']
                assert not trainable_storage['student'] & trainable_storage['fake']
                aliases = 0
                for first, second, _ in pairs:
                    a = resolve_module(state, first[0]).get_parameter(first[1])
                    b = resolve_module(state, second[0]).get_parameter(second[1])
                    assert not a.requires_grad and not b.requires_grad
                    same = a is b
                    assert same == (storage_key(a) == storage_key(b))
                    assert same == (execution['mode'] == 'manual')
                    aliases += same
                assert session.effects['shared_parameter_aliases'] == aliases
                rows.append(dict(state=pack_state(state), outputs=cpu_tree(outputs), gradients=deepcopy(gradients),
                    rng=rng_state(device=device), versions=versions, sample_id=dataset.sample_id(cycle),
                    feeds=cpu_tree(workload.feeds), aliases=aliases, storage=session.initial_storage))
            assert len({row['sample_id'] for row in rows}) == 3
            assert (rows[0]['storage'] < before_storage) == (execution['mode'] == 'manual')
            if execution['mode'] == 'manual':
                for invalid in CUDA_UNSUPPORTED:
                    with pytest.raises(ValueError, match='CUDA.*trial.*snapshot'):
                        components.admit_training(state=state, request=replace(request, execution=invalid),
                            prepared_identity=dataset.identity, prepared_manifest=dataset.manifest)
                next_workload = build_cycle(request, components, dataset, algorithm, cycle=3, cursor=0, versions=versions)
                before = pack_state(state); outer = rng_state(device=device)
                # Actual inventory and live optimizer identity gates remain required.
                with pytest.raises(ValueError, match='capability predicates'):
                    CyclePolicy(CUDA_MANUAL).run(replace(next_workload, immutable_modules=()), root / 'no-inventory')
                optimizer = state['fake_optimizer']; original = optimizer.param_groups[0]['params'][0]
                optimizer.param_groups[0]['params'][0] = next(p for p in state['student'].parameters() if p.requires_grad)
                try:
                    with pytest.raises(ValueError, match='optimizer must exactly own disjoint role trainables'):
                        build_cycle(request, components, dataset, algorithm, cycle=3, cursor=0, versions=versions)
                finally:
                    optimizer.param_groups[0]['params'][0] = original
                with monkeypatch.context() as patch:
                    patch.setattr(planning, 'source_identity', lambda: '0' * 64)
                    with pytest.raises(ValueError, match='source'):
                        policy.run(next_workload, root / 'wrong-source')
                    resumed = CyclePolicy(CUDA_MANUAL); resumed.restore(policy.bound)
                    with pytest.raises(ValueError, match='source'):
                        resumed.prepare_resume(next_workload)
                assert equal(before, pack_state(state)) and equal(outer, rng_state(device=device))
                # Mutation detection is the existing post-execution invariant,
                # not a promise that a malicious write rolls back the cycle.
                first = pairs[0][0]
                with torch.no_grad():
                    resolve_module(state, first[0]).get_parameter(first[1]).add_(.001)
                with pytest.raises(ValueError, match='declared immutable parameters changed'):
                    policy.run(next_workload, root / 'mutated-immutable')
        finally:
            for handle in handles:
                handle.remove()
        return dict(initial=initial, initial_rng=initial_rng, rows=rows)

    reference = cycles(dict(mode='reference'))
    gc.collect()  # All native GPU owners have left the previous function scope.
    manual = cycles(deepcopy(CUDA_MANUAL))
    gc.collect()
    assert equal(reference['initial'], manual['initial']) and equal(reference['initial_rng'], manual['initial_rng'])
    for left, right in zip(reference['rows'], manual['rows']):
        for key in ('state', 'outputs', 'gradients', 'rng', 'versions', 'sample_id', 'feeds'):
            assert equal(left[key], right[key]), key
        assert right['aliases'] > left['aliases'] == 0 and right['storage'] < left['storage']
    (root / 'manual-sharing-proof.json').write_text(json.dumps(dict(evidence_kind='SYNTHETIC_TEST_ONLY',
        execution=CUDA_MANUAL, cycles=3, exact_state_gradients_rng_versions_outputs=True,
        alias_counts=[row['aliases'] for row in manual['rows']],
        reference_tracked_storage_bytes=reference['rows'][0]['storage'],
        manual_tracked_storage_bytes=manual['rows'][0]['storage'],
        capacity_scope='tracked resident tensors only; construction still materializes all roles',
        pretrained_capacity='NOT_RUN', quality_validation='NOT_RUN', claimable_speedup=False), indent=2))


def assert_manual_restart_policy(config, partial, resumed, whole, root):
    """Check actual effects and reject cold/resumed public policy substitutions."""
    from cuda_test_policy import run_cli
    if config.execution.get('mode') != 'manual':
        return
    for result in (partial, resumed, whole):
        assert result['policy']['implementation'] == CUDA_MANUAL['implementation']
        assert result['numerical_status'] == 'NOT_RUN', 'manual selection is not automatic paired admission'
        metrics = Path(result['checkpoint']).parent.parent / 'metrics.jsonl'
        for row in map(json.loads, metrics.read_text().splitlines()):
            receipt = row['execution']
            assert receipt['effects']['shared_parameter_aliases'] > 0
            assert receipt['effects']['checkpoint_calls'] == receipt['effects']['recompute_calls'] == 0
            assert receipt['dispatcher']['executed_operations'] == result['policy']['order']
    for index, execution in enumerate((dict(CUDA_MANUAL, order=[]),
            dict(CUDA_MANUAL, implementation=dict(CUDA_MANUAL['implementation'], share_immutable=False)),
            dict(CUDA_MANUAL, implementation=dict(CUDA_MANUAL['implementation'], owner_execution={})))):
        value = config.to_dict(); value['execution'] = execution
        request = root / f'wrong-policy-{index}.json'; request.write_text(json.dumps(value))
        for resume in ((), ('--resume', partial['checkpoint'])):
            output = root / f'refused-{index}-{bool(resume)}'
            run_cli('train', request, *resume, '--output', output, expected_error='CUDA execution requires')
            assert not output.exists()
    value = config.to_dict(); value['execution'] = dict(mode='reference')
    request = root / 'wrong-resume-mode.json'; request.write_text(json.dumps(value))
    run_cli('train', request, '--resume', partial['checkpoint'], '--output', root / 'wrong-resume-mode',
            expected_error='identity')


def _native_flow_oracle(state, feeds, policy, seed):
    """Direct native roles + shared objectives + Torch AdamW, without Program.

    Seed each stochastic boundary with the existing logical-call RNG contract.
    This recipe explicitly reuses the pre-update Student sample for all Fakes.
    """
    from distillschedule.open_workload.torch_backend import random_scope
    from distillschedule.training.dmd import student_objective, fake_objective
    student, fake, teacher = (state[name] for name in ('student', 'fake', 'teacher'))
    assert policy['fake_sample_policy'] == 'reuse_student' and policy['fake_updates'] == 5
    device = student.device
    losses, gradients = {}, []

    def broadcast(sigma, value):
        return sigma.reshape((-1,) + (1,) * (value.ndim - 1))

    def noise(generated, call):
        with torch.no_grad(), random_scope(seed, call, device):
            first = next(iter(generated.values()))
            sigma = torch.rand(first.shape[0], device=device, dtype=first.dtype)
            sigma = policy['sigma_min'] + (policy['sigma_max'] - policy['sigma_min']) * sigma
            epsilon = {name: torch.randn_like(value) for name, value in generated.items()}
            noised = {name: (1 - broadcast(sigma, value)) * value + broadcast(sigma, value) * epsilon[name]
                      for name, value in generated.items()}
            return student.apply_conditioning(noised, feeds), sigma, {
                name: epsilon[name] - value for name, value in generated.items()}

    def update(role):
        with torch.no_grad():
            params = [parameter for parameter in state[role].parameters() if parameter.requires_grad]
            torch.nn.utils.clip_grad_norm_(params, policy['max_grad_norm'], error_if_nonfinite=True)
            gradients.append((role, cpu_tree({name: p.grad for name, p in state[role].named_parameters()
                                             if p.requires_grad})))
            state[role + '_optimizer'].step()
            state[role + '_optimizer'].zero_grad(set_to_none=True)
            state[role + '_scheduler'].step()
            state['dmd_counters'][1 if role == 'student' else 2] += 1
            if role == 'student':
                for name, parameter in student.named_parameters():
                    state['student_ema'][name].lerp_(parameter.detach(), 1 - state['dmd_lifecycle']['ema_decay'])

    with random_scope(seed, 'dmd.c0.student_forward', device):
        generated = student.rollout(feeds, policy['rollout_sigmas'], policy['gradient_policy'])
    detached = {name: value.detach() for name, value in generated.items()}
    masks = student.loss_masks(feeds, generated)
    noised, sigma, _ = noise(detached, 'dmd.c0.noise')
    with torch.no_grad():
        conditional = teacher.denoise(noised, sigma, feeds, conditional=True)
        if policy['teacher_cfg'] == 1:
            velocity = conditional
        else:
            unconditional = teacher.denoise(noised, sigma, feeds, conditional=False)
            velocity = {name: unconditional[name] + policy['teacher_cfg'] * (value - unconditional[name])
                        for name, value in conditional.items()}
        teacher_x0 = {name: value - broadcast(sigma, value) * velocity[name] for name, value in noised.items()}
        fake_velocity = fake.denoise(noised, sigma, feeds, conditional=True)
        fake_x0 = {name: value - broadcast(sigma, value) * fake_velocity[name] for name, value in noised.items()}
    loss = student_objective(generated, teacher_x0, fake_x0, masks,
        weights=policy['modality_weights'], normalizer=policy['normalizer'], epsilon=policy['normalizer_epsilon'])
    losses['dmd.c0.student_loss'] = cpu_tree(loss)
    loss.backward(); update('student')
    del loss, generated, teacher_x0, fake_x0, conditional, velocity, fake_velocity
    for index in range(policy['fake_updates']):
        noised, sigma, target = noise(detached, f'dmd.c0.f{index}.noise')
        prediction = fake.denoise(noised, sigma, feeds, conditional=True)
        loss = fake_objective(prediction, target, masks, weights=policy['modality_weights'])
        losses[f'dmd.c0.f{index}.loss'] = cpu_tree(loss)
        loss.backward(); update('fake')
    state['dmd_counters'][0] += 1
    return losses, gradients

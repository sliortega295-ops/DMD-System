"""SYNTHETIC_TEST_ONLY independent sequential B1 replica oracle.

No Program, reducer, DMD hook or wrapper is called here. Native role methods,
shared mathematical objectives and native AdamW are retained. Local gradients
are collected separately and averaged BEFORE clipping and the ONE global step.
"""
import torch

from distillschedule.training.checkpoint import cpu_tree


def rank_scope(seed, call, device, rank, world):
    # Reproduce the documented hash, not the Session's rank-selection logic.
    from distillschedule.open_workload.torch_backend import random_scope
    namespace = dict(schema='rank_independent_v1', rank=rank, world_size=world)
    return random_scope(seed, call, device, namespace=namespace)


def sequential_b1_oracle(state, feeds_by_rank, policy, seed):
    from distillschedule.training.dmd import student_objective, fake_objective
    student, fake, teacher = (state[name] for name in ('student', 'fake', 'teacher'))
    world = len(feeds_by_rank)
    device = next(student.parameters()).device
    assert world >= 2 and policy['fake_sample_policy'] == 'reuse_student'
    assert policy['teacher_cfg'] == policy['fake_cfg'] == 1
    losses = [{} for _ in feeds_by_rank]
    generated, masks, trace, stochastic = [], [], [], []

    def expand(sigma, value):
        return sigma.reshape((-1,) + (1,) * (value.ndim - 1))

    def noised(rank, sample, call):
        feeds = feeds_by_rank[rank]
        with torch.no_grad(), rank_scope(seed, call, device, rank, world):
            first = next(iter(sample.values()))
            assert first.shape[0] == 1
            sigma = torch.rand(1, device=device, dtype=first.dtype)
            sigma = policy['sigma_min'] + (policy['sigma_max'] - policy['sigma_min']) * sigma
            epsilon = {name: torch.randn_like(value) for name, value in sample.items()}
            noisy = {name: (1 - expand(sigma, value)) * value + expand(sigma, value) * epsilon[name]
                     for name, value in sample.items()}
            stochastic.append((rank, call, cpu_tree(sigma), cpu_tree(epsilon)))
            return student.apply_conditioning(noisy, feeds), sigma, {
                name: epsilon[name] - value for name, value in sample.items()}

    def local_backward(role, loss):
        loss.backward()
        result = {name: p.grad.detach().clone() if p.grad is not None else None
                  for name, p in state[role].named_parameters() if p.requires_grad}
        state[role + '_optimizer'].zero_grad(set_to_none=True)
        return result

    def update(role, local):
        parameters = {name: p for name, p in state[role].named_parameters() if p.requires_grad}
        for name, parameter in parameters.items():
            values = [row[name] for row in local]
            parameter.grad = (sum(value if value is not None else torch.zeros_like(parameter) for value in values)
                              / world if any(value is not None for value in values) else None)
        mean = cpu_tree({name: p.grad for name, p in parameters.items()})
        torch.nn.utils.clip_grad_norm_(list(parameters.values()), policy['max_grad_norm'], error_if_nonfinite=True)
        clipped = cpu_tree({name: p.grad for name, p in parameters.items()})
        trace.append(dict(role=role, local=cpu_tree(local), mean=mean, clipped=clipped))
        state[role + '_optimizer'].step()
        state[role + '_optimizer'].zero_grad(set_to_none=True)
        state[role + '_scheduler'].step()
        state['dmd_counters'][1 if role == 'student' else 2] += 1
        if role == 'student' and 'student_ema' in state:
            for name, parameter in student.named_parameters():
                state['student_ema'][name].lerp_(parameter.detach(), 1 - state['dmd_lifecycle']['ema_decay'])

    local = []
    for rank, feeds in enumerate(feeds_by_rank):
        with rank_scope(seed, 'dmd.c0.student_forward', device, rank, world):
            sample = student.rollout(feeds, policy['rollout_sigmas'], policy['gradient_policy'])
        generated.append({key: value.detach() for key, value in sample.items()})
        masks.append(student.loss_masks(feeds, sample))
        noisy, sigma, _ = noised(rank, generated[-1], 'dmd.c0.noise')
        with torch.no_grad(), rank_scope(seed, 'dmd.c0.teacher', device, rank, world):
            teacher_velocity = teacher.denoise(noisy, sigma, feeds, conditional=True)
            teacher_x0 = {key: value - expand(sigma, value) * teacher_velocity[key] for key, value in noisy.items()}
        with torch.no_grad(), rank_scope(seed, 'dmd.c0.fake', device, rank, world):
            fake_velocity = fake.denoise(noisy, sigma, feeds, conditional=True)
            fake_x0 = {key: value - expand(sigma, value) * fake_velocity[key] for key, value in noisy.items()}
        loss = student_objective(sample, teacher_x0, fake_x0, masks[-1], weights=policy['modality_weights'],
            normalizer=policy['normalizer'], epsilon=policy['normalizer_epsilon'])
        losses[rank]['dmd.c0.student_loss'] = cpu_tree(loss)
        local.append(local_backward('student', loss))
    update('student', local)
    for index in range(policy['fake_updates']):
        local = []
        for rank, feeds in enumerate(feeds_by_rank):
            noisy, sigma, target = noised(rank, generated[rank], f'dmd.c0.f{index}.noise')
            with rank_scope(seed, f'dmd.c0.f{index}.forward', device, rank, world):
                prediction = fake.denoise(noisy, sigma, feeds, conditional=True)
            loss = fake_objective(prediction, target, masks[rank], weights=policy['modality_weights'])
            losses[rank][f'dmd.c0.f{index}.loss'] = cpu_tree(loss)
            local.append(local_backward('fake', loss))
        update('fake', local)
    state['dmd_counters'][0] += 1
    return dict(losses=losses, gradients=trace, stochastic=stochastic)


def observe_gradients(session):
    """Observe reducer output before clipping and optimizer input after clipping."""
    rows, handles = [], []
    actual_after_call = session.lifecycle.after_call
    def after_call(current, call):
        actual_after_call(current, call)
        if call.id in current.workload.replica_semantics.backward_calls:
            role = 'student' if call.attributes['phase'] == 'student' else 'fake'
            rows.append(dict(role=role, mean=cpu_tree({name: p.grad for name, p in current.state[role].named_parameters()
                                                     if p.requires_grad})))
    session.lifecycle.after_call = after_call
    def observe(role):
        def before_step(optimizer, args, kwargs):
            assert rows[-1]['role'] == role
            rows[-1]['clipped'] = cpu_tree({name: p.grad for name, p in session.state[role].named_parameters()
                                           if p.requires_grad})
        return before_step
    for role in ('student', 'fake'):
        handles.append(session.state[role + '_optimizer'].register_step_pre_hook(observe(role)))
    return rows, handles

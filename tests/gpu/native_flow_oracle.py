"""One direct native-role flow-DMD oracle shared by AV and image GPU tests."""
import torch

from distillschedule.training.checkpoint import cpu_tree


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

    def noise(generated, call):
        with torch.no_grad(), random_scope(seed, call, device):
            first = next(iter(generated.values()))
            sigma = torch.rand(first.shape[0], device=device, dtype=first.dtype)
            sigma = policy['sigma_min'] + (policy['sigma_max'] - policy['sigma_min']) * sigma
            epsilon = {name: torch.randn_like(value) for name, value in generated.items()}
            noised = {name: (1 - sigma[:, None, None]) * value + sigma[:, None, None] * epsilon[name]
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
        teacher_x0 = {name: value - sigma[:, None, None] * velocity[name] for name, value in noised.items()}
        fake_velocity = fake.denoise(noised, sigma, feeds, conditional=True)
        fake_x0 = {name: value - sigma[:, None, None] * fake_velocity[name] for name, value in noised.items()}
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

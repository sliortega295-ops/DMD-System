"""Reusable, explicit No-GAN rectified-flow DMD on the existing Program ABI.

This is NOT an alias for official DMD2 or official Self-Forcing. A model owns
``sample_template/validate_feeds/rollout/denoise/loss_masks/apply_conditioning``
on its three role modules. Payloads are named FP32 tensor dictionaries of ANY
rank >= 2. There is no model-name, latent-layout or fixed 5-D boundary here.

For z=(1-s)*x+s*e, flow v=e-x, x0=z-s*v. The student minimizes the detached
x0-score surrogate; Fake regresses the actual flow target e-x. Every mathematical
policy is required in the algorithm parameters, not borrowed from another paper.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
from torch.optim.lr_scheduler import ExponentialLR

from distillschedule.open_workload.dmd import DMDHook, DMDHooks, build_dmd_workload, dmd_inputs
from distillschedule.training.prediction import PredictionSpec
from distillschedule.workloads.program import Call


def require(condition, message):
    if not condition:
        raise ValueError('flow DMD: ' + message)


def _number(value, name, *, minimum=0., maximum=float('inf'), inclusive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value >= minimum if inclusive else value > minimum) and value <= maximum,
            'invalid finite ' + name)
    return float(value)


def sigmas(values):
    require(isinstance(values, (list, tuple)) and len(values) >= 2
            and all(type(x) in (int, float) and math.isfinite(x) for x in values)
            and values[0] == 1. and values[-1] == 0.
            and all(0 <= b < a <= 1 for a, b in zip(values, values[1:])),
            'rollout_sigmas must explicitly descend from 1 through every Euler step to 0')
    return tuple(float(x) for x in values)


def _policy(parameters):
    keys = {'fake_updates', 'fake_sample_policy', 'rollout_sigmas', 'gradient_policy', 'sigma_min',
            'sigma_max', 'teacher_cfg', 'fake_cfg', 'student_loss', 'fake_loss', 'normalizer',
            'normalizer_epsilon', 'modality_weights', 'max_grad_norm'}
    require(isinstance(parameters, dict) and set(parameters) == keys,
            'all loss/CFG/noise/normalizer/gradient/Fake policies must be explicit; unknown fields rejected')
    p = dict(parameters)
    require(type(p['fake_updates']) is int and 1 <= p['fake_updates'] <= 32, 'fake_updates must be 1..32')
    require(p['fake_sample_policy'] in ('reuse_student', 'fresh_student'), 'explicit Fake sample policy required')
    p['rollout_sigmas'] = sigmas(p['rollout_sigmas'])
    require(p['gradient_policy'] in ('all_steps', 'last_step'), 'gradient_policy must be all_steps or last_step')
    p['sigma_min'] = _number(p['sigma_min'], 'sigma_min', maximum=1.)
    p['sigma_max'] = _number(p['sigma_max'], 'sigma_max', maximum=1.)
    require(p['sigma_min'] < p['sigma_max'], 'score noise must have 0 < sigma_min < sigma_max <= 1')
    p['teacher_cfg'] = _number(p['teacher_cfg'], 'teacher_cfg', inclusive=True)
    require(p['fake_cfg'] == 1 and type(p['fake_cfg']) in (int, float),
            'this Fake objective is conditional flow regression: fake_cfg must be 1')
    require(p['student_loss'] == 'flow_x0_surrogate' and p['fake_loss'] == 'flow_velocity_mse',
            'unsupported loss; official DMD2/Self-Forcing policies are separate factories')
    require(p['normalizer'] in ('none', 'per_modality_mean_abs'), 'explicit supported normalizer required')
    p['normalizer_epsilon'] = _number(p['normalizer_epsilon'], 'normalizer_epsilon')
    weights = p['modality_weights']
    require(isinstance(weights, dict) and weights and all(isinstance(k, str) and k for k in weights),
            'named modality loss weights required')
    p['modality_weights'] = {k: _number(v, 'modality weight') for k, v in weights.items()}
    p['max_grad_norm'] = _number(p['max_grad_norm'], 'max_grad_norm')
    return p


@dataclass(frozen=True)
class _CycleProgress:
    """Live-only immutable contract derived from this algorithm's Program writes."""
    fake_updates: int
    version_strides: tuple[tuple[str, int], ...]

    def validate(self, progress):
        require(isinstance(progress, dict) and type(progress.get('cycle')) is int and progress['cycle'] >= 0,
                'complete-cycle checkpoint progress required')
        cycle, versions = progress['cycle'], progress.get('versions')
        expected = {name: cycle * stride for name, stride in self.version_strides}
        require(isinstance(versions, dict) and versions.keys() == expected.keys()
                and all(type(value) is int for value in versions.values()) and versions == expected,
                'logical state versions do not describe the declared complete-cycle updates')
        return cycle


class DMDClocks:
    """One owned counter tensor, validated before checkpoint restore or updates."""
    SCHEMA = 'distillschedule.dmd_clocks.v1'

    def __init__(self):
        self._counters = torch.zeros(3, dtype=torch.int64)
        self._progress = None

    def bind_progress(self, progress):
        require(self._progress is None or self._progress == progress, 'complete-cycle clock contract changed')
        self._progress = progress

    def __getitem__(self, index):
        return self._counters[index]

    def __setitem__(self, index, value):
        self._counters[index] = value

    def tolist(self):
        return self._counters.tolist()

    def runtime_tensors(self):
        yield self._counters

    def state_dict(self):
        return {'schema_version': self.SCHEMA, 'counters': self._counters}

    def validate_state_dict(self, value):
        require(isinstance(value, dict) and set(value) == {'schema_version', 'counters'}
                and value['schema_version'] == self.SCHEMA, 'complete-cycle clock state schema changed')
        counter = value['counters']
        require(type(counter) is torch.Tensor and counter.dtype == torch.int64 and counter.shape == (3,)
                and not counter.is_meta and counter.layout == torch.strided and not counter.requires_grad
                and bool((counter >= 0).all()), 'complete-cycle counters must be a nonnegative int64 triple')

    def load_state_dict(self, value):
        self.validate_state_dict(value)
        self._counters.copy_(value['counters'].to(self._counters.device))

    def _validate_cycle(self, value, cycle):
        self.validate_state_dict(value)
        require(self._progress is not None, 'complete-cycle clock contract is not bound to a Program')
        require(value['counters'].tolist() == [cycle, cycle, cycle * self._progress.fake_updates],
                'complete-cycle counters must equal [cycle, cycle, cycle * fake_updates]')

    def validate_cycle(self, cycle):
        self._validate_cycle(self.state_dict(), cycle)

    def validate_checkpoint_boundary(self):
        self.validate_cycle(int(self._counters[0]))

    def validate_checkpoint_progress(self, value, progress):
        require(self._progress is not None, 'complete-cycle clock contract is not bound to a Program')
        self._validate_cycle(value, self._progress.validate(progress))


class DMDExponentialLR(ExponentialLR):
    """The actual native scheduler, with per-role complete-cycle admission only.

    Native step/get_lr arithmetic is unchanged. Role/Program bindings are live
    immutable configuration, not a second serialized scheduler or mutable cache.
    """
    SCHEMA = 'distillschedule.dmd_exponential_lr.v1'

    def __init__(self, optimizer, *, gamma, role):
        self._dmd_role, self._dmd_progress = role, None
        super().__init__(optimizer, gamma=gamma)

    def bind_progress(self, progress):
        require(self._dmd_progress is None or self._dmd_progress == progress, 'complete-cycle scheduler contract changed')
        self._dmd_progress = progress

    def state_dict(self):
        value = {key: item for key, item in super().state_dict().items()
                 if key not in ('_dmd_role', '_dmd_progress')}
        return dict(value, dmd_schema_version=self.SCHEMA, dmd_role=self._dmd_role)

    def validate_state_dict(self, value):
        require(isinstance(value, dict) and value.keys() == self.state_dict().keys()
                and value['dmd_schema_version'] == self.SCHEMA and value['dmd_role'] == self._dmd_role
                and value['gamma'] == self.gamma and value['base_lrs'] == self.base_lrs,
                'scheduler clock schema/role/policy changed')
        require(type(value['last_epoch']) is int and value['last_epoch'] >= 0
                and type(value['_step_count']) is int and value['_step_count'] >= 1,
                'scheduler clock must contain actual nonnegative update counts')

    def load_state_dict(self, value):
        self.validate_state_dict(value)
        super().load_state_dict({key: item for key, item in value.items()
                                if key not in ('dmd_schema_version', 'dmd_role')})

    def _validate_cycle(self, value, cycle):
        self.validate_state_dict(value)
        require(self._dmd_progress is not None, 'complete-cycle scheduler contract is not bound to a Program')
        updates = cycle * (1 if self._dmd_role == 'student' else self._dmd_progress.fake_updates)
        require(value['last_epoch'] == updates and value['_step_count'] == updates + 1,
                'scheduler clock differs from actual complete-cycle role updates: ' + self._dmd_role)

    def validate_cycle(self, cycle):
        self._validate_cycle(self.state_dict(), cycle)

    def validate_checkpoint_progress(self, value, progress):
        require(self._dmd_progress is not None, 'complete-cycle scheduler contract is not bound to a Program')
        self._validate_cycle(value, self._dmd_progress.validate(progress))


def training_state(student, fake, teacher, *, student_lr, fake_lr, scheduler_gamma, ema_decay=None):
    """Visible ordinary modules, AdamW, actual per-update schedulers and clocks.

    Model construction owns freezing and the trainable parameter scope. EMA is
    explicitly optional, updated from CURRENT student parameters after G commits.
    """
    import torch
    gamma = _number(scheduler_gamma, 'scheduler_gamma', maximum=1.)
    require(ema_decay is None or type(ema_decay) in (int, float) and 0 <= ema_decay < 1,
            'ema_decay must be None or finite in [0,1)')
    require(all(not p.requires_grad for p in teacher.parameters()), 'Teacher must already be frozen')
    state = {'student': student, 'fake': fake, 'teacher': teacher,
             'student_grad': None, 'fake_grad': None,
             'dmd_counters': DMDClocks(),
             'dmd_lifecycle': {'ema_decay': ema_decay, 'scheduler': 'per_actual_role_update'}}
    for role, lr in (('student', student_lr), ('fake', fake_lr)):
        params = [p for p in state[role].parameters() if p.requires_grad]
        require(bool(params), 'role has no actual trainable parameters: ' + role)
        optimizer = torch.optim.AdamW(params, lr=_number(lr, role + '_lr'),
                                      betas=(.9, .999), eps=1e-8, weight_decay=0.)
        state[role + '_optimizer'] = optimizer
        state[role + '_scheduler'] = DMDExponentialLR(optimizer, gamma=gamma, role=role)
    if ema_decay is not None:
        state['student_ema'] = {name: p.detach().clone() for name, p in student.named_parameters()}
    return state


def _payload(value, *, like=None):
    import torch
    require(isinstance(value, dict) and value and all(isinstance(k, str) and k for k in value),
            'model must return a named tensor payload')
    require(all(isinstance(t, torch.Tensor) and t.dtype == torch.float32 and t.ndim >= 2
                and min(t.shape) > 0 and not t.is_meta and bool(torch.isfinite(t).all()) for t in value.values()),
            'payload must be nonempty finite real FP32 tensors')
    first = next(iter(value.values()))
    require(all(t.shape[0] == first.shape[0] and t.device == first.device for t in value.values()),
            'payload modalities must share batch and device')
    if like is not None:
        require(value.keys() == like.keys() and all(value[k].shape == like[k].shape for k in value),
                'model prediction changed payload layout')
    return value


def _masks(value, sample):
    import torch
    require(isinstance(value, dict) and value.keys() == sample.keys(), 'loss mask modalities changed')
    require(all(isinstance(value[k], torch.Tensor) and value[k].dtype == torch.bool
                and value[k].device == sample[k].device and value[k].shape == sample[k].shape for k in sample),
            'loss masks must explicitly match every element; clean conditioning must be excluded')
    return value


def _mean(value, mask):
    axes = tuple(range(1, value.ndim))
    # Empty/fully conditioned modalities contribute zero, never NaN. Weight
    # normalization remains the explicitly configured sum across ALL modalities.
    return (value * mask).sum(axes, keepdim=True) / mask.sum(axes, keepdim=True).clamp_min(1)


def student_objective(sample, teacher_x0, fake_x0, masks, *, weights, normalizer, epsilon):
    """Detached score-difference target; no score-parameter or mask gradients."""
    _payload(sample); _payload(teacher_x0, like=sample); _payload(fake_x0, like=sample); _masks(masks, sample)
    require(weights.keys() == sample.keys(), 'loss weights must match model modalities exactly')
    terms = []
    for key, x in sample.items():
        denominator = (_mean((x.detach() - teacher_x0[key].detach()).abs(), masks[key]).clamp_min(epsilon)
                       if normalizer == 'per_modality_mean_abs' else 1.)
        gradient = ((fake_x0[key] - teacher_x0[key]) / denominator).detach()
        target = (x - gradient).detach()
        terms.append(.5 * weights[key] * _mean((x - target).square(), masks[key]).mean())
    return sum(terms) / sum(weights.values())


def fake_objective(prediction, target, masks, *, weights):
    _payload(prediction); _payload(target, like=prediction); _masks(masks, prediction)
    require(weights.keys() == prediction.keys(), 'loss weights must match model modalities exactly')
    return sum(weights[k] * _mean((value - target[k].detach()).square(), masks[k]).mean()
               for k, value in prediction.items()) / sum(weights.values())


def _expand(sigma, tensor):
    return sigma.reshape(sigma.shape[0], *([1] * (tensor.ndim - 1)))


def _student(ctx):
    values = dmd_inputs(ctx)
    counters = ctx.state['dmd_counters']
    require(int(counters[0]) == ctx.attributes['cycle'], 'complete-cycle clock differs from requested cycle')
    if ctx.attributes['phase'] == 'student':
        counters.validate_cycle(ctx.attributes['cycle'])
        for role in ('student', 'fake'):
            ctx.state[role + '_scheduler'].validate_cycle(ctx.attributes['cycle'])
    p = ctx.attributes['flow_policy']
    return _payload(ctx.state['student'].rollout(values['feeds'], p['rollout_sigmas'], p['gradient_policy']))


def _noise(ctx):
    import torch
    values = dmd_inputs(ctx); generated = _payload(values['generated'])
    template = next(iter(generated.values())); p = ctx.attributes['flow_policy']
    sigma = torch.rand(template.shape[0], device=template.device, dtype=template.dtype)
    sigma = p['sigma_min'] + (p['sigma_max'] - p['sigma_min']) * sigma
    epsilon = {name: torch.randn_like(x) for name, x in generated.items()}
    noisy = {k: (1 - _expand(sigma, x)) * x + _expand(sigma, x) * epsilon[k] for k, x in generated.items()}
    noisy = ctx.state['student'].apply_conditioning(noisy, values['feeds'])
    masks = _masks(ctx.state['student'].loss_masks(values['feeds'], generated), generated)
    return {'sample': _payload(noisy, like=generated), 'sigma': sigma, 'masks': masks,
            'target': {k: epsilon[k] - x for k, x in generated.items()}}


def _teacher(ctx):
    values = dmd_inputs(ctx); noise = values['noise']; model = ctx.state['teacher']
    conditional = _payload(model.denoise(noise['sample'], noise['sigma'], values['feeds'], conditional=True),
                           like=noise['sample'])
    scale = ctx.attributes['flow_policy']['teacher_cfg']
    if scale == 1:
        velocity = conditional
    else:
        unconditional = _payload(model.denoise(noise['sample'], noise['sigma'], values['feeds'], conditional=False),
                                 like=noise['sample'])
        velocity = {k: unconditional[k] + scale * (v - unconditional[k]) for k, v in conditional.items()}
    return {k: x - _expand(noise['sigma'], x) * velocity[k] for k, x in noise['sample'].items()}


def _fake(ctx):
    values = dmd_inputs(ctx); noise = values['noise']
    velocity = _payload(ctx.state['fake'].denoise(noise['sample'], noise['sigma'], values['feeds'], conditional=True),
                        like=noise['sample'])
    if ctx.attributes['phase'] == 'student':
        return {k: x - _expand(noise['sigma'], x) * velocity[k] for k, x in noise['sample'].items()}
    return velocity


def _student_loss(ctx):
    values = dmd_inputs(ctx); p = ctx.attributes['flow_policy']
    return student_objective(values['generated'], values['teacher'], values['fake'], values['noise']['masks'],
        weights=p['modality_weights'], normalizer=p['normalizer'], epsilon=p['normalizer_epsilon'])


def _fake_loss(ctx):
    values = dmd_inputs(ctx)
    return fake_objective(values['fake'], values['noise']['target'], values['noise']['masks'],
                          weights=ctx.attributes['flow_policy']['modality_weights'])


def _update(ctx):
    import torch
    role = 'student' if ctx.attributes['phase'] == 'student' else 'fake'
    state = ctx.state
    params = [p for p in state[role].parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(params, ctx.attributes['flow_policy']['max_grad_norm'], error_if_nonfinite=True)
    require(bool(torch.isfinite(norm)), 'nonfinite gradient')
    state[role + '_optimizer'].step()
    state[role + '_optimizer'].zero_grad(set_to_none=True)
    state[role + '_scheduler'].step()
    state['dmd_counters'][1 if role == 'student' else 2] += 1
    if role == 'student' and 'student_ema' in state:
        decay = state['dmd_lifecycle']['ema_decay']
        for name, parameter in state['student'].named_parameters():
            state['student_ema'][name].lerp_(parameter.detach(), 1 - decay)


def _commit(ctx):
    cycle = ctx.attributes['cycle']; ratio = ctx.attributes['fake_updates']
    require(ctx.state['dmd_counters'].tolist() == [cycle, cycle + 1, (cycle + 1) * ratio],
            'actual update counters do not describe a complete Student1/FakeN cycle')
    ctx.state['dmd_counters'][0] += 1


def build_algorithm(components, feeds, parameters, cycle, seed):
    """Build one genuine reusable Student1/FakeN graph; never selects a trainer."""
    from distillschedule.training.distributed import rank_world
    replicated = rank_world()[1] > 1
    require(not replicated or 'dmd_flow_private_replicated_b1_v1' in components.capabilities,
            'distributed flow requires explicit private-method B1 replication capability')
    require(components.prediction == PredictionSpec('flow') and 'dmd_flow_v1' in components.capabilities,
            'model must explicitly implement the flow DMD protocol')
    require(type(cycle) is int and cycle >= 0, 'nonnegative cycle required')
    p = _policy(parameters); state = components.state
    methods = ('sample_template', 'validate_feeds', 'rollout', 'denoise', 'loss_masks', 'apply_conditioning')
    for role in ('student', 'fake', 'teacher'):
        require(role in state and all(callable(getattr(state[role], name, None)) for name in methods),
                'role must provide model-owned diffusion seams: ' + role)
        state[role].validate_feeds(feeds)
    template = _payload(state['student'].sample_template(feeds))
    require(not replicated or next(iter(template.values())).shape[0] == 1,
            'private flow replication requires equal local batch-one B1')
    require(template.keys() == p['modality_weights'].keys(), 'modality weights differ from the model payload')
    _masks(state['student'].loss_masks(feeds, template), template)
    base_names = {'student', 'fake', 'teacher', 'student_optimizer', 'fake_optimizer', 'student_grad', 'fake_grad'}
    required = {'dmd_counters', 'dmd_lifecycle', 'student_scheduler', 'fake_scheduler'}
    require(base_names | required <= state.keys(), 'complete role/optimizer/scheduler/counter state required')
    require(type(state['dmd_counters']) is DMDClocks
            and all(type(state[role + '_scheduler']) is DMDExponentialLR
                    and state[role + '_scheduler'].optimizer is state[role + '_optimizer']
                    and state[role + '_scheduler']._dmd_role == role for role in ('student', 'fake')),
            'owned complete-cycle clocks and native per-role schedulers required')
    require(('student_ema' in state) == (state['dmd_lifecycle']['ema_decay'] is not None), 'EMA state/policy mismatch')
    student_effects = ('student_scheduler', 'dmd_counters', *(('student_ema',) if 'student_ema' in state else ()))
    hooks = DMDHooks(student_forward=DMDHook(_student, reads=('dmd_counters', 'student_scheduler', 'fake_scheduler')),
        teacher_forward=_teacher, fake_forward=_fake,
        noise=DMDHook(_noise, reads=('student',)), student_loss=_student_loss, fake_loss=_fake_loss,
        student_update=DMDHook(_update, reads=(*student_effects, 'dmd_lifecycle'), writes=student_effects),
        fake_update=DMDHook(_update, reads=('fake_scheduler', 'dmd_counters'), writes=('fake_scheduler', 'dmd_counters')))
    workload = build_dmd_workload(student=state['student'], fake=state['fake'], teacher=state['teacher'],
        student_optimizer=state['student_optimizer'], fake_optimizer=state['fake_optimizer'],
        hooks=hooks, feeds=feeds, cycles=1, fake_updates=p['fake_updates'], fake_sample_policy=p['fake_sample_policy'],
        extra_state={k: v for k, v in state.items() if k not in base_names}, seed=seed,
        immutable_modules=components.immutable_modules,
        identity_extra={'algorithm': 'explicit_nogan_rectified_flow_dmd.v1', 'policy': p})
    calls = tuple(replace(call, attributes={**call.attributes, 'cycle': cycle, 'flow_policy': p})
                  for call in workload.program.calls)
    commit = Call('dmd.commit_cycle', 'commit_cycle', (), ('dmd_counters',), ('dmd_counters',),
        after=tuple(call.id for call in calls), attributes={'cycle': cycle, 'fake_updates': p['fake_updates']})
    workload.program = replace(workload.program, calls=(*calls, commit),
        parameters={**workload.program.parameters, 'algorithm': 'explicit_nogan_rectified_flow_dmd.v1', 'policy': p})
    workload.bindings = {**workload.bindings, 'commit_cycle': _commit}
    if replicated:
        from distillschedule.open_workload.replica_contract import ReplicaSemantics
        workload.replica_semantics = ReplicaSemantics(
            {'student_forward': 'student', 'fake_forward': 'fake'},
            tuple(call.id for call in calls if call.backward_of), 1,
            forward_interface='private_methods', random_streams='rank_independent_v1')
    progress = _CycleProgress(p['fake_updates'], tuple((name,
        sum(name in call.writes for call in workload.program.calls)) for name in sorted(state)))
    for name in ('dmd_counters', 'student_scheduler', 'fake_scheduler'):
        state[name].bind_progress(progress)
    return workload.validate()

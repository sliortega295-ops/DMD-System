"""Explicit Student1/FakeN semantics lowered to the existing open Program ABI.

The caller owns every forward, noise construction and loss. This module chooses
neither a DMD formula nor an execution strategy, and never imports model assets.
``build_dmd_program`` is Torch-free; ``build_dmd_workload`` binds actual models
and optimizers for the existing callable backend/planner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from distillschedule.core.contracts import require
from distillschedule.workloads.program import Call, Program


_STATES = ('student', 'fake', 'teacher', 'student_optimizer', 'fake_optimizer',
           'student_grad', 'fake_grad')


@dataclass(frozen=True)
class DMDHook:
    """Inspectable Context function plus any additional logical state effects.

    Role model/optimizer/gradient state is declared by the builder. Other state
    must be listed here and bound by the caller. ``replay_safe`` is an explicit
    purity assertion for gradient-bearing forward regions only, never inferred.
    All state must flow through Context, not captured models/tensors/optimizers.
    """
    function: Callable
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    replay_safe: bool = False

    def __post_init__(self):
        require(callable(self.function), 'callable DMD hook required', path='open.dmd')
        for name in ('reads', 'writes'):
            values = getattr(self, name)
            require(isinstance(values, (tuple, list)) and len(values) == len(set(values))
                    and all(isinstance(value, str) and value for value in values),
                    f'invalid hook {name}', path='open.dmd')
            object.__setattr__(self, name, tuple(values))
        require(set(self.writes) <= set(self.reads), 'hook writes must also read state', path='open.dmd')
        require('teacher' not in self.writes, 'Teacher parameters are immutable', path='open.dmd')
        require(type(self.replay_safe) is bool, 'replay_safe must be boolean', path='open.dmd')


def dmd_inputs(ctx):
    """Return semantic input names, independent of generated graph call IDs.

    ``feeds`` contains the caller's complete feed mapping. Other keys depend on
    the hook (generated/noise/teacher/fake/history/current/previous/loss).
    Cycle, phase and Fake substep are in ``ctx.attributes``.
    """
    return {'feeds': {key: ctx.inputs[key] for key in ctx.attributes['dmd_feeds']},
            **{key: ctx.inputs[value] for key, value in ctx.attributes['dmd_inputs'].items()}}


def _detach(ctx):
    # Session detaches every output tensor tree of a no_grad Call, including a
    # tensor returned unchanged here. This explicit graph node erases provenance.
    return dmd_inputs(ctx)['generated']


def _backward(ctx):
    dmd_inputs(ctx)['loss'].backward()


def _update(ctx):
    optimizer = ctx.state[ctx.attributes['optimizer']]
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


@dataclass(frozen=True)
class DMDHooks:
    """Required user math; no default squared-error/pseudo-DMD objective.

    Forward hooks use ctx.state['student'/'fake'/'teacher']; model input/output
    structure is unrestricted. Noise and loss hooks interpret that structure.
    Optional history builds the caller's previous/current latent context under
    no_grad. Optional backward/update hooks can preserve scaling or update math,
    provided every additional mutable state effect is declared with DMDHook.
    Plain inspectable functions are accepted as shorthand for pure DMDHook.
    """
    student_forward: DMDHook | Callable
    fake_forward: DMDHook | Callable
    teacher_forward: DMDHook | Callable
    noise: DMDHook | Callable
    student_loss: DMDHook | Callable
    fake_loss: DMDHook | Callable
    history: DMDHook | Callable | None = None
    student_backward: DMDHook | Callable = _backward
    fake_backward: DMDHook | Callable = _backward
    student_update: DMDHook | Callable = _update
    fake_update: DMDHook | Callable = _update

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name == 'history' and value is None:
                continue
            object.__setattr__(self, name, value if isinstance(value, DMDHook) else DMDHook(value))
        for name in ('teacher_forward', 'noise', 'history', 'student_backward',
                     'fake_backward', 'student_update', 'fake_update'):
            hook = getattr(self, name)
            require(hook is None or not hook.replay_safe,
                    f'{name} is not a replayable forward', path='open.dmd')

    def bindings(self, program: Program):
        """Exact bindings for Workload; no closure captures a second model."""
        available = {name: hook.function for name in self.__dataclass_fields__
                     if (hook := getattr(self, name)) is not None}
        available['detach'] = _detach
        return {name: available[name] for name in {call.binding for call in program.calls}}


def build_dmd_program(*, hooks: DMDHooks, feed_names: tuple[str, ...], cycles: int,
                      fake_updates: int, fake_sample_policy: str,
                      history_feed: str | None = None,
                      initial_versions: Mapping[str, int] | None = None) -> Program:
    """Instantiate Student1/FakeN without choosing layout/checkpoint/solver.

    Each cycle runs one differentiable Student update, then N Fake updates.
    ``reuse_student`` uses that cycle's PRE-update detached Student sample;
    ``fresh_student`` calls the UPDATED Student under no_grad for each Fake
    update. This choice is required: they are different training workloads.

    A history hook sees ``current`` (detached generated output) and optional
    ``previous`` (the prior cycle history or initial history feed). It returns
    arbitrary nested tensor context under no_grad. Student forwards consume
    the previous cycle context, not history generated later in the same cycle.
    Fresh Fake rollouts use that same previous context and do not advance the
    next cycle's history. These are latent contexts, not persistent KV caches.
    """
    require(isinstance(hooks, DMDHooks), 'typed DMDHooks required', path='open.dmd')
    require(type(cycles) is int and cycles > 0 and type(fake_updates) is int and fake_updates > 0,
            'positive cycles and Fake update count required', path='open.dmd')
    require(fake_sample_policy in ('reuse_student', 'fresh_student'),
            'explicit Fake sample policy required', path='open.dmd')
    require(isinstance(feed_names, (tuple, list)) and len(set(feed_names)) == len(feed_names)
            and all(isinstance(name, str) and name for name in feed_names),
            'unique feed names required', path='open.dmd')
    feed_names = tuple(feed_names)
    require(history_feed is None or (hooks.history is not None and history_feed in feed_names),
            'history_feed requires a history hook and a declared feed', path='open.dmd')
    versions = dict.fromkeys(_STATES, 0) if initial_versions is None else dict(initial_versions)
    require(set(_STATES) <= set(versions), 'all DMD logical states must be declared', path='open.dmd')
    # Bound construction before allocating a huge list; Program also enforces 511.
    per_cycle = 8 + int(hooks.history is not None)
    per_fake = 5 + (1 + int(hooks.history is not None) if fake_sample_policy == 'fresh_student' else 0)
    require(cycles * (per_cycle + fake_updates * per_fake) <= 511,
            'DMD program exceeds the 511-call open Program bound', path='open.dmd')
    calls = []; outputs = []

    def emit(call_id, binding, named_inputs, *, cycle, phase, fake_index=None,
             reads=(), writes=(), grad='no_grad', backward_of=(), optimizer=None, after=()):
        hook = None if binding == 'detach' else getattr(hooks, binding)
        extra_reads = hook.reads if hook is not None else ()
        extra_writes = hook.writes if hook is not None else ()
        attributes = {'cycle': cycle, 'phase': phase, 'fake_index': fake_index,
                      'dmd_inputs': dict(named_inputs), 'dmd_feeds': list(feed_names)}
        if optimizer is not None:
            attributes['optimizer'] = optimizer
        calls.append(Call(call_id, binding,
            tuple(dict.fromkeys((*feed_names, *named_inputs.values()))),
            tuple(dict.fromkeys((*reads, *extra_reads))),
            tuple(dict.fromkeys((*writes, *extra_writes))), grad=grad,
            backward_of=tuple(backward_of), after=tuple(after),
            replay_safe=bool(hook is not None and hook.replay_safe and grad == 'grad'),
            attributes=attributes))
        return call_id

    previous = history_feed
    for cycle in range(cycles):
        prefix = f'dmd.c{cycle}'; phase = dict(cycle=cycle, phase='student')
        old_history = {'history': previous} if previous is not None else {}
        generated = emit(prefix + '.student_forward', 'student_forward', old_history,
                         reads=('student',), grad='grad', **phase)
        sample = emit(prefix + '.detach', 'detach', {'generated': generated}, **phase)
        history = None
        if hooks.history is not None:
            history_inputs = {'current': sample}
            if previous is not None:
                history_inputs['previous'] = previous
            history = emit(prefix + '.history', 'history', history_inputs, **phase)
        context = {'generated': sample, **({'history': history} if history is not None else {})}
        noised = emit(prefix + '.noise', 'noise', context, **phase)
        score_inputs = {**context, 'noise': noised}
        teacher = emit(prefix + '.teacher', 'teacher_forward', score_inputs, reads=('teacher',), **phase)
        fake = emit(prefix + '.fake', 'fake_forward', score_inputs, reads=('fake',), **phase)
        loss = emit(prefix + '.student_loss', 'student_loss',
                    {**score_inputs, 'generated': generated, 'teacher': teacher, 'fake': fake},
                    grad='grad', **phase)
        backward = emit(prefix + '.student_backward', 'student_backward', {'loss': loss},
                        reads=('student_grad',), writes=('student_grad',),
                        grad='grad', backward_of=(loss,), **phase)
        student_commit = emit(prefix + '.student_update', 'student_update', {'backward': backward},
             reads=('student', 'student_optimizer', 'student_grad'),
             writes=('student', 'student_optimizer', 'student_grad'),
             optimizer='student_optimizer', **phase)
        outputs.append(loss)
        for index in range(fake_updates):
            sub = prefix + f'.f{index}'
            fake_phase = dict(cycle=cycle, phase='fake', fake_index=index)
            fake_context = dict(context)
            if fake_sample_policy == 'fresh_student':
                fake_sample = emit(sub + '.rollout', 'student_forward', old_history,
                                   reads=('student',), **fake_phase)
                fake_context = {'generated': fake_sample}
                if hooks.history is not None:
                    history_inputs = {'current': fake_sample}
                    if previous is not None:
                        history_inputs['previous'] = previous
                    fake_context['history'] = emit(sub + '.history', 'history', history_inputs, **fake_phase)
            fake_noise = emit(sub + '.noise', 'noise', fake_context, **fake_phase)
            prediction_inputs = {**fake_context, 'noise': fake_noise}
            prediction = emit(sub + '.forward', 'fake_forward', prediction_inputs,
                              reads=('fake',), grad='grad', **fake_phase)
            loss = emit(sub + '.loss', 'fake_loss', {**prediction_inputs, 'fake': prediction},
                        grad='grad', **fake_phase)
            backward = emit(sub + '.backward', 'fake_backward', {'loss': loss},
                            reads=('fake_grad',), writes=('fake_grad',),
                            grad='grad', backward_of=(loss,), **fake_phase)
            emit(sub + '.update', 'fake_update', {'backward': backward},
                 reads=('fake', 'fake_optimizer', 'fake_grad'),
                 writes=('fake', 'fake_optimizer', 'fake_grad'), optimizer='fake_optimizer',
                 after=(student_commit,), **fake_phase)
            outputs.append(loss)
        previous = history
    if previous is not None:
        outputs.append(previous)
    return Program(tuple(calls), versions, feed_names, tuple(outputs), {
        'builder': 'explicit_dmd_student1_faken_v1', 'cycles': cycles,
        'fake_updates': fake_updates, 'fake_sample_policy': fake_sample_policy,
        'history_feed': history_feed, 'history': 'explicit_no_grad_hook' if hooks.history else 'none'})


def build_dmd_workload(*, student, fake, teacher, student_optimizer, fake_optimizer,
                       hooks: DMDHooks, feeds: Mapping[str, Any], cycles: int,
                       fake_updates: int, fake_sample_policy: str,
                       history_feed: str | None = None, extra_state: Mapping[str, Any] | None = None,
                       initial_versions: Mapping[str, int] | None = None, seed: int = 0,
                       immutable_modules: tuple[str, ...] = (),
                       identity_extra: Mapping[str, Any] | None = None,
                       replica_batch_size: int | None = None):
    """Bind caller models/optimizers; return a normal Workload for the SAME planner.

    Nothing is frozen, reinitialized, downloaded or moved as a side effect.
    Teacher must already be frozen, and each optimizer must exactly own its
    role's trainable parameters. Initial history tensors must already be detached.
    Default updates are step + zero_grad(None).
    Advanced lifecycle/EMA/scaler effects require explicit hook state effects;
    contracts outside this bounded loop can still use Program/Call directly.
    """
    from .torch_backend import Workload, tensors
    import torch
    state = {'student': student, 'fake': fake, 'teacher': teacher,
             'student_optimizer': student_optimizer, 'fake_optimizer': fake_optimizer,
             'student_grad': None, 'fake_grad': None}
    require(not set(extra_state or {}) & set(state), 'extra_state conflicts with DMD state', path='open.dmd')
    state.update(extra_state or {})
    for role in ('student', 'fake', 'teacher'):
        require(isinstance(state[role], torch.nn.Module), 'DMD role must be a module', path='open.dmd')
    require(all(not parameter.requires_grad for parameter in teacher.parameters()),
            'Teacher must already be frozen; builder never changes user models', path='open.dmd')
    for role in ('student', 'fake'):
        optimizer = state[role + '_optimizer']
        require(isinstance(optimizer, torch.optim.Optimizer), 'Torch optimizer required', path='open.dmd')
        owned = [id(parameter) for group in optimizer.param_groups for parameter in group['params']]
        expected = {id(parameter) for parameter in state[role].parameters() if parameter.requires_grad}
        require(expected and len(owned) == len(set(owned)) and set(owned) == expected,
                'optimizer must exactly own disjoint role trainables', path='open.dmd')
    versions = dict.fromkeys(state, 0) if initial_versions is None else initial_versions
    program = build_dmd_program(hooks=hooks, feed_names=tuple(feeds), cycles=cycles,
        fake_updates=fake_updates, fake_sample_policy=fake_sample_policy,
        history_feed=history_feed, initial_versions=versions)
    if history_feed is not None:
        require(all(not tensor.requires_grad and tensor.grad_fn is None
                    for tensor in tensors(feeds[history_feed])),
                'initial history must contain only detached tensors', path='open.dmd')
    semantics = None
    if replica_batch_size is not None:
        from .replica_contract import ReplicaSemantics
        semantics = ReplicaSemantics({'student_forward': 'student', 'fake_forward': 'fake'},
            tuple(call.id for call in program.calls if call.backward_of), replica_batch_size)
    return Workload(program, hooks.bindings(program), state, dict(feeds), seed=seed,
        immutable_modules=tuple(dict.fromkeys(('teacher', *immutable_modules))),
        identity_extra={'dmd_builder': dict(program.parameters), 'user': dict(identity_extra or {})},
        replica_semantics=semantics).validate()

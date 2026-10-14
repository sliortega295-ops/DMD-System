"""Stage the pinned native W6 method at optimizer boundaries, not new DMD math.

world8 needs two independent B1 calls to represent global batch16. Running the
native update twice would instead commit Fake and Student twice. This opt-in
adapter changes only the two step_fn call sites into generator yield sites.
Every expression, branch, forward and returned metric comes from the original
function. A source SHA mismatch stops before any model work.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from contextlib import nullcontext

PINNED_NATIVE_UPDATE_SHA256 = 'a8455b053ad278fd87d2dd0b14a5d86e4eba6152935bda46a04b5650157f5f96'


def staged_native_update(native_update):
    source = inspect.getsource(native_update)
    if hashlib.sha256(source.encode()).hexdigest() != PINNED_NATIVE_UPDATE_SHA256:
        raise ValueError('W6 native update source identity changed')
    tree = ast.parse(textwrap.dedent(source))
    sites = []

    class StepBoundary(ast.NodeTransformer):
        def visit_Expr(self, node):
            call = node.value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name) and call.func.value.id == 'self'
                    and call.func.attr == 'step_fn'):
                if len(call.args) != 2 or call.keywords:
                    raise ValueError('unexpected native step callback signature')
                role = call.args[0]
                if not isinstance(role, ast.Constant) or role.value not in ('critic', 'student'):
                    raise ValueError('unexpected native optimizer role')
                sites.append(role.value)
                return ast.copy_location(ast.Expr(ast.Yield(ast.Tuple(call.args, ast.Load()))), node)
            return self.generic_visit(node)

    tree = StepBoundary().visit(tree)
    if sites != ['critic', 'student']:
        raise ValueError('native Fake->Student optimizer boundaries changed')
    tree.body[0].name = '_staged_native_update'
    ast.fix_missing_locations(tree)
    namespace = dict(native_update.__globals__)
    exec(compile(tree, inspect.getsourcefile(native_update) + ':h100-b16-boundaries', 'exec'), namespace)
    return namespace['_staged_native_update']


def run_native_microbatches(trainers, batches, *, backward, commit,
                           rng_scope=lambda slot: nullcontext(),
                           logical_slots=None):
    """Independent native B1 work, mean gradient, exactly one commit per role.

backward(role, loss/number_of_rows) must consume the graph immediately, while
the matching adapter remains selected. commit handles the existing distributed
mean, clipping and optimizer semantics. Two losses NEVER stay alive together.
RNG scopes are caller-owned immutable logical-rank streams.
"""
    if len(trainers) != len(batches) or len(trainers) not in (1, 2):
        raise ValueError('exactly one native canary row or two physical-world8 B1 rows required')
    logical_slots = list(range(len(trainers))) if logical_slots is None else list(logical_slots)
    if len(logical_slots) != len(trainers) or len(set(logical_slots)) != len(trainers):
        raise ValueError('logical sample slots must be independent')
    first = trainers[0]
    for trainer in trainers:
        if (trainer.config.critic_updates != 1 or trainer.config.num_inference_steps != 8
                or trainer.adapter is not first.adapter or trainer.discriminator is not first.discriminator
                or (trainer.generator_step, trainer.critic_step) != (first.generator_step, first.critic_step)):
            raise ValueError('W6 AR8 G1F1 shared state/version contract changed')
    native = type(first).update
    execute = staged_native_update(native)
    generators = [execute(trainer, *batch) for trainer, batch in zip(trainers, batches)]
    for expected in ('critic', 'student'):
        for slot, generator in zip(logical_slots, generators):
            with rng_scope(slot):
                role, loss = next(generator)
                if role != expected:
                    raise RuntimeError('native stage order changed')
                selected = 'fake_score' if role == 'critic' else 'student'
                if any(layer.role != selected for _, layer in first.adapter.layers):
                    raise RuntimeError('checkpoint backward adapter role mismatch')
                backward(role, loss / len(generators))
                if any(layer.role != selected for _, layer in first.adapter.layers):
                    raise RuntimeError('checkpoint replay changed active adapter')
                del loss
        commit(expected)
    results = []
    for slot, generator in zip(logical_slots, generators):
        with rng_scope(slot):
            try:
                next(generator)
            except StopIteration as finished:
                results.append(finished.value)
            else:
                raise RuntimeError('unexpected third optimizer boundary')
    return results

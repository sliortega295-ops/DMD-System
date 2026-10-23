"""Reusable selective-recompute policies on native PyTorch operations.

No model names or custom attention replacement. The entire native output tuple
is cached and accounted (not advertised as H3 compact output/LSE caching).
State/version/purity admission remains the public Program's responsibility;
PyTorch retains its cache-mutation detection. No cross-call cache reuse.
"""
from __future__ import annotations
from contextlib import contextmanager
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts
from distillschedule.core.contracts import require

POLICIES = ('save_matmul', 'save_attention')
_NAMES = {
    'save_matmul': ('mm', 'addmm', 'bmm'),
    'save_attention': ('_scaled_dot_product_flash_attention_for_cpu',
                       '_scaled_dot_product_flash_attention', '_scaled_dot_product_efficient_attention'),
}


def operations(policy):
    require(policy in POLICIES, 'unknown selective checkpoint policy', path='open.checkpoint')
    result = []
    for name in _NAMES[policy]:
        try: result.append(getattr(torch.ops.aten, name).default)
        except AttributeError: pass
    return frozenset(result)


def contexts(policy, saved, excluded, effects):
    """Measure real native saves; explicit replay visits are version-dependent.

    PyTorch can satisfy checkpoint backward directly from saved native outputs
    without invoking the policy on recompute (observed on CPU Torch 2.14).
    Therefore selective_save_requests is the observed cache-admission event;
    selective_replay_hits counts only policy callbacks actually visited and
    must never be synthesized from requests. Native forward-op counts and full
    gradient/optimizer parity provide the separate recompute-effect gate.
    """
    from .torch_backend import tensors, storage_key, physical_tensor
    ops = operations(policy)
    require(ops, 'native selective operations unavailable', path='open.checkpoint')
    def choose(ctx, op, *args, **kwargs):
        if op in ops:
            key = 'selective_replay_hits' if ctx.is_recompute else 'selective_save_requests'
            effects[key] = effects.get(key, 0) + 1
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    class ObserveCache(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if func in ops:
                new_bytes = 0
                for value in tensors(result):
                    key = storage_key(value)
                    if key not in excluded and key not in saved:
                        size = physical_tensor(value).untyped_storage().nbytes()
                        saved[key] = size
                        new_bytes += size
                effects['selective_native_cache_bytes'] = effects.get('selective_native_cache_bytes', 0) + new_bytes
            return result

    forward, replay = create_selective_checkpoint_contexts(choose, allow_cache_entry_mutation=False)
    @contextmanager
    def measured_forward():
        with ObserveCache(), forward:
            yield
    return measured_forward(), replay


def native_effect_evidence(implementation, effects):
    """Fail closed on strategies whose target native op did not actually run.

    The candidate inventory is intentionally conservative and may include an
    operator which is supported by installed PyTorch but absent from a user's
    program. Admission uses measured native forward saves/bytes on every rank.
    PyTorch 2.14 can skip replay callbacks, so replay hits are NOT a gate.
    """
    local = implementation.get('local', implementation)
    policy = local.get('checkpoint_policy', 'recompute')
    owner_options = bool(local.get('owner_execution'))
    if policy == 'recompute' and not owner_options:
        return {'status': 'NOT_REQUIRED', 'policy': policy, 'active': True}
    rows = effects.get('replica_effects', [effects])
    if owner_options:
        from .owner_execution import evidence as owner_evidence
        owner = owner_evidence(local, rows)
        if policy == 'recompute':
            return {'status': owner['status'], 'policy': policy, 'active': owner['active'],
                    'owner_execution': owner}
    evidence = [{'native_save_requests': int(row.get('selective_save_requests', 0)),
                 'native_cached_bytes': int(row.get('selective_native_cache_bytes', 0)),
                 'explicit_replay_callbacks': int(row.get('selective_replay_hits', 0))}
                for row in rows]
    active = bool(evidence) and all(row['native_save_requests'] > 0 and
                                    row['native_cached_bytes'] > 0 for row in evidence)
    result = {'status': 'ACTIVE_NATIVE_SELECTIVE_SAVE' if active else 'INAPPLICABLE_NATIVE_OP',
              'policy': policy, 'active': active, 'rank_evidence': evidence}
    if owner_options:
        result.update(owner_execution=owner, active=active and owner['active'])
        if not owner['active']: result['status'] = owner['status']
    return result

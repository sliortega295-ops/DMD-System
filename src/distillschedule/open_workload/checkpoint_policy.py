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
    """Measure actual cache storage under SAC; never keep extra tensor references."""
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

"""Cycle-local finite checks without rescanning unchanged owned tensors."""
from __future__ import annotations

import weakref


class FiniteCheck:
    """Remember successful checks using live tensor identity and mutation tokens.

    Updated parameters, optimizer slots, gradients and outputs are still checked
    each cycle. Frozen state is scanned once, and again if its version, storage
    or layout changes. ``audit=True`` also detects raw writes that bypass tensor
    version counters. No tensor or model-sized storage is retained by this cache.
    """

    def __init__(self):
        self._checked = {}

    def __call__(self, state, outputs, *, audit=False):
        import torch
        from distillschedule.open_workload.torch_backend import (
            _immutable_parameter_state, physical_tensor, tensors,
        )

        checks, seen, pending = {}, set(), {}
        with torch.no_grad():
            # Outputs take precedence over cached state when both reference
            # the same tensor; every output must be checked on every call.
            for is_output, tree in ((True, outputs), (False, state)):
                for tensor in tensors(tree):
                    key = id(tensor)
                    if key in seen or not (tensor.is_floating_point() or tensor.is_complex()):
                        continue
                    seen.add(key)
                    token = _immutable_parameter_state(tensor)
                    cacheable = None not in token[1][1]
                    previous = self._checked.get(key)
                    if (not audit and not is_output and cacheable and previous is not None
                            and previous[0]() is tensor and previous[1] == token):
                        pending[key] = previous
                        continue
                    value = physical_tensor(tensor)
                    finite = torch.isfinite(value).all()
                    if value.device in checks:
                        checks[value.device].logical_and_(finite)
                    else:
                        checks[value.device] = finite
                    if cacheable and not is_output:
                        pending[key] = (weakref.ref(tensor), token)
            # One scalar read per device; all state is local-shard only.
            valid = all([bool(check) for check in checks.values()])
        self._checked = pending if valid else {}
        return valid

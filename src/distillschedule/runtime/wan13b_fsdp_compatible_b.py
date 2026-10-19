"""Observed scope guard for R2's independent FSDP-compatible B subset.

No tensor/parameter caches, FSDP hooks, collectives, residency or update changes.
Every PEFT call consumes the current block's freshly unsharded parameters.
"""
from __future__ import annotations

from typing import Any


def install(model: Any, *, role: str, placement: str, linear_type: Any = None) -> dict[str, Any]:
    if placement != 'world16-fsdp' or role not in {'generator', 'critic'}:
        raise ValueError('compatible B scope requires world16 FSDP Generator/Critic')
    import torch
    if linear_type is None:
        from peft.tuners.lora.layer import Linear
        linear_type = Linear
    receipt = dict(enabled=True, role=role, placement=placement,
                   lifetime='CALL_LOCAL_CURRENT_UNSHARDED_PARAMETERS_NO_WEIGHT_OR_CAST_CACHE',
                   parameter_check_calls=0, grad_enabled_calls=0, no_grad_calls=0,
                   adapter_disabled_calls=0, contract_failures=0, cached_parameter_bytes=0)
    linears = [m for m in model.modules() if isinstance(m, linear_type)]
    if not linears:
        raise RuntimeError('compatible B found no PEFT Linear modules')
    for child in linears:
        # Only immutable host metadata survives setup, never a tensor reference.
        expected = {n: (p.dtype, p.requires_grad) for n, p in child.named_parameters()}
        if not all(dtype == torch.float32 for name, (dtype, grad) in expected.items() if grad):
            raise ValueError('compatible B requires original FP32 LoRA')

        def check(module: Any, args: Any, *, _expected: Any = expected) -> None:
            actual = dict(module.named_parameters())
            disabled = bool(getattr(module, 'disable_adapters', False))
            # PEFT disables inactive LoRA requires_grad; FSDP may restore the
            # original full-param metadata when unsharding. Neither is consumed
            # on adapter-off Teacher calls. Active/grad calls remain strict.
            valid = actual.keys() == _expected.keys() and all(
                not hasattr(p, 'device_mesh') and p.dtype == _expected[n][0]
                and (p.requires_grad == _expected[n][1] or (disabled and _expected[n][1]))
                for n, p in actual.items())
            if not valid:
                receipt['contract_failures'] += 1
                raise RuntimeError('compatible B called outside FSDP materialization/dtype scope')
            receipt['parameter_check_calls'] += 1
            receipt['adapter_disabled_calls'] += int(disabled)
            receipt['grad_enabled_calls' if torch.is_grad_enabled() else 'no_grad_calls'] += 1
        child.register_forward_pre_hook(check)
    receipt['instrumented_module_count'] = len(linears)
    return receipt


def validate(build: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for role in ('generator', 'critic'):
        meta = build[role]['metadata']
        row = meta.get('fsdp_compatible_b', {})
        if (row.get('enabled') is not True or row.get('placement') != 'world16-fsdp'
                or row.get('contract_failures') != 0 or row.get('cached_parameter_bytes') != 0
                or any(int(row.get(k, 0)) <= 0 for k in
                       ('instrumented_module_count', 'parameter_check_calls', 'grad_enabled_calls', 'no_grad_calls'))):
            raise AssertionError(f'{role}: compatible B has no actual safe FSDP calls')
        if (role == 'generator' and row.get('adapter_disabled_calls') != 0
                or role == 'critic' and int(row.get('adapter_disabled_calls', 0)) <= 0):
            raise AssertionError(f'{role}: compatible B adapter/Teacher role scope changed')
        nr = meta['modulated_norm_fusion']
        if nr.get('native_chunk_tokens') != 512 or nr.get('fsdp_compatible_b') is not True:
            raise AssertionError('compatible B normalization changed chunk/placement')
        result[role] = row
    return result

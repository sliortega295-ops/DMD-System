"""Default-off exact unit-scale LoRA sum, preserving native module calls.

The frozen Linear output has no other consumer before the native sum. Linear
backward saves its input/weight, not its output. Optional input selection only
composes the separately admitted activation-cast elision; it does not cast any
parameter, change the scale, or fuse a matmul.
"""
from collections import Counter
import functools
import hashlib
from pathlib import Path

import torch


class UnitScaleInplaceLoRA:
    def __init__(self, native, adapter, *, enabled=False, input_selector=None):
        self.enabled = enabled
        self.counts = Counter()
        self.cls = native.DualLoRALinear
        self.original = self.cls.forward
        self.input_selector = input_selector
        if not enabled:
            return
        if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() != (
                '77744f5ec391be813d42ebaf6a2692c18ea58f5ca7fb41398fd287d2a177ca62'):
            raise ValueError('exact private native LoRA source required')
        eligible = {id(layer) for _, layer in adapter.layers}
        if not eligible or any(layer.scale != 1.0 or any(
                p.requires_grad for p in layer.base.parameters()) for _, layer in adapter.layers):
            raise ValueError('frozen base and exact unit LoRA scale required')
        original, counts = self.original, self.counts

        @functools.wraps(original)
        def forward(layer, value):
            if id(layer) not in eligible or layer.role == 'teacher':
                return original(layer, value)
            if layer.scale != 1.0:
                raise RuntimeError('unit LoRA scale changed')
            pair = layer.adapters[layer.role]
            result = layer.base(value)
            selected = (value.to(pair['A'].weight.dtype) if input_selector is None
                        else input_selector(value, pair['A'].weight))
            update = pair['B'](pair['A'](selected))
            result.add_(update.to(result.dtype))
            counts['physical_dual_lora_forward'] += 1
            counts['grad_forward' if torch.is_grad_enabled() else 'nograd_forward'] += 1
            counts['elided_unit_multiply'] += 1
            counts['inplace_base_add'] += 1
            counts['elided_full_width_intermediate_logical_bytes'] += (
                2 * result.numel() * result.element_size())
            return result

        self.cls.forward = forward

    def finish_cycle(self):
        if self.enabled and not self.counts['physical_dual_lora_forward']:
            raise RuntimeError('LoRA sum optimization never executed')
        result = {'enabled': self.enabled, 'mechanism': 'unit-scale in-place native LoRA sum',
                  'counters': dict(self.counts), 'A_B_module_calls_preserved': True,
                  'input_selector_composed': self.input_selector is not None,
                  'parameter_dtype_changed': False, 'optimizer_or_loss_changed': False,
                  'elided_bytes_scope': 'logical allocation bound, not measured HBM traffic'}
        self.counts.clear()
        return result

    def close(self):
        if self.enabled:
            self.cls.forward = self.original

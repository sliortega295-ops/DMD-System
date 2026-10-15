"""Default-off removal of a redundant activation cast under BF16 autocast.

Keep native A/B module calls, FP32 parameters, autograd, role and scale/add.
CPU autocast is supported for executable unit tests; target runs are CUDA-only.
"""
from collections import Counter
import functools
import hashlib
import inspect
from pathlib import Path
import textwrap

DMD_SHA = '77744f5ec391be813d42ebaf6a2692c18ea58f5ca7fb41398fd287d2a177ca62'


def select_input(value, weight, counts):
    import torch
    device = value.device.type
    phase = 'grad' if torch.is_grad_enabled() else 'nograd'
    if (device in ('cpu', 'cuda') and value.dtype == torch.bfloat16
            and weight.dtype == torch.float32
            and torch.is_autocast_enabled(device)
            and torch.get_autocast_dtype(device) == torch.bfloat16):
        counts[device + '_' + phase + '_elided'] += 1
        counts['avoided_FP32_activation_bytes'] += value.numel() * 4
        return value
    counts[device + '_' + phase + '_original_cast'] += 1
    return value.to(weight.dtype)


class NativeLoRAInputCast:
    def __init__(self, module):
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != DMD_SHA:
            raise ValueError('PINNED_NATIVE_DMD_SOURCE_MISMATCH')
        cls = module.DualLoRALinear
        original = cls.forward
        if getattr(original, '_h3_lora_input_cast_elision', False):
            raise ValueError('NATIVE_LORA_INPUT_CAST_ALREADY_INSTALLED')
        body = textwrap.dedent(inspect.getsource(original))
        site = 'value.to(pair["A"].weight.dtype)'
        if body.count(site) != 1:
            raise ValueError('PINNED_NATIVE_LORA_INPUT_CAST_SITE_MISMATCH')
        self.counts = Counter()
        self.previous = Counter()

        def choose(value, weight):
            return select_input(value, weight, self.counts)

        namespace = dict(original.__globals__, _h100_native_lora_input=choose)
        exec(compile(body.replace(site, '_h100_native_lora_input(value, pair["A"].weight)'),
                     'pinned_native_lora_activation_cast_elision', 'exec'), namespace)
        cls.forward = functools.wraps(original)(namespace[original.__name__])
        cls.forward._h3_lora_input_cast_elision = True
        self.original = original
        self.cls = cls

    def finish_cycle(self):
        current = self.counts.copy()
        delta = current - self.previous
        self.previous = current
        return dict(enabled=True, source_sha256=DMD_SHA, counts=dict(current),
                    cycle_delta=dict(delta), parameter_dtype_changed=False,
                    native_A_B_module_calls_preserved=True,
                    scope='BF16 activation direct to native FP32-weight Linear under BF16 autocast only')

    def restore(self):
        self.cls.forward = self.original


def install(module, *, enabled=False):
    if not enabled:
        return None
    return NativeLoRAInputCast(module)

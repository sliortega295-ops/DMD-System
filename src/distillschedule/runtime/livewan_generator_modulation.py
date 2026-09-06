"""C32 canary-only exact no-grad G modulation; no installation on import."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect
import textwrap
from typing import Any

SOURCE_SHA256 = 'f244152360d40d6745e28bed4add081604312d04fe2f342842a32432f4bd20ad'


def build(torch: Any):
    import triton
    import triton.language as tl

    @triton.jit
    def modulation(X, SCALE, SHIFT, Y, N: tl.constexpr, D: tl.constexpr,
                   BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + i, i < N, other=0).to(tl.float32)
        scale = tl.load(SCALE + i % D).to(tl.float32)
        shift = tl.load(SHIFT + i % D).to(tl.float32)
        # Eager G uses FP32 scale (unlike frozen Score's BF16 scale).
        one = tl.full((BLOCK,), 1.0, tl.float32)
        factor = tl.inline_asm_elementwise('add.rn.f32 $0, $1, $2;',
            '=f,f,f', [scale, one], dtype=tl.float32, is_pure=True, pack=1)
        product = tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;',
            '=f,f,f', [x, factor], dtype=tl.float32, is_pure=True, pack=1)
        result = tl.inline_asm_elementwise('add.rn.f32 $0, $1, $2;',
            '=f,f,f', [product, shift], dtype=tl.float32, is_pure=True, pack=1)
        tl.store(Y + i, result, i < N)

    receipt = dict(name='generator_nograd_fp32_modulation_v1', calls=0,
        contract_failures=0, native_norm_preserved=True, grad_path_changed=False,
        scope='single_prepare_rollout_generator_only', scale_dtype='float32',
        shift_dtype='float32', output_dtype='float32', fma=False,
        source_sha256=SOURCE_SHA256, shape_dtype_histogram={}, triton=triton.__version__)

    def op(normalized: Any, scale: Any, shift: Any):
        if (torch.is_grad_enabled() or not normalized.is_cuda
            or scale.device != normalized.device or shift.device != normalized.device
            or normalized.dtype not in (torch.bfloat16, torch.float32)
            or scale.dtype != torch.float32 or shift.dtype != torch.float32
            or normalized.ndim != 3 or normalized.shape[0] != 1
            or normalized.shape[-1] != 1536 or scale.shape != (1, 1, 1536)
            or shift.shape != scale.shape or scale.stride(-1) != 1
            or shift.stride(-1) != 1 or not normalized.is_contiguous()):
            receipt['contract_failures'] += 1
            raise RuntimeError('C32 no-grad G modulation source dtype/shape/stride changed')
        out = torch.empty(normalized.shape, device=normalized.device, dtype=torch.float32)
        modulation[(triton.cdiv(normalized.numel(), 1024),)](
            normalized, scale, shift, out, N=normalized.numel(), D=1536,
            BLOCK=1024, num_warps=4, enable_fp_fusion=False)
        receipt['calls'] += 1
        key = f'{normalized.dtype}/{tuple(normalized.shape)}'
        receipt['shape_dtype_histogram'][key] = receipt['shape_dtype_histogram'].get(key, 0) + 1
        return out

    return op, receipt


@contextmanager
def canary_scope(torch: Any, student: Any, op: Any, *, enabled: bool = False):
    if not enabled:
        yield
        return
    if torch.is_grad_enabled():
        raise RuntimeError('C32 scope requires no_grad')
    from wanstreamer import blockcausal as bc
    original = bc._modulate
    sha = hashlib.sha256(textwrap.dedent(inspect.getsource(original)).strip().encode()).hexdigest()
    if sha != SOURCE_SHA256:
        raise RuntimeError('C32 _modulate source mismatch or recursive scope')
    if (len(student.blocks) != 30 or hasattr(student, 'causal_layers')
        or not all(p.is_cuda and p.dtype == torch.float32 and p.requires_grad
                   and not hasattr(p, 'to_local') for p in student.parameters())):
        raise RuntimeError('C32 requires unwrapped fully trainable FP32 resident G')

    def fused(x, norm, shift, scale, per_frame, n_frames):
        if per_frame or n_frames != 1 or torch.is_grad_enabled():
            raise RuntimeError('C32 per-frame/grad execution not allowed')
        # Keep the actual PyTorch norm, including its exact reduction and dtype.
        normalized = norm(x)
        return op(normalized, scale, shift)

    bc._modulate = fused
    try:
        yield
    finally:
        bc._modulate = original

"""Opt-in C33 no-grad Generator residual gate; native grad path is untouched."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect
import textwrap
from typing import Any

SOURCE_SHA256 = '4665c81a5dfbb7b7b75590c94a9c3acb2027bf7474f708033d1d1c96583ce527'
ATTRIBUTE = '_distillschedule_generator_gate'
PLAN_CONTRACT = dict(
    enabled=True, name='generator_nograd_fp32_residual_gate_v1',
    scope='single_prepare_rollout_generator_only', source_sha256=SOURCE_SHA256,
    product_dtype='float32', output_dtype='float32', fma=False,
    out_of_place=True, grad_path_changed=False,
)


def build(torch: Any):
    import triton
    import triton.language as tl

    @triton.jit
    def gate(X, Y, G, OUT, N: tl.constexpr, D: tl.constexpr,
             XS: tl.constexpr, XD: tl.constexpr, YS: tl.constexpr,
             YD: tl.constexpr, OS: tl.constexpr, OD: tl.constexpr,
             BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        # Iterate along output's physically contiguous dimension, including
        # the transposed patch-embedding residual at the first block.
        if OD == 1:
            row, col = i // D, i % D
        else:
            row, col = i % (N // D), i // (N // D)
        x = tl.load(X + row * XS + col * XD, i < N, other=0).to(tl.float32)
        y = tl.load(Y + row * YS + col * YD, i < N, other=0).to(tl.float32)
        g = tl.load(G + col, i < N, other=0).to(tl.float32)
        product = tl.inline_asm_elementwise(
            'mul.rn.f32 $0, $1, $2;', '=f,f,f', [y, g],
            dtype=tl.float32, is_pure=True, pack=1)
        result = tl.inline_asm_elementwise(
            'add.rn.f32 $0, $1, $2;', '=f,f,f', [x, product],
            dtype=tl.float32, is_pure=True, pack=1)
        tl.store(OUT + row * OS + col * OD, result, i < N)

    receipt = dict(PLAN_CONTRACT, calls=0, contract_failures=0,
                   shape_dtype_stride_histogram={}, triton=triton.__version__)

    def dense_b1(t):
        return (t.ndim == 3 and t.shape[0] == 1 and t.shape[2] == 1536
                and (t.stride()[1:] in ((1536, 1), (1, t.shape[1]))))

    def op(x: Any, y: Any, g: Any):
        if (torch.is_grad_enabled() or not x.is_cuda or y.device != x.device
            or g.device != x.device or not dense_b1(x) or not dense_b1(y)
            or x.shape != y.shape or x.dtype not in (torch.bfloat16, torch.float32)
            or y.dtype != torch.bfloat16 or g.dtype != torch.float32
            or g.shape != (1, 1, 1536) or g.stride(-1) != 1):
            receipt['contract_failures'] += 1
            raise RuntimeError('C33 source dtype/shape/stride/grad contract changed')
        out = torch.empty_like(x, dtype=torch.float32)
        gate[(triton.cdiv(x.numel(), 1024),)](
            x, y, g, out, N=x.numel(), D=1536,
            XS=x.stride(1), XD=x.stride(2), YS=y.stride(1), YD=y.stride(2),
            OS=out.stride(1), OD=out.stride(2), BLOCK=1024, num_warps=4,
            enable_fp_fusion=False)
        receipt['calls'] += 1
        key = f'{x.dtype}/{tuple(x.shape)}/{tuple(x.stride())}/{tuple(y.stride())}'
        hist = receipt['shape_dtype_stride_histogram']
        hist[key] = hist.get(key, 0) + 1
        return out

    return op, receipt


@contextmanager
def canary_scope(torch: Any, student: Any, op: Any, *, enabled=False):
    if not enabled:
        yield
        return
    if torch.is_grad_enabled():
        raise RuntimeError('C33 scope requires no_grad')
    from wanstreamer import blockcausal as bc
    original = bc._gate
    sha = hashlib.sha256(textwrap.dedent(inspect.getsource(original)).strip().encode()).hexdigest()
    if sha != SOURCE_SHA256:
        raise RuntimeError('C33 _gate source mismatch or recursive scope')
    if (len(student.blocks) != 30 or hasattr(student, 'causal_layers')
        or not all(p.is_cuda and p.dtype == torch.float32 and p.requires_grad
                   and not hasattr(p, 'to_local') for p in student.parameters())):
        raise RuntimeError('C33 requires unwrapped fully trainable FP32 resident G')

    def fused(x, y, g, per_frame, n_frames):
        if per_frame or n_frames != 1 or torch.is_grad_enabled():
            raise RuntimeError('C33 per-frame/grad execution not allowed')
        return op(x, y, g)

    bc._gate = fused
    try:
        yield
    finally:
        bc._gate = original

"""C29 standalone-only Generator RMSNorm epilogue experiment.

Keep native pow/mean reduction, BF16 intermediate rounding and FP32 affine/output.
Nothing is installed by production plans. No CUDA import/allocation on import.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect
import types
from typing import Any

REFERENCE_CLASS_SHA256 = '31bf960deb2ee55c33ed4656a0ca3e9e4e9632cd41fd04a8677e70e86edbff2e'


def validate_inputs(torch: Any, x: Any, weight: Any, eps: float) -> None:
    if torch.is_grad_enabled():
        raise RuntimeError('Generator RMS epilogue requires no_grad')
    if (not x.is_cuda or not weight.is_cuda or x.device != weight.device
        or x.dtype != torch.bfloat16 or weight.dtype != torch.float32
        or x.ndim != 3 or x.shape[0] != 1 or x.shape[-1] != 1536
        or weight.shape != (1536,) or not x.is_contiguous()
        or not weight.is_contiguous() or float(eps) != 1e-6):
        raise RuntimeError('Generator RMS epilogue source dtype/shape/device/epsilon mismatch')


def build_nograd_generator_rms_epilogue(torch: Any) -> tuple[Any, dict[str, Any]]:
    import triton
    import triton.language as tl

    @triton.jit
    def epilogue(X, V, W, Y, N: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + i, i < N, other=0).to(tl.float32)
        var = tl.load(V + i // D, i < N, other=0)
        weight = tl.load(W + i % D, i < N, other=0)
        inv = tl.rsqrt(var + 1.0e-6)
        normalized = (x * inv).to(tl.bfloat16).to(tl.float32)
        tl.store(Y + i, normalized * weight, i < N)

    receipt: dict[str, Any] = dict(name='generator_nograd_rms_epilogue_v1',
        status='CANARY_ONLY_NOT_PRODUCTION_INSTALLED', calls=0, contract_failures=0,
        native_pow_mean_preserved=True, input_dtype='bfloat16', output_dtype='float32',
        affine_dtype='float32', intermediate_bf16_rounding=True, grad_path_changed=False,
        shape_histogram={}, triton_version=triton.__version__)

    def op(x: Any, weight: Any, eps: float) -> Any:
        try:
            validate_inputs(torch, x, weight, eps)
        except RuntimeError:
            receipt['contract_failures'] += 1
            raise
        # Exactly the original reduction operations/order; only fuse the tail.
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        y = torch.empty(x.shape, device=x.device, dtype=torch.float32)
        epilogue[(triton.cdiv(x.numel(), 1024),)](x, var, weight, y,
            N=x.numel(), D=x.shape[-1], BLOCK=1024, num_warps=4, enable_fp_fusion=False)
        receipt['calls'] += 1
        shape = 'x'.join(str(int(v)) for v in x.shape)
        receipt['shape_histogram'][shape] = receipt['shape_histogram'].get(shape, 0) + 1
        return y

    return op, receipt


@contextmanager
def model_epilogue_canary_scope(torch: Any, student: Any, op: Any, *, enabled: bool = False):
    """Temporary instance-level canary patch; never affects a grad-enabled call."""
    if not enabled:
        yield
        return
    if torch.is_grad_enabled():
        raise RuntimeError('Generator RMS canary requires no_grad')
    from wan.modules.model import WanRMSNorm
    observed = hashlib.sha256(inspect.getsource(WanRMSNorm).encode()).hexdigest()
    if observed != REFERENCE_CLASS_SHA256:
        raise RuntimeError('Generator RMS source mismatch')
    modules = [m for m in student.modules() if type(m) is WanRMSNorm]
    if len(modules) != 120 or len(student.blocks) != 30:
        raise RuntimeError('Generator RMS module topology mismatch')
    if any('forward' in m.__dict__ or m.weight.dtype != torch.float32
           or not m.weight.requires_grad for m in modules):
        raise RuntimeError('Generator RMS existing instance override/affine mismatch')

    def forward(module: Any, x: Any) -> Any:
        return op(x, module.weight, module.eps)

    installed = []
    try:
        for m in modules:
            m.forward = types.MethodType(forward, m)
            installed.append(m)
        yield
    finally:
        for m in installed:
            del m.forward

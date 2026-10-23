"""C29 opt-in Generator no-grad RMSNorm epilogue experiment.

Keep native pow/mean reduction, BF16 intermediate rounding and FP32 affine/output.
Nothing is installed by default plans. No CUDA import/allocation on import.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect
import types
from typing import Any

REFERENCE_CLASS_SHA256 = '31bf960deb2ee55c33ed4656a0ca3e9e4e9632cd41fd04a8677e70e86edbff2e'
ATTRIBUTE = '_distillschedule_generator_rms_epilogue'
PLAN_CONTRACT = dict(enabled=True, name='generator_nograd_rms_epilogue_v1',
    scope='single_prepare_rollout_generator_only', native_pow_mean_preserved=True,
    input_dtype='bfloat16', affine_dtype='float32', output_dtype='float32',
    intermediate_bf16_rounding=True, grad_path_changed=False,
    source_class_sha256=REFERENCE_CLASS_SHA256)
CAST_SQUARE_PLAN_CONTRACT = dict(PLAN_CONTRACT,
    name='generator_nograd_rms_cast_square_v1', native_pow_mean_preserved=False,
    native_mean_preserved=True, fp32_square_preserved=True, fused_cast_square=True)


def validate_inputs(torch: Any, x: Any, weight: Any, eps: float) -> None:
    if torch.is_grad_enabled():
        raise RuntimeError('Generator RMS epilogue requires no_grad')
    if (not x.is_cuda or not weight.is_cuda or x.device != weight.device
        or x.dtype != torch.bfloat16 or weight.dtype != torch.float32
        or x.ndim != 3 or x.shape[0] != 1 or x.shape[-1] != 1536
        or weight.shape != (1536,) or not x.is_contiguous()
        or not weight.is_contiguous() or float(eps) != 1e-6):
        raise RuntimeError('Generator RMS epilogue source dtype/shape/device/epsilon mismatch')


def build_cast_square(torch: Any) -> Any:
    """C31 canary primitive: FP32 square without an intermediate FP32 cast tensor."""
    import triton
    import triton.language as tl

    @triton.jit
    def cast_square(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(X + i, i < N, other=0).to(tl.float32)
        tl.store(Y + i, value * value, i < N)

    def square(x: Any) -> Any:
        if (torch.is_grad_enabled() or not x.is_cuda or x.dtype != torch.bfloat16
            or not x.is_contiguous() or x.ndim != 3 or x.shape[0] != 1
            or x.shape[-1] != 1536):
            raise RuntimeError('C31 cast-square requires source-pinned no-grad BF16 G input')
        y = torch.empty(x.shape, device=x.device, dtype=torch.float32)
        cast_square[(triton.cdiv(x.numel(), 1024),)](
            x, y, N=x.numel(), BLOCK=1024, num_warps=4, enable_fp_fusion=False)
        return y

    return square


def build_nograd_generator_rms_epilogue(torch: Any, *, fuse_cast_square: bool = False) -> tuple[Any, dict[str, Any]]:
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
    square = build_cast_square(torch) if fuse_cast_square else None
    if fuse_cast_square:
        receipt.update(name='generator_nograd_rms_cast_square_v1',
            native_pow_mean_preserved=False, native_mean_preserved=True,
            fp32_square_preserved=True, fused_cast_square=True, square_calls=0)

    def op(x: Any, weight: Any, eps: float) -> Any:
        try:
            validate_inputs(torch, x, weight, eps)
        except RuntimeError:
            receipt['contract_failures'] += 1
            raise
        # Exactly the original reduction operations/order; only fuse the tail.
        if square is None:
            var = x.float().pow(2).mean(dim=-1, keepdim=True)
        else:
            var = square(x).mean(dim=-1, keepdim=True)
            receipt['square_calls'] += 1
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


class GeneratorRMSEpilogue:
    def __init__(self, torch: Any, student: Any, *, fuse_cast_square: bool = False):
        self.student = student
        self.op, self.kernel_receipt = build_nograd_generator_rms_epilogue(
            torch, fuse_cast_square=fuse_cast_square)
        self.entries = self.exits = self.failures = 0
        self.active = False
        self.scope_calls: list[int] = []

    @contextmanager
    def scope(self, torch: Any):
        if self.active:
            self.failures += 1
            raise RuntimeError('recursive Generator RMS epilogue scope')
        self.active = True
        self.entries += 1
        before = self.kernel_receipt['calls']
        try:
            with model_epilogue_canary_scope(torch, self.student, self.op, enabled=True):
                yield
        except BaseException:
            self.failures += 1
            raise
        finally:
            self.scope_calls.append(self.kernel_receipt['calls'] - before)
            self.active = False
            self.exits += 1

    def receipt(self):
        return dict(self.kernel_receipt, enabled=True, status='INSTALLED', execution_status='PASS',
            source_class_sha256=REFERENCE_CLASS_SHA256,
            scope_entries=self.entries, scope_exits=self.exits,
            scope_forward_calls=self.scope_calls, active=self.active,
            scope_failures=self.failures)


def install(torch: Any, student: Any, *, fuse_cast_square: bool = False):
    if (str(torch.__version__) != '2.10.0+cu128' or hasattr(student, ATTRIBUTE)
        or hasattr(student, 'causal_layers') or len(student.blocks) != 30):
        raise RuntimeError('C29 requires validated unwrapped resident 30-block Generator')
    if not all(p.is_cuda and p.dtype == torch.float32 and p.requires_grad
               and not hasattr(p, 'to_local') for p in student.parameters()):
        raise RuntimeError('C29 Generator dtype/trainability/placement changed')
    helper = GeneratorRMSEpilogue(torch, student, fuse_cast_square=fuse_cast_square)
    setattr(student, ATTRIBUTE, helper)
    return helper


@contextmanager
def model_runtime_scope(torch: Any, student: Any):
    helper = getattr(student, ATTRIBUTE, None)
    if helper is None:
        yield
    else:
        with helper.scope(torch):
            yield


def runtime_receipt_errors(r: dict[str, Any], *, rank: int, cycles: int,
                           fuse_cast_square: bool = False, request=None) -> list[str]:
    from distillschedule.search.livewan_request import LiveWanRequest
    allowed = (request or LiveWanRequest()).kernel_scope_calls("rms")
    if r.get("name") == "generator_nograd_fused_rms_row_v1":
        if request is not None and request != LiveWanRequest():
            return ["C51 fused RMS supports only the measured 81-frame geometry"]
        from distillschedule.runtime.livewan_generator_fused_rms import runtime_receipt_errors as fused_errors
        return fused_errors(r, rank=rank, cycles=cycles)
    if rank >= 8:
        return [] if r.get('execution_status') == 'NOT_APPLICABLE' else ['C29 changed Score scope']
    scope_calls = r.get('scope_forward_calls', [])
    expected = CAST_SQUARE_PLAN_CONTRACT if fuse_cast_square else PLAN_CONTRACT
    if fuse_cast_square and (r.get('square_calls') != r.get('calls')
        or r.get('native_mean_preserved') is not True
        or r.get('fp32_square_preserved') is not True
        or r.get('fused_cast_square') is not True):
        return ['C31 actual cast-square/native-mean contract failed']
    if not fuse_cast_square and r.get('fused_cast_square', False):
        return ['C29/C30 cannot silently enable C31']
    if (r.get('name') != expected['name'] or r.get('execution_status') != 'PASS'
        or r.get('source_class_sha256') != REFERENCE_CLASS_SHA256
        or r.get('scope_entries') != 10 * cycles or r.get('scope_exits') != 10 * cycles
        or r.get('active') is not False or r.get('scope_failures') != 0
        or r.get('contract_failures') != 0 or r.get('grad_path_changed') is not False
        or r.get('native_pow_mean_preserved') is not expected['native_pow_mean_preserved']
        or r.get('input_dtype') != 'bfloat16' or r.get('affine_dtype') != 'float32'
        or r.get('output_dtype') != 'float32' or r.get('intermediate_bf16_rounding') is not True
        or len(scope_calls) != 10 * cycles
        or any(v not in allowed for v in scope_calls)
        or r.get('calls') != sum(scope_calls)):
        return ['C29 actual norm call/scope/precision contract failed']
    return []

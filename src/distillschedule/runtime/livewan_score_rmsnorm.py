"""Opt-in frozen Score RMSNorm candidate; not installed by any default plan.

The canary targets the pinned WanRMSNorm's two BF16 roundings, not a LoRA
precision change. Reduction ordering is validated numerically, not assumed
bitwise identical. No monkeypatch, CUDA import, or allocation occurs on import.
"""
from __future__ import annotations

from typing import Any


def validate_frozen_rms_inputs(torch: Any, x: Any, weight: Any, eps: float) -> None:
    if (
        not x.is_cuda or not weight.is_cuda or x.device != weight.device
        or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
        or x.ndim != 3 or x.shape[-1] not in {1536, 5120}
        or x.shape[0] not in {1, 2} or weight.shape != (x.shape[-1],)
        or not x.is_contiguous() or not weight.is_contiguous()
        or weight.requires_grad or float(eps) != 1e-6
    ):
        raise RuntimeError("frozen Score RMSNorm device/dtype/shape/affine/epsilon contract changed")


def build_frozen_score_rmsnorm(torch: Any) -> tuple[Any, dict[str, Any]]:
    """Build a standalone canary operator. Production integration is separate."""
    import triton
    import triton.language as tl

    @triton.jit
    def forward_kernel(X, W, Y, R, D: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        x = tl.load(X + row * D + col, col < D, other=0).to(tl.float32)
        weight = tl.load(W + col, col < D, other=0).to(tl.float32)
        variance = tl.sum(x * x, 0) / D
        inv = tl.rsqrt(variance + 1.0e-6)
        # Source: _norm(x.float()).type_as(x) * weight.
        normalized = (x * inv).to(tl.bfloat16).to(tl.float32)
        y = (normalized * weight).to(tl.bfloat16)
        tl.store(Y + row * D + col, y, col < D)
        tl.store(R + row, inv)

    @triton.jit
    def backward_kernel(X, W, R, DY, DX, D: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        x = tl.load(X + row * D + col, col < D, other=0).to(tl.float32)
        w = tl.load(W + col, col < D, other=0).to(tl.float32)
        dy = tl.load(DY + row * D + col, col < D, other=0).to(tl.float32)
        inv = tl.load(R + row)
        # Backward through BF16 affine multiplication rounds before .float().
        g = (dy * w).to(tl.bfloat16).to(tl.float32)
        dot = tl.sum(g * x, 0)
        variance_grad = ((-0.5 * (inv * inv * inv)) * dot) / D
        dx = (g * inv + (2.0 * x) * variance_grad).to(tl.bfloat16)
        tl.store(DX + row * D + col, dx, col < D)

    receipt: dict[str, Any] = {
        "name": "score_frozen_bf16_rmsnorm_v1",
        "status": "CANARY_ONLY_NOT_PRODUCTION_INSTALLED",
        "forward_calls": 0, "backward_calls": 0, "shape_histogram": {},
        "triton_version": triton.__version__, "contract_failures": 0,
        "intermediate_bf16_rounding": True, "weight_gradient": False,
    }

    class FrozenScoreRMSNorm(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, x: Any, weight: Any, eps: float) -> Any:
            try:
                validate_frozen_rms_inputs(torch, x, weight, eps)
            except RuntimeError:
                receipt["contract_failures"] += 1
                raise
            d = int(x.shape[-1])
            rows = x.numel() // d
            y = torch.empty_like(x)
            inv = torch.empty((rows,), device=x.device, dtype=torch.float32)
            forward_kernel[(rows,)](
                x, weight, y, inv, D=d, BLOCK=triton.next_power_of_2(d),
                num_warps=4 if d == 1536 else 8, enable_fp_fusion=False,
            )
            ctx.save_for_backward(x, weight, inv)
            receipt["forward_calls"] += 1
            shape = "x".join(str(int(v)) for v in x.shape)
            receipt["shape_histogram"][shape] = receipt["shape_histogram"].get(shape, 0) + 1
            return y

        @staticmethod
        def backward(ctx: Any, dy: Any) -> tuple[Any, None, None]:
            x, weight, inv = ctx.saved_tensors
            if dy.dtype != torch.bfloat16 or not dy.is_contiguous() or dy.shape != x.shape:
                receipt["contract_failures"] += 1
                raise RuntimeError("frozen Score RMSNorm hidden-gradient contract changed")
            d = int(x.shape[-1])
            dx = torch.empty_like(x)
            backward_kernel[(x.numel() // d,)](
                x, weight, inv, dy, dx, D=d, BLOCK=triton.next_power_of_2(d),
                num_warps=4 if d == 1536 else 8, enable_fp_fusion=False,
            )
            receipt["backward_calls"] += 1
            return dx, None, None

    return FrozenScoreRMSNorm.apply, receipt

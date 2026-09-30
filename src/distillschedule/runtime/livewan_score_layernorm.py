"""Standalone frozen Score LayerNorm canary, never installed by a plan.

Fuse only BF16 row-major non-affine LayerNorm. Preserve the pinned operation:
BF16 input -> FP32 statistics and normalization -> BF16 output. Backward
computes FP32 dInput then the original BF16 cast. No model/parameter mutation,
RNG, synchronization, CPU tensor access or collective is introduced here.
"""
from __future__ import annotations

import math
import hashlib
import inspect
from typing import Any


SOURCE_SHA256 = "17d4c9908c060202755a6f1bfd7830ec5aca655c28ff90a3e4278ee483c0282b"
CANARY_CONTRACT = dict(
    name="score_frozen_non_affine_bf16_layernorm_canary_v1",
    production_installed=False, input_dtype="bfloat16", output_dtype="bfloat16",
    statistics_dtype="float32", input_gradient_dtype="bfloat16",
    elementwise_affine=False, output_layout="row_major", eps=1e-6,
    numerical_gate="UNCHANGED_DEFAULT_P0_PARITY",
)
PLAN_CONTRACT = dict(
    enabled=True, name="score_frozen_non_affine_bf16_layernorm_v1",
    scope="score_ranks_only_critic_and_teacher", source_sha256=SOURCE_SHA256,
    input_dtype="bfloat16", output_dtype="bfloat16", statistics_dtype="float32",
    input_gradient_dtype="bfloat16", elementwise_affine=False, eps=1e-6,
    output_layout="row_major", native_affine_and_unsupported_preserved=True,
    numerical_gate="UNCHANGED_DEFAULT_P0_PARITY", application_calls_unchanged=True,
)


def install_frozen_score_layernorm(torch: Any) -> dict[str, Any]:
    """Score-process-only opt-in; all other norm signatures keep their source."""
    from wan.modules.model import WanLayerNorm

    if hasattr(WanLayerNorm, "_distillschedule_score_layernorm"):
        raise RuntimeError("Score LayerNorm already installed")
    if hashlib.sha256(inspect.getsource(WanLayerNorm).encode()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Score LayerNorm source mismatch")
    native = WanLayerNorm.forward
    op, receipt = build(torch)
    receipt.update(PLAN_CONTRACT, status="INSTALLED", install_status="PASS",
        execution_status="PASS", production_installed=True,
        total_norm_calls=0, selected_entries=0, native_fallback_calls=0,
        native_fallback_histogram={})

    def forward(norm: Any, x: Any) -> Any:
        receipt["total_norm_calls"] += 1
        if norm.weight is not None or norm.bias is not None:
            reason = "affine_preserved"
        elif not x.is_cuda:
            reason = "device_preserved"
        elif x.dtype != torch.bfloat16:
            reason = "dtype_preserved"
        elif x.ndim != 3 or x.shape[0] not in (1, 2) or x.shape[2] not in (1536, 5120):
            reason = "shape_preserved"
        elif x.stride() != (x.shape[1] * x.shape[2], x.shape[2], 1):
            reason = "layout_preserved"
        elif norm.eps != 1e-6:
            reason = "epsilon_preserved"
        else:
            receipt["selected_entries"] += 1
            return op(x, norm.eps)
        receipt["native_fallback_calls"] += 1
        h = receipt["native_fallback_histogram"]
        h[reason] = h.get(reason, 0) + 1
        return native(norm, x)

    WanLayerNorm.forward = forward
    WanLayerNorm._distillschedule_score_layernorm = receipt
    return receipt


def runtime_receipt_errors(receipt: dict[str, Any], *, rank: int,
                           inventory: dict[str, int]) -> list[str]:
    if rank < 8:
        return [] if receipt.get("execution_status") == "NOT_APPLICABLE" else ["Score LayerNorm changed Generator"]
    teacher = int(inventory.get("teacher/nograd/plain", 0))
    critic_ng = int(inventory.get("critic/nograd/plain", 0))
    critic_grad = int(inventory.get("critic/grad/checkpoint", 0))
    # Pinned Teacher:40 blocks x3 norms +head. Critic:30x3 +head;
    # block checkpoints replay their3 norms, while the root head does not.
    expected = teacher * 121 + critic_ng * 91 + critic_grad * 181
    if (any(receipt.get(k) != v for k, v in PLAN_CONTRACT.items())
        or receipt.get("install_status") != "PASS"
        or receipt.get("execution_status") != "PASS"
        or receipt.get("contract_failures") != 0
        or receipt.get("total_norm_calls") != expected or expected <= 0
        or receipt.get("selected_entries", 0) <= 0
        or receipt.get("forward_calls") != receipt.get("selected_entries")
        or receipt.get("selected_entries", 0) + receipt.get("native_fallback_calls", 0) != expected
        or sum(receipt.get("native_fallback_histogram", {}).values()) != receipt.get("native_fallback_calls")
        or not 0 < receipt.get("backward_calls", 0) <= critic_grad * 61):
        return ["Score LayerNorm source/scope/operation census incomplete"]
    return []


def validate(torch: Any, x: Any, eps: float) -> None:
    if (
        not x.is_cuda or x.dtype != torch.bfloat16 or x.ndim != 3
        or x.shape[0] not in (1, 2) or x.shape[1] <= 0
        or x.shape[2] not in (1536, 5120)
        or x.stride() != (x.shape[1] * x.shape[2], x.shape[2], 1)
        or not isinstance(eps, (float, int)) or not math.isfinite(eps)
        or eps != 1e-6
    ):
        raise RuntimeError("Score non-affine LayerNorm dtype/shape/layout/epsilon contract changed")


def build(torch: Any) -> tuple[Any, dict[str, Any]]:
    """Lazy standalone builder. Production/default behavior is unchanged."""
    import triton
    import triton.language as tl

    @triton.jit
    def forward_kernel(X, OUT, MEAN, RSTD, D: tl.constexpr,
                       EPS: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        x = tl.load(X + row * D + col, col < D, other=0).to(tl.float32)
        mean = tl.sum(x, 0) / D
        centered = tl.where(col < D, x - mean, 0)
        variance = tl.sum(centered * centered, 0) / D
        rstd = tl.rsqrt(variance + EPS)
        result = centered * rstd
        tl.store(OUT + row * D + col, result.to(tl.bfloat16), col < D)
        tl.store(MEAN + row, mean)
        tl.store(RSTD + row, rstd)

    @triton.jit
    def backward_kernel(X, DY, MEAN, RSTD, DX, D: tl.constexpr,
                        S: tl.constexpr, DB: tl.constexpr,
                        DS: tl.constexpr, DD: tl.constexpr,
                        BLOCK: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        x = tl.load(X + row * D + col, col < D, other=0).to(tl.float32)
        dy = tl.load(DY + (row // S) * DB + (row % S) * DS + col * DD,
                     col < D, other=0).to(tl.float32)
        mean = tl.load(MEAN + row)
        rstd = tl.load(RSTD + row)
        normalized = tl.where(col < D, (x - mean) * rstd, 0)
        mean_dy = tl.sum(dy, 0) / D
        mean_dy_normalized = tl.sum(dy * normalized, 0) / D
        dx = (dy - normalized * mean_dy_normalized - mean_dy) * rstd
        tl.store(DX + row * D + col, dx.to(tl.bfloat16), col < D)

    receipt: dict[str, Any] = dict(
        **CANARY_CONTRACT, status="CANARY_ONLY_NOT_PRODUCTION_INSTALLED",
        forward_calls=0, backward_calls=0, contract_failures=0,
        shape_stride_histogram={}, triton_version=triton.__version__,
    )

    class ScoreLayerNorm(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, x: Any, eps: float) -> Any:
            try:
                validate(torch, x, eps)
            except RuntimeError:
                receipt["contract_failures"] += 1
                raise
            b, s, d = x.shape
            out = torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)
            mean = torch.empty((b * s,), device=x.device, dtype=torch.float32)
            rstd = torch.empty_like(mean)
            block = triton.next_power_of_2(d)
            forward_kernel[(b * s,)](
                x, out, mean, rstd, D=d, EPS=eps, BLOCK=block,
                num_warps=4 if d <= 1536 else 8, enable_fp_fusion=False,
            )
            # Count before save_for_backward: non-reentrant replay may stop
            # early while restoring saved tensors rather than return normally.
            receipt["forward_calls"] += 1
            key = f"{tuple(x.shape)}/{tuple(x.stride())}"
            h = receipt["shape_stride_histogram"]
            h[key] = h.get(key, 0) + 1
            ctx.save_for_backward(x, mean, rstd)
            return out

        @staticmethod
        def backward(ctx: Any, dy: Any) -> tuple[Any, None]:
            if torch.is_grad_enabled():
                receipt["contract_failures"] += 1
                raise RuntimeError("Score LayerNorm canary does not support higher-order backward")
            x, mean, rstd = ctx.saved_tensors
            if not dy.is_cuda or dy.device != x.device or dy.dtype != x.dtype or dy.shape != x.shape:
                receipt["contract_failures"] += 1
                raise RuntimeError("Score LayerNorm incoming gradient contract changed")
            b, s, d = x.shape
            dx = torch.empty(x.shape, device=x.device, dtype=x.dtype)
            backward_kernel[(b * s,)](
                x, dy, mean, rstd, dx, D=d, S=s,
                DB=dy.stride(0), DS=dy.stride(1), DD=dy.stride(2),
                BLOCK=triton.next_power_of_2(d),
                num_warps=4 if d <= 1536 else 8, enable_fp_fusion=False,
            )
            receipt["backward_calls"] += 1
            return dx, None

    return ScoreLayerNorm.apply, receipt

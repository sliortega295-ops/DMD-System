"""Standalone Score BF16 residual/layout canary; not installed by any plan.

Unlike the FP32 fully trainable Generator gate, Score's pinned pointwise
operation rounds the product and the sum separately to BF16. Keep those exact
roundings and gradients while emitting a row-major residual for native norms.
"""
from __future__ import annotations

import hashlib
import inspect
import textwrap
from typing import Any

SOURCE_SHA256 = "4665c81a5dfbb7b7b75590c94a9c3acb2027bf7474f708033d1d1c96583ce527"
PLAN_CONTRACT = dict(
    enabled=True, name="score_bf16_row_major_residual_gate_v1",
    scope="score_ranks_only_critic_and_teacher", source_sha256=SOURCE_SHA256,
    product_dtype="bfloat16", output_dtype="bfloat16", fma=False,
    gate_requires_grad=False, output_layout="row_major", out_of_place=True,
    native_layernorm_unchanged=True, application_calls_unchanged=True,
)


def install_score_gate(torch: Any) -> dict[str, Any]:
    """Explicit Score-process installation; checkpoint replay uses the same op."""
    from wanstreamer import blockcausal as bc
    if hasattr(bc, "_distillschedule_score_gate"):
        raise RuntimeError("Score residual gate already installed")
    source = textwrap.dedent(inspect.getsource(bc._gate)).strip()
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Score residual gate source mismatch")
    op, receipt = build(torch)

    def fused(x: Any, y: Any, g: Any, per_frame: bool, n_frames: int) -> Any:
        if per_frame or n_frames != 1:
            receipt["contract_failures"] += 1
            raise RuntimeError("Score residual gate frame contract changed")
        return op(x, y, g)

    bc._gate = fused
    receipt.update(PLAN_CONTRACT, status="INSTALLED", install_status="PASS",
        execution_status="PASS", production_installed=True)
    bc._distillschedule_score_gate = receipt
    return receipt


def runtime_receipt_errors(receipt: dict[str, Any], *, rank: int,
                           inventory: dict[str, int]) -> list[str]:
    if rank < 8:
        return [] if receipt.get("execution_status") == "NOT_APPLICABLE" else ["Score gate changed Generator"]
    ng_teacher = int(inventory.get("teacher/nograd/plain", 0)) * 80
    ng_critic = int(inventory.get("critic/nograd/plain", 0)) * 60
    grad_critic = int(inventory.get("critic/grad/checkpoint", 0)) * 60
    expected = ng_teacher + ng_critic + 2 * grad_critic
    if (any(receipt.get(k) != v for k,v in PLAN_CONTRACT.items())
        or receipt.get("install_status") != "PASS"
        or receipt.get("execution_status") != "PASS"
        or receipt.get("contract_failures") != 0
        or receipt.get("forward_calls") != expected
        or receipt.get("backward_calls") != grad_critic
        or expected <= 0 or grad_critic <= 0):
        return ["Score residual gate runtime source/rounding/call contract failed"]
    return []


def validate(torch: Any, x: Any, y: Any, gate: Any) -> None:
    def dense(t: Any) -> bool:
        return (
            t.ndim == 3 and t.shape[0] in (1, 2)
            and t.shape[2] in (1536, 5120)
            and t.stride(0) == t.shape[1] * t.shape[2]
            and t.stride()[1:] in ((t.shape[2], 1), (1, t.shape[1]))
        )
    if (
        not x.is_cuda or not y.is_cuda or not gate.is_cuda
        or y.device != x.device or gate.device != x.device
        or x.dtype != torch.bfloat16 or y.dtype != torch.bfloat16
        or gate.dtype != torch.bfloat16 or x.shape != y.shape
        or not dense(x) or not dense(y)
        or gate.ndim != 3 or gate.shape[0] not in (1, x.shape[0])
        or gate.shape[1:] != (1, x.shape[2]) or gate.stride(-1) != 1
        or gate.requires_grad
    ):
        raise RuntimeError("Score BF16 gate dtype/shape/stride/frozen-gate contract changed")


def build(torch: Any) -> tuple[Any, dict[str, Any]]:
    """Build lazily; no import-time patch or production registration."""
    import triton
    import triton.language as tl

    @triton.jit
    def forward_kernel(X, Y, G, OUT, N: tl.constexpr, S: tl.constexpr,
                       D: tl.constexpr, XB: tl.constexpr, XS: tl.constexpr,
                       XD: tl.constexpr, YB: tl.constexpr, YS: tl.constexpr,
                       YD: tl.constexpr, GB: tl.constexpr, GR: tl.constexpr,
                       BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        col = i % D
        row = (i // D) % S
        batch = i // (S * D)
        x = tl.load(X + batch * XB + row * XS + col * XD, i < N, other=0).to(tl.float32)
        y = tl.load(Y + batch * YB + row * YS + col * YD, i < N, other=0).to(tl.float32)
        g = tl.load(G + tl.where(GR == 1, 0, batch) * GB + col,
                    i < N, other=0).to(tl.float32)
        product = tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;", "=f,f,f", [y, g],
            dtype=tl.float32, is_pure=True, pack=1,
        ).to(tl.bfloat16).to(tl.float32)
        result = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;", "=f,f,f", [x, product],
            dtype=tl.float32, is_pure=True, pack=1,
        ).to(tl.bfloat16)
        tl.store(OUT + i, result, i < N)

    @triton.jit
    def backward_kernel(DY, G, DG_Y, N: tl.constexpr, S: tl.constexpr,
                        D: tl.constexpr, DB: tl.constexpr, DS: tl.constexpr,
                        DD: tl.constexpr, GB: tl.constexpr, GR: tl.constexpr,
                        BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        col = i % D
        row = (i // D) % S
        batch = i // (S * D)
        dy = tl.load(DY + batch * DB + row * DS + col * DD, i < N, other=0).to(tl.float32)
        g = tl.load(G + tl.where(GR == 1, 0, batch) * GB + col,
                    i < N, other=0).to(tl.float32)
        product = tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;", "=f,f,f", [dy, g],
            dtype=tl.float32, is_pure=True, pack=1,
        ).to(tl.bfloat16)
        tl.store(DG_Y + i, product, i < N)

    receipt: dict[str, Any] = dict(
        name="score_bf16_row_major_residual_gate_canary_v1",
        status="CANARY_ONLY_NOT_PRODUCTION_INSTALLED",
        source_sha256=SOURCE_SHA256, forward_calls=0, backward_calls=0,
        contract_failures=0, output_layout="row_major", product_dtype="bfloat16",
        output_dtype="bfloat16", gate_requires_grad=False, fma=False,
        triton_version=triton.__version__, shape_stride_histogram={},
    )

    class ScoreGate(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, x: Any, y: Any, gate: Any) -> Any:
            try:
                validate(torch, x, y, gate)
            except RuntimeError:
                receipt["contract_failures"] += 1
                raise
            out = torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)
            forward_kernel[(triton.cdiv(x.numel(), 1024),)](
                x, y, gate, out, N=x.numel(), S=x.shape[1], D=x.shape[2],
                XB=x.stride(0), XS=x.stride(1), XD=x.stride(2),
                YB=y.stride(0), YS=y.stride(1), YD=y.stride(2),
                GB=gate.stride(0), GR=gate.shape[0], BLOCK=1024,
                num_warps=4, enable_fp_fusion=False,
            )
            # Count actual launches before a non-reentrant early-stop can
            # interrupt saved-tensor processing during checkpoint replay.
            receipt["forward_calls"] += 1
            key = f"{tuple(x.shape)}/{tuple(x.stride())}/{tuple(y.stride())}/{tuple(gate.shape)}"
            h = receipt["shape_stride_histogram"]
            h[key] = h.get(key, 0) + 1
            ctx.save_for_backward(gate)
            ctx.shape = tuple(x.shape)
            return out

        @staticmethod
        def backward(ctx: Any, dy: Any) -> tuple[Any, Any, None]:
            gate, = ctx.saved_tensors
            if dy.shape != ctx.shape:
                receipt["contract_failures"] += 1
                raise RuntimeError("Score gate incoming gradient shape changed")
            try:
                validate(torch, dy, dy, gate)
            except RuntimeError:
                receipt["contract_failures"] += 1
                raise
            grad_y = None
            if ctx.needs_input_grad[1]:
                grad_y = torch.empty(dy.shape, device=dy.device, dtype=dy.dtype)
                backward_kernel[(triton.cdiv(dy.numel(), 1024),)](
                    dy, gate, grad_y, N=dy.numel(), S=dy.shape[1], D=dy.shape[2],
                    DB=dy.stride(0), DS=dy.stride(1), DD=dy.stride(2),
                    GB=gate.stride(0), GR=gate.shape[0], BLOCK=1024,
                    num_warps=4, enable_fp_fusion=False,
                )
            receipt["backward_calls"] += 1
            return dy if ctx.needs_input_grad[0] else None, grad_y, None

    return ScoreGate.apply, receipt

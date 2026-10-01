"""C40 opt-in shared cast; no installation or GPU import at module import.

CUDA BF16 autocast already converts FP32 Score inputs twice: frozen base and
LoRA-A on nonleaf inputs. Share the payload without summing their input gradients in BF16. The custom
cast has two alias outputs; their BF16 gradients are independently cast and
added in FP32 exactly as in the parent. No parameter or optimizer change.
"""
from __future__ import annotations
from typing import Any

VARIANT = "c40-continuous-b16-score-shared-input"
PLAN_KERNEL_CONTRACT = {
    "enabled": True,
    "name": "scale_one_autocast_shared_bf16_input_v1",
    "scale": 1.0,
    "base_input_output_dtype": "bfloat16",
    "lora_parameter_compute_dtype": "float32",
    "source_forward_sha256": "70d00d496aa2bd8ea06d64e6b04af873644eff52539396f74ec586a4cbbe97a3",
    "math_change": "share_existing_bf16_input_cast_preserve_fp32_branch_gradient_sum",
    "autocast_dtype": "bfloat16",
    "required_lora_output_dtype": "bfloat16",
    "gradient_accumulation_dtype": "float32",
    "grad_leaf_input_policy": "unchanged_parent_autocast",
    "shape_scope": "cuda_contiguous_rank3_b1_b2_width1536",
}


def eligible(torch: Any, x: Any) -> bool:
    # CUDA autocast can cache a grad leaf conversion, accumulating its branches
    # in BF16. Never substitute our two-branch FP32 rule for that parent graph.
    return (x.is_cuda and x.dtype == torch.float32 and x.ndim == 3
        and x.is_contiguous() and x.shape[0] in (1, 2) and x.shape[-1] == 1536
        and not (torch.is_grad_enabled() and x.requires_grad and x.is_leaf))


def runtime_receipt_errors(receipt: dict[str, Any], *, rank: int) -> list[str]:
    errors = []
    if receipt.get("name") != PLAN_KERNEL_CONTRACT["name"]:
        errors.append("C40 LoRA specialization identity missing")
    if receipt.get("shared_autocast_input") is not True:
        errors.append("C40 shared input disabled")
    shared = receipt.get("shared_input_kernel", {})
    if shared.get("production_installed") is not True:
        errors.append("C40 shared kernel not installed")
    for key in ("contract_failures",):
        if shared.get(key) != 0:
            errors.append(f"C40 {key} changed")
    if shared.get("gradient_accumulation_dtype") != "float32":
        errors.append("C40 input gradient precision changed")
    for key in ("forward_calls", "backward_calls"):
        count = shared.get(key, -1)
        if not isinstance(count, int) or (count > 0) != (rank >= 8):
            errors.append(f"C40 {key} actual runtime count mismatch")
    return errors


def build(torch: Any):
    import triton
    import triton.language as tl

    @triton.jit
    def sum_cast(A, B, OUT, N: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
                 AB: tl.constexpr, AS: tl.constexpr, AD: tl.constexpr,
                 BB: tl.constexpr, BS: tl.constexpr, BD: tl.constexpr,
                 HAS_A: tl.constexpr, HAS_B: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        col = i % D
        row = (i // D) % S
        batch = i // (S * D)
        a = tl.full((BLOCK,), 0.0, tl.float32)
        b = tl.full((BLOCK,), 0.0, tl.float32)
        if HAS_A:
            a = tl.load(A + batch * AB + row * AS + col * AD, i < N, other=0).to(tl.float32)
        if HAS_B:
            b = tl.load(B + batch * BB + row * BS + col * BD, i < N, other=0).to(tl.float32)
        result = tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;",
            "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1)
        tl.store(OUT + i, result, i < N)

    receipt = dict(name="score_lora_shared_bf16_input_v1",
        production_installed=False, forward_calls=0, backward_calls=0,
        contract_failures=0, gradient_accumulation_dtype="float32",
        compute_dtype="existing_cuda_bf16_autocast", weight_storage_dtype="float32")

    class SharedInput(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            if (not x.is_cuda or x.dtype != torch.float32 or x.ndim != 3
                or not x.is_contiguous() or x.shape[0] not in (1, 2) or x.shape[-1] != 1536):
                receipt["contract_failures"] += 1
                raise RuntimeError("C40 Score input canary contract changed")
            ctx.shape, ctx.device = tuple(x.shape), x.device
            ctx.set_materialize_grads(False)
            y = x.to(torch.bfloat16)
            receipt["forward_calls"] += 1
            # Distinct Tensor outputs with shared storage; no second payload copy.
            return y.view_as(y), y.view_as(y)

        @staticmethod
        def backward(ctx, a, b):
            if torch.is_grad_enabled():
                raise RuntimeError("C40 higher-order backward is not supported")
            for g in (a, b):
                if g is not None and (g.device != ctx.device or g.dtype != torch.bfloat16 or tuple(g.shape) != ctx.shape):
                    receipt["contract_failures"] += 1
                    raise RuntimeError("C40 incoming gradient contract changed")
            receipt["backward_calls"] += 1
            if a is None and b is None:
                return None
            out = torch.empty(ctx.shape, device=ctx.device, dtype=torch.float32)
            s, d = ctx.shape[1:]
            ast = a.stride() if a is not None else (0, 0, 0)
            bst = b.stride() if b is not None else (0, 0, 0)
            sum_cast[(triton.cdiv(out.numel(), 1024),)](
                a if a is not None else out, b if b is not None else out, out,
                N=out.numel(), S=s, D=d, AB=ast[0], AS=ast[1], AD=ast[2],
                BB=bst[0], BS=bst[1], BD=bst[2], HAS_A=a is not None, HAS_B=b is not None,
                BLOCK=1024, num_warps=4, enable_fp_fusion=False)
            return out

    return SharedInput.apply, receipt

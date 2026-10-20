"""C51 opt-in no-grad Generator RMS; grad/Score scopes remain native."""
from __future__ import annotations

from typing import Any, Mapping

from distillschedule.runtime.livewan_generator_rms_epilogue import (
    ATTRIBUTE, GeneratorRMSEpilogue, REFERENCE_CLASS_SHA256, validate_inputs,
)

VARIANT = "c51-continuous-b16-generator-fused-rms"
CONTRACT = dict(
    enabled=True, name="generator_nograd_fused_rms_row_v1",
    scope="single_prepare_rollout_generator_only",
    input_dtype="bfloat16", affine_dtype="float32", output_dtype="float32",
    intermediate_bf16_rounding=True, grad_path_changed=False,
    source_class_sha256=REFERENCE_CLASS_SHA256,
    native_pow_mean_preserved=False, native_mean_preserved=False,
    fp32_square_preserved=True, fused_cast_square=True, accumulation_dtype="float32",
    reduction_tree_changed=True, semantic_math_changed=False,
    num_warps=4, row_block_size=2048,
)


def enabled(plan: Mapping[str, Any], *, mode: str) -> bool:
    value = plan.get("metadata", {}).get("generator_rms_epilogue")
    if value is None or value.get("name") != CONTRACT["name"]:
        return False
    from distillschedule.runtime.livewan_generator_splitkv import enabled as splitkv_enabled
    if mode != "pipeline" or dict(value) != CONTRACT or not splitkv_enabled(plan, mode=mode):
        raise RuntimeError("C51 requires exact fused RMS and unchanged C49 parent")
    return True


def build(torch: Any) -> tuple[Any, dict[str, Any]]:
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(X, W, Y, D: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        x = tl.load(X + row * D + col, col < D, other=0).to(tl.float32)
        variance = tl.sum(x * x, 0) / D
        normalized = (x * tl.rsqrt(variance + 1.0e-6)).to(tl.bfloat16).to(tl.float32)
        weight = tl.load(W + col, col < D, other=0)
        tl.store(Y + row * D + col, normalized * weight, col < D)

    receipt = dict(CONTRACT, status="CANARY_ONLY_NOT_PRODUCTION_INSTALLED",
        calls=0, contract_failures=0, shape_histogram={}, triton_version=triton.__version__)

    def op(x: Any, weight: Any, eps: float) -> Any:
        try:
            validate_inputs(torch, x, weight, eps)
        except RuntimeError:
            receipt["contract_failures"] += 1
            raise
        y = torch.empty(x.shape, device=x.device, dtype=torch.float32)
        kernel[(x.numel() // 1536,)](x, weight, y, D=1536, BLOCK=2048,
            num_warps=4, enable_fp_fusion=False)
        receipt["calls"] += 1
        shape = "x".join(str(int(v)) for v in x.shape)
        receipt["shape_histogram"][shape] = receipt["shape_histogram"].get(shape, 0) + 1
        return y

    return op, receipt


class FusedGeneratorRMS(GeneratorRMSEpilogue):
    """Reuse source-bound scope/restore/audit; replace only its norm kernel."""

    def __init__(self, torch: Any, student: Any) -> None:
        self.student = student
        self.op, self.kernel_receipt = build(torch)
        self.entries = self.exits = self.failures = 0
        self.active = False
        self.scope_calls: list[int] = []


def install(torch: Any, student: Any) -> FusedGeneratorRMS:
    if (str(torch.__version__) != "2.10.0+cu128" or hasattr(student, ATTRIBUTE)
            or hasattr(student, "causal_layers") or len(student.blocks) != 30):
        raise RuntimeError("C51 requires validated unwrapped resident 30-block Generator")
    if not all(p.is_cuda and p.dtype == torch.float32 and p.requires_grad
               and not hasattr(p, "to_local") for p in student.parameters()):
        raise RuntimeError("C51 Generator dtype/trainability/placement changed")
    helper = FusedGeneratorRMS(torch, student)
    setattr(student, ATTRIBUTE, helper)
    return helper


def runtime_receipt_errors(r: Mapping[str, Any], *, rank: int, cycles: int) -> list[str]:
    if rank >= 8:
        return [] if r.get("execution_status") == "NOT_APPLICABLE" else ["C51 changed Score scope"]
    if any(r.get(k) != v for k, v in CONTRACT.items()):
        return ["C51 actual norm kernel/math/precision contract failed"]
    counts = r.get("scope_forward_calls", [])
    if (r.get("execution_status") != "PASS"
            or r.get("scope_entries") != 10 * cycles or r.get("scope_exits") != 10 * cycles
            or r.get("active") is not False or r.get("scope_failures") != 0
            or r.get("contract_failures") != 0 or len(counts) != 10 * cycles
            or any(v not in (1920, 2370, 2820) for v in counts)
            or r.get("calls") != sum(counts)
            or sum(r.get("shape_histogram", {}).values()) != r.get("calls")):
        return ["C51 actual scope/call/source receipt incomplete"]
    return []

"""Opt-in removal of PEFT's identity LoRA scaling multiplication."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import inspect
import types
from typing import Any, Callable

import torch

from .triton_lora_epilogue import BLOCK_N as LORA_EPILOGUE_BLOCK_N
from .triton_lora_epilogue import fused_lora_b_residual


PINNED_PEFT_LINEAR_FORWARD_SHA256 = (
    "ec0a80f5c5ce05f5dd90027952bb5d29aaf910e9d2ab40486a90e8a03c7e1cd9"
)
EXPECTED_MODULE_COUNT = 52 * 6
EXPECTED_TOTAL_CALLS_PER_CYCLE = EXPECTED_MODULE_COUNT * 37
EXPECTED_DISABLED_CALLS_PER_CYCLE = EXPECTED_MODULE_COUNT
EXPECTED_NOGRAD_ELISIONS_PER_CYCLE = EXPECTED_MODULE_COUNT * 24
EXPECTED_GRAD_ELISIONS_PER_CYCLE = EXPECTED_MODULE_COUNT * 12
EXPECTED_GRAD_BACKWARDS_PER_CYCLE = EXPECTED_MODULE_COUNT * 6


@dataclasses.dataclass
class LoRAScale1Stats:
    total_calls: int = 0
    elided_calls: int = 0
    no_grad_elided_calls: int = 0
    grad_elided_calls: int = 0
    disabled_reference_calls: int = 0
    unsupported_reference_calls: int = 0
    invalid_contract_calls: int = 0
    fused_nograd_epilogue_calls: int = 0
    fused_grad_epilogue_calls: int = 0
    fused_grad_epilogue_backward_calls: int = 0
    reference_grad_epilogue_calls: int = 0


@dataclasses.dataclass
class LoRAScale1Registration:
    enabled: bool
    epilogue_enabled: bool
    grad_epilogue_enabled: bool
    source_sha256: str | None
    module_count: int
    stats: LoRAScale1Stats

    def snapshot(self) -> dict[str, int]:
        return dataclasses.asdict(self.stats)

    def receipt(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "epilogue_enabled": self.epilogue_enabled,
            "grad_epilogue_enabled": self.grad_epilogue_enabled,
            "source_sha256": self.source_sha256,
            "module_count": self.module_count,
            "epilogue_block_n": LORA_EPILOGUE_BLOCK_N,
            "stats": self.snapshot(),
        }


def _source_sha256(function: Callable[..., Any]) -> str:
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def _scale_is_exact_one(value: Any) -> bool:
    return isinstance(value, (int, float)) and float(value) == 1.0


class _FusedLoRABResidualAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base, projected, weight, stats):
        ctx.save_for_backward(projected, weight)
        ctx.stats = stats
        return fused_lora_b_residual(base, projected, weight)

    @staticmethod
    def backward(ctx, grad_output):
        projected, weight = ctx.saved_tensors
        ctx.stats.fused_grad_epilogue_backward_calls += 1
        columns = int(weight.shape[0])
        rows = int(grad_output.numel() // columns)
        grad_2d = grad_output.reshape(rows, columns)
        projected_2d = projected.reshape(rows, int(weight.shape[1]))
        grad_projected = torch.mm(grad_2d, weight)
        grad_weight = torch.mm(grad_2d.transpose(0, 1), projected_2d)
        return (
            grad_output,
            grad_projected.reshape(projected.shape),
            grad_weight,
            None,
        )


def _patched_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    stats: LoRAScale1Stats = self._h3_lora_scale1_stats
    stats.total_calls += 1

    # Teacher execution deliberately disables all adapters. Preserve PEFT's
    # reference branch rather than duplicating its merge/disable behavior.
    if self.disable_adapters:
        stats.disabled_reference_calls += 1
        return self._h3_original_lora_forward(x, *args, **kwargs)

    # The controlled H3 callsite passes only x. Keep every more general PEFT
    # feature on the pinned reference implementation and fail the cycle census
    # if it unexpectedly appears in the measured workload.
    if args or kwargs or self.merged:
        stats.unsupported_reference_calls += 1
        return self._h3_original_lora_forward(x, *args, **kwargs)

    active = [name for name in self.active_adapters if name in self.lora_A]
    if len(active) != 1:
        stats.invalid_contract_calls += 1
        raise RuntimeError(
            "H3 LoRA scale-one elision requires exactly one active adapter; "
            f"observed {active!r}"
        )
    adapter = active[0]
    if adapter in self.lora_variant:
        stats.invalid_contract_calls += 1
        raise RuntimeError("H3 LoRA scale-one elision does not support LoRA variants")
    scaling = self.scaling[adapter]
    if not _scale_is_exact_one(scaling):
        stats.invalid_contract_calls += 1
        raise RuntimeError(
            "H3 LoRA scale-one elision observed non-identity scaling: "
            f"adapter={adapter!r} scaling={scaling!r}"
        )
    dropout = self.lora_dropout[adapter]
    if not isinstance(dropout, torch.nn.Identity):
        stats.invalid_contract_calls += 1
        raise RuntimeError(
            "H3 LoRA scale-one elision requires zero-dropout Identity, got "
            f"{type(dropout).__name__}"
        )

    # Count path entry before executing tensor operations. Non-reentrant
    # checkpoint replay may stop by raising PyTorch's internal early-stop
    # exception from inside a nested module once all required tensors have
    # been reconstructed, so code after the final operation is not guaranteed
    # to run even though this scale-one path was selected and executed.
    stats.elided_calls += 1
    grad_enabled = torch.is_grad_enabled()
    if grad_enabled:
        stats.grad_elided_calls += 1
    else:
        stats.no_grad_elided_calls += 1
    if self._h3_lora_epilogue_enabled:
        if grad_enabled:
            if self._h3_lora_grad_epilogue_enabled:
                stats.fused_grad_epilogue_calls += 1
            else:
                stats.reference_grad_epilogue_calls += 1
        else:
            stats.fused_nograd_epilogue_calls += 1

    result = self.base_layer(x)
    result_dtype = result.dtype
    lora_A = self.lora_A[adapter]
    lora_B = self.lora_B[adapter]
    cast_x = self._cast_input_dtype(x, lora_A.weight.dtype)
    if self._h3_lora_epilogue_enabled and (
        not grad_enabled or self._h3_lora_grad_epilogue_enabled
    ):
        if (
            lora_B.bias is not None
            or lora_B._forward_pre_hooks
            or lora_B._forward_hooks
        ):
            stats.invalid_contract_calls += 1
            raise RuntimeError(
                "H3 LoRA epilogue requires a hook-free bias-free LoRA-B Linear"
            )
        projected = lora_A(cast_x)
        if grad_enabled:
            return _FusedLoRABResidualAutograd.apply(
                result, projected, lora_B.weight, stats
            ).to(result_dtype)
        return fused_lora_b_residual(result, projected, lora_B.weight).to(result_dtype)

    # PEFT's pinned expression is `result + lora_B(lora_A(x)) * 1.0`.
    # Removing only the identity multiplication preserves both GEMMs and the
    # residual add while eliminating one full-output elementwise CUDA kernel.
    result = result + lora_B(lora_A(cast_x))
    return result.to(result_dtype)


def install_lora_scale1_elision(
    transformer: torch.nn.Module,
    *,
    enabled: bool,
    epilogue_enabled: bool = False,
    grad_epilogue_enabled: bool = False,
) -> LoRAScale1Registration:
    """Patch the pinned PEFT Linear modules after both adapters are injected."""

    if not enabled:
        if epilogue_enabled or grad_epilogue_enabled:
            raise RuntimeError("H3 LoRA epilogue requires scale-one elision")
        return LoRAScale1Registration(
            False, False, False, None, 0, LoRAScale1Stats()
        )
    if grad_epilogue_enabled and not epilogue_enabled:
        raise RuntimeError("H3 grad LoRA epilogue requires base LoRA epilogue enabled")

    from peft.tuners.lora.layer import Linear

    source_sha256 = _source_sha256(Linear.forward)
    if source_sha256 != PINNED_PEFT_LINEAR_FORWARD_SHA256:
        raise RuntimeError(
            "H3 PEFT Linear source identity mismatch: "
            f"observed={source_sha256} "
            f"expected={PINNED_PEFT_LINEAR_FORWARD_SHA256}"
        )
    modules = [module for module in transformer.modules() if isinstance(module, Linear)]
    if len(modules) != EXPECTED_MODULE_COUNT:
        raise RuntimeError(
            f"H3 LoRA scale-one elision requires {EXPECTED_MODULE_COUNT} PEFT "
            f"Linear modules, got {len(modules)}"
        )

    stats = LoRAScale1Stats()
    for module in modules:
        if hasattr(module, "_h3_original_lora_forward"):
            raise RuntimeError("H3 LoRA scale-one elision was installed more than once")
        object.__setattr__(module, "_h3_original_lora_forward", module.forward)
        object.__setattr__(module, "_h3_lora_scale1_stats", stats)
        object.__setattr__(module, "_h3_lora_epilogue_enabled", epilogue_enabled)
        object.__setattr__(
            module, "_h3_lora_grad_epilogue_enabled", grad_epilogue_enabled
        )
        object.__setattr__(module, "forward", types.MethodType(_patched_forward, module))
    return LoRAScale1Registration(
        True,
        epilogue_enabled,
        grad_epilogue_enabled,
        source_sha256,
        len(modules),
        stats,
    )


def validate_cycle(
    registration: LoRAScale1Registration, start: dict[str, int]
) -> dict[str, int]:
    if not registration.enabled:
        return {}
    current = registration.snapshot()
    delta = {key: int(current[key]) - int(start.get(key, 0)) for key in current}
    expected = {
        "total_calls": EXPECTED_TOTAL_CALLS_PER_CYCLE,
        "elided_calls": (
            EXPECTED_NOGRAD_ELISIONS_PER_CYCLE + EXPECTED_GRAD_ELISIONS_PER_CYCLE
        ),
        "no_grad_elided_calls": EXPECTED_NOGRAD_ELISIONS_PER_CYCLE,
        "grad_elided_calls": EXPECTED_GRAD_ELISIONS_PER_CYCLE,
        "disabled_reference_calls": EXPECTED_DISABLED_CALLS_PER_CYCLE,
        "unsupported_reference_calls": 0,
        "invalid_contract_calls": 0,
        "fused_nograd_epilogue_calls": (
            EXPECTED_NOGRAD_ELISIONS_PER_CYCLE if registration.epilogue_enabled else 0
        ),
        "fused_grad_epilogue_calls": (
            EXPECTED_GRAD_ELISIONS_PER_CYCLE
            if registration.grad_epilogue_enabled
            else 0
        ),
        "fused_grad_epilogue_backward_calls": (
            EXPECTED_GRAD_BACKWARDS_PER_CYCLE
            if registration.grad_epilogue_enabled
            else 0
        ),
        "reference_grad_epilogue_calls": (
            EXPECTED_GRAD_ELISIONS_PER_CYCLE
            if registration.epilogue_enabled and not registration.grad_epilogue_enabled
            else 0
        ),
    }
    errors = [
        f"{key}={delta[key]} expected={value}"
        for key, value in expected.items()
        if delta[key] != value
    ]
    if errors:
        raise RuntimeError(
            f"H3 LoRA scale-one elision cycle contract failed: {'; '.join(errors)}"
        )
    return delta

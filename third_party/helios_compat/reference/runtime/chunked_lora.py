"""Memory-bounded LoRA-B accumulation with one gradient edge per parameter.

The standard PEFT expression materializes a full ``lora_B(lora_A(x))`` tensor
before adding it to the frozen base result.  Helios' long-token FFN projection can
make that temporary hundreds of MiB.  This wrapper tiles only the LoRA-B output
dimension and adds each tile into the already allocated base result.  A custom
autograd function returns one gradient tensor for each LoRA parameter, avoiding the
repeated parameter-reduction hooks caused by invoking the module once per token tile.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MethodType
from typing import Any, Iterator, Mapping

import torch
import torch.nn.functional as F


_CACHE_RELEASE_MARGIN_BYTES = 16 * 1024 * 1024


@dataclass
class ChunkedLoraBGradientAccumulator:
    """One outer LoRA-B gradient buffer shared by all recompute tiles.

    ``safe_tiled_ffn`` owns the buffer and ultimately returns it to the real
    parameter's autograd edge.  The chunked LoRA-B backward writes each tile's
    matrix product directly into this buffer instead of allocating a second
    full-size parameter gradient and adding it afterwards.
    """

    tensor: torch.Tensor
    contribution_count: int = 0


_ACTIVE_WEIGHT_GRADIENT_ACCUMULATORS: ContextVar[
    Mapping[int, ChunkedLoraBGradientAccumulator] | None
] = ContextVar(
    "distillschedule_chunked_lora_b_gradient_accumulators",
    default=None,
)


@contextmanager
def activate_chunked_lora_b_gradient_accumulators(
    accumulators_by_weight_id: Mapping[int, ChunkedLoraBGradientAccumulator],
) -> Iterator[None]:
    """Expose outer recompute accumulators to patched LoRA-B modules.

    Tensor identity is scoped to a single ``functional_call`` recompute tile;
    the context is reset even when forward/backward raises, so ordinary LoRA
    calls retain the original independent-gradient behavior.
    """

    token = _ACTIVE_WEIGHT_GRADIENT_ACCUMULATORS.set(accumulators_by_weight_id)
    try:
        yield
    finally:
        _ACTIVE_WEIGHT_GRADIENT_ACCUMULATORS.reset(token)


def _active_weight_gradient_accumulator(
    weight: torch.Tensor,
) -> ChunkedLoraBGradientAccumulator | None:
    accumulators = _ACTIVE_WEIGHT_GRADIENT_ACCUMULATORS.get()
    if accumulators is None:
        return None
    return accumulators.get(id(weight))


def _linear_output_bytes(reference: torch.Tensor, output_features: int) -> int:
    if output_features <= 0:
        raise ValueError("linear output feature count must be positive")
    rows = reference.numel() // int(reference.shape[-1])
    return rows * int(output_features) * reference.element_size()


def _release_cache_for_base_output(
    reference: torch.Tensor,
    required_bytes: int,
) -> bool:
    """Return cached blocks only when the next full base output lacks headroom."""
    if not reference.is_cuda:
        return False
    free_bytes, _total_bytes = torch.cuda.mem_get_info(reference.device)
    if free_bytes >= required_bytes + _CACHE_RELEASE_MARGIN_BYTES:
        return False
    torch.cuda.nvtx.range_push(
        "helios_runtime|op=allocator_cache_release|target=chunked_lora_base_output"
    )
    try:
        torch.cuda.empty_cache()
    finally:
        torch.cuda.nvtx.range_pop()
    return True


class _ChunkedLoraAdd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        base_result: torch.Tensor,
        lora_input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        scaling: float,
        output_chunk_size: int,
        weight_gradient_accumulator: ChunkedLoraBGradientAccumulator | None,
    ) -> torch.Tensor:
        if output_chunk_size <= 0:
            raise ValueError("LoRA output chunk size must be positive")
        ctx.scaling = float(scaling)
        ctx.has_bias = bias is not None
        ctx.output_chunk_size = int(output_chunk_size)
        ctx.weight_gradient_accumulator = weight_gradient_accumulator
        ctx.save_for_backward(lora_input, weight)
        ctx.mark_dirty(base_result)

        output_features = int(weight.shape[0])
        for start in range(0, output_features, output_chunk_size):
            end = min(output_features, start + output_chunk_size)
            contribution = F.linear(
                lora_input,
                weight[start:end],
                None if bias is None else bias[start:end],
            )
            base_result[..., start:end].add_(contribution, alpha=ctx.scaling)
        return base_result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        lora_input, weight = ctx.saved_tensors
        scaling = ctx.scaling
        grad_lora_input = F.linear(grad_output, weight.t()) * scaling
        flat_grad = grad_output.reshape(-1, grad_output.shape[-1])
        flat_input = lora_input.reshape(-1, lora_input.shape[-1])
        accumulator = ctx.weight_gradient_accumulator
        if accumulator is None:
            grad_weight = flat_grad.t().matmul(flat_input) * scaling
        else:
            if accumulator.tensor.shape != weight.shape:
                raise RuntimeError(
                    "LoRA-B gradient accumulator shape mismatch: "
                    f"expected {tuple(weight.shape)}, got {tuple(accumulator.tensor.shape)}"
                )
            if accumulator.tensor.device != weight.device:
                raise RuntimeError(
                    "LoRA-B gradient accumulator device mismatch: "
                    f"expected {weight.device}, got {accumulator.tensor.device}"
                )
            if accumulator.tensor.dtype != weight.dtype:
                raise RuntimeError(
                    "LoRA-B gradient accumulator dtype mismatch: "
                    f"expected {weight.dtype}, got {accumulator.tensor.dtype}"
                )
            # The full LoRA-B matrix can itself be large enough to fail at the
            # last few MiB of Generator backward. Keep the single outer
            # accumulator, but compute its rows in the same bounded chunks as
            # the forward LoRA output. Each out= target is a view into the
            # preallocated buffer, so no full-size per-tile grad_weight is ever
            # materialized.
            row_chunk_size = min(ctx.output_chunk_size, int(weight.shape[0]))
            for row_start in range(0, int(weight.shape[0]), row_chunk_size):
                row_end = min(int(weight.shape[0]), row_start + row_chunk_size)
                target = accumulator.tensor[row_start:row_end]
                row_grad = flat_grad[:, row_start:row_end]
                if accumulator.contribution_count == 0:
                    torch.mm(row_grad.t(), flat_input, out=target)
                    target.mul_(scaling)
                else:
                    torch.addmm(
                        target,
                        row_grad.t(),
                        flat_input,
                        beta=1.0,
                        alpha=scaling,
                        out=target,
                    )
            accumulator.contribution_count += 1
            # The owning safe-tiled autograd function returns the shared buffer
            # once. Returning a per-tile gradient here would recreate the peak
            # this path is designed to remove.
            grad_weight = None
        grad_bias = (
            flat_grad.sum(dim=0) * scaling if ctx.has_bias else None
        )
        return (
            grad_output,
            grad_lora_input,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
        )


def install_chunked_lora_b(
    model: torch.nn.Module,
    *,
    output_chunk_size: int = 512,
    min_output_features: int = 4096,
) -> int:
    """Patch large vanilla PEFT Linear adapters in ``model`` and return their count.

    Helios' attention output projection has 6,144 output features, so the default
    threshold intentionally covers attention as well as the wider FFN projections.
    """
    if output_chunk_size <= 0:
        raise ValueError("LoRA output chunk size must be positive")
    if min_output_features <= 0:
        raise ValueError("LoRA minimum output feature count must be positive")

    from peft.tuners.lora.layer import Linear, VARIANT_KWARG_KEYS

    patched = 0
    for module in model.modules():
        if not isinstance(module, Linear):
            continue
        if getattr(module, "_distillschedule_chunked_lora_b", False):
            continue
        base_layer = module.get_base_layer()
        if int(getattr(base_layer, "out_features", 0)) < min_output_features:
            continue

        module._distillschedule_original_forward = module.forward
        module._distillschedule_chunked_lora_b = True
        module._distillschedule_lora_output_chunk_size = int(output_chunk_size)

        def chunked_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
            original_kwargs = dict(kwargs)
            adapter_names = kwargs.get("adapter_names")
            has_variant = any(
                adapter in self.lora_variant for adapter in self.active_adapters
            )
            if self.disable_adapters or adapter_names is not None or self.merged or has_variant:
                return self._distillschedule_original_forward(
                    x, *args, **original_kwargs
                )

            self._check_forward_args(x, *args, **kwargs)
            kwargs = dict(kwargs)
            kwargs.pop("adapter_names", None)
            for key in VARIANT_KWARG_KEYS:
                kwargs.pop(key, None)
            base_layer = self.get_base_layer()
            _release_cache_for_base_output(
                x,
                _linear_output_bytes(x, int(base_layer.out_features)),
            )
            result = self.base_layer(x, *args, **kwargs)
            result_dtype = result.dtype

            lora_a_keys = self.lora_A.keys()
            for active_adapter in self.active_adapters:
                if active_adapter not in lora_a_keys:
                    continue
                lora_a = self.lora_A[active_adapter]
                lora_b = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                cast_x = self._cast_input_dtype(x, lora_a.weight.dtype)
                lora_input = lora_a(dropout(cast_x))
                result = _ChunkedLoraAdd.apply(
                    result,
                    lora_input,
                    lora_b.weight,
                    lora_b.bias,
                    self.scaling[active_adapter],
                    self._distillschedule_lora_output_chunk_size,
                    _active_weight_gradient_accumulator(lora_b.weight),
                )
            return result.to(result_dtype)

        module.forward = MethodType(chunked_forward, module)
        patched += 1
    return patched

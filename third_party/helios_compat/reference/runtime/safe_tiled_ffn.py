"""Sequence-tiled FFN execution with one outer gradient per trainable parameter.

Calling an FFN once per token tile in an ordinary autograd graph makes DeepSpeed
observe the same LoRA parameter repeatedly.  This wrapper instead exposes one custom
autograd edge.  Its backward recomputes each tile with detached functional parameter
aliases, accumulates their gradients locally, and returns one tensor per real
parameter to the outer graph.
"""
from __future__ import annotations

from contextlib import contextmanager
from math import prod
from types import MethodType

import torch
from torch.func import functional_call

from runtime.chunked_lora import (
    ChunkedLoraBGradientAccumulator,
    activate_chunked_lora_b_gradient_accumulators,
)


_CACHE_RELEASE_MARGIN_BYTES = 64 * 1024 * 1024


def _distributed_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return 0


@contextmanager
def _transfer_range(
    reference: torch.Tensor,
    *,
    component: str,
    direction: str,
    byte_count: int,
):
    if not reference.is_cuda:
        yield
        return
    torch.cuda.nvtx.range_push(
        "|".join(
            [
                "helios_transfer",
                f"rank={_distributed_rank()}",
                f"component={component}",
                "kind=ffn_input_reuse",
                f"direction={direction}",
                f"bytes={int(byte_count)}",
            ]
        )
    )
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _release_cache_for_allocation(
    reference: torch.Tensor,
    required_bytes: int,
    *,
    target: str,
) -> None:
    """Release fragmented cached blocks only when driver headroom is insufficient."""
    if not reference.is_cuda:
        return
    free_bytes, _total_bytes = torch.cuda.mem_get_info(reference.device)
    if free_bytes >= required_bytes + _CACHE_RELEASE_MARGIN_BYTES:
        return
    torch.cuda.nvtx.range_push(
        "helios_runtime|op=allocator_cache_release|target=" + target
    )
    try:
        torch.cuda.empty_cache()
    finally:
        torch.cuda.nvtx.range_pop()


class _SafeTiledModule(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        module: torch.nn.Module,
        hidden_states: torch.Tensor,
        chunk_tokens: int,
        parameter_names: tuple[str, ...],
        *parameters: torch.Tensor,
    ) -> torch.Tensor:
        if chunk_tokens <= 0:
            raise ValueError("safe tiled FFN chunk size must be positive")
        ctx.module = module
        ctx.chunk_tokens = int(chunk_tokens)
        ctx.parameter_names = parameter_names
        ctx.direct_lora_b_parameter_names = frozenset(
            getattr(
                module,
                "_distillschedule_direct_lora_b_parameter_names",
                (),
            )
        )
        ctx.hidden_requires_grad = bool(hidden_states.requires_grad)
        ctx.component = str(
            getattr(module, "_distillschedule_ffn_component", "unknown")
        )
        ctx.input_shape = tuple(hidden_states.shape)
        ctx.input_dtype = hidden_states.dtype
        ctx.input_device = hidden_states.device
        ctx.inplace_output = False
        sequence_length = int(hidden_states.shape[-2])
        ctx.sequence_length = sequence_length
        if sequence_length == 0:
            ctx.save_for_backward(hidden_states, *parameters)
            return module._distillschedule_original_forward(hidden_states)
        with torch.no_grad():
            shape_probe = module._distillschedule_original_forward(
                hidden_states[..., :0, :]
            )
            output_shape = list(shape_probe.shape)
            output_shape[-2] = sequence_length
            output_bytes = (
                shape_probe.element_size()
                * int(prod(output_shape))
            )
            same_storage_shape = (
                tuple(output_shape) == tuple(hidden_states.shape)
                and shape_probe.dtype == hidden_states.dtype
                and shape_probe.device == hidden_states.device
            )
            can_reuse_input = same_storage_shape and not hidden_states.is_leaf
            if can_reuse_input:
                byte_count = hidden_states.numel() * hidden_states.element_size()
                with _transfer_range(
                    hidden_states,
                    component=ctx.component,
                    direction="d2h",
                    byte_count=byte_count,
                ):
                    if hidden_states.is_cuda:
                        ctx.hidden_states_host = hidden_states.detach().to(
                            "cpu", non_blocking=False
                        )
                    else:
                        ctx.hidden_states_host = hidden_states.detach().clone()
                ctx.save_for_backward(*parameters)
                ctx.mark_dirty(hidden_states)
                output = hidden_states
                ctx.inplace_output = True
            else:
                ctx.save_for_backward(hidden_states, *parameters)
                _release_cache_for_allocation(
                    hidden_states,
                    output_bytes,
                    target="safe_tiled_ffn_output",
                )
                output = shape_probe.new_empty(output_shape)
            del shape_probe
            for start in range(0, sequence_length, chunk_tokens):
                end = min(sequence_length, start + chunk_tokens)
                output[..., start:end, :].copy_(
                    module._distillschedule_original_forward(
                        hidden_states[..., start:end, :]
                    )
                )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.inplace_output:
            parameters = list(ctx.saved_tensors)
            hidden_states = None
        else:
            hidden_states, *parameters = ctx.saved_tensors
        sequence_length = ctx.sequence_length
        if ctx.hidden_requires_grad:
            _release_cache_for_allocation(
                grad_output,
                int(prod(ctx.input_shape))
                * torch.empty((), dtype=ctx.input_dtype).element_size(),
                target="safe_tiled_ffn_hidden_gradient",
            )
            hidden_gradient = torch.empty(
                ctx.input_shape,
                device=ctx.input_device,
                dtype=ctx.input_dtype,
            )
        else:
            hidden_gradient = None
        parameter_gradients: list[torch.Tensor | None] = [None] * len(parameters)
        direct_lora_b_accumulators = {
            index: ChunkedLoraBGradientAccumulator(torch.empty_like(parameter))
            for index, (name, parameter) in enumerate(
                zip(ctx.parameter_names, parameters, strict=True)
            )
            if sequence_length > 0 and name in ctx.direct_lora_b_parameter_names
        }

        for start in range(0, sequence_length, ctx.chunk_tokens):
            end = min(sequence_length, start + ctx.chunk_tokens)
            if ctx.inplace_output:
                host_chunk = ctx.hidden_states_host[..., start:end, :]
                byte_count = host_chunk.numel() * host_chunk.element_size()
                with _transfer_range(
                    grad_output,
                    component=ctx.component,
                    direction="h2d",
                    byte_count=byte_count,
                ):
                    chunk = host_chunk.to(
                        device=ctx.input_device,
                        dtype=ctx.input_dtype,
                        non_blocking=False,
                    )
            else:
                chunk = hidden_states[..., start:end, :].detach()
            chunk.requires_grad_(ctx.hidden_requires_grad)
            functional_parameters = {
                name: parameter.detach().requires_grad_(True)
                for name, parameter in zip(ctx.parameter_names, parameters, strict=True)
            }
            active_accumulators = {
                id(functional_parameters[ctx.parameter_names[index]]): accumulator
                for index, accumulator in direct_lora_b_accumulators.items()
            }
            module = ctx.module
            module._distillschedule_functional_recompute = True
            try:
                with activate_chunked_lora_b_gradient_accumulators(
                    active_accumulators
                ):
                    with torch.enable_grad():
                        output = functional_call(
                            module,
                            functional_parameters,
                            (chunk,),
                            strict=False,
                        )
                    gradient_inputs = (
                        [chunk] if ctx.hidden_requires_grad else []
                    ) + list(functional_parameters.values())
                    gradients = torch.autograd.grad(
                        output,
                        gradient_inputs,
                        grad_output[..., start:end, :],
                        allow_unused=True,
                    )
            finally:
                module._distillschedule_functional_recompute = False

            offset = 0
            if ctx.hidden_requires_grad:
                hidden_gradient[..., start:end, :].copy_(gradients[0])
                offset = 1
            for index, gradient in enumerate(gradients[offset:]):
                if index in direct_lora_b_accumulators:
                    if gradient is not None:
                        raise RuntimeError(
                            "direct LoRA-B accumulation unexpectedly returned a per-tile gradient "
                            f"for {ctx.parameter_names[index]}"
                        )
                    continue
                if gradient is None:
                    continue
                if parameter_gradients[index] is None:
                    parameter_gradients[index] = gradient
                else:
                    parameter_gradients[index].add_(gradient)

        expected_contributions = (
            (sequence_length + ctx.chunk_tokens - 1) // ctx.chunk_tokens
            if sequence_length > 0
            else 0
        )
        for index, accumulator in direct_lora_b_accumulators.items():
            if accumulator.contribution_count != expected_contributions:
                raise RuntimeError(
                    "direct LoRA-B accumulation did not cover every recompute tile for "
                    f"{ctx.parameter_names[index]}: expected {expected_contributions}, "
                    f"observed {accumulator.contribution_count}"
                )
            parameter_gradients[index] = accumulator.tensor
        ctx.module._distillschedule_direct_lora_b_last_contribution_counts = {
            ctx.parameter_names[index]: accumulator.contribution_count
            for index, accumulator in direct_lora_b_accumulators.items()
        }

        if ctx.inplace_output:
            ctx.hidden_states_host = None
        return (None, hidden_gradient, None, None, *parameter_gradients)


def install_safe_tiled_ffn(
    model: torch.nn.Module,
    *,
    chunk_tokens: int = 512,
    logical_component: str = "unknown",
) -> int:
    """Patch Diffusers-style FeedForward modules and return the number patched."""
    if chunk_tokens <= 0:
        raise ValueError("safe tiled FFN chunk size must be positive")
    patched = 0
    for module in model.modules():
        if type(module).__name__ != "FeedForward":
            continue
        if getattr(module, "_distillschedule_safe_tiled_ffn", False):
            continue
        parameter_names = tuple(
            name for name, parameter in module.named_parameters() if parameter.requires_grad
        )
        if not parameter_names:
            continue
        module._distillschedule_original_forward = module.forward
        module._distillschedule_safe_tiled_ffn = True
        module._distillschedule_ffn_chunk_tokens = int(chunk_tokens)
        module._distillschedule_ffn_component = str(logical_component)
        module._distillschedule_ffn_parameter_names = parameter_names
        direct_lora_b_parameter_names = []
        for child_name, child in module.named_modules():
            if not getattr(child, "_distillschedule_chunked_lora_b", False):
                continue
            prefix = f"{child_name}." if child_name else ""
            active_adapters = tuple(child.active_adapters)
            for adapter_name in active_adapters:
                if adapter_name not in child.lora_B:
                    continue
                lora_b = child.lora_B[adapter_name]
                if lora_b.weight.requires_grad:
                    direct_lora_b_parameter_names.append(
                        f"{prefix}lora_B.{adapter_name}.weight"
                    )
        unknown_direct_names = set(direct_lora_b_parameter_names).difference(
            parameter_names
        )
        if unknown_direct_names:
            raise RuntimeError(
                "chunked LoRA-B parameters are absent from safe-tiled parameter list: "
                + ", ".join(sorted(unknown_direct_names))
            )
        module._distillschedule_direct_lora_b_parameter_names = tuple(
            direct_lora_b_parameter_names
        )
        module._distillschedule_functional_recompute = False

        def tiled_forward(self, hidden_states: torch.Tensor, *args, **kwargs):
            if self._distillschedule_functional_recompute:
                return self._distillschedule_original_forward(
                    hidden_states, *args, **kwargs
                )
            if args or kwargs:
                return self._distillschedule_original_forward(
                    hidden_states, *args, **kwargs
                )
            named_parameters = dict(self.named_parameters())
            parameters = tuple(
                named_parameters[name]
                for name in self._distillschedule_ffn_parameter_names
            )
            return _SafeTiledModule.apply(
                self,
                hidden_states,
                self._distillschedule_ffn_chunk_tokens,
                self._distillschedule_ffn_parameter_names,
                *parameters,
            )

        module.forward = MethodType(tiled_forward, module)
        patched += 1
    return patched

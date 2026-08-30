from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from distillschedule.runtime.helios_modulated_norm_fusion import (
    install_exact_modulated_norm_epilogue,
)
from distillschedule.runtime.p0_parity import tensor_parity_metrics


def _materialize(table, temb, index, start=None, end=None):
    del start, end
    return (table[:, index, :] + temb[:, index, :].float()).unsqueeze(1)


def _rebuild_temb(descriptor, inputs):
    assert descriptor == ("tensor", ())
    assert len(inputs) == 1
    return inputs[0]


class _SourceChunkedNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden,
        norm,
        table,
        scale_index,
        shift_index,
        chunk_tokens,
        descriptor,
        *temb_inputs,
    ):
        ctx.norm = norm
        ctx.scale_shift_table = table
        ctx.scale_index = int(scale_index)
        ctx.shift_index = int(shift_index)
        ctx.chunk_tokens = int(chunk_tokens)
        ctx.descriptor = descriptor
        ctx.sequence_length = int(hidden.shape[1])
        ctx.save_for_backward(hidden, *temb_inputs)
        temb = _rebuild_temb(descriptor, temb_inputs)
        chunks = []
        for start in range(0, ctx.sequence_length, ctx.chunk_tokens):
            end = min(ctx.sequence_length, start + ctx.chunk_tokens)
            scale = _materialize(table, temb, ctx.scale_index, start, end)
            shift = _materialize(table, temb, ctx.shift_index, start, end)
            chunks.append(
                (
                    norm(hidden[:, start:end].float()) * (1 + scale) + shift
                ).to(hidden.dtype)
            )
        return torch.cat(chunks, dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        hidden, *temb_inputs = ctx.saved_tensors
        hidden_grad = torch.empty_like(hidden)
        temb_grad = torch.zeros_like(temb_inputs[0])
        for start in range(0, ctx.sequence_length, ctx.chunk_tokens):
            end = min(ctx.sequence_length, start + ctx.chunk_tokens)
            hidden_chunk = hidden[:, start:end].detach().requires_grad_(True)
            temb = temb_inputs[0].detach().requires_grad_(True)
            with torch.enable_grad():
                scale = _materialize(
                    ctx.scale_shift_table, temb, ctx.scale_index, start, end
                )
                shift = _materialize(
                    ctx.scale_shift_table, temb, ctx.shift_index, start, end
                )
                output = (
                    ctx.norm(hidden_chunk.float()) * (1 + scale) + shift
                ).to(hidden.dtype)
            chunk_grad, chunk_temb_grad = torch.autograd.grad(
                output,
                (hidden_chunk, temb),
                grad_output[:, start:end],
            )
            hidden_grad[:, start:end].copy_(chunk_grad)
            temb_grad.add_(chunk_temb_grad)
        return hidden_grad, None, None, None, None, None, None, temb_grad


_SOURCE_FORWARD = _SourceChunkedNorm.forward
_SOURCE_BACKWARD = _SourceChunkedNorm.backward


def _module():
    class FreshSourceChunkedNorm(torch.autograd.Function):
        forward = staticmethod(_SOURCE_FORWARD)
        backward = staticmethod(_SOURCE_BACKWARD)

    return SimpleNamespace(
        __file__=__file__,
        _TembAwareChunkedNorm=FreshSourceChunkedNorm,
        _materialize_scale_shift_component=_materialize,
        _rebuild_temb=_rebuild_temb,
    )


def _run(function, hidden, norm, table, temb):
    return function.apply(
        hidden,
        norm,
        table,
        1,
        2,
        3,
        ("tensor", ()),
        temb,
    )


def test_install_is_default_off() -> None:
    module = _module()
    original_forward = module._TembAwareChunkedNorm.forward
    receipt = install_exact_modulated_norm_epilogue(
        enabled=False,
        role="generator",
        transformer_module=module,
        fused_epilogue=lambda *_args: None,
    )
    assert receipt["enabled"] is False
    assert receipt["disabled_reason"] == "NOT_REQUESTED"
    assert module._TembAwareChunkedNorm.forward is original_forward


def test_forward_and_gradients_are_exact_and_source_backward_is_unchanged() -> None:
    torch.manual_seed(31)
    module = _module()
    function = module._TembAwareChunkedNorm
    norm_reference = torch.nn.LayerNorm(8, elementwise_affine=False)
    norm_candidate = copy.deepcopy(norm_reference)
    table = torch.randn(1, 6, 8, dtype=torch.float32)
    hidden_reference = torch.randn(1, 7, 8, dtype=torch.bfloat16).requires_grad_()
    hidden_candidate = hidden_reference.detach().clone().requires_grad_()
    temb_reference = torch.randn(1, 6, 8, dtype=torch.bfloat16).requires_grad_()
    temb_candidate = temb_reference.detach().clone().requires_grad_()
    upstream = torch.randn_like(hidden_reference)

    expected = _run(
        function, hidden_reference, norm_reference, table, temb_reference
    )
    expected.backward(upstream)
    source_backward = function.backward

    def fused(normalized, scale, shift, output):
        output.copy_((normalized * (1 + scale) + shift).to(output.dtype))
        return output

    receipt = install_exact_modulated_norm_epilogue(
        enabled=True,
        role="generator",
        transformer_module=module,
        fused_epilogue=fused,
    )
    actual = _run(function, hidden_candidate, norm_candidate, table, temb_candidate)
    actual.backward(upstream)

    assert torch.equal(expected, actual)
    assert torch.equal(hidden_reference.grad, hidden_candidate.grad)
    assert torch.equal(temb_reference.grad, temb_candidate.grad)
    assert function.backward is source_backward
    assert receipt["enabled"] is True
    assert receipt["fused_call_count"] == 1
    assert receipt["fused_chunk_count"] == 3
    assert receipt["reference_fallback_count"] == 0


@pytest.mark.parametrize("batch_size", [1, 3])
def test_table_component_materialization_is_exact(batch_size: int) -> None:
    torch.manual_seed(41)
    module = _module()
    function = module._TembAwareChunkedNorm
    norm_reference = torch.nn.LayerNorm(8, elementwise_affine=False)
    norm_candidate = copy.deepcopy(norm_reference)
    table = torch.randn(1, 6, 8, dtype=torch.float32)
    hidden_reference = torch.randn(
        batch_size, 7, 8, dtype=torch.bfloat16
    ).requires_grad_()
    hidden_candidate = hidden_reference.detach().clone().requires_grad_()
    temb_reference = torch.randn(
        batch_size, 6, 8, dtype=torch.bfloat16
    ).requires_grad_()
    temb_candidate = temb_reference.detach().clone().requires_grad_()
    upstream = torch.randn_like(hidden_reference)

    expected = _run(
        function, hidden_reference, norm_reference, table, temb_reference
    )
    expected.backward(upstream)

    def fused_epilogue(normalized, scale, shift, output):
        output.copy_((normalized * (1 + scale) + shift).to(output.dtype))
        return output

    def fused_materialization(
        normalized,
        source_table,
        scale_index,
        shift_index,
        scale_component,
        shift_component,
        output,
    ):
        scale = source_table[:, scale_index, :].unsqueeze(1) + scale_component.float()
        shift = source_table[:, shift_index, :].unsqueeze(1) + shift_component.float()
        output.copy_((normalized * (1 + scale) + shift).to(output.dtype))
        return output

    receipt = install_exact_modulated_norm_epilogue(
        enabled=True,
        role="generator",
        fuse_materialization=True,
        transformer_module=module,
        fused_epilogue=fused_epilogue,
        fused_materialization=fused_materialization,
    )
    actual = _run(function, hidden_candidate, norm_candidate, table, temb_candidate)
    actual.backward(upstream)

    assert torch.equal(expected, actual)
    assert torch.equal(hidden_reference.grad, hidden_candidate.grad)
    assert torch.equal(temb_reference.grad, temb_candidate.grad)
    assert receipt["materialization_requested"] is True
    assert receipt["materialization_fused_call_count"] == 1
    assert receipt["materialization_fused_chunk_count"] == 3
    assert receipt["reference_fallback_count"] == 0
    assert receipt["scope"] == (
        "ALL_ROLE_REGISTERED_B1_B3_MODULATED_NORM_FORWARD_EPILOGUE"
    )


def test_explicit_backward_matches_source_under_frozen_p0_gate() -> None:
    torch.manual_seed(47)
    module = _module()
    function = module._TembAwareChunkedNorm
    norm_reference = torch.nn.LayerNorm(8, elementwise_affine=False)
    norm_candidate = copy.deepcopy(norm_reference)
    table = torch.randn(1, 6, 8, dtype=torch.float32)
    hidden_reference = torch.randn(1, 7, 8, dtype=torch.bfloat16).requires_grad_()
    hidden_candidate = hidden_reference.detach().clone().requires_grad_()
    temb_reference = torch.randn(1, 6, 8, dtype=torch.bfloat16).requires_grad_()
    temb_candidate = temb_reference.detach().clone().requires_grad_()
    upstream = torch.randn_like(hidden_reference)

    expected = _run(
        function, hidden_reference, norm_reference, table, temb_reference
    )
    expected.backward(upstream)
    source_backward = function.backward

    def fused_epilogue(normalized, scale, shift, output):
        output.copy_((normalized * (1 + scale) + shift).to(output.dtype))
        return output

    receipt = install_exact_modulated_norm_epilogue(
        enabled=True,
        role="generator",
        fuse_backward=True,
        transformer_module=module,
        fused_epilogue=fused_epilogue,
    )
    actual = _run(function, hidden_candidate, norm_candidate, table, temb_candidate)
    actual.backward(upstream)

    assert torch.equal(expected, actual)
    assert tensor_parity_metrics(
        hidden_reference.grad, hidden_candidate.grad
    )["status"] == "PASS"
    assert tensor_parity_metrics(
        temb_reference.grad, temb_candidate.grad
    )["status"] == "PASS"
    assert function.backward is not source_backward
    assert receipt["backward_requested"] is True
    assert receipt["backward_call_count"] == 1
    assert receipt["backward_chunk_count"] == 3
    assert receipt["backward_layout_counts"] == {"tensor_3d": 3, "tensor": 3}
    assert receipt["backward_hidden_element_count"] == hidden_candidate.numel()


def test_launcher_controller_and_summarizer_are_fail_closed() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (
        root / "scripts/distillschedule/launch_static_fsdp2_p0_world16_node.sh"
    ).read_text()
    controller = (
        root / "scripts/distillschedule/run_static_fsdp2_p0_world16.py"
    ).read_text()
    summarizer = (
        root / "scripts/distillschedule/summarize_static_fsdp2_p0_world16.py"
    ).read_text()
    for source in (launcher, controller, summarizer):
        assert "exact_modulated_norm_epilogue" in source
        assert "exact_modulated_norm_backward" in source
    assert "DISTILLSCHEDULE_HELIOS_EXACT_MODULATED_NORM_EPILOGUE" in launcher
    assert "DISTILLSCHEDULE_HELIOS_EXACT_MODULATED_NORM_MATERIALIZATION" in launcher
    assert "DISTILLSCHEDULE_HELIOS_EXACT_MODULATED_NORM_BACKWARD" in launcher
    assert "rank_exact_modulated_norm_epilogue" in summarizer
    assert "materialization_fused_chunk_count" in summarizer
    assert "backward_chunk_count" in summarizer
    assert "reference_fallback_count" in summarizer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only Triton parity")
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("sequence_length", [1, 511, 512])
@pytest.mark.parametrize("broadcast_modulation", [False, True])
def test_real_triton_kernel_is_bitwise_exact(
    sequence_length: int, broadcast_modulation: bool, batch_size: int
) -> None:
    from distillschedule.runtime.helios_modulated_norm_kernel import (
        fused_modulated_norm_epilogue,
    )

    generator = torch.Generator(device="cuda").manual_seed(37)
    normalized = torch.randn(
        (batch_size, sequence_length, 5120),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    modulation_length = 1 if broadcast_modulation else sequence_length
    scale = torch.randn(
        (batch_size, modulation_length, 5120),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    shift = torch.randn_like(scale)
    if broadcast_modulation:
        scale = scale.expand_as(normalized)
        shift = shift.expand_as(normalized)
    expected = (normalized * (1 + scale) + shift).to(torch.bfloat16)
    actual = torch.empty_like(expected)
    fused_modulated_norm_epilogue(normalized, scale, shift, actual)
    torch.cuda.synchronize()
    assert torch.equal(expected, actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only Triton parity")
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("sequence_length", [1, 127, 512])
@pytest.mark.parametrize("broadcast_modulation", [False, True])
@pytest.mark.parametrize("table_dtype", [torch.bfloat16, torch.float32])
def test_real_materialization_kernel_is_bitwise_exact(
    sequence_length: int,
    broadcast_modulation: bool,
    table_dtype: torch.dtype,
    batch_size: int,
) -> None:
    from distillschedule.runtime.helios_modulated_norm_kernel import (
        fused_modulated_norm_materialization,
    )

    generator = torch.Generator(device="cuda").manual_seed(43 + sequence_length)
    normalized = torch.randn(
        (batch_size, sequence_length, 5120),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    table = torch.randn(
        (1, 6, 5120),
        device="cuda",
        dtype=table_dtype,
        generator=generator,
    )
    if broadcast_modulation:
        temb = torch.randn(
            (batch_size, 6, 5120),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        scale_component = temb[:, 1, :].unsqueeze(1).expand_as(normalized)
        shift_component = temb[:, 2, :].unsqueeze(1).expand_as(normalized)
    else:
        temb = torch.randn(
            (batch_size, sequence_length, 6, 5120),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        scale_component = temb[:, :, 1, :]
        shift_component = temb[:, :, 2, :]
    scale = table[:, 1, :].unsqueeze(1) + scale_component.float()
    shift = table[:, 2, :].unsqueeze(1) + shift_component.float()
    expected = (normalized * (1 + scale) + shift).to(torch.bfloat16)
    actual = torch.empty_like(expected)
    fused_modulated_norm_materialization(
        normalized,
        table,
        1,
        2,
        scale_component,
        shift_component,
        actual,
    )
    torch.cuda.synchronize()
    assert torch.equal(expected, actual)

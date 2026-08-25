from __future__ import annotations

from types import SimpleNamespace

import pytest
torch = pytest.importorskip("torch")

from distillschedule.runtime.helios_residual_fusion import (
    generator_nograd_residual_fusion_scope,
    install_generator_nograd_residual_fusion,
)


class _PhaseLocalTemb:
    def __init__(self, segments):
        self.segments = tuple(segments)
        self.total_length = sum(int(length) for _value, length in self.segments)

    def materialize_component(self, index, token_start=None, token_end=None):
        start = 0 if token_start is None else int(token_start)
        end = self.total_length if token_end is None else int(token_end)
        views = []
        segment_start = 0
        for values, length in self.segments:
            segment_end = segment_start + int(length)
            overlap_start = max(start, segment_start)
            overlap_end = min(end, segment_end)
            if overlap_start < overlap_end:
                views.append(
                    values[:, index, :]
                    .unsqueeze(1)
                    .expand(-1, overlap_end - overlap_start, -1)
                )
            segment_start = segment_end
        if not views:
            return self.segments[0][0][:, index, :].unsqueeze(1)[:, :0, :]
        return views[0] if len(views) == 1 else torch.cat(views, dim=1)


def _reference(
    hidden,
    residual,
    scale_shift_table,
    temb,
    component_index,
    chunk_tokens,
    logical_component="unknown",
    residual_storage="cpu_inplace",
):
    del logical_component, residual_storage
    chunks = []
    for start in range(0, int(hidden.shape[1]), int(chunk_tokens)):
        end = min(int(hidden.shape[1]), start + int(chunk_tokens))
        gate = _materialize(
            scale_shift_table,
            temb,
            component_index,
            start,
            end,
        )
        chunks.append(
            (
                hidden[:, start:end].float()
                + residual[:, start:end].float() * gate
            ).to(hidden.dtype)
        )
    return torch.cat(chunks, dim=1)


def _materialize(table, temb, component_index, start=None, end=None):
    if isinstance(temb, _PhaseLocalTemb):
        component = temb.materialize_component(component_index, start, end)
        return table[:, component_index, :] + component.float()
    if temb.ndim == 4:
        return (
            table[:, component_index, :]
            + temb[:, slice(start, end), component_index, :].float()
        )
    return (
        table[:, component_index, :] + temb[:, component_index, :].float()
    ).unsqueeze(1)


def _flatten_temb(temb):
    if isinstance(temb, _PhaseLocalTemb):
        return (
            ("phase_local", tuple(int(length) for _value, length in temb.segments)),
            tuple(value for value, _length in temb.segments),
        )
    return ("tensor", ()), (temb,)


def _rebuild_temb(descriptor, inputs):
    kind, lengths = descriptor
    if kind == "tensor":
        assert lengths == ()
        assert len(inputs) == 1
        return inputs[0]
    assert kind == "phase_local"
    assert len(lengths) == len(inputs)
    return _PhaseLocalTemb(list(zip(inputs, lengths, strict=True)))


class _SourceTembAwareResidual(torch.autograd.Function):
    """Test copy of the pinned Helios cuda_out_of_place backward contract."""

    @staticmethod
    def forward(
        ctx,
        hidden,
        residual,
        table,
        component_index,
        chunk_tokens,
        descriptor,
        logical_component,
        residual_storage,
        *temb_inputs,
    ):
        del logical_component
        assert residual_storage == "cuda_out_of_place"
        ctx.table = table
        ctx.component_index = int(component_index)
        ctx.chunk_tokens = int(chunk_tokens)
        ctx.descriptor = descriptor
        ctx.hidden_dtype = hidden.dtype
        ctx.residual_dtype = residual.dtype
        ctx.sequence_length = int(hidden.shape[1])
        ctx.save_for_backward(residual, *temb_inputs)
        temb = _rebuild_temb(descriptor, tuple(temb_inputs))
        output = torch.empty_like(hidden)
        for start in range(0, ctx.sequence_length, ctx.chunk_tokens):
            end = min(ctx.sequence_length, start + ctx.chunk_tokens)
            gate = _materialize(
                table, temb, ctx.component_index, start, end
            )
            output[:, start:end].copy_(
                (
                    hidden[:, start:end].float()
                    + residual[:, start:end].float() * gate
                ).to(hidden.dtype)
            )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        residual, *saved_temb_inputs = ctx.saved_tensors
        hidden_gradient = (
            grad_output.to(ctx.hidden_dtype) if ctx.needs_input_grad[0] else None
        )
        residual_gradient = (
            torch.empty_like(residual) if ctx.needs_input_grad[1] else None
        )
        temb_gradients = [None for _ in saved_temb_inputs]
        temb_needs_grad = list(ctx.needs_input_grad[8:])
        if residual_gradient is not None or any(temb_needs_grad):
            for start in range(0, ctx.sequence_length, ctx.chunk_tokens):
                end = min(ctx.sequence_length, start + ctx.chunk_tokens)
                residual_chunk = residual[:, start:end].detach()
                residual_chunk.requires_grad_(ctx.needs_input_grad[1])
                functional_inputs = tuple(
                    value.detach().requires_grad_(needs_grad)
                    for value, needs_grad in zip(
                        saved_temb_inputs, temb_needs_grad, strict=True
                    )
                )
                functional_temb = _rebuild_temb(ctx.descriptor, functional_inputs)
                with torch.enable_grad():
                    gate = _materialize(
                        ctx.table,
                        functional_temb,
                        ctx.component_index,
                        start,
                        end,
                    )
                    product = residual_chunk.float() * gate
                gradient_inputs = (
                    [residual_chunk] if ctx.needs_input_grad[1] else []
                ) + [
                    value
                    for value, needs_grad in zip(
                        functional_inputs, temb_needs_grad, strict=True
                    )
                    if needs_grad
                ]
                gradients = torch.autograd.grad(
                    product,
                    gradient_inputs,
                    grad_output[:, start:end].float(),
                    allow_unused=True,
                )
                offset = 0
                if residual_gradient is not None:
                    residual_gradient[:, start:end].copy_(gradients[0])
                    offset = 1
                gradient_index = offset
                for index, needs_grad in enumerate(temb_needs_grad):
                    if not needs_grad:
                        continue
                    gradient = gradients[gradient_index]
                    current = temb_gradients[index]
                    if gradient is not None:
                        temb_gradients[index] = (
                            gradient if current is None else current + gradient
                        )
                    gradient_index += 1
        return (
            hidden_gradient,
            residual_gradient,
            None,
            None,
            None,
            None,
            None,
            None,
            *temb_gradients,
        )


def _source_reference(
    hidden,
    residual,
    table,
    temb,
    component_index,
    chunk_tokens,
    logical_component="unknown",
    residual_storage="cuda_out_of_place",
):
    descriptor, inputs = _flatten_temb(temb)
    return _SourceTembAwareResidual.apply(
        hidden,
        residual,
        table,
        component_index,
        chunk_tokens,
        descriptor,
        logical_component,
        residual_storage,
        *inputs,
    )


def _module() -> SimpleNamespace:
    return SimpleNamespace(
        __file__=__file__,
        _apply_chunked_gated_residual=_reference,
        _materialize_scale_shift_component=_materialize,
        _flatten_temb_inputs=_flatten_temb,
        _rebuild_temb=_rebuild_temb,
    )


def _fused(hidden, residual, gate, output):
    output.copy_((hidden.float() + residual.float() * gate).to(hidden.dtype))
    return output


def _fused_backward(grad_output, gate, grad_residual):
    grad_residual.copy_((grad_output.float() * gate).to(grad_residual.dtype))
    return grad_residual


def _fused_trainable_backward(
    grad_output, residual, gate, grad_residual, temb_product
):
    grad_residual.copy_((grad_output.float() * gate).to(grad_residual.dtype))
    temb_product.copy_(
        (residual.float() * grad_output.float()).to(temb_product.dtype)
    )
    return grad_residual, temb_product


def test_residual_fusion_is_default_off() -> None:
    module = _module()
    receipt = install_generator_nograd_residual_fusion(
        enabled=False,
        role="generator",
        transformer_module=module,
        fused_chunk=_fused,
    )
    assert receipt["enabled"] is False
    assert receipt["disabled_reason"] == "NOT_REQUESTED"
    assert module._apply_chunked_gated_residual is _reference


def test_residual_fusion_is_generator_nograd_only_and_exact() -> None:
    module = _module()
    torch.manual_seed(17)
    hidden = torch.randn((1, 7, 5120), dtype=torch.bfloat16)
    residual = torch.randn_like(hidden)
    table = torch.randn((1, 6, 5120), dtype=torch.float32)
    temb = torch.randn((1, 6, 5120), dtype=torch.float32)
    with torch.no_grad():
        expected = _reference(hidden, residual, table, temb, 2, 4)
    receipt = install_generator_nograd_residual_fusion(
        enabled=True,
        role="generator",
        transformer_module=module,
        fused_chunk=_fused,
    )
    with torch.no_grad(), generator_nograd_residual_fusion_scope():
        actual = module._apply_chunked_gated_residual(
            hidden,
            residual,
            table,
            temb,
            2,
            4,
            "unknown_after_fsdp_wrap",
            "cuda_out_of_place",
        )
    assert torch.equal(expected, actual)
    assert receipt["fused_call_count"] == 1
    assert receipt["fused_chunk_count"] == 2
    assert receipt["scope_active_call_count"] == 1
    assert receipt["reference_fallback_count"] == 0

    # Grad-enabled Generator and every Critic call stay on the source helper.
    module._apply_chunked_gated_residual(
        hidden,
        residual,
        table,
        temb,
        2,
        4,
        "generator",
        "cuda_out_of_place",
    )
    with torch.no_grad():
        module._apply_chunked_gated_residual(
            hidden,
            residual,
            table,
            temb,
            2,
            4,
            "critic",
            "cuda_out_of_place",
        )
    assert receipt["grad_bypass_count"] == 1
    assert receipt["non_generator_bypass_count"] == 1


def test_exact_grad_residual_fusion_matches_forward_and_gradients() -> None:
    torch.manual_seed(29)
    hidden_reference = torch.randn(
        (1, 7, 5120), dtype=torch.bfloat16, requires_grad=True
    )
    residual_reference = torch.randn_like(hidden_reference, requires_grad=True)
    table = torch.randn((1, 6, 5120), dtype=torch.float32)
    temb = torch.randn((1, 6, 5120), dtype=torch.float32)
    upstream = torch.randn_like(hidden_reference)
    expected = _source_reference(
        hidden_reference,
        residual_reference,
        table,
        temb,
        2,
        4,
        "generator",
        "cuda_out_of_place",
    )
    expected.backward(upstream)

    hidden_actual = hidden_reference.detach().clone().requires_grad_(True)
    residual_actual = residual_reference.detach().clone().requires_grad_(True)
    module = _module()
    receipt = install_generator_nograd_residual_fusion(
        enabled=False,
        exact_grad_enabled=True,
        role="generator",
        transformer_module=module,
        fused_chunk=_fused,
        fused_backward_chunk=_fused_backward,
    )
    actual = module._apply_chunked_gated_residual(
        hidden_actual,
        residual_actual,
        table,
        temb,
        2,
        4,
        "generator",
        "cuda_out_of_place",
    )
    actual.backward(upstream)
    assert torch.equal(expected, actual)
    assert torch.equal(hidden_reference.grad, hidden_actual.grad)
    assert torch.equal(residual_reference.grad, residual_actual.grad)
    assert receipt["grad_fused_call_count"] == 1
    assert receipt["grad_fused_chunk_count"] == 2
    assert receipt["grad_backward_call_count"] == 1
    assert receipt["grad_backward_chunk_count"] == 2
    assert receipt["grad_saved_residual_tensor_count"] == 1
    assert receipt["grad_frozen_temb_call_count"] == 1
    assert receipt["grad_trainable_temb_call_count"] == 0
    assert receipt["grad_temb_gradient_tensor_count"] == 0
    assert receipt["grad_reference_fallback_count"] == 0


@pytest.mark.parametrize("temb_rank", [3, 4])
def test_exact_grad_fusion_matches_trainable_tensor_temb(temb_rank: int) -> None:
    torch.manual_seed(41 + temb_rank)
    module = _module()
    receipt = install_generator_nograd_residual_fusion(
        enabled=False,
        exact_grad_enabled=True,
        role="generator",
        transformer_module=module,
        fused_chunk=_fused,
        fused_backward_chunk=_fused_backward,
    )
    hidden_reference = torch.randn(
        (1, 7, 32), dtype=torch.bfloat16, requires_grad=True
    )
    residual_reference = torch.randn_like(hidden_reference, requires_grad=True)
    table = torch.randn((1, 6, 32), dtype=torch.float32)
    temb_shape = (1, 6, 32) if temb_rank == 3 else (1, 7, 6, 32)
    temb_reference = torch.randn(
        temb_shape, dtype=torch.float32, requires_grad=True
    )
    upstream = torch.randn_like(hidden_reference)
    expected = _source_reference(
        hidden_reference,
        residual_reference,
        table,
        temb_reference,
        2,
        4,
        "generator",
        "cuda_out_of_place",
    )
    expected.backward(upstream)

    hidden_actual = hidden_reference.detach().clone().requires_grad_(True)
    residual_actual = residual_reference.detach().clone().requires_grad_(True)
    temb_actual = temb_reference.detach().clone().requires_grad_(True)
    actual = module._apply_chunked_gated_residual(
        hidden_actual,
        residual_actual,
        table,
        temb_actual,
        2,
        4,
        "generator",
        "cuda_out_of_place",
    )
    actual.backward(upstream)
    assert torch.equal(expected, actual)
    assert torch.equal(hidden_reference.grad, hidden_actual.grad)
    assert torch.equal(residual_reference.grad, residual_actual.grad)
    assert torch.equal(temb_reference.grad, temb_actual.grad)
    assert receipt["grad_fused_call_count"] == 1
    assert receipt["grad_trainable_temb_call_count"] == 1
    assert receipt["grad_temb_descriptor_counts"] == {"tensor": 1}
    assert receipt["grad_temb_dtype_counts"] == {"torch.float32": 1}
    assert receipt["grad_temb_gradient_tensor_count"] == 2
    assert receipt["grad_reference_fallback_count"] == 0


@pytest.mark.parametrize("temb_dtype", [torch.float32, torch.bfloat16])
def test_exact_grad_fusion_matches_trainable_phase_local_temb(temb_dtype) -> None:
    torch.manual_seed(47)
    hidden_reference = torch.randn(
        (1, 7, 32), dtype=torch.bfloat16, requires_grad=True
    )
    residual_reference = torch.randn_like(hidden_reference, requires_grad=True)
    table = torch.randn((1, 6, 32), dtype=torch.float32)
    first_reference = torch.randn(
        (1, 6, 32), dtype=temb_dtype, requires_grad=True
    )
    second_reference = torch.randn(
        (1, 6, 32), dtype=temb_dtype, requires_grad=True
    )
    temb_reference = _PhaseLocalTemb(
        [(first_reference, 3), (second_reference, 4)]
    )
    upstream = torch.randn_like(hidden_reference)
    expected = _source_reference(
        hidden_reference,
        residual_reference,
        table,
        temb_reference,
        4,
        4,
        "generator",
        "cuda_out_of_place",
    )
    expected.backward(upstream)

    hidden_actual = hidden_reference.detach().clone().requires_grad_(True)
    residual_actual = residual_reference.detach().clone().requires_grad_(True)
    first_actual = first_reference.detach().clone().requires_grad_(True)
    second_actual = second_reference.detach().clone().requires_grad_(True)
    temb_actual = _PhaseLocalTemb([(first_actual, 3), (second_actual, 4)])
    module = _module()
    receipt = install_generator_nograd_residual_fusion(
        enabled=False,
        exact_grad_enabled=True,
        role="generator",
        transformer_module=module,
        fused_chunk=_fused,
        fused_backward_chunk=_fused_backward,
        fused_trainable_backward_chunk=(
            _fused_trainable_backward
            if temb_dtype is torch.bfloat16
            else None
        ),
    )
    actual = module._apply_chunked_gated_residual(
        hidden_actual,
        residual_actual,
        table,
        temb_actual,
        4,
        4,
        "generator",
        "cuda_out_of_place",
    )
    actual.backward(upstream)
    assert torch.equal(expected, actual)
    assert torch.equal(hidden_reference.grad, hidden_actual.grad)
    assert torch.equal(residual_reference.grad, residual_actual.grad)
    assert torch.equal(first_reference.grad, first_actual.grad)
    assert torch.equal(second_reference.grad, second_actual.grad)
    assert receipt["grad_fused_call_count"] == 1
    assert receipt["grad_trainable_temb_call_count"] == 1
    assert receipt["grad_temb_descriptor_counts"] == {"phase_local": 1}
    assert receipt["grad_temb_dtype_counts"] == {str(temb_dtype): 2}
    assert receipt["grad_temb_gradient_tensor_count"] == 3
    assert receipt["grad_phase_local_fused_staging_chunk_count"] == (
        2 if temb_dtype is torch.bfloat16 else 0
    )
    assert receipt["grad_reference_fallback_count"] == 0


def test_launcher_and_policy_are_fail_closed() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    launcher = (
        root / "scripts/distillschedule/launch_static_fsdp2_p0_world16_node.sh"
    ).read_text()
    summarizer = (
        root / "scripts/distillschedule/summarize_static_fsdp2_p0_world16.py"
    ).read_text()
    assert "DISTILLSCHEDULE_HELIOS_GENERATOR_NOGRAD_RESIDUAL_FUSION" in launcher
    assert "DISTILLSCHEDULE_HELIOS_EXACT_GRAD_RESIDUAL_FUSION" in launcher
    assert '"${GENERATOR_NOGRAD_RESIDUAL_FUSION}" != on || "${ROLLOUT_BATCH_SIZE}" == 1' in launcher
    assert '"${EXACT_GRAD_RESIDUAL_FUSION}" != on || "${CRITIC_RESIDUAL_STORAGE}" == cuda_out_of_place' in launcher
    assert "generator_nograd_residual_fusion" in summarizer
    assert "exact_grad_residual_fusion" in summarizer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only Triton parity")
@pytest.mark.parametrize("contiguous_gate", [False, True])
def test_real_triton_kernel_is_bitwise_exact(contiguous_gate: bool) -> None:
    from distillschedule.runtime.helios_residual_kernel import (
        fused_gated_residual_backward_chunk,
        fused_gated_residual_chunk,
        fused_gated_residual_trainable_temb_backward_chunk,
    )

    generator = torch.Generator(device="cuda").manual_seed(23)
    hidden = torch.randn(
        (1, 511, 5120), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    residual = torch.randn_like(hidden)
    gate = torch.randn(
        (1, 1, 5120), device="cuda", dtype=torch.float32, generator=generator
    ).expand_as(hidden)
    if contiguous_gate:
        gate = gate.contiguous()
    expected = (hidden.float() + residual.float() * gate).to(torch.bfloat16)
    actual = torch.empty_like(hidden)
    fused_gated_residual_chunk(hidden, residual, gate, actual)
    grad_output = torch.randn_like(hidden)
    expected_grad = (grad_output.float() * gate).to(torch.bfloat16)
    actual_grad = torch.empty_like(hidden)
    fused_gated_residual_backward_chunk(grad_output, gate, actual_grad)
    expected_temb_product = (
        residual.float() * grad_output.float()
    ).to(torch.bfloat16)
    actual_temb_product = torch.empty_like(hidden)
    actual_trainable_grad = torch.empty_like(hidden)
    fused_gated_residual_trainable_temb_backward_chunk(
        grad_output,
        residual,
        gate,
        actual_trainable_grad,
        actual_temb_product,
    )
    torch.cuda.synchronize()
    assert torch.equal(expected, actual)
    assert torch.equal(expected_grad, actual_grad)
    assert torch.equal(expected_grad, actual_trainable_grad)
    assert torch.equal(expected_temb_product, actual_temb_product)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only autograd parity")
def test_real_triton_exact_grad_autograd_is_bitwise_exact() -> None:
    from distillschedule.runtime.helios_residual_kernel import (
        fused_gated_residual_backward_chunk,
        fused_gated_residual_chunk,
    )

    generator = torch.Generator(device="cuda").manual_seed(31)
    hidden_reference = torch.randn(
        (1, 511, 5120),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=True,
    )
    residual_reference = torch.randn_like(hidden_reference, requires_grad=True)
    table = torch.randn(
        (1, 6, 5120), device="cuda", dtype=torch.float32, generator=generator
    )
    temb = torch.randn(
        (1, 6, 5120),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
        requires_grad=True,
    )
    upstream = torch.randn_like(hidden_reference)
    expected = _source_reference(
        hidden_reference,
        residual_reference,
        table,
        temb,
        2,
        127,
        "generator",
        "cuda_out_of_place",
    )
    expected.backward(upstream)

    hidden_actual = hidden_reference.detach().clone().requires_grad_(True)
    residual_actual = residual_reference.detach().clone().requires_grad_(True)
    temb_actual = temb.detach().clone().requires_grad_(True)
    module = _module()
    receipt = install_generator_nograd_residual_fusion(
        enabled=False,
        exact_grad_enabled=True,
        role="generator",
        transformer_module=module,
        fused_chunk=fused_gated_residual_chunk,
        fused_backward_chunk=fused_gated_residual_backward_chunk,
    )
    actual = module._apply_chunked_gated_residual(
        hidden_actual,
        residual_actual,
        table,
        temb_actual,
        2,
        127,
        "generator",
        "cuda_out_of_place",
    )
    actual.backward(upstream)
    torch.cuda.synchronize()
    assert torch.equal(expected, actual)
    assert torch.equal(hidden_reference.grad, hidden_actual.grad)
    assert torch.equal(residual_reference.grad, residual_actual.grad)
    assert torch.equal(temb.grad, temb_actual.grad)
    assert receipt["grad_fused_call_count"] == 1
    assert receipt["grad_fused_chunk_count"] == 5
    assert receipt["grad_backward_call_count"] == 1
    assert receipt["grad_backward_chunk_count"] == 5
    assert receipt["grad_trainable_temb_call_count"] == 1
    assert receipt["grad_temb_dtype_counts"] == {"torch.float32": 1}
    assert receipt["grad_temb_gradient_tensor_count"] == 5
    assert receipt["grad_reference_fallback_count"] == 0

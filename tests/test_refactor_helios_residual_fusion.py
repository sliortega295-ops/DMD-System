from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest

from distillschedule.runtime.helios_residual_fusion import (
    generator_nograd_residual_fusion_scope,
    install_generator_nograd_residual_fusion,
)


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
    del component_index, chunk_tokens, logical_component, residual_storage
    gate = scale_shift_table + temb
    return (hidden.float() + residual.float() * gate).to(hidden.dtype)


def _materialize(table, temb, component_index, start=None, end=None):
    del component_index
    gate = table + temb
    if int(gate.shape[1]) == 1:
        gate = gate.expand(1, 7, 5120)
    return gate[:, start:end] if start is not None else gate


def _flatten_temb(temb):
    return ("tensor", ()), (temb,)


def _rebuild_temb(descriptor, inputs):
    assert descriptor == ("tensor", ())
    assert len(inputs) == 1
    return inputs[0]


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
    table = torch.randn((1, 1, 5120), dtype=torch.float32)
    temb = torch.randn((1, 1, 5120), dtype=torch.float32)
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
    table = torch.randn((1, 1, 5120), dtype=torch.float32)
    temb = torch.randn((1, 1, 5120), dtype=torch.float32)
    upstream = torch.randn_like(hidden_reference)
    expected = _reference(
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
    assert receipt["grad_saved_residual_tensor_count"] == 0
    assert receipt["grad_reference_fallback_count"] == 0


def test_exact_grad_fusion_fails_closed_for_trainable_temb() -> None:
    module = _module()
    receipt = install_generator_nograd_residual_fusion(
        enabled=False,
        exact_grad_enabled=True,
        role="generator",
        transformer_module=module,
        fused_chunk=_fused,
        fused_backward_chunk=_fused_backward,
    )
    hidden = torch.randn((1, 7, 5120), dtype=torch.bfloat16, requires_grad=True)
    residual = torch.randn_like(hidden, requires_grad=True)
    table = torch.randn((1, 1, 5120), dtype=torch.float32)
    temb = torch.randn((1, 1, 5120), dtype=torch.float32, requires_grad=True)
    module._apply_chunked_gated_residual(
        hidden,
        residual,
        table,
        temb,
        2,
        4,
        "generator",
        "cuda_out_of_place",
    ).sum().backward()
    assert receipt["grad_fused_call_count"] == 0
    assert receipt["grad_reference_fallback_count"] == 1
    assert receipt["grad_reference_fallback_reasons"] == {
        "TRAINABLE_TEMB_INPUT": 1
    }


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
    torch.cuda.synchronize()
    assert torch.equal(expected, actual)
    assert torch.equal(expected_grad, actual_grad)


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
        (1, 1, 5120), device="cuda", dtype=torch.float32, generator=generator
    )
    temb = torch.randn(
        (1, 511, 5120), device="cuda", dtype=torch.float32, generator=generator
    )
    upstream = torch.randn_like(hidden_reference)
    expected = _reference(
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
        temb,
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
    assert receipt["grad_fused_call_count"] == 1
    assert receipt["grad_fused_chunk_count"] == 5
    assert receipt["grad_backward_call_count"] == 1
    assert receipt["grad_backward_chunk_count"] == 5
    assert receipt["grad_reference_fallback_count"] == 0

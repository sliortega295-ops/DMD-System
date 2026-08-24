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
    gate = (table + temb).expand(1, 7, 5120)
    return gate[:, start:end] if start is not None else gate


def _module() -> SimpleNamespace:
    return SimpleNamespace(
        __file__=__file__,
        _apply_chunked_gated_residual=_reference,
        _materialize_scale_shift_component=_materialize,
    )


def _fused(hidden, residual, gate, output):
    output.copy_((hidden.float() + residual.float() * gate).to(hidden.dtype))
    return output


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
    assert '"${GENERATOR_NOGRAD_RESIDUAL_FUSION}" != on || "${ROLLOUT_BATCH_SIZE}" == 1' in launcher
    assert "generator_nograd_residual_fusion" in summarizer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only Triton parity")
@pytest.mark.parametrize("contiguous_gate", [False, True])
def test_real_triton_kernel_is_bitwise_exact(contiguous_gate: bool) -> None:
    from distillschedule.runtime.helios_residual_kernel import (
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
    torch.cuda.synchronize()
    assert torch.equal(expected, actual)

from __future__ import annotations

import torch

from distillschedule.runtime.helios_lora_epilogue import (
    PEFT_0181_LINEAR_FORWARD_SHA256,
    install_generator_nograd_lora_fused_epilogue,
)


class FakeLinear(torch.nn.Module):
    disable_adapters = False
    merged = False
    active_adapters = ["default"]
    lora_variant = {}

    def __init__(self) -> None:
        super().__init__()
        self.base_layer = torch.nn.Linear(2, 2, bias=False).to(torch.bfloat16)
        self.lora_A = torch.nn.ModuleDict(
            {"default": torch.nn.Linear(2, 1, bias=False).to(torch.float32)}
        )
        self.lora_B = torch.nn.ModuleDict(
            {"default": torch.nn.Linear(1, 2, bias=False).to(torch.float32)}
        )
        self.lora_dropout = torch.nn.ModuleDict({"default": torch.nn.Identity()})
        self.scaling = {"default": 0.5}

    def _check_forward_args(self, value, *args, **kwargs):
        del value, args, kwargs

    def _cast_input_dtype(self, value, dtype):
        return value.to(dtype=dtype)

    def forward(self, value, *args, **kwargs):
        del args, kwargs
        result = self.base_layer(value)
        update = self.lora_B["default"](
            self.lora_A["default"](value.to(torch.float32))
        )
        return (result + update * self.scaling["default"]).to(result.dtype)


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = FakeLinear()

    def forward(self, value):
        return self.linear(value)


class FakeLinearBF16(FakeLinear):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = self.lora_A.to(torch.bfloat16)
        self.lora_B = self.lora_B.to(torch.bfloat16)


class ModelBF16(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = FakeLinearBF16()

    def forward(self, value):
        return self.linear(value)


def test_epilogue_is_default_off() -> None:
    model = Model()
    receipt = install_generator_nograd_lora_fused_epilogue(
        model, enabled=False, role="generator", linear_type=FakeLinear
    )
    assert receipt["enabled"] is False
    assert receipt["disabled_reason"] == "NOT_REQUESTED"


def test_epilogue_is_generator_nograd_only_and_exact() -> None:
    calls = []

    def fused(base, update, scaling):
        calls.append((base.dtype, update.dtype, scaling))
        return (base + update * scaling).to(base.dtype)

    model = Model()
    value = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    with torch.no_grad():
        reference = model(value)
    receipt = install_generator_nograd_lora_fused_epilogue(
        model,
        enabled=True,
        role="generator",
        linear_type=FakeLinear,
        fused_epilogue=fused,
    )
    with torch.no_grad():
        candidate = model(value)
    assert torch.equal(reference, candidate)
    assert receipt["model_call_count"] == 1
    assert receipt["fused_invocation_count"] == 1
    assert receipt["reference_fallback_count"] == 0
    assert calls == [(torch.bfloat16, torch.float32, 0.5)]

    before = dict(receipt)
    model(value)
    assert receipt["model_call_count"] == before["model_call_count"]
    assert receipt["fused_invocation_count"] == before["fused_invocation_count"]


def test_epilogue_accepts_real_fsdp_bf16_update_contract() -> None:
    calls = []

    def fused(base, update, scaling):
        calls.append((base.dtype, update.dtype, scaling))
        # Keep both eager BF16 roundings in this injected reference.
        scaled = update * scaling
        return base + scaled

    model = ModelBF16()
    value = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    with torch.no_grad():
        reference = model(value)
    receipt = install_generator_nograd_lora_fused_epilogue(
        model,
        enabled=True,
        role="generator",
        linear_type=FakeLinearBF16,
        fused_epilogue=fused,
    )
    with torch.no_grad():
        candidate = model(value)
    assert torch.equal(reference, candidate)
    assert calls == [(torch.bfloat16, torch.bfloat16, 0.5)]
    assert receipt["fused_invocation_count"] == 1
    assert receipt["fused_update_dtype_counts"] == {"torch.bfloat16": 1}
    assert receipt["reference_fallback_count"] == 0


def test_peft_source_and_launcher_contract_are_pinned() -> None:
    assert PEFT_0181_LINEAR_FORWARD_SHA256 == (
        "ec0a80f5c5ce05f5dd90027952bb5d29aaf910e9d2ab40486a90e8a03c7e1cd9"
    )
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    launcher = (
        root / "scripts/distillschedule/launch_static_fsdp2_p0_world16_node.sh"
    ).read_text()
    summarizer = (
        root / "scripts/distillschedule/summarize_static_fsdp2_p0_world16.py"
    ).read_text()
    assert "DISTILLSCHEDULE_HELIOS_GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE" in launcher
    assert "generator_nograd_lora_fused_epilogue" in summarizer

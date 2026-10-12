from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from distillschedule.runtime.wan13b_rollout_shadow import (
    ENV, PLACEMENT, MergedRolloutShadow, expected_epilogue_counts, install,
    requested, validate,
)


def test_default_off_does_not_access_model(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert requested() is False
    assert install(object(), role="generator", placement=PLACEMENT) == {
        "requested": False, "enabled": False, "role": "generator",
        "disabled_reason": "NOT_REQUESTED",
    }
    monkeypatch.setenv(ENV, "yes")
    with pytest.raises(ValueError, match="0 or 1"):
        requested()


def _fixture():
    torch = pytest.importorskip("torch")

    class Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = torch.nn.Linear(7, 5, dtype=torch.bfloat16)
            self.base_layer.requires_grad_(False)
            self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(7, 2, bias=False)})
            self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 5, bias=False)})
            self.lora_dropout = torch.nn.ModuleDict({"default": torch.nn.Dropout(0)})
            self.lora_bias = {"default": False}
            self.lora_variant = {}
            self.active_adapters = ["default"]
            self.disable_adapters = False
            self.merged = False

        def get_delta_weight(self, adapter):
            return self.lora_B[adapter].weight @ self.lora_A[adapter].weight * 0.125

        def forward(self, x):
            base = self.base_layer(x)
            delta = self.lora_B["default"](self.lora_A["default"](x.float())) * 0.125
            return (base + delta).to(x.dtype)

    source = torch.nn.Sequential(Linear())
    return torch, source, Linear


def test_independent_merged_storage_keeps_training_outputs_gradients_and_rng():
    torch, source, linear = _fixture()
    x = torch.randn(3, 7, dtype=torch.bfloat16)
    before = {name: p.detach().clone() for name, p in source.named_parameters()}
    y = source(x)
    y.float().square().sum().backward()
    gradients = {name: p.grad.clone() for name, p in source.named_parameters() if p.grad is not None}
    source.zero_grad(set_to_none=True)
    rng = torch.get_rng_state().clone()
    shadow = MergedRolloutShadow(source, linear_type=linear)
    shadow.begin_rollout(version=3, logical_steps=(10,))
    assert torch.equal(rng, torch.get_rng_state())
    with torch.no_grad():
        result = shadow(x)
        expected_weight = (source[0].base_layer.weight.float() + source[0].get_delta_weight("default")).to(torch.bfloat16)
        expected = torch.nn.functional.linear(x, expected_weight, source[0].base_layer.bias)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
    shadow.end_rollout()
    assert not any("lora_" in name for name, _ in shadow.model.named_parameters())
    assert not any(p.requires_grad for p in shadow.model.parameters())
    assert all(p.data_ptr() != before[name].data_ptr() for name, p in source.named_parameters())
    assert shadow.model[0].weight.data_ptr() != source[0].base_layer.weight.data_ptr()
    for name, parameter in source.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    new_y = source(x)
    torch.testing.assert_close(new_y, y, rtol=0, atol=0)
    new_y.float().square().sum().backward()
    for name, p in source.named_parameters():
        if name in gradients:
            torch.testing.assert_close(p.grad, gradients[name], rtol=0, atol=0)


def test_refresh_rebuilds_without_accumulation_and_rejects_stale_version():
    torch, source, linear = _fixture()
    shadow = MergedRolloutShadow(source, linear_type=linear)
    shadow.begin_rollout(version=3, logical_steps=(10,))
    shadow.end_rollout()
    original = shadow.model[0].weight.clone()
    shadow.begin_rollout(version=3, logical_steps=(11,))
    shadow.end_rollout()
    assert shadow.receipt["refresh_count"] == 1
    torch.testing.assert_close(shadow.model[0].weight, original, rtol=0, atol=0)
    with torch.no_grad():
        source[0].lora_B["default"].weight.add_(0.2)
    shadow.begin_rollout(version=4, logical_steps=(10,))
    shadow.end_rollout()
    expected = (source[0].base_layer.weight.float() + source[0].get_delta_weight("default")).to(torch.bfloat16)
    torch.testing.assert_close(shadow.model[0].weight, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="advance exactly once"):
        shadow.begin_rollout(version=3, logical_steps=(10,))
    with pytest.raises(ValueError, match="B1"):
        shadow.begin_rollout(version=4, logical_steps=(10, 11))


def test_execution_is_fail_closed_outside_post_commit_no_grad_scope():
    torch, source, linear = _fixture()
    shadow = MergedRolloutShadow(source, linear_type=linear)
    x = torch.zeros(1, 7, dtype=torch.bfloat16)
    with torch.no_grad(), pytest.raises(RuntimeError, match="post-G"):
        shadow(x)
    shadow.begin_rollout(version=3, logical_steps=(10,))
    with pytest.raises(RuntimeError, match="post-G"):
        shadow(x)
    with pytest.raises(RuntimeError, match="nested"):
        shadow.begin_rollout(version=3, logical_steps=(11,))
    shadow.end_rollout()


@pytest.mark.parametrize("invalid", ["dropout", "dtype", "other_trainable"])
def test_unsupported_training_contract_rejected(invalid):
    torch, source, linear = _fixture()
    if invalid == "dropout":
        source[0].lora_dropout["default"].p = 0.1
    elif invalid == "dtype":
        source[0].lora_A["default"].to(torch.bfloat16)
    else:
        source[0].base_layer.weight.requires_grad_(True)
    with pytest.raises(ValueError):
        MergedRolloutShadow(source, linear_type=linear)


def test_shadow_and_epilogue_census_are_joint_not_weakened():
    assert expected_epilogue_counts(cycles=2, shadow_enabled=False) == {
        "model_call_count": 56, "fused_invocation_count": 17080, "reference_fallback_count": 224,
    }
    assert expected_epilogue_counts(cycles=2, shadow_enabled=True) == {
        "model_call_count": 2, "fused_invocation_count": 610, "reference_fallback_count": 8,
    }
    generator = {
        "requested": True, "enabled": True, "refresh_count": 2,
        "rollout_entry_count": 10, "model_call_count": 54,
        "shadow_lora_parameter_count": 0, "active": False,
        "training_master_unchanged": True, "optimizer_reducer_ema_unchanged": True,
        "refresh_versions": [3, 4],
        "entries": [{"version": 3+c, "logical_steps": [step]} for c in range(2) for step in range(10, 15)],
    }
    build = {
        "generator": {"metadata": {"generator_rollout_merged_shadow": generator}},
        "critic": {"metadata": {"generator_rollout_merged_shadow": {"enabled": False, "disabled_reason": "GENERATOR_ONLY"}}},
    }
    assert validate(build, enabled=True, cycles=2)["enabled"]
    generator["model_call_count"] = 53
    with pytest.raises(AssertionError, match="model_call_count"):
        validate(build, enabled=True, cycles=2)


def test_real_peft_pyramid_conv3d_adapters_use_peft_delta_layout():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from peft.tuners.lora.layer import Conv3d

    base = torch.nn.Conv3d(2, 3, kernel_size=(1, 2, 2), dtype=torch.float32)
    layer = Conv3d(base, "default", r=2, lora_alpha=2, lora_dropout=0, init_lora_weights=False)
    layer.base_layer.requires_grad_(False)
    source = torch.nn.Sequential(layer)
    shadow = MergedRolloutShadow(source)
    shadow.begin_rollout(version=3, logical_steps=(10,))
    with torch.no_grad():
        expected_weight = base.weight + layer.get_delta_weight("default")
        torch.testing.assert_close(shadow.model[0].weight, expected_weight, rtol=0, atol=0)
        x = torch.randn(1, 2, 2, 4, 4)
        torch.testing.assert_close(shadow(x), torch.nn.functional.conv3d(x, expected_weight, base.bias), rtol=0, atol=0)
    shadow.end_rollout()
    assert shadow.receipt["merged_module_inventory"][0]["class"] == "Conv3d"
    assert shadow.receipt["shadow_lora_parameter_count"] == 0


@pytest.mark.parametrize("enabled", [False, True])
def test_controller_explicit_opt_in_and_isolated_ssh_tty(enabled):
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "scripts/distillschedule/run_wan13b_fair_world16.py"),
        "--run-id", "shadow-plan", "--mode", "trajectory",
        "--placement", PLACEMENT, "--expected-head", "a"*40,
        "--model-root", "/model", "--bootstrap", "/bootstrap", "--bootstrap-hash", "b"*64,
        "--master-addr", "127.0.0.1", "--trajectory-cycles", "2", "--trajectory-arm-id", "resident",
        "--generator-nograd-lora-fused-epilogue", "--grad-lora-fused-epilogue",
        "--tf32-cublas-override", "1", "--ssh-tty", "--dry-run"]
    if enabled:
        command.append("--generator-rollout-merged-shadow")
    plan = json.loads(subprocess.check_output(command, env=dict(os.environ, PYTHONPATH=str(root / "src")), text=True))
    for row in plan["commands"]:
        assert "-tt" in row
        assert f"{ENV}={int(enabled)}" in " ".join(row)


def test_routing_is_only_the_five_post_generator_rollouts():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/distillschedule/probe_static_fsdp2_p0_world16.py").read_text()
    before, function = source.split("def _precompute_critic_rollouts(", 1)
    function, after = function.split("\ndef _", 1)
    assert "_distillschedule_merged_rollout_shadow" in function
    assert "shadow.begin_rollout(" in function
    assert "finally:" in function and "shadow.end_rollout()" in function
    assert "_distillschedule_merged_rollout_shadow" not in before + after

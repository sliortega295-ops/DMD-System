from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from distillschedule.runtime.wan13b_rollout_cuda_graph import (
    ENV, PLACEMENT, RolloutCudaGraph, clone_tree, copy_tree, install,
    requested, signature, validate_receipt,
)
from distillschedule.runtime.wan13b_rollout_shadow import expected_epilogue_counts


def test_default_off_and_forbidden_joint_candidate(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert not requested()
    assert install(object(), role="generator", placement=PLACEMENT)["disabled_reason"] == "NOT_REQUESTED"
    monkeypatch.setenv(ENV, "invalid")
    with pytest.raises(ValueError, match="0 or 1"):
        requested()
    monkeypatch.setenv(ENV, "1")
    with pytest.raises(ValueError, match="Resident"):
        install(object(), role="generator", placement="world16-fsdp")
    monkeypatch.setenv("DISTILLSCHEDULE_WAN13B_GENERATOR_ROLLOUT_MERGED_SHADOW", "1")
    with pytest.raises(ValueError, match="merged"):
        install(object(), role="generator", placement=PLACEMENT)


def test_signature_keys_layout_not_values_and_clone_is_non_aliasing():
    torch = pytest.importorskip("torch")
    x = torch.randn(2, 3, 4).transpose(0, 1)
    tree = {"hidden_states": x, "flag": True, "other": [None, torch.ones(2)]}
    result = clone_tree(tree)
    assert signature(tree) == signature(result)
    assert result["hidden_states"].stride() == x.stride()
    assert result["hidden_states"].data_ptr() != x.data_ptr()
    changed = clone_tree(tree)
    changed["hidden_states"].add_(1)
    assert signature(changed) == signature(tree)
    copy_tree(result, changed)
    torch.testing.assert_close(result["hidden_states"], changed["hidden_states"], rtol=0, atol=0)
    assert signature({**tree, "flag": False}) != signature(tree)
    with pytest.raises(ValueError, match="unsupported"):
        signature(object())
    with pytest.raises(ValueError, match="detached"):
        signature(torch.ones(2, requires_grad=True))


def test_dispatch_protocol_live_inputs_weights_and_output_lifetime_cpu(monkeypatch):
    """CPU protocol unit only; actual CUDA capture is tested by real GPU canary."""
    torch = pytest.importorskip("torch")
    source = torch.nn.Linear(4, 4)
    monkeypatch.setattr(RolloutCudaGraph, "_audit_source", lambda self: None)

    # Test protocol with a toy keyword-compatible source, not GPU performance.
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = source

        def forward(self, hidden_states):
            return self.linear(hidden_states)

    model = Model()
    controller = RolloutCudaGraph(model)

    def capture(self, args, kwargs):
        sa, sk = clone_tree(args), clone_tree(kwargs)
        self.receipt["warmup_model_call_count"] += 3
        self.receipt["capture_model_call_count"] += 1
        output = model(*sa, **sk).clone()

        class ProtocolReplay:
            def replay(self):
                output.copy_(model(*sa, **sk))

        return sa, sk, ProtocolReplay(), output

    monkeypatch.setattr(RolloutCudaGraph, "_capture", capture)
    x = torch.randn(2, 4)
    rng = torch.get_rng_state().clone()
    controller.begin_rollout(version=3, logical_steps=(10,))
    with pytest.raises(ValueError, match="no-grad"):
        controller(hidden_states=x)
    with torch.no_grad():
        first = controller(hidden_states=x)
        saved = first.clone()
        second = controller(hidden_states=x + 1)
        torch.testing.assert_close(second, model(hidden_states=x + 1), rtol=0, atol=0)
        torch.testing.assert_close(first, saved, rtol=0, atol=0)
    controller.end_rollout()
    with torch.no_grad():
        source.weight.add_(0.01)
    controller.begin_rollout(version=4, logical_steps=(10,))
    with torch.no_grad():
        updated = controller(hidden_states=x)
        torch.testing.assert_close(updated, model(hidden_states=x), rtol=0, atol=0)
        with pytest.raises(ValueError, match="after warmup"):
            controller(hidden_states=torch.ones(3, 4))
    controller.end_rollout()
    assert torch.equal(rng, torch.get_rng_state())
    with pytest.raises(ValueError, match="consecutive"):
        controller.begin_rollout(version=6, logical_steps=(10,))
    source.weight = torch.nn.Parameter(source.weight.clone())
    with pytest.raises(ValueError, match="addresses"):
        controller.begin_rollout(version=5, logical_steps=(10,))


def _receipt(*, reference=False):
    entries = [{"version": 3 + c, "logical_steps": [step]}
               for c in range(2) for step in range(10, 15)]
    return {"requested": True, "enabled": True, "weights_merged": False,
            "active": False, "versions": [3, 4], "entries": entries,
            "model_call_count": 54, "graph_replay_model_call_count": 54,
            "rollout_entry_count": 10, "cache_entry_count": 3,
            "cache_entries": [{"latent_spatial_shape": list(s)} for s in [(12, 20), (24, 40), (48, 80)]],
            "capture_model_call_count": 3, "warmup_model_call_count": 9,
            "diagnostic_reference_model_call_count": 54 if reference else 0,
            "diagnostic_reference_entry_count": 10 if reference else 0,
            "diagnostic_reference_parity": [{"status": "PASS", "identity": e} for e in entries] if reference else []}


@pytest.mark.parametrize("reference", [False, True])
def test_counts_separate_replay_python_cold_and_correctness_reference(reference):
    receipt = _receipt(reference=reference)
    validate_receipt(receipt, enabled=True, cycles=2, diagnostic_reference=reference)
    expected = expected_epilogue_counts(cycles=2, shadow_enabled=False, graph_receipt=receipt)
    calls = 14 + (54 if reference else 0)
    assert expected == {"model_call_count": calls, "fused_invocation_count": 305 * calls,
                        "reference_fallback_count": 4 * calls}
    receipt["graph_replay_model_call_count"] = 53
    with pytest.raises(AssertionError, match="census"):
        validate_receipt(receipt, enabled=True, cycles=2, diagnostic_reference=reference)


def test_reference_cannot_be_missing_or_in_native_timer():
    receipt = _receipt()
    with pytest.raises(AssertionError, match="reference census"):
        validate_receipt(receipt, enabled=True, cycles=2, diagnostic_reference=True)
    receipt = _receipt(reference=True)
    with pytest.raises(AssertionError, match="reference census"):
        validate_receipt(receipt, enabled=True, cycles=2, diagnostic_reference=False)
    receipt["diagnostic_reference_parity"][0]["status"] = "FAIL"
    with pytest.raises(AssertionError, match="parity"):
        validate_receipt(receipt, enabled=True, cycles=2, diagnostic_reference=True)


@pytest.mark.parametrize("graph", [False, True])
def test_controller_graph_explicit_export_with_norm_background(graph):
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "scripts/distillschedule/run_wan13b_fair_world16.py"),
               "--run-id", "graph-plan", "--mode", "trajectory", "--placement", PLACEMENT,
               "--expected-head", "a" * 40, "--model-root", "/model", "--bootstrap", "/bootstrap",
               "--bootstrap-hash", "b" * 64, "--master-addr", "127.0.0.1", "--trajectory-cycles", "2",
               "--trajectory-arm-id", "resident", "--generator-nograd-lora-fused-epilogue",
               "--grad-lora-fused-epilogue", "--modulated-norm-fusion", "--dry-run"]
    if graph:
        command.append("--generator-rollout-cuda-graph")
    plan = json.loads(subprocess.check_output(command, env=dict(os.environ, PYTHONPATH=str(root / "src")), text=True))
    for row in plan["commands"]:
        assert f"{ENV}={int(graph)}" in " ".join(row)
    if graph:
        result = subprocess.run(command + ["--generator-rollout-merged-shadow"], capture_output=True, text=True)
        assert result.returncode and "merged" in result.stderr


def test_only_post_generator_routing_and_no_timed_reference():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/distillschedule/probe_static_fsdp2_p0_world16.py").read_text()
    before, function = source.split("def _precompute_critic_rollouts(", 1)
    function, after = function.split("\ndef _", 1)
    assert "_distillschedule_rollout_cuda_graph" in function
    assert "_distillschedule_rollout_cuda_graph" not in before + after
    assert "require_tensor_parity" in function
    trajectory = (root / "scripts/distillschedule/probe_wan13b_trajectory_world16.py").read_text()
    assert "cuda_graph_parity_check=graph_enabled" in trajectory
    native = (root / "scripts/distillschedule/probe_wan13b_two_cycle_timing_world16.py").read_text()
    assert "cuda_graph_parity_check=True" not in native
    assert "cuda_graph_parity_check=graph_enabled" not in native

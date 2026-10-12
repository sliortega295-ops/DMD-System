from __future__ import annotations

from types import SimpleNamespace

import pytest

from distillschedule.runtime.wan13b_score_cuda_graph import (
    ENV, PLACEMENT, ScoreCudaGraph, check_admission, install, requested,
    validate_receipt,
)
from distillschedule.runtime.wan13b_rollout_cuda_graph import clone_tree


def test_default_off_no_torch_and_placement_rejection(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert not requested()
    assert not install(object(), role="critic", placement=PLACEMENT,
                       reference_checks=True)["enabled"]
    monkeypatch.setenv(ENV, "maybe")
    with pytest.raises(ValueError, match="0 or 1"):
        requested()
    monkeypatch.setenv(ENV, "1")
    with pytest.raises(ValueError, match="Resident"):
        check_admission("world16-fsdp", enabled=True, role="critic")
    assert not check_admission(PLACEMENT, enabled=True, role="generator")


def test_cpu_dispatch_modes_live_input_weight_update_lifetime_rng(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(ScoreCudaGraph, "_audit_source", lambda self: None)
    monkeypatch.setenv(ENV, "1")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)
            self._distillschedule_resident_clean_state = SimpleNamespace(
                training=True, adapter_enabled=True)

        def forward(self, hidden_states):
            if self._distillschedule_resident_clean_state.adapter_enabled:
                return (self.linear(hidden_states),)
            return (hidden_states * 2,)

    model = Model()
    original = model.forward

    def capture(self, args, kwargs):
        sa, sk = clone_tree(args), clone_tree(kwargs)
        output = clone_tree(self.original_forward(*sa, **sk))
        self.receipt["warmup_calls"] += 3
        self.receipt["capture_calls"] += 1

        class ProtocolReplay:
            def replay(self):
                output[0].copy_(self_controller.original_forward(*sa, **sk)[0])

        self_controller = self
        return sa, sk, ProtocolReplay(), output

    monkeypatch.setattr(ScoreCudaGraph, "_capture", capture)
    receipt = install(model, role="critic", placement=PLACEMENT, reference_checks=True)
    controller = model._distillschedule_score_cuda_graph
    x = torch.randn(2, 4)
    rng = torch.get_rng_state().clone()
    for cycle in range(2):
        with torch.no_grad():
            if cycle:
                model.linear.weight.add_(0.125)
            for enabled in [True, False, False]:
                model._distillschedule_resident_clean_state.adapter_enabled = enabled
                first = model(hidden_states=x + cycle)
                torch.testing.assert_close(first[0], original(hidden_states=x + cycle)[0],
                                           rtol=0, atol=0)
                saved = first[0].clone()
            torch.testing.assert_close(first[0], saved, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    validate_receipt(receipt, enabled=True, cycles=2, reference_checks=True)
    # Backward/autograd remains the direct original, not a graph replay.
    model._distillschedule_resident_clean_state.adapter_enabled = True
    model(hidden_states=x)[0].sum().backward()
    assert model.linear.weight.grad is not None
    assert receipt["replay_calls"] == 6
    with torch.no_grad(), pytest.raises(ValueError, match="after cold"):
        model(hidden_states=torch.ones(3, 4))
    model.linear.weight = torch.nn.Parameter(model.linear.weight.clone())
    with torch.no_grad(), pytest.raises(ValueError, match="address"):
        model(hidden_states=x)


def test_receipt_rejects_missing_replay_extra_calls_or_reference():
    receipt = {"requested": True, "enabled": True, "replay_calls": 6,
               "adapter_on_calls": 2, "adapter_off_calls": 4,
               "pointer_identity_checks": 2, "reference_calls": 0,
               "cache_entries": [{}, {}], "capture_calls": 2, "warmup_calls": 6}
    validate_receipt(receipt, enabled=True, cycles=2, reference_checks=False)
    for field in ["replay_calls", "adapter_on_calls", "capture_calls", "warmup_calls"]:
        changed = {**receipt, field: 0}
        with pytest.raises(AssertionError):
            validate_receipt(changed, enabled=True, cycles=2, reference_checks=False)
    with pytest.raises(AssertionError):
        validate_receipt(receipt, enabled=True, cycles=2, reference_checks=True)


def test_launcher_default_off_explicit_optin_and_receipt_binding():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    launch = (root / 'scripts/distillschedule/launch_wan13b_fair_world16_node.sh').read_text()
    worker = (root / 'scripts/distillschedule/probe_wan13b_two_cycle_timing_world16.py').read_text()
    trajectory = (root / 'scripts/distillschedule/probe_wan13b_trajectory_world16.py').read_text()
    driver = (root / 'scripts/distillschedule/run_wan13b_fair_world16.py').read_text()
    assert 'DISTILLSCHEDULE_WAN13B_SCORE_CUDA_GRAPH' in launch
    assert '--score-cuda-graph' in driver
    assert '"score_cuda_graph"' in worker
    assert '"score_cuda_graph"' in trajectory
    assert 'reference_checks=False' in worker
    assert 'reference_checks=True' in trajectory

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

import runtime.h3_diffsynth_checkpoint_boundary_cpu as policy


def native_checkpoint(model, checkpoint, offload, *args, **kwargs):
    assert not offload
    if checkpoint:
        return torch_checkpoint(model, *args, **kwargs, use_reentrant=False)
    return model(*args, **kwargs)


def fixture(monkeypatch):
    # Test-only small real PyTorch checkpoint, not a production source override.
    monkeypatch.setattr(policy, "PINNED_NATIVE_CHECKPOINT_FILE_SHA256",
                        hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return SimpleNamespace(gradient_checkpoint_forward=native_checkpoint)


def test_disabled_identity():
    module = SimpleNamespace(gradient_checkpoint_forward=native_checkpoint)
    context = policy.NativeCheckpointBoundaryCPU(module, [], lambda: "student")
    assert module.gradient_checkpoint_forward is native_checkpoint
    assert not context.snapshot()["enabled"]
    context.close()


def test_real_nonreentrant_output_gradient_and_replay_scope(monkeypatch):
    torch.manual_seed(10)
    reference = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.SiLU(), torch.nn.Linear(16, 8))
    candidate = copy.deepcopy(reference)
    x = torch.randn(2, 8)
    a = x.clone().requires_grad_(); b = x.clone().requires_grad_()
    module = fixture(monkeypatch)
    context = policy.NativeCheckpointBoundaryCPU(module, [candidate], lambda: "student", enabled=True)
    expected = native_checkpoint(reference, True, False, a)
    actual = module.gradient_checkpoint_forward(candidate, True, False, b)
    packed_at_return = context.snapshot()["counters"]["saved_input_count"]
    expected.sum().backward(); actual.sum().backward()
    assert torch.equal(expected, actual) and torch.equal(a.grad, b.grad)
    for left, right in zip(reference.parameters(), candidate.parameters()):
        assert torch.equal(left.grad, right.grad)
    assert context.snapshot()["counters"]["saved_input_count"] == packed_at_return
    assert context.snapshot()["calls_by_role"] == {"student": 1}
    context.close(); assert module.gradient_checkpoint_forward is native_checkpoint


def test_no_grad_and_unregistered_units_are_not_offloaded(monkeypatch):
    module = fixture(monkeypatch)
    registered = torch.nn.Linear(8, 8); other = torch.nn.Linear(8, 8)
    context = policy.NativeCheckpointBoundaryCPU(module, [registered], lambda: "teacher", enabled=True)
    with torch.no_grad():
        module.gradient_checkpoint_forward(registered, True, False, torch.randn(2, 8))
    module.gradient_checkpoint_forward(other, True, False, torch.randn(2, 8))
    assert context.snapshot()["counters"] == {}
    context.close()


def test_source_and_conflicting_policy_fail_closed(monkeypatch):
    module = SimpleNamespace(gradient_checkpoint_forward=native_checkpoint)
    with pytest.raises(ValueError, match="source identity"):
        policy.NativeCheckpointBoundaryCPU(module, [torch.nn.Linear(8, 8)], lambda: "student", enabled=True)
    module = fixture(monkeypatch); block = torch.nn.Linear(8, 8)
    context = policy.NativeCheckpointBoundaryCPU(module, [block], lambda: "student", enabled=True)
    with pytest.raises(RuntimeError, match="nest two"):
        module.gradient_checkpoint_forward(block, True, True, torch.randn(2, 8))
    context.close()

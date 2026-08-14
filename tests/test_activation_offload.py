from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from runtime.activation_offload import install_activation_offload_hooks


class _SaveTwice(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.save_for_backward(value, value)
        return value * 2

    @staticmethod
    def backward(ctx, grad_output):
        first, second = ctx.saved_tensors
        return grad_output * (first * 0 + second * 0 + 2)


class _SaveButDoNotUnpack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.save_for_backward(value)
        return value * 3

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * 3


class _SaveTwoShapes(torch.autograd.Function):
    @staticmethod
    def forward(ctx, first, second):
        ctx.save_for_backward(first, second)
        return first.sum() + second.sum()

    @staticmethod
    def backward(ctx, grad_output):
        first, second = ctx.saved_tensors
        return torch.ones_like(first) * grad_output, torch.ones_like(second) * grad_output


class _ToyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))
        self.register_buffer("offset", torch.randn(1, 4))

    def _checkpointed(self, value):
        return torch.nn.functional.linear(value, self.weight) + self.offset

    def forward(self, value):
        expanded = value[:, :1].expand(-1, 4)
        duplicated = _SaveTwice.apply(value)
        expanded_saved = _SaveTwice.apply(expanded)
        parameter_term = (value * self.weight[0]).sum(dim=-1, keepdim=True)
        return (
            checkpoint(self._checkpointed, expanded_saved + duplicated, use_reentrant=False)
            + parameter_term
        )


def _events(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_audited_offload_preserves_grad_and_audits_storage(tmp_path: Path) -> None:
    torch.manual_seed(5)
    reference = _ToyModule()
    candidate = _ToyModule()
    candidate.load_state_dict(reference.state_dict())
    event_path = tmp_path / "events.jsonl"
    registration = install_activation_offload_hooks(
        candidate,
        logical_component="generator_selected",
        event_path=event_path,
        pin_memory=False,
    )
    ref_input = torch.randn(2, 4, requires_grad=True)
    test_input = ref_input.detach().clone().requires_grad_(True)
    ref_output = reference(ref_input)
    test_output = candidate(test_input)
    torch.testing.assert_close(test_output, ref_output)
    ref_output.sum().backward()
    test_output.sum().backward()
    torch.testing.assert_close(test_input.grad, ref_input.grad)
    torch.testing.assert_close(candidate.weight.grad, reference.weight.grad)
    registration.remove()

    events = _events(event_path)
    packs = [event for event in events if event["event"] == "pack"]
    assert packs
    assert all(event["save_generation"] == "ORIGINAL_FORWARD" for event in packs)
    assert any(event["policy"] == "KEEP_EXCLUDED_STORAGE" for event in packs)
    assert any(event["policy"] == "KEEP_ZERO_STRIDE_VIEW" for event in packs)
    assert registration.offloader.stats["duplicate_pack_count"] >= 1
    assert registration.offloader.stats["cpu_copy_count"] < registration.offloader.stats["pack_count"]
    assert registration.offloader.stats["live_cpu_bytes"] == 0


def test_no_grad_call_does_not_open_offload_scope(tmp_path: Path) -> None:
    module = _ToyModule()
    event_path = tmp_path / "events.jsonl"
    registration = install_activation_offload_hooks(
        module,
        logical_component="critic_train",
        event_path=event_path,
        pin_memory=False,
    )
    with torch.no_grad():
        module(torch.randn(2, 4))
    registration.remove()
    assert not event_path.exists()
    assert registration.offloader.stats["scope_count"] == 0


def test_offload_threshold_keeps_small_saved_storage(tmp_path: Path) -> None:
    module = _ToyModule()
    event_path = tmp_path / "events.jsonl"
    registration = install_activation_offload_hooks(
        module,
        logical_component="generator_selected",
        event_path=event_path,
        pin_memory=False,
        min_offload_bytes=2**30,
    )
    value = torch.randn(2, 4, requires_grad=True)
    module(value).sum().backward()
    assert registration.offloader.stats["cpu_copy_count"] == 0
    assert registration.offloader.stats["kept_below_threshold_count"] > 0
    registration.remove()


def test_offload_signature_allowlist_is_exact(tmp_path: Path) -> None:
    class _TwoShapeModule(nn.Module):
        def forward(self, first, second):
            return _SaveTwoShapes.apply(first, second)

    module = _TwoShapeModule()
    event_path = tmp_path / "events.jsonl"
    registration = install_activation_offload_hooks(
        module,
        logical_component="generator_selected",
        event_path=event_path,
        pin_memory=False,
        allowed_offload_signatures=frozenset(
            {((2, 4), "torch.float32", 2 * 4 * 4)}
        ),
    )
    first = torch.randn(2, 4, requires_grad=True)
    second = torch.randn(2, 3, requires_grad=True)
    module(first, second).backward()
    registration.remove()

    assert registration.offloader.stats["cpu_copy_count"] == 1
    assert registration.offloader.stats["kept_policy_filter_count"] == 1
    assert first.grad is not None
    assert second.grad is not None
    packs = [event for event in _events(event_path) if event["event"] == "pack"]
    assert [event["policy"] for event in packs].count("OFFLOAD_CPU") == 1
    assert [event["policy"] for event in packs].count("KEEP_POLICY_FILTER") == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pinned_signature_allowlist_cuda_parity(tmp_path: Path) -> None:
    class _TwoShapeModule(nn.Module):
        def forward(self, first, second):
            return _SaveTwoShapes.apply(first, second)

    torch.manual_seed(91)
    reference_first = torch.randn(64, 64, device="cuda", requires_grad=True)
    reference_second = torch.randn(32, 64, device="cuda", requires_grad=True)
    candidate_first = reference_first.detach().clone().requires_grad_(True)
    candidate_second = reference_second.detach().clone().requires_grad_(True)
    module = _TwoShapeModule().cuda()
    registration = install_activation_offload_hooks(
        module,
        logical_component="generator_selected",
        event_path=tmp_path / "cuda_events.jsonl",
        pin_memory=True,
        allowed_offload_signatures=frozenset(
            {((64, 64), "torch.float32", 64 * 64 * 4)}
        ),
    )

    reference = _SaveTwoShapes.apply(reference_first, reference_second)
    candidate = module(candidate_first, candidate_second)
    reference.backward()
    candidate.backward()
    torch.cuda.synchronize()
    registration.remove()

    assert torch.equal(candidate.detach(), reference.detach())
    assert torch.equal(candidate_first.grad, reference_first.grad)
    assert torch.equal(candidate_second.grad, reference_second.grad)
    assert registration.offloader.stats["cpu_copy_count"] == 1
    assert registration.offloader.stats["live_cpu_bytes"] == 0


def test_aggregate_mode_omits_per_tensor_events(tmp_path: Path) -> None:
    module = _ToyModule()
    event_path = tmp_path / "events.jsonl"
    registration = install_activation_offload_hooks(
        module,
        logical_component="generator_selected",
        event_path=event_path,
        pin_memory=False,
        detailed_events=False,
    )
    value = torch.randn(2, 4, requires_grad=True)
    module(value).sum().backward()
    registration.remove()
    events = _events(event_path)
    assert {event["event"] for event in events} == {"scope_start", "scope_end"}
    assert registration.offloader.stats["pack_count"] > 0


def test_unused_saved_tensor_does_not_leak_alias_registry(
    tmp_path: Path,
) -> None:
    import gc

    class _UnusedSaveModule(nn.Module):
        def forward(self, value):
            return _SaveButDoNotUnpack.apply(value)

    module = _UnusedSaveModule()
    registration = install_activation_offload_hooks(
        module,
        logical_component="generator_selected",
        event_path=tmp_path / "events.jsonl",
        pin_memory=False,
    )
    value = torch.randn(2, 4, requires_grad=True)
    output = module(value)
    output.sum().backward()
    output = None
    gc.collect()
    registration.offloader.assert_scope_closed()
    assert registration.offloader.stats["unpack_count"] == 0
    assert registration.offloader.stats["live_cpu_bytes"] == 0
    registration.remove()

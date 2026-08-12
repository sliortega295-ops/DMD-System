import json

import torch

from runtime.saved_tensor_census import install_saved_tensor_census_hooks


def test_saved_tensor_census_is_pass_through_and_metadata_only(tmp_path) -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.GELU(),
        torch.nn.Linear(16, 4),
    )
    path = tmp_path / "events.jsonl"
    registration = install_saved_tensor_census_hooks(
        model,
        event_path=path,
        logical_component="generator_selected",
    )
    value = torch.randn(3, 8, requires_grad=True)
    with torch.no_grad():
        expected = model(value.detach().clone()).detach()
    observed = model(value)
    observed.sum().backward()
    summary = registration.remove()

    assert torch.equal(observed.detach(), expected)
    assert value.grad is not None
    assert summary["scope_count"] == 1
    assert summary["pack_count"] > 0
    assert summary["unpack_count"] > 0
    assert summary["pack_returns_original_tensor"] is True
    assert summary["unpack_returns_same_tensor"] is True
    assert summary["tensor_references_retained_by_census"] == 0
    assert registration.context_stack == []
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert events[0]["event"] == "scope_start"
    assert events[-1]["event"] in {"scope_end", "unpack"}
    assert all("tensor" not in event for event in events)


def test_saved_tensor_census_ignores_no_grad_call(tmp_path) -> None:
    model = torch.nn.Linear(4, 4)
    registration = install_saved_tensor_census_hooks(
        model,
        event_path=tmp_path / "events.jsonl",
        logical_component="generator_selected",
    )
    with torch.no_grad():
        model(torch.randn(2, 4))
    summary = registration.remove()
    assert summary["scope_count"] == 0
    assert summary["pack_count"] == 0
    assert summary["signatures"] == []

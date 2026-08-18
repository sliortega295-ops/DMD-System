from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.utils.checkpoint  # noqa: F401

from runtime.lightx2v_checkpoint_boundary_offload import (
    BoundaryManifest,
    SCHEMA_VERSION,
    install_boundary_census,
    install_checkpoint_boundary_offload,
    write_boundary_manifest,
)


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 4))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value @ self.weight


class _ToyMiniMax(torch.nn.Module):
    gradient_checkpointing = True

    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([_Block() for _ in range(50)])
        # The production adapter wraps the pinned MiniMax checkpoint function.
        # A direct function call keeps this CPU unit test deterministic and
        # avoids PyTorch's very slow no-grad checkpoint path on shared hosts;
        # the exact checkpoint call-count/manifest contract remains exercised.
        self._gradient_checkpointing_func = lambda fn, value: fn(value)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.transformer_blocks:
            def run(hidden: torch.Tensor, block: torch.nn.Module = block) -> torch.Tensor:
                return block(hidden)

            value = self._gradient_checkpointing_func(run, value)
        return value


def _manifest(tmp_path: Path) -> BoundaryManifest:
    expected = {
        "shape": [2, 4],
        "stride": [4, 1],
        "dtype": "torch.float32",
        "storage_bytes": 32,
        "logical_bytes": 32,
    }
    census = [
        {
            "block_index": index,
            **expected,
        }
        for index in range(50)
    ]
    path = write_boundary_manifest(
        tmp_path / "boundary.json",
        boundaries=census,
        selected_block_indices=list(range(12)),
        source={"test": True},
    )
    return BoundaryManifest.load(path)


def test_manifest_is_explicit_and_hashable(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    assert manifest.checkpoint_segment == 0
    assert manifest.block_count == 50
    assert manifest.selected_block_indices == tuple(range(12))
    assert len(manifest.payload_sha256) == 64
    payload = json.loads(manifest.path.read_text())
    assert payload["schema_version"] == SCHEMA_VERSION


def test_selected_boundary_offload_closes_before_backward_and_preserves_gradients(
    tmp_path: Path,
) -> None:
    torch.manual_seed(3)
    reference = _ToyMiniMax()
    candidate = _ToyMiniMax()
    candidate.load_state_dict(copy.deepcopy(reference.state_dict()))
    input_ref = torch.randn(2, 4, requires_grad=True)
    input_candidate = input_ref.detach().clone().requires_grad_()

    ref_output = reference(input_ref)
    ref_loss = ref_output.square().mean()
    ref_loss.backward()
    ref_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in reference.named_parameters()
    }

    registration = install_checkpoint_boundary_offload(
        candidate,
        _manifest(tmp_path),
        role="student",
        event_path=tmp_path / "events.jsonl",
        pin_memory=False,
        require_cuda=False,
    )
    candidate_output = candidate(input_candidate)
    # The hook context has already closed when the top-level forward returns;
    # backward therefore cannot recursively trigger the pack hook.
    assert registration.offloader._active_scope is False
    candidate_output.square().mean().backward()

    torch.testing.assert_close(candidate_output, ref_output, rtol=0, atol=0)
    for name, parameter in candidate.named_parameters():
        torch.testing.assert_close(parameter.grad, ref_grads[name], rtol=0, atol=0)
    assert registration.grad_forward_count == 1
    assert registration.call_count == 50
    assert registration.selected_call_count == 12
    assert registration.recompute_pack_count == 0
    registration.remove()
    assert registration.closed
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert sum(event["event"] == "scope_start" for event in events) == 12
    assert sum(event["event"] == "scope_end" for event in events) == 12


def test_boundary_policy_rejects_non_native_segment(tmp_path: Path) -> None:
    payload = json.loads((_manifest(tmp_path).path).read_text())
    payload["checkpoint_segment"] = 8
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    try:
        BoundaryManifest.load(path)
    except RuntimeError as error:
        assert "segment=0" in str(error)
    else:
        raise AssertionError("non-native checkpoint segment was accepted")


def test_census_flushes_each_no_grad_forward_without_retaining_rows(tmp_path: Path) -> None:
    model = _ToyMiniMax()
    # The census itself is independent of checkpoint semantics.  Avoid the
    # PyTorch no-grad checkpoint path in this tiny CPU unit test (the real
    # MiniMax canary exercises native checkpointing on CUDA).
    model._gradient_checkpointing_func = lambda fn, value: fn(value)
    output = tmp_path / "census.jsonl"
    registration = install_boundary_census(model, output_path=output)
    with torch.no_grad():
        model(torch.randn(2, 4))
    assert output.is_file()
    assert len(output.read_text().splitlines()) == 50
    # A second no-grad forward appends rather than overwrites the first one.
    with torch.no_grad():
        model(torch.randn(2, 4))
    assert len(output.read_text().splitlines()) == 100
    registration.remove()
    assert len(output.read_text().splitlines()) == 100

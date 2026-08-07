from __future__ import annotations

import torch
from torch import nn

from runtime.zero3_checkpoint import (
    configure_block_gradient_checkpointing,
    zero3_metadata_tolerant_checkpoint,
)


class _RecordingModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def enable_gradient_checkpointing(self, **kwargs: object) -> None:
        self.calls.append(kwargs)


class _ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.dropout = nn.Dropout(0.25)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.dropout(torch.tanh(self.linear(value)))


def test_strict_policy_keeps_diffusers_default() -> None:
    model = _RecordingModel()
    configure_block_gradient_checkpointing(model, zero_stage=2, policy="strict")
    assert model.calls == [{}]


def test_zero3_policy_installs_metadata_tolerant_block_function() -> None:
    model = _RecordingModel()
    configure_block_gradient_checkpointing(
        model,
        zero_stage=3,
        policy="zero3_metadata_tolerant",
    )
    assert model.calls == [
        {"gradient_checkpointing_func": zero3_metadata_tolerant_checkpoint}
    ]


def test_zero3_policy_rejects_other_zero_stages() -> None:
    model = _RecordingModel()
    try:
        configure_block_gradient_checkpointing(
            model,
            zero_stage=2,
            policy="zero3_metadata_tolerant",
        )
    except ValueError as error:
        assert "requires ZeRO-3" in str(error)
    else:
        raise AssertionError("ZeRO-2 unexpectedly accepted the ZeRO-3 checkpoint policy")


def test_metadata_tolerant_checkpoint_matches_uncheckpointed_block() -> None:
    torch.manual_seed(29)
    reference = _ToyBlock()
    candidate = _ToyBlock()
    candidate.load_state_dict(reference.state_dict())
    reference.train()
    candidate.train()

    reference_input = torch.randn(3, 4, requires_grad=True)
    candidate_input = reference_input.detach().clone().requires_grad_(True)

    torch.manual_seed(31)
    reference_output = reference(reference_input)
    torch.manual_seed(31)
    candidate_output = zero3_metadata_tolerant_checkpoint(candidate, candidate_input)

    torch.testing.assert_close(candidate_output, reference_output)
    reference_output.sum().backward()
    candidate_output.sum().backward()
    torch.testing.assert_close(candidate_input.grad, reference_input.grad)
    torch.testing.assert_close(
        candidate.linear.weight.grad,
        reference.linear.weight.grad,
    )

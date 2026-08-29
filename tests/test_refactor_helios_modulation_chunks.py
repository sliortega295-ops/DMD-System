from __future__ import annotations

import pytest
import torch

from distillschedule.runtime.fsdp2_p0 import StaticFSDP2P0Policy
from distillschedule.runtime.helios_modulation_chunks import (
    configure_helios_modulation_chunk_tokens,
)


HeliosTransformerBlock = type(
    "HeliosTransformerBlock",
    (torch.nn.Module,),
    {"__init__": lambda self: torch.nn.Module.__init__(self)},
)


class FakeHelios(torch.nn.Module):
    def __init__(self, count: int = 40) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [HeliosTransformerBlock() for _ in range(count)]
        )


def test_configure_chunk_tokens_sets_all_40_blocks() -> None:
    model = FakeHelios()
    receipt = configure_helios_modulation_chunk_tokens(
        model, chunk_tokens=2048, role="generator"
    )
    assert receipt == {
        "status": "PASS",
        "role": "generator",
        "requested_chunk_tokens": 2048,
        "block_count": 40,
        "prior_chunk_tokens": [512],
        "observed_chunk_tokens": [2048],
        "semantic_scope": (
            "TOKEN_INDEPENDENT_MODULATION_RESIDUAL_AND_CROSS_NORM_LAUNCH_GROUPING"
        ),
    }
    assert {
        block._distillschedule_modulation_chunk_tokens for block in model.blocks
    } == {2048}


def test_chunk_policy_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="unsupported Helios modulation chunk"):
        configure_helios_modulation_chunk_tokens(
            FakeHelios(), chunk_tokens=4096, role="generator"
        )
    with pytest.raises(RuntimeError, match="block inventory"):
        configure_helios_modulation_chunk_tokens(
            FakeHelios(39), chunk_tokens=2048, role="generator"
        )
    model = FakeHelios()
    model.blocks[7]._distillschedule_modulation_chunk_tokens = 1024
    with pytest.raises(RuntimeError, match="already changed"):
        configure_helios_modulation_chunk_tokens(
            model, chunk_tokens=2048, role="critic"
        )


def test_static_policy_records_chunk_candidate() -> None:
    assert StaticFSDP2P0Policy().modulation_chunk_tokens == 512
    assert StaticFSDP2P0Policy(modulation_chunk_tokens=2048).id.endswith(
        "chunk2048"
    )
    with pytest.raises(ValueError, match="modulation_chunk_tokens"):
        StaticFSDP2P0Policy(modulation_chunk_tokens=4096)

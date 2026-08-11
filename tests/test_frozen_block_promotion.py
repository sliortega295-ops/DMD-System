from pathlib import Path

import pytest
import torch

from runtime.frozen_block_streaming import install_frozen_block_streamer


class _TinyBlocks(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [torch.nn.Linear(8, 8, bias=False) for _ in range(4)]
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_streamed_blocks_can_be_promoted_to_resident(tmp_path: Path) -> None:
    device = torch.device("cuda", 0)
    model = _TinyBlocks().to(device)
    expected = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    streamer = install_frozen_block_streamer(
        model,
        logical_component="promotion_test",
        device=device,
        window_size=1,
        streamed_block_indices=(0, 2),
        host_policy="DROP_RELOAD_FROM_MMAP",
        checkpoint_identity="promotion-test-v1",
        event_path=tmp_path / "events.jsonl",
        mmap_cache_root=tmp_path / "mmap",
        detailed_events=False,
    )
    streamer.assert_idle()
    assert model.blocks[0].weight.device.type == "cpu"
    assert model.blocks[0].weight.numel() == 0

    promoted = streamer.promote_to_resident()
    assert promoted["streamed_block_count"] == 2
    assert promoted["promoted_bytes"] == 2 * 8 * 8 * 4
    for name, parameter in model.named_parameters():
        assert parameter.device == device
        assert torch.equal(parameter.detach().cpu(), expected[name])

    # Cleanup must not undo the deliberate ownership transfer.
    streamer.remove()
    for parameter in model.parameters():
        assert parameter.device == device

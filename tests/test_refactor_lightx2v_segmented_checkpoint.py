from __future__ import annotations

from pathlib import Path

from runtime.lightx2v_segmented_checkpoint import SUPPORTED_SEGMENTS, _validate_segment


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "runtime/lightx2v_segmented_checkpoint.py"
LAUNCHER = ROOT / "scripts/distillschedule/launch_lightx2v_minimax_matched_world16_node.sh"


def test_segmented_checkpoint_has_bounded_registered_segments() -> None:
    assert SUPPORTED_SEGMENTS == (2, 4, 5, 8)
    for value in SUPPORTED_SEGMENTS:
        assert _validate_segment(value) == value


def test_segmented_checkpoint_rejects_unregistered_segments() -> None:
    for value in (0, 1, 3, 6, 10, 50):
        try:
            _validate_segment(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unexpected segment accepted: {value}")


def test_segmented_policy_keeps_fsdps_per_block_and_disables_offload() -> None:
    source = HELPER.read_text()
    assert "one_fully_shard_per_transformer_block" in source
    assert "activation_offload" in source
    assert "len(blocks) != 50" in source
    assert "torch.utils.checkpoint" not in source
    launch = LAUNCHER.read_text()
    assert 'LIGHTX2V_MINIMAX_CHECKPOINT_SEGMENT' in launch
    assert 'LIGHTX2V_MINIMAX_ACTIVATION_OFFLOAD' in launch

from __future__ import annotations

import copy
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.utils.checkpoint  # noqa: F401,E402

from runtime.lightx2v_segmented_checkpoint import (  # noqa: E402
    DEBUG_REFERENCE_SEGMENT,
    SUPPORTED_SEGMENTS,
    _install_transformer_forward,
    _validate_segment,
)


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "runtime/lightx2v_segmented_checkpoint.py"
LAUNCHER = ROOT / "scripts/distillschedule/launch_lightx2v_minimax_matched_world16_node.sh"


def test_segmented_checkpoint_has_bounded_registered_segments() -> None:
    assert SUPPORTED_SEGMENTS == (2, 4, 5, 8)
    for value in SUPPORTED_SEGMENTS:
        assert _validate_segment(value) == value


def test_segmented_checkpoint_rejects_unregistered_segments() -> None:
    for value in (0, DEBUG_REFERENCE_SEGMENT, 3, 6, 10, 50):
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


def _install_fake_diffusers_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide only the pinned helper symbols for the CPU semantic test."""
    diffusers = types.ModuleType("diffusers")
    models = types.ModuleType("diffusers.models")
    modeling_utils = types.ModuleType("diffusers.models.modeling_utils")
    transformers = types.ModuleType("diffusers.models.transformers")
    minimax = types.ModuleType("diffusers.models.transformers.transformer_minimax_h3")
    utils = types.ModuleType("diffusers.utils")

    modeling_utils.get_parameter_dtype = lambda module: next(module.parameters()).dtype
    minimax.MINIMAX_H3_MODALITY_NUM = 2

    class MiniMaxH3TransformerOutput:
        def __init__(self, sample: torch.Tensor, audio_sample: torch.Tensor):
            self.sample = sample
            self.audio_sample = audio_sample

    minimax.MiniMaxH3TransformerOutput = MiniMaxH3TransformerOutput
    utils.apply_lora_scale = lambda _name: (lambda fn: fn)
    for name, value in {
        "diffusers": diffusers,
        "diffusers.models": models,
        "diffusers.models.modeling_utils": modeling_utils,
        "diffusers.models.transformers": transformers,
        "diffusers.models.transformers.transformer_minimax_h3": minimax,
        "diffusers.utils": utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, value)


class MiniMaxH3Transformer3DModel(torch.nn.Module):
    """Tiny shape-compatible stand-in used only for wrapper semantics."""

    def __init__(self, state: dict[str, torch.Tensor] | None = None):
        super().__init__()
        self.gradient_checkpointing = True
        self.rope = torch.nn.Identity()
        self.proj_in = torch.nn.Linear(4, 4)
        self.audio_proj_in = torch.nn.Linear(4, 4)
        self.context_embedder = torch.nn.Linear(4, 4)
        self.token_refiner = torch.nn.Identity()
        self.time_proj = torch.nn.Linear(1, 4)
        self.time_embedder = torch.nn.Linear(4, 4)
        self.transformer_blocks = torch.nn.ModuleList([_FakeBlock() for _ in range(50)])
        self.norm_out = _FakeNorm()
        self.proj_out = torch.nn.Linear(4, 4)
        self.audio_proj_out = torch.nn.Linear(4, 4)
        self.checkpoint_calls = 0
        self._install_checkpoint_function()
        if state is not None:
            self.load_state_dict(state)

    def _install_checkpoint_function(self) -> None:
        def checkpoint(function, value):
            self.checkpoint_calls += 1
            return torch.utils.checkpoint.checkpoint(function, value, use_reentrant=False)

        self._gradient_checkpointing_func = checkpoint

    def forward(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        timestep,
        timestep_indices,
        token_tags,
        position_ids,
        video_indices,
        audio_indices,
        text_indices,
        attention_kwargs=None,
        return_dict=True,
    ):
        rotary_emb = self.rope(position_ids)
        video_embeds = self.proj_in(hidden_states)
        audio_embeds = self.audio_proj_in(audio_hidden_states)
        text_embeds = self.token_refiner(self.context_embedder(encoder_hidden_states))
        hidden_states = text_embeds.new_zeros((text_embeds.shape[0], position_ids.shape[0], 4))
        hidden_states = hidden_states.index_copy(1, text_indices, text_embeds)
        hidden_states = hidden_states.index_copy(1, video_indices, video_embeds)
        hidden_states = hidden_states.index_copy(1, audio_indices, audio_embeds)
        temb = self.time_embedder(self.time_proj(timestep))
        adaln_indices = timestep_indices * 2 + token_tags
        for block in self.transformer_blocks:
            def run_one(value, block=block):
                return block(value, temb, adaln_indices, rotary_emb)

            hidden_states = self._gradient_checkpointing_func(run_one, hidden_states)
        normalized = self.norm_out(hidden_states, temb, timestep_indices)
        video_output = self.proj_out(normalized).index_select(1, video_indices)
        audio_output = self.audio_proj_out(normalized).index_select(1, audio_indices)
        return types.SimpleNamespace(sample=video_output, audio_sample=audio_output)


class _FakeNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(4)

    def forward(self, value, _temb, _timestep_indices):
        return self.norm(value)


class _FakeBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(()))
        self.lora_A = torch.nn.Parameter(torch.randn(()))

    def forward(self, value, temb, _adaln_indices, _rotary_emb):
        return value + self.weight * self.lora_A + temb[..., : value.shape[-1]].unsqueeze(1) * 0.01


def _test_inputs() -> dict[str, torch.Tensor]:
    return {
        "hidden_states": torch.randn(1, 2, 4),
        "audio_hidden_states": torch.randn(1, 1, 4),
        "encoder_hidden_states": torch.randn(1, 1, 4),
        "timestep": torch.randn(1, 1),
        "timestep_indices": torch.tensor([0, 1, 0, 1]),
        "token_tags": torch.tensor([0, 1, 0, 1]),
        "position_ids": torch.zeros(4, 3),
        "video_indices": torch.tensor([1, 2]),
        "audio_indices": torch.tensor([3]),
        "text_indices": torch.tensor([0]),
    }


def _run_reference_or_segmented(model: MiniMaxH3Transformer3DModel, inputs: dict[str, torch.Tensor]):
    hooks = [block.register_forward_hook(lambda *_args: None) for block in model.transformer_blocks]
    counts = [0] * 50
    for index, block in enumerate(model.transformer_blocks):
        hooks[index].remove()
        hooks[index] = block.register_forward_hook(lambda *_args, i=index: counts.__setitem__(i, counts[i] + 1))
    output = model(**inputs)
    loss = output.sample.square().mean() + output.audio_sample.square().mean()
    loss.backward()
    for hook in hooks:
        hook.remove()
    gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters() if parameter.grad is not None}
    return output.sample.detach(), output.audio_sample.detach(), gradients, counts, model.checkpoint_calls


def test_segmented_forward_debug_reference_and_checkpoint_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute the wrapper: segment=1 is the native semantic reference."""
    _install_fake_diffusers_imports(monkeypatch)
    assert _validate_segment(DEBUG_REFERENCE_SEGMENT, allow_debug_reference=True) == 1
    torch.manual_seed(7)
    reference = MiniMaxH3Transformer3DModel()
    state = copy.deepcopy(reference.state_dict())
    inputs = _test_inputs()
    ref_result = _run_reference_or_segmented(reference, inputs)
    assert ref_result[4] == 50
    expected_calls = {1: 50, 2: 25, 4: 13, 5: 10, 8: 7}

    for segment, call_count in expected_calls.items():
        candidate = MiniMaxH3Transformer3DModel(state)
        debug_events: list[dict[str, object]] = []
        candidate._distillschedule_segment_debug_hook = debug_events.append
        _install_transformer_forward(candidate, segment, validate_contract=False, debug=True)
        result = _run_reference_or_segmented(candidate, inputs)
        torch.testing.assert_close(result[0], ref_result[0], rtol=0, atol=0)
        torch.testing.assert_close(result[1], ref_result[1], rtol=0, atol=0)
        assert result[2].keys() == ref_result[2].keys()
        for name in result[2]:
            torch.testing.assert_close(result[2][name], ref_result[2][name], rtol=0, atol=0)
        if segment == DEBUG_REFERENCE_SEGMENT:
            assert result[3] == ref_result[3]
        else:
            # Grouped checkpoints can legitimately change which blocks are
            # replayed before non-reentrant early-stop; every logical block
            # still executes in the original forward and is observed here.
            assert all(count >= 1 for count in result[3])
        assert result[4] == call_count
        assert candidate.transformer_blocks[0].lora_A.grad is not None
        segment_events = [event for event in debug_events if event["event"] == "segment_enter"]
        assert len(segment_events) == call_count
        assert sum(event["event"] == "block_enter" for event in debug_events) == 100
        for event in debug_events:
            if event["event"] == "block_enter":
                assert event["segment_start"] <= event["block"] < event["segment_end"]
        # Non-reentrant checkpoint intentionally raises its internal early-stop
        # control exception after the last needed output during recompute, so
        # the final block in each segment has no matching exit marker.
        assert sum(event["event"] == "block_exit" for event in debug_events) == 100 - call_count
        assert not any(event["event"] in {"segment_error", "block_error"} for event in debug_events)

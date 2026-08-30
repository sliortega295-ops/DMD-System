from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from distillschedule.runtime.helios_rollout_context_cache import (
    generator_nograd_static_context_cache_scope,
    install_generator_nograd_static_context_cache,
)


class CountingPatch(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = float(scale)
        self.calls = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        # Three deterministic channels, retaining the T/H/W geometry.
        base = value[:, :1].float() * self.scale
        return torch.cat((base, base + 1.0, base - 1.0), dim=1)


class CountingText(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return value * 2.0 + 1.0


class MockHelios(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embedding = CountingPatch(1.0)
        self.patch_short = CountingPatch(2.0)
        self.patch_mid = CountingPatch(3.0)
        self.patch_long = CountingPatch(4.0)
        self.condition_embedder = SimpleNamespace(text_embedder=CountingText())

    @staticmethod
    def gradient_checkpointing_method(block: torch.nn.Module, *args: object) -> object:
        return block(*args)

    @staticmethod
    def rope(
        *,
        frame_indices: torch.Tensor,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        batch, frames = frame_indices.shape
        tokens = int(frames) * int(height) * int(width)
        return torch.arange(tokens * 2, device=device, dtype=torch.float32).view(
            1, 2, frames, height, width
        ).expand(batch, -1, -1, -1, -1)

    def process_input_hidden_states(
        self,
        latents: torch.Tensor,
        indices_hidden_states: torch.Tensor | None = None,
        indices_latents_history_short: torch.Tensor | None = None,
        indices_latents_history_mid: torch.Tensor | None = None,
        indices_latents_history_long: torch.Tensor | None = None,
        latents_history_short: torch.Tensor | None = None,
        latents_history_mid: torch.Tensor | None = None,
        latents_history_long: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int], list[int], list[int]]:
        assert indices_hidden_states is not None
        assert indices_latents_history_short is not None
        assert indices_latents_history_mid is not None
        assert indices_latents_history_long is not None
        assert latents_history_short is not None
        assert latents_history_mid is not None
        assert latents_history_long is not None
        current = self.patch_embedding(latents)
        short = self.patch_short(latents_history_short)
        mid = self.patch_mid(latents_history_mid)
        long = self.patch_long(latents_history_long)
        _batch, _channels, frames, height, width = current.shape

        def tokens(value: torch.Tensor) -> torch.Tensor:
            return value.flatten(2).transpose(1, 2)

        hidden = torch.cat((tokens(long), tokens(mid), tokens(short), tokens(current)), dim=1)
        def rope_tokens(indices: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            _b, _c, _t, h, w = value.shape
            return self.rope(
                frame_indices=indices,
                height=h,
                width=w,
                device=value.device,
            ).flatten(2).transpose(1, 2)

        rope = torch.cat(
            (
                rope_tokens(indices_latents_history_long, long),
                rope_tokens(indices_latents_history_mid, mid),
                rope_tokens(indices_latents_history_short, short),
                rope_tokens(indices_hidden_states, current),
            ),
            dim=1,
        )
        return hidden, rope, [height], [width], [frames], [frames * height * width]

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        **static: torch.Tensor,
    ) -> torch.Tensor:
        del timestep
        hidden, rope, *_ = self.process_input_hidden_states(
            latents=latents, **static
        )
        text = self.condition_embedder.text_embedder(encoder_hidden_states)
        return torch.cat((hidden.flatten(), rope.flatten(), text.flatten()))


def _static_inputs() -> dict[str, torch.Tensor]:
    index = torch.tensor([[0]], dtype=torch.int64)
    return {
        "indices_hidden_states": index.clone(),
        "indices_latents_history_short": index.clone(),
        "indices_latents_history_mid": index.clone(),
        "indices_latents_history_long": index.clone(),
        "latents_history_short": torch.randn(1, 1, 1, 2, 2),
        "latents_history_mid": torch.randn(1, 1, 1, 2, 2),
        "latents_history_long": torch.randn(1, 1, 1, 2, 2),
    }


def _install(model: MockHelios, *, enabled: bool) -> dict[str, object]:
    source = Path(__file__).resolve()
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    return install_generator_nograd_static_context_cache(
        model,
        enabled=enabled,
        role="generator",
        transformer_module=SimpleNamespace(__file__=str(source)),
        expected_source_sha256=sha,
    )


def test_static_context_cache_is_default_off() -> None:
    model = MockHelios()
    receipt = _install(model, enabled=False)
    assert receipt["enabled"] is False
    assert receipt["disabled_reason"] == "NOT_REQUESTED"
    assert not hasattr(model, "_distillschedule_rollout_context_original_process")


def test_static_context_cache_matches_uncached_and_clears() -> None:
    torch.manual_seed(7)
    reference = MockHelios()
    candidate = MockHelios()
    static = _static_inputs()
    encoder = torch.randn(1, 5, 4)
    # Real Helios uses three pyramid layouts within one rollout.  Storage and
    # values change every call, while each layout repeats at least once.
    layouts = ((2, 2), (2, 2), (4, 4), (4, 4), (8, 8))
    latents = [torch.randn(1, 1, 1, h, w) for h, w in layouts]
    timestep = torch.tensor([1])
    install = _install(candidate, enabled=True)

    with torch.no_grad():
        expected = [
            reference(
                current,
                timestep=timestep,
                encoder_hidden_states=encoder,
                **static,
            )
            for current in latents
        ]
        with generator_nograd_static_context_cache_scope(
            candidate,
            enabled=True,
            expected_model_calls=5,
            expected_current_layouts=3,
            source_encoder_hidden_states=encoder,
        ) as receipt:
            observed = [
                candidate(
                    current,
                    timestep=timestep,
                    encoder_hidden_states=encoder,
                    **static,
                )
                for current in latents
            ]

    assert all(torch.equal(left, right) for left, right in zip(expected, observed))
    assert receipt["status"] == "PASS"
    assert receipt["process_fill_count"] == 1
    assert receipt["process_reuse_count"] == 4
    assert receipt["text_fill_count"] == 1
    assert receipt["text_reuse_count"] == 4
    assert receipt["current_layout_count"] == 3
    assert receipt["current_rope_fill_count"] == 3
    assert receipt["current_rope_reuse_count"] == 2
    assert receipt["cache_entries_before_clear"] == 6
    assert receipt["cache_entries_after_clear"] == 0
    assert candidate.patch_embedding.calls == 5
    assert candidate.patch_short.calls == 1
    assert candidate.patch_mid.calls == 1
    assert candidate.patch_long.calls == 1
    assert candidate.condition_embedder.text_embedder.calls == 1
    assert install["rollout_scope_count"] == 1
    assert install["process_reuse_count"] == 4
    assert install["cache_resident_bytes_sum"] > 0


def test_static_context_cache_fails_on_history_identity_change() -> None:
    model = MockHelios()
    _install(model, enabled=True)
    static = _static_inputs()
    encoder = torch.randn(1, 5, 4)
    current = torch.randn(1, 1, 1, 2, 2)
    with torch.no_grad(), pytest.raises(RuntimeError, match="history/index identity"):
        with generator_nograd_static_context_cache_scope(
            model,
            enabled=True,
            expected_model_calls=2,
            expected_current_layouts=1,
            source_encoder_hidden_states=encoder,
        ):
            model(
                current,
                timestep=torch.tensor([1]),
                encoder_hidden_states=encoder,
                **static,
            )
            changed = dict(static)
            changed["latents_history_short"] = static[
                "latents_history_short"
            ].clone()
            model(
                current,
                timestep=torch.tensor([2]),
                encoder_hidden_states=encoder,
                **changed,
            )


def test_static_context_flag_is_wired_end_to_end() -> None:
    root = Path(__file__).resolve().parents[1]
    required = (
        root / "scripts/distillschedule/run_static_fsdp2_p0_world16.py",
        root / "scripts/distillschedule/launch_static_fsdp2_p0_world16_node.sh",
        root / "scripts/distillschedule/probe_static_fsdp2_p0_world16.py",
        root / "scripts/distillschedule/probe_static_fsdp2_p0_world16_hardened.py",
        root / "scripts/distillschedule/summarize_static_fsdp2_p0_world16.py",
    )
    for path in required:
        assert "generator-nograd-static-context-cache" in path.read_text()

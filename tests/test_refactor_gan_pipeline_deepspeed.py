from __future__ import annotations

from runtime.gan_pipeline_deepspeed import backward_low_vram_gan_micro_wave


class FakeEngine:
    def __init__(self) -> None:
        self.calls = []

    def set_gradient_accumulation_boundary(self, *, is_boundary: bool) -> None:
        self.calls.append(("boundary", is_boundary))

    def backward(self, loss) -> None:
        self.calls.append(("backward", loss))

    def step(self) -> None:
        self.calls.append(("step", None))


def test_two_gan_micro_waves_produce_one_deepspeed_step() -> None:
    engine = FakeEngine()
    first = backward_low_vram_gan_micro_wave(
        engine=engine,
        denoising_loss="d0",
        gan_regularization_loss="g0",
        micro_wave=0,
    )
    second = backward_low_vram_gan_micro_wave(
        engine=engine,
        denoising_loss="d1",
        gan_regularization_loss="g1",
        micro_wave=1,
    )
    assert first.optimizer_committed is False
    assert second.optimizer_committed is True
    assert engine.calls == [
        ("boundary", False),
        ("backward", "d0"),
        ("boundary", False),
        ("backward", "g0"),
        ("boundary", False),
        ("backward", "d1"),
        ("boundary", True),
        ("backward", "g1"),
        ("step", None),
    ]

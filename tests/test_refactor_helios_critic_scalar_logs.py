from __future__ import annotations

import hashlib
import types

from distillschedule.runtime.helios_critic_scalar_logs import (
    TIMING_OMITTED,
    install_critic_scalar_log_elision,
)


def _critic_loss(
    denoising_loss,
    gan_D_loss,
    r1_loss,
    r2_loss,
    pipeline_micro_wave: int = None,
):
    critic_log_dict = {}
    if True:
        critic_log_dict["denoising_loss"] = denoising_loss.detach().item()
        critic_log_dict["gan_D_loss"] = gan_D_loss.detach().item()
        critic_log_dict["r1_loss"] = r1_loss.detach().item()
        critic_log_dict["r2_loss"] = r2_loss.detach().item()
    return denoising_loss + gan_D_loss + r1_loss + r2_loss, critic_log_dict


class Scalar:
    def __init__(self, value: float) -> None:
        self.value = value
        self.item_count = 0

    def detach(self):
        return self

    def item(self):
        self.item_count += 1
        return self.value

    def __add__(self, other):
        return float(self.value) + float(other.value if isinstance(other, Scalar) else other)

    def __radd__(self, other):
        return float(other) + float(self.value)


def _module() -> tuple[types.SimpleNamespace, str]:
    value = types.SimpleNamespace(_critic_loss=_critic_loss)
    sha = hashlib.sha256(__import__("inspect").getsource(_critic_loss).encode()).hexdigest()
    return value, sha


def test_default_off_does_not_patch() -> None:
    module, sha = _module()
    before = module._critic_loss
    receipt = install_critic_scalar_log_elision(
        enabled=False, module=module, expected_source_sha256=sha
    )
    assert module._critic_loss is before
    assert receipt["enabled"] is False


def test_correctness_collects_and_timing_elides_only_four_scalars() -> None:
    module, sha = _module()
    receipt = install_critic_scalar_log_elision(
        enabled=True, module=module, expected_source_sha256=sha
    )
    values = [Scalar(i + 1.0) for i in range(4)]
    total, logs = module._critic_loss(*values, collect_scalar_logs=True)
    assert total == 10.0
    assert list(logs.values()) == [1.0, 2.0, 3.0, 4.0]
    assert [value.item_count for value in values] == [1, 1, 1, 1]

    values = [Scalar(i + 1.0) for i in range(4)]
    total, logs = module._critic_loss(*values, collect_scalar_logs=False)
    assert total == 10.0
    assert set(logs.values()) == {TIMING_OMITTED}
    assert [value.item_count for value in values] == [0, 0, 0, 0]
    assert receipt["call_count"] == 2
    assert receipt["collected_scalar_count"] == 4
    assert receipt["elided_scalar_count"] == 4
    assert receipt["shared_source_modified"] is False

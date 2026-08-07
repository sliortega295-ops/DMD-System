import pytest

from runtime.deepspeed_backend import (
    build_client_optimizer_offload_overrides,
    build_zero_optimization_config,
)


def build(**overrides):
    values = {
        "stage": 3,
        "reduce_bucket_mib": 64,
        "allgather_bucket_mib": 32,
        "overlap_comm": False,
        "contiguous_gradients": False,
        "stage3_param_persistence_threshold": 0,
    }
    values.update(overrides)
    return build_zero_optimization_config(**values)


def test_zero3_cpu_parameter_offload_is_explicit() -> None:
    config = build(param_offload_device="cpu", param_offload_pin_memory=True)
    assert config["offload_param"] == {"device": "cpu", "pin_memory": True}
    assert config["reduce_bucket_size"] == 64 * 2**20 // 4
    assert config["stage3_prefetch_bucket_size"] == 32 * 2**20 // 4


def test_zero3_default_has_no_parameter_offload() -> None:
    config = build()
    assert "offload_param" not in config
    assert "offload_optimizer" not in config


def test_cpu_optimizer_offload_is_explicit() -> None:
    config = build(
        optimizer_offload_device="cpu", optimizer_offload_pin_memory=True
    )
    assert config["offload_optimizer"] == {"device": "cpu", "pin_memory": True}


def test_zero1_cpu_optimizer_keeps_parameters_and_gradients_replicated() -> None:
    config = build(stage=1, optimizer_offload_device="cpu")
    assert config["stage"] == 1
    assert config["offload_optimizer"] == {"device": "cpu", "pin_memory": False}
    assert "offload_param" not in config
    assert not any(key.startswith("stage3_") for key in config)


def test_client_optimizer_is_preserved_for_cpu_offload() -> None:
    assert build_client_optimizer_offload_overrides("cpu") == {
        "zero_force_ds_cpu_optimizer": False
    }
    assert build_client_optimizer_offload_overrides("none") == {}


def test_parameter_offload_rejects_zero2() -> None:
    with pytest.raises(ValueError, match="requires ZeRO-3"):
        build(stage=2, param_offload_device="cpu")


def test_pin_memory_requires_cpu_offload() -> None:
    with pytest.raises(ValueError, match="requires the CPU"):
        build(param_offload_pin_memory=True)


def test_optimizer_pin_memory_requires_cpu_offload() -> None:
    with pytest.raises(ValueError, match="requires the CPU"):
        build(optimizer_offload_pin_memory=True)

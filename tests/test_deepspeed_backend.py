import pytest
from types import SimpleNamespace

from runtime.deepspeed_backend import (
    build_client_optimizer_offload_overrides,
    build_zero_optimization_config,
    deepspeed_subgroup_default_barriers,
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


def test_zero3_prefetch_live_and_reuse_overrides_are_explicit() -> None:
    config = build(
        stage3_prefetch_bucket_elements=402_653_184,
        stage3_max_live_parameters=805_306_368,
        stage3_max_reuse_distance=268_435_456,
    )
    assert config["stage3_prefetch_bucket_size"] == 402_653_184
    assert config["stage3_max_live_parameters"] == 805_306_368
    assert config["stage3_max_reuse_distance"] == 268_435_456



def test_zero3_module_granularity_threshold_is_explicit() -> None:
    config = build(stage3_module_granularity_threshold=8_000_000)
    assert config["stage3_module_granularity_threshold"] == 8_000_000


def test_zero3_module_granularity_threshold_rejects_negative() -> None:
    with pytest.raises(ValueError, match="module_granularity"):
        build(stage3_module_granularity_threshold=-1)

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


def test_deepspeed_subgroup_scope_redirects_only_bare_barriers() -> None:
    calls = []

    def barrier(*args, **kwargs):
        calls.append((args, kwargs))
        return len(calls)

    comm = SimpleNamespace(barrier=barrier)
    subgroup = object()
    explicit = object()
    with deepspeed_subgroup_default_barriers(
        subgroup, comm_module=comm
    ) as audit:
        assert comm.barrier() == 1
        assert comm.barrier(group=explicit) == 2

    assert comm.barrier is barrier
    assert calls == [((), {"group": subgroup}), ((), {"group": explicit})]
    assert audit == {"redirected_bare_barriers": 1, "explicit_barriers": 1}


def test_deepspeed_subgroup_scope_is_fail_closed() -> None:
    comm = SimpleNamespace(barrier=lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="concrete"):
        with deepspeed_subgroup_default_barriers(None, comm_module=comm):
            pass

    with deepspeed_subgroup_default_barriers(object(), comm_module=comm):
        with pytest.raises(RuntimeError, match="nested"):
            with deepspeed_subgroup_default_barriers(object(), comm_module=comm):
                pass

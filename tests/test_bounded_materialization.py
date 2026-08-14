import pytest
import torch

from runtime.bounded_materialization import (
    _unique_registered_tensors,
    _validate_cpu_state,
    drain_deferred_host_allocator_cleanups,
    materialize_module_from_cpu_with_bounded_pinned_staging,
    materialize_module_from_persistent_pinned_cpu,
    pin_module_cpu_registered_state,
    start_deferred_host_allocator_cleanup,
)


class _MixedState(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(64, 32, dtype=torch.bfloat16))
        self.scale = torch.nn.Parameter(torch.randn(32, dtype=torch.float32))
        self.register_buffer("index", torch.arange(11, dtype=torch.int64))


def test_cpu_inventory_deduplicates_registered_objects() -> None:
    module = _MixedState()
    module.alias = module.weight
    values = _unique_registered_tensors(module)
    inventory = _validate_cpu_state(values)
    assert inventory["tensor_count"] == 3
    assert inventory["unique_storage_count"] == 3
    assert inventory["logical_bytes"] == sum(
        tensor.numel() * tensor.element_size() for _kind, _name, tensor in values
    )


def test_cpu_inventory_rejects_distinct_storage_aliases() -> None:
    module = torch.nn.Module()
    storage = torch.randn(32)
    module.left = torch.nn.Parameter(storage[:16])
    module.right = torch.nn.Parameter(storage[16:])
    with pytest.raises(ValueError, match="alias one CPU storage"):
        _validate_cpu_state(_unique_registered_tensors(module))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bounded_pinned_materialization_is_value_and_metadata_exact() -> None:
    module = _MixedState()
    before = {
        name: (
            tensor.detach().clone(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
        )
        for _kind, name, tensor in _unique_registered_tensors(module)
    }
    result = materialize_module_from_cpu_with_bounded_pinned_staging(
        module,
        device=torch.device("cuda", 0),
        chunk_mib=1,
        slot_count=2,
    )
    after = {name: tensor for _kind, name, tensor in _unique_registered_tensors(module)}
    assert result["policy"] == "BOUNDED_PINNED_PING_PONG_H2D"
    assert result["pinned_arena_bytes"] <= 6 * 2**20
    assert result["ready_on_return"] is True
    for name, (value, shape, stride, dtype) in before.items():
        observed = after[name]
        assert observed.is_cuda
        assert tuple(observed.shape) == shape
        assert tuple(observed.stride()) == stride
        assert observed.dtype == dtype
        assert torch.equal(observed.cpu(), value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_persistent_pinned_materialization_is_value_object_and_metadata_exact() -> None:
    module = _MixedState()
    before = {
        name: (
            id(tensor),
            tensor.detach().clone(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            bool(getattr(tensor, "requires_grad", False)),
        )
        for _kind, name, tensor in _unique_registered_tensors(module)
    }
    pinned = pin_module_cpu_registered_state(module)
    pinned_state = {
        name: tensor for _kind, name, tensor in _unique_registered_tensors(module)
    }
    assert pinned["policy"] == "PER_RANK_PRIVATE_PERSISTENT_PINNED_REGISTERED_STATE"
    assert pinned["runtime_owned_active_pinned_bytes"] == pinned["logical_bytes"]
    assert all(tensor.is_pinned() for tensor in pinned_state.values())
    assert all(id(pinned_state[name]) == expected[0] for name, expected in before.items())

    result = materialize_module_from_persistent_pinned_cpu(
        module,
        device=torch.device("cuda", 0),
    )
    after = {name: tensor for _kind, name, tensor in _unique_registered_tensors(module)}
    assert result["policy"] == "PERSISTENT_PINNED_DIRECT_H2D"
    assert result["host_allocator_cleanup_policy"] == "immediate"
    assert result["released_runtime_owned_active_pinned_bytes"] == pinned["logical_bytes"]
    assert result["runtime_owned_active_pinned_bytes_after_return"] == 0
    assert result["ready_on_return"] is True
    for name, (object_id, value, shape, stride, dtype, requires_grad) in before.items():
        observed = after[name]
        assert id(observed) == object_id
        assert observed.is_cuda
        assert tuple(observed.shape) == shape
        assert tuple(observed.stride()) == stride
        assert observed.dtype == dtype
        assert observed.requires_grad == requires_grad
        assert torch.equal(observed.cpu(), value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_persistent_materialization_rejects_pageable_state() -> None:
    with pytest.raises(ValueError, match="pageable state"):
        materialize_module_from_persistent_pinned_cpu(
            _MixedState(),
            device=torch.device("cuda", 0),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_deferred_pinned_cleanup_is_joined_and_value_exact() -> None:
    module = _MixedState()
    before = {
        name: tensor.detach().clone()
        for _kind, name, tensor in _unique_registered_tensors(module)
    }
    pinned = pin_module_cpu_registered_state(module)
    result = materialize_module_from_persistent_pinned_cpu(
        module,
        device=torch.device("cuda", 0),
        cleanup_policy="deferred",
    )
    assert result["host_allocator_cleanup_policy"] == "deferred"
    assert isinstance(result["deferred_host_cleanup_token"], int)
    assert result["allocator_trim_host_wall_seconds"] == 0.0
    assert (
        result["runtime_owned_cached_pinned_bytes_pending_cleanup"]
        == pinned["logical_bytes"]
    )
    cleanups = drain_deferred_host_allocator_cleanups()
    assert len(cleanups) == 1
    assert cleanups[0]["status"] == "PASS"
    assert cleanups[0]["join_wait_host_wall_seconds"] >= 0.0
    assert drain_deferred_host_allocator_cleanups() == []
    after = {name: tensor for _kind, name, tensor in _unique_registered_tensors(module)}
    assert all(torch.equal(after[name].cpu(), value) for name, value in before.items())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_late_deferred_cleanup_requires_explicit_start_then_drain() -> None:
    module = _MixedState()
    pin_module_cpu_registered_state(module)
    result = materialize_module_from_persistent_pinned_cpu(
        module,
        device=torch.device("cuda", 0),
        cleanup_policy="late_deferred",
    )
    assert result["deferred_host_cleanup_token"] is None
    assert result["host_allocator_trimmed"] is False
    assert drain_deferred_host_allocator_cleanups() == []
    token = start_deferred_host_allocator_cleanup()
    cleanups = drain_deferred_host_allocator_cleanups()
    assert [cleanup["token"] for cleanup in cleanups] == [token]
    assert cleanups[0]["status"] == "PASS"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_post_full5_cleanup_requires_explicit_start_then_drain() -> None:
    module = _MixedState()
    pin_module_cpu_registered_state(module)
    result = materialize_module_from_persistent_pinned_cpu(
        module,
        device=torch.device("cuda", 0),
        cleanup_policy="post_full5",
    )
    assert result["deferred_host_cleanup_token"] is None
    assert result["host_allocator_trimmed"] is False
    assert result["runtime_owned_cached_pinned_bytes_pending_cleanup"] > 0
    assert drain_deferred_host_allocator_cleanups() == []
    token = start_deferred_host_allocator_cleanup()
    cleanups = drain_deferred_host_allocator_cleanups()
    assert [cleanup["token"] for cleanup in cleanups] == [token]
    assert cleanups[0]["status"] == "PASS"

"""Bounded FSDP2 full-parameter reuse for a no-grad rollout phase.

The scope changes only ``reshard_after_forward`` while several independent B1
eval forwards use the same parameter version.  It restores every FSDP unit to
the original sharded policy before returning to training.  There is no sample
batching, parameter offload, or persistent topology change.
"""
from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Any


EXPECTED_HELIOS_FSDP_MODULES = 41  # 40 transformer blocks plus the root.


def _memory_snapshot() -> dict[str, int]:
    import torch

    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _parameter_group(module: Any) -> Any | None:
    state = module._get_fsdp_state()
    return state._fsdp_param_group


def _count_unsharded(modules: list[Any]) -> int:
    return sum(
        int(bool(group.is_unsharded))
        for module in modules
        if (group := _parameter_group(module)) is not None
    )


@contextlib.contextmanager
def generator_nograd_persistent_unshard_scope(
    module: Any,
    *,
    enabled: bool,
    flush_allocator_cache: bool = True,
    module_type: type[Any] | tuple[type[Any], ...] | None = None,
    empty_cache: Callable[[], None] | None = None,
    memory_snapshot: Callable[[], dict[str, int]] | None = None,
) -> Iterator[dict[str, Any]]:
    """Retain all Generator full parameters across one no-grad rollout phase.

    ``FSDPModule.set_reshard_after_forward(False)`` is the public PyTorch API
    for repeated eval forwards.  The private ``is_unsharded`` state is read
    only as a version-pinned fail-closed audit; no private state is mutated.
    """

    if not enabled:
        yield {
            "enabled": False,
            "flush_allocator_cache": bool(flush_allocator_cache),
            "scope": "GENERATOR_NOGRAD_ROLLOUTS_ONLY",
            "expected_fsdp_module_count": EXPECTED_HELIOS_FSDP_MODULES,
            "rollout_batch_count": 0,
            "restored": True,
        }
        return

    if module_type is None:
        from torch.distributed.fsdp import FSDPModule

        module_type = FSDPModule
    if flush_allocator_cache and empty_cache is None:
        import torch

        empty_cache = torch.cuda.empty_cache
    snapshot = memory_snapshot or _memory_snapshot
    modules = [value for value in module.modules() if isinstance(value, module_type)]
    if len(modules) != EXPECTED_HELIOS_FSDP_MODULES:
        raise AssertionError(
            "Generator FSDP module count changed: "
            f"{len(modules)} != {EXPECTED_HELIOS_FSDP_MODULES}"
        )
    parameter_groups = [
        group for value in modules if (group := _parameter_group(value)) is not None
    ]
    if len(parameter_groups) != EXPECTED_HELIOS_FSDP_MODULES:
        raise AssertionError(
            "Generator FSDP parameter-group count changed: "
            f"{len(parameter_groups)} != {EXPECTED_HELIOS_FSDP_MODULES}"
        )
    if _count_unsharded(modules):
        raise AssertionError("Generator entered persistent rollout scope unsharded")

    receipt: dict[str, Any] = {
        "enabled": True,
        "flush_allocator_cache": bool(flush_allocator_cache),
        "scope": "GENERATOR_NOGRAD_ROLLOUTS_ONLY",
        "expected_fsdp_module_count": EXPECTED_HELIOS_FSDP_MODULES,
        "fsdp_module_count": len(modules),
        "parameter_group_count": len(parameter_groups),
        "rollout_batch_count": 0,
        "pre_empty_cache_count": 0,
        "post_empty_cache_count": 0,
        "set_no_reshard_count": 0,
        "restore_reshard_policy_count": 0,
        "explicit_reshard_count": 0,
        "fully_unsharded_before_exit_count": None,
        "unsharded_after_exit_count": None,
        "restored": False,
        "memory_before": snapshot(),
        "memory_after_pre_empty_cache": None,
        "memory_before_exit": None,
        "memory_after_restore": None,
    }

    if flush_allocator_cache:
        assert empty_cache is not None
        empty_cache()
        receipt["pre_empty_cache_count"] = 1
    receipt["memory_after_pre_empty_cache"] = snapshot()
    for value in modules:
        value.set_reshard_after_forward(False, recurse=False)
        receipt["set_no_reshard_count"] += 1

    completed = False
    try:
        yield receipt
        completed = True
    finally:
        receipt["memory_before_exit"] = snapshot()
        receipt["fully_unsharded_before_exit_count"] = _count_unsharded(modules)

        # Restore the policy before forcing reshard so cleanup also works if an
        # exception interrupted a module while its training state was FORWARD.
        for value in modules:
            value.set_reshard_after_forward(True, recurse=False)
            receipt["restore_reshard_policy_count"] += 1
        for value in reversed(modules):
            value.reshard()
            receipt["explicit_reshard_count"] += 1

        if flush_allocator_cache:
            assert empty_cache is not None
            empty_cache()
            receipt["post_empty_cache_count"] = 1
        receipt["unsharded_after_exit_count"] = _count_unsharded(modules)
        receipt["memory_after_restore"] = snapshot()
        receipt["restored"] = receipt["unsharded_after_exit_count"] == 0

        if receipt["unsharded_after_exit_count"] != 0:
            raise AssertionError("Generator FSDP parameters remained unsharded")
        if completed and (
            receipt["fully_unsharded_before_exit_count"]
            != EXPECTED_HELIOS_FSDP_MODULES
        ):
            raise AssertionError(
                "Generator no-grad phase did not materialize every FSDP unit: "
                f"{receipt['fully_unsharded_before_exit_count']} != "
                f"{EXPECTED_HELIOS_FSDP_MODULES}"
            )


__all__ = [
    "EXPECTED_HELIOS_FSDP_MODULES",
    "generator_nograd_persistent_unshard_scope",
]

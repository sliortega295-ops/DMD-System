"""Manual opt-in, phase-local reuse of selected Generator FSDP parameters.

This is not full-model persistence and does not add a search-space dimension.
Only public FSDP mutation APIs are used; pinned private state is read for audit.
"""
from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Any

from .fsdp2_nograd_persistence import (
    EXPECTED_HELIOS_FSDP_MODULES, _count_unsharded, _memory_snapshot, _parameter_group,
)

POLICY = "generator-first24-parameters.v1"
BLOCK_INDICES = tuple(range(24))


def validate_receipt(receipt: dict[str, Any], *, expected_calls: int) -> None:
    """Audit the actual scope, not just its requested configuration."""
    p = receipt.get("partial_parameter_retention", {})
    wanted = dict(enabled=True, policy=POLICY, block_indices=list(BLOCK_INDICES),
                  selected_unit_count=24, set_no_reshard_count=24,
                  restore_policy_count=24, explicit_reshard_count=24,
                  selected_unsharded_before_exit_count=24,
                  total_unsharded_before_exit_count=24,
                  total_unsharded_after_exit_count=0,
                  actual_generator_forward_count=expected_calls,
                  grad_enabled_failures=0, parameter_version_failures=0,
                  restored=True, weights_merged=False, parameter_dtype_changed=False,
                  parameter_offload=False)
    if not (receipt.get("enabled") is False and receipt.get("restored") is True
            and receipt.get("rollout_batch_count") == 5
            and all(type(p.get(k)) is type(v) and p.get(k) == v for k, v in wanted.items())
            and type(p.get("generator_version_before")) is int
            and p["generator_version_before"] == p.get("generator_version_after")):
        raise AssertionError("partial Generator parameter scope actual census/version/lifecycle failed")


@contextlib.contextmanager
def partial_generator_nograd_scope(
    module: Any, *, retained_block_count: int,
    parameter_version_reader: Callable[[], int],
    module_type: type[Any] | tuple[type[Any], ...] | None = None,
    memory_snapshot: Callable[[], dict[str, int]] | None = None,
) -> Iterator[dict[str, Any]]:
    if type(retained_block_count) is not int or retained_block_count != 24:
        raise ValueError("only the versioned first24 parameter policy is admitted")
    if not callable(parameter_version_reader):
        raise ValueError("partial parameter reuse needs an actual Generator version reader")
    import torch

    if module_type is None:
        from torch.distributed.fsdp import FSDPModule
        module_type = FSDPModule
    snapshot = memory_snapshot or _memory_snapshot
    modules = [m for m in module.modules() if isinstance(m, module_type)]
    blocks = list(getattr(module, "blocks", ()))
    if (len(modules) != EXPECTED_HELIOS_FSDP_MODULES or len(blocks) != 40
            or len({id(m) for m in blocks}) != 40 or module in blocks
            or any(not isinstance(m, module_type) for m in blocks)):
        raise AssertionError("partial reuse requires exactly40 FSDP blocks plus its root")
    groups = [_parameter_group(m) for m in modules]
    if (any(g is None for g in groups) or len({id(g) for g in groups}) != 41
            or any(getattr(g, "_reshard_after_forward", None) is not True for g in groups)
            or _count_unsharded(modules) != 0):
        raise AssertionError("Generator must enter with41 unique sharded reshard-after-forward units")
    selected = [blocks[i] for i in BLOCK_INDICES]
    version = parameter_version_reader()
    if type(version) is not int or version < 0:
        raise AssertionError("invalid physical Generator parameter version")
    p: dict[str, Any] = dict(
        enabled=True, policy=POLICY, block_indices=list(BLOCK_INDICES),
        selected_unit_count=24, set_no_reshard_count=0, restore_policy_count=0,
        explicit_reshard_count=0, selected_unsharded_before_exit_count=None,
        total_unsharded_before_exit_count=None, total_unsharded_after_exit_count=None,
        actual_generator_forward_count=0, grad_enabled_failures=0,
        parameter_version_failures=0, generator_version_before=version,
        generator_version_after=None, restored=False, weights_merged=False,
        parameter_dtype_changed=False, parameter_offload=False,
        memory_before=snapshot(), memory_before_exit=None, memory_after_restore=None,
    )
    receipt = dict(enabled=False, scope="GENERATOR_NOGRAD_ROLLOUTS_ONLY",
                   flush_allocator_cache=False, expected_fsdp_module_count=41,
                   rollout_batch_count=0, restored=False, partial_parameter_retention=p)

    def before_forward(_module, _args):
        if torch.is_grad_enabled():
            p["grad_enabled_failures"] += 1
            raise AssertionError("partial parameter persistence reached a grad-enabled Generator call")
        if parameter_version_reader() != version:
            p["parameter_version_failures"] += 1
            raise AssertionError("Generator version changed inside partial reuse scope")
        p["actual_generator_forward_count"] += 1

    handle = module.register_forward_pre_hook(before_forward)
    changed = []
    completed = False
    try:
        for m in selected:
            m.set_reshard_after_forward(False, recurse=False)
            changed.append(m)
            p["set_no_reshard_count"] += 1
        yield receipt
        completed = True
    finally:
        handle.remove()
        p["memory_before_exit"] = snapshot()
        p["selected_unsharded_before_exit_count"] = _count_unsharded(selected)
        p["total_unsharded_before_exit_count"] = _count_unsharded(modules)
        for m in changed:
            m.set_reshard_after_forward(True, recurse=False)
            p["restore_policy_count"] += 1
        for m in reversed(changed):
            m.reshard()
            p["explicit_reshard_count"] += 1
        p["generator_version_after"] = parameter_version_reader()
        p["total_unsharded_after_exit_count"] = _count_unsharded(modules)
        p["memory_after_restore"] = snapshot()
        p["restored"] = (p["total_unsharded_after_exit_count"] == 0
                         and all(g._reshard_after_forward is True for g in groups))
        receipt["restored"] = p["restored"]
        if not p["restored"]:
            raise AssertionError("partial Generator parameter reuse did not fully restore/release")
        if completed:
            validate_receipt(receipt, expected_calls=p["actual_generator_forward_count"])
            if p["actual_generator_forward_count"] <= 0:
                raise AssertionError("partial parameter reuse never executed a Generator call")

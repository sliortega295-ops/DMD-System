"""Bounded elision of official Helios GAN Generator allocator flushes.

The pinned official GAN Generator Score/DMD/GAN-G path calls Diffusers'
``free_memory`` four times.  Each call performs Python garbage collection and
``torch.cuda.empty_cache`` even though the static FSDP2 worker retains enough
allocated-memory headroom and immediately continues model execution.

This module does not edit the shared Helios checkout.  It source-pins the three
installed functions involved, then temporarily replaces only the Helios module
global named ``free_memory`` while the one official Generator helper runs.
Correctness delegates every call to the original function; low-overhead
timing/profile may count and elide exactly four calls.  The global is restored
in ``finally`` and an unexpected call count fails closed.
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import inspect
from typing import Any, Iterator


HELIOS_DISTRIBUTION_MATCHING_SHA256 = (
    "c479e4f3eec2730d1e991ecb19e06588ec30510dabe48e2daf3846d9f7960db1"
)
HELIOS_GAN_CACHED_GRAD_SHA256 = (
    "8455f5d265b189208491681e6372790f7b5809773953a6c855a668f10b632643"
)
DIFFUSERS_FREE_MEMORY_SHA256 = (
    "d16707d1b9935038a99f9a927043844d7eea8d22a9b6e008fbfc014c49885df3"
)
EXPECTED_CALLS_PER_GENERATOR_UPDATE = 4


def source_sha256(value: Any) -> str:
    return hashlib.sha256(inspect.getsource(value).encode()).hexdigest()


def prepare_generator_free_memory_receipt(
    *,
    enabled: bool,
    module: Any | None = None,
) -> dict[str, Any]:
    """Verify installed source identity and return a mutable run receipt."""

    receipt: dict[str, Any] = {
        "requested": bool(enabled),
        "enabled": False,
        "scope": "OFFICIAL_GAN_GENERATOR_SCORE_DMD_GAN_G_ONLY",
        "expected_calls_per_generator_update": (
            EXPECTED_CALLS_PER_GENERATOR_UPDATE
        ),
        "scope_entry_count": 0,
        "call_count": 0,
        "executed_count": 0,
        "elided_count": 0,
        "restored_count": 0,
        "shared_source_modified": False,
        "semantic_contract": (
            "MODEL_LOSS_BACKWARD_OPTIMIZER_TENSORS_COLLECTIVES_AND_ORDER_UNCHANGED;"
            "CORRECTNESS_EXECUTES_OFFICIAL_FREE_MEMORY;"
            "TIMING_PROFILE_MAY_ELIDE_EXACTLY_FOUR_GENERATOR_ONLY_CALLS"
        ),
        "source_sha256": {},
        "expected_source_sha256": {
            "compute_distribution_matching_loss": (
                HELIOS_DISTRIBUTION_MATCHING_SHA256
            ),
            "Gan_D_Loss_With_Cached_Grad": HELIOS_GAN_CACHED_GRAD_SHA256,
            "free_memory": DIFFUSERS_FREE_MEMORY_SHA256,
        },
    }
    if not enabled:
        receipt["disabled_reason"] = "NOT_REQUESTED"
        return receipt

    if module is None:
        from helios.utils import utils_helios_post as module

    observed = {
        "compute_distribution_matching_loss": source_sha256(
            module.compute_distribution_matching_loss
        ),
        "Gan_D_Loss_With_Cached_Grad": source_sha256(
            module.Gan_D_Loss_With_Cached_Grad
        ),
        "free_memory": source_sha256(module.free_memory),
    }
    receipt["source_sha256"] = observed
    for name, expected in receipt["expected_source_sha256"].items():
        if observed[name] != expected:
            raise RuntimeError(
                f"installed {name} source changed: {observed[name]} != {expected}"
            )
    if getattr(module, "_distillschedule_generator_free_memory_scope_active", False):
        raise RuntimeError("Generator free-memory scope was already active")
    receipt["enabled"] = True
    return receipt


@contextlib.contextmanager
def generator_free_memory_scope(
    receipt: dict[str, Any] | None,
    *,
    elide: bool,
    module: Any | None = None,
) -> Iterator[None]:
    """Count and optionally elide the four calls in one Generator update."""

    if receipt is None or not receipt.get("enabled", False):
        yield
        return
    if module is None:
        from helios.utils import utils_helios_post as module

    if getattr(module, "_distillschedule_generator_free_memory_scope_active", False):
        raise RuntimeError("nested Generator free-memory scope is not supported")
    original = module.free_memory
    if source_sha256(original) != DIFFUSERS_FREE_MEMORY_SHA256:
        raise RuntimeError("Helios free_memory global changed after source validation")
    call_count_before = int(receipt["call_count"])

    @functools.wraps(original)
    def wrapped() -> Any:
        receipt["call_count"] += 1
        if elide:
            receipt["elided_count"] += 1
            return None
        receipt["executed_count"] += 1
        return original()

    receipt["scope_entry_count"] += 1
    module._distillschedule_generator_free_memory_scope_active = True
    module.free_memory = wrapped
    try:
        yield
    finally:
        module.free_memory = original
        module._distillschedule_generator_free_memory_scope_active = False
        receipt["restored_count"] += 1

    delta = int(receipt["call_count"]) - call_count_before
    if delta != EXPECTED_CALLS_PER_GENERATOR_UPDATE:
        raise RuntimeError(
            "official GAN Generator free_memory call count changed: "
            f"{delta} != {EXPECTED_CALLS_PER_GENERATOR_UPDATE}"
        )


__all__ = [
    "DIFFUSERS_FREE_MEMORY_SHA256",
    "EXPECTED_CALLS_PER_GENERATOR_UPDATE",
    "HELIOS_DISTRIBUTION_MATCHING_SHA256",
    "HELIOS_GAN_CACHED_GRAD_SHA256",
    "generator_free_memory_scope",
    "prepare_generator_free_memory_receipt",
    "source_sha256",
]

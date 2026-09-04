"""Opt-in removal of official Helios Critic scalar logging synchronizations.

The pinned GAN Critic helper converts four detached CUDA scalars to Python
numbers after every optimizer commit.  The P0 timing worker discards those
values, but ``Tensor.item()`` still inserts a host allocation, a tiny D2H copy,
and a stream synchronization.  This module source-pins the external helper and
adds one default-true keyword.  Correctness runs retain the official behavior;
low-overhead timing/profile runs may return an explicit omission sentinel.

No loss, backward, optimizer, tensor, collective, or update-order expression is
changed.  The patch is installed in memory in each worker process and never
edits the shared Helios checkout.
"""
from __future__ import annotations

import functools
import hashlib
import inspect
from typing import Any


HELIOS_CRITIC_LOSS_SHA256 = (
    "7078d12bb1c59913bf5f8438785fe097cf22114796f53a489394d3645db4c282"
)
TIMING_OMITTED = "omitted_in_low_overhead_timing_mode"
_SIGNATURE_NEEDLE = "    pipeline_micro_wave: int = None,\n):"
_SCALAR_NAMES = ("denoising_loss", "gan_D_loss", "r1_loss", "r2_loss")


def critic_scalar_log_call_kwargs(
    *,
    collect_scalar_logs: bool,
    module: Any,
) -> dict[str, bool]:
    """Return the optional scalar-log keyword for the live Helios helper.

    The pinned official ``_critic_loss`` predates ``collect_scalar_logs``.  A
    correctness/trajectory call wants that helper's default behavior, so it
    must *omit* the newer keyword rather than passing ``True``.  Low-overhead
    timing calls request ``False`` only after the source-pinned in-process
    elision patch has been installed (or if a future helper explicitly exposes
    the keyword).

    This is deliberately capability-checked instead of catching ``TypeError``:
    retrying a training helper after it has started could duplicate backward or
    optimizer side effects.
    """

    if collect_scalar_logs:
        return {}

    function = module._critic_loss
    signature = inspect.signature(function, follow_wrapped=False)
    explicitly_supported = "collect_scalar_logs" in signature.parameters
    installed_patch = bool(
        getattr(module, "_distillschedule_critic_scalar_log_elision", False)
    )
    if not (explicitly_supported or installed_patch):
        raise RuntimeError(
            "collect_scalar_logs=False requires the source-pinned Helios "
            "Critic scalar-log elision patch or an explicit live API"
        )
    return {"collect_scalar_logs": False}


def source_sha256(function: Any) -> str:
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def _rewrite_critic_loss_source(source: str) -> str:
    if source.count(_SIGNATURE_NEEDLE) != 1:
        raise RuntimeError("pinned Helios _critic_loss signature changed")
    rewritten = source.replace(
        _SIGNATURE_NEEDLE,
        "    pipeline_micro_wave: int = None,\n"
        "    collect_scalar_logs: bool = True,\n"
        "):",
    )
    replacements = 0
    for name in _SCALAR_NAMES:
        old = f'{name}.detach().item()'
        new = (
            f'({old} if collect_scalar_logs else '
            f'"{TIMING_OMITTED}")'
        )
        count = rewritten.count(old)
        if count != 1:
            raise RuntimeError(
                f"pinned Helios scalar log expression changed for {name}: {count}"
            )
        rewritten = rewritten.replace(old, new)
        replacements += count
    if replacements != 4:
        raise RuntimeError("Helios Critic scalar-log replacement count changed")
    return rewritten


def install_critic_scalar_log_elision(
    *,
    enabled: bool,
    module: Any | None = None,
    expected_source_sha256: str = HELIOS_CRITIC_LOSS_SHA256,
) -> dict[str, Any]:
    """Install the source-pinned helper replacement and return a live receipt."""

    receipt: dict[str, Any] = {
        "requested": bool(enabled),
        "enabled": False,
        "scope": "OFFICIAL_GAN_CRITIC_SCALAR_LOGS_ONLY",
        "source_sha256": None,
        "expected_source_sha256": str(expected_source_sha256),
        "call_count": 0,
        "collected_scalar_count": 0,
        "elided_scalar_count": 0,
        "scalar_names": list(_SCALAR_NAMES),
        "semantic_contract": (
            "LOSS_BACKWARD_OPTIMIZER_TENSORS_COLLECTIVES_AND_ORDER_UNCHANGED;"
            "CORRECTNESS_COLLECTS_OFFICIAL_SCALARS;TIMING_RETURNS_SENTINELS"
        ),
        "shared_source_modified": False,
    }
    if not enabled:
        receipt["disabled_reason"] = "NOT_REQUESTED"
        return receipt

    if module is None:
        from helios.utils import utils_helios_post as module

    original = module._critic_loss
    observed_sha = source_sha256(original)
    receipt["source_sha256"] = observed_sha
    if observed_sha != expected_source_sha256:
        raise RuntimeError(
            "installed Helios _critic_loss source changed: "
            f"{observed_sha} != {expected_source_sha256}"
        )
    if getattr(module, "_distillschedule_critic_scalar_log_elision", False):
        raise RuntimeError("Helios Critic scalar-log elision installed twice")

    rewritten = _rewrite_critic_loss_source(inspect.getsource(original))
    namespace = dict(module.__dict__)
    exec(compile(rewritten, inspect.getsourcefile(original) or "<helios_critic_loss>", "exec"), namespace)
    rewritten_function = namespace["_critic_loss"]

    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        collect = bool(kwargs.get("collect_scalar_logs", True))
        receipt["call_count"] += 1
        key = "collected_scalar_count" if collect else "elided_scalar_count"
        receipt[key] += len(_SCALAR_NAMES)
        return rewritten_function(*args, **kwargs)

    module._critic_loss = wrapped
    module._distillschedule_critic_scalar_log_elision = True
    receipt["enabled"] = True
    return receipt


__all__ = [
    "HELIOS_CRITIC_LOSS_SHA256",
    "TIMING_OMITTED",
    "critic_scalar_log_call_kwargs",
    "install_critic_scalar_log_elision",
]

"""Exact AdaLN-cache scope restoration for native checkpoint replay.

The shared model precomputes exact AdaLN modulation before every grad-enabled DiT forward,
but PyTorch's native non-reentrant checkpoint does not preserve our ContextVar
scope when it later replays a block.  Without the scope, the parameter-free
AdaLN handle falls back to its ~520 MB projection FSDP unit and materializes one
projection per replayed block.

This module wraps the *already installed* native checkpoint function.  It
captures the exact AdaLN cache key at checkpoint creation time and restores it
around both the original block execution and backward replay.  It fails closed
if the key is missing, incomplete, or if executing a block causes any AdaLN
cache miss.  Continuous DMD sigma, model math, application forward counts,
FSDP placement and checkpoint boundaries are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from .adaln_cache import AdaLNCacheController


@dataclass
class ExactAdaLNReplayRegistration:
    transformer: Any
    controller: AdaLNCacheController
    original_checkpoint: Callable[..., Any]
    stats: dict[str, int]
    closed: bool = False

    def receipt(self) -> dict[str, Any]:
        return {
            "policy": "exact_adaln_checkpoint_replay_scope",
            "stats": dict(self.stats),
            "closed": self.closed,
        }

    def remove(self) -> None:
        if self.closed:
            return
        self.transformer._gradient_checkpointing_func = self.original_checkpoint
        self.closed = True


def install_exact_adaln_checkpoint_replay_scope(
    transformer: Any,
    controller: AdaLNCacheController,
    *,
    expected_block_count: int = 50,
) -> ExactAdaLNReplayRegistration:
    """Restore the precomputed AdaLN key whenever checkpoint executes/replays.

    Install this *after* checkpoint-boundary CPU staging, so the wrapper chain
    is:

        exact AdaLN scope -> FA3 replay -> boundary staging / native checkpoint

    The nested scoped function is what torch checkpoint stores and replays.
    PyTorch non-reentrant checkpoint may terminate replay through its internal
    early-stop exception before the user function returns, so replay auditing
    is performed from ``finally`` rather than after a normal return.
    """

    original = getattr(transformer, "_gradient_checkpointing_func", None)
    if not callable(original):
        raise RuntimeError("MiniMax transformer has no callable checkpoint function")
    if not bool(getattr(transformer, "gradient_checkpointing", False)):
        raise RuntimeError("exact AdaLN replay scope requires gradient checkpointing")

    stats = {
        "checkpoint_wrap_count": 0,
        "scoped_execution_count": 0,
        "captured_key_missing_count": 0,
        "captured_key_incomplete_count": 0,
        "cache_hit_count": 0,
        "cache_miss_count": 0,
        "unexpected_hit_delta_count": 0,
    }

    def wrapped(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        active = controller.current_scope()
        key = active.key
        if key is None:
            stats["captured_key_missing_count"] += 1
            raise RuntimeError(
                "Exact AdaLN checkpoint replay has no active cache key. "
                "Refusing to fall back to the giant AdaLN projection bank."
            )
        if not controller.has_complete_key(key, expected_block_count):
            stats["captured_key_incomplete_count"] += 1
            raise RuntimeError(
                f"Exact AdaLN cache key {key!r} is incomplete before checkpoint creation"
            )
        persistent = bool(active.persistent)
        stats["checkpoint_wrap_count"] += 1

        def scoped_function(*fn_args: Any, **fn_kwargs: Any) -> Any:
            hits_before, misses_before = controller.access_counts()
            stats["scoped_execution_count"] += 1
            try:
                with controller.scope(key, persistent=persistent):
                    return function(*fn_args, **fn_kwargs)
            finally:
                hits_after, misses_after = controller.access_counts()
                hit_delta = hits_after - hits_before
                miss_delta = misses_after - misses_before
                stats["cache_hit_count"] += hit_delta
                stats["cache_miss_count"] += miss_delta
                # AdaLN is the first block-level conditioning operation, so
                # even non-reentrant early-stop replay must have consumed the
                # one cached modulation entry before checkpoint stops.
                if miss_delta != 0:
                    raise RuntimeError(
                        f"Exact AdaLN replay cache miss for key={key!r}; "
                        f"miss_delta={miss_delta}. Projection fallback is forbidden."
                    )
                if hit_delta != 1:
                    stats["unexpected_hit_delta_count"] += 1
                    raise RuntimeError(
                        f"Exact AdaLN replay expected one cache hit per block, got {hit_delta} "
                        f"for key={key!r}"
                    )

        return original(scoped_function, *args, **kwargs)

    transformer._gradient_checkpointing_func = wrapped
    registration = ExactAdaLNReplayRegistration(
        transformer=transformer,
        controller=controller,
        original_checkpoint=original,
        stats=stats,
    )
    logger.info(
        "[h3-a100][exact-adaln-replay] installed expected_blocks={} wrapper_chain=scope->boundary->checkpoint",
        expected_block_count,
    )
    return registration

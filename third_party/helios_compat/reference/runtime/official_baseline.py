"""Opt-in helpers for the measured official Helios A100 strong baseline.

These helpers are inert unless train_helios.py explicitly enables them through the
DISTILLSCHEDULE_* environment contract. They make the existing hand-written baseline
measurable and executable and expose one narrow, profile-derived host admission rule;
they are not the complete DistillSchedule planner.
"""
from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import math
import os
import resource
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TypeVar

import torch


_TRUE = {"1", "true", "yes", "on"}
_T = TypeVar("_T")


try:
    _LIBC = ctypes.CDLL(None)
    _MALLOC_TRIM = _LIBC.malloc_trim
    _MALLOC_TRIM.argtypes = [ctypes.c_size_t]
    _MALLOC_TRIM.restype = ctypes.c_int
except (AttributeError, OSError):
    _MALLOC_TRIM = None


def trim_host_allocator() -> bool:
    """Return unused glibc arenas to the host cgroup.

    PyTorch frees the old CPU storages after ``Module.to(cuda)``, but glibc may
    retain their pages in per-process arenas.  With eight rank-local 14B model
    replicas this retained memory is large enough to OOM the node on a later
    D2H swap.  The helper is explicit and opt-in because trimming trades CPU
    allocator reuse for a lower residency watermark.
    """

    gc.collect()
    host_empty_cache = getattr(torch._C, "_host_emptyCache", None)
    if host_empty_cache is not None:
        host_empty_cache()
    if _MALLOC_TRIM is None:
        return False
    return bool(_MALLOC_TRIM(0))


@contextmanager
def suspend_transformers_zero3_init() -> Iterator[None]:
    """Temporarily keep short-lived preprocessing models out of ZeRO-3 init.

    Accelerate registers its ZeRO-3 config globally with Transformers. That is
    desirable for the 14B backbones, but Transformers rejects the explicit
    device_map used by Helios for its short-lived text encoder. Preserve and
    restore the exact global config while the VAE/text encoder are loaded,
    used once, and released.
    """

    from transformers.integrations import deepspeed as transformers_deepspeed

    config_ref = transformers_deepspeed._hf_deepspeed_config_weak_ref
    config = config_ref() if config_ref is not None else None
    if config is not None:
        transformers_deepspeed.unset_hf_deepspeed_config()
    try:
        yield
    finally:
        if config is not None:
            transformers_deepspeed.set_hf_deepspeed_config(config)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE


def _env_nonnegative_int(name: str, default: int = 0) -> int:
    raw = os.environ.get(name)
    value = default if raw is None or not raw.strip() else int(raw)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _env_optional_nonnegative_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _env_nonnegative_int_tuple(name: str) -> tuple[int, ...]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return ()
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if any(value < 0 for value in values):
        raise ValueError(f"{name} entries must be non-negative")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} entries must be unique")
    return values


def _env_name_tuple(
    name: str,
    *,
    default: tuple[str, ...] = (),
) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if len(set(values)) != len(values):
        raise ValueError(f"{name} entries must be unique")
    return values


def swap_wave_ranges(local_world_size: int, wave_size: int) -> tuple[tuple[int, int], ...]:
    """Partition local ranks into bounded contiguous residency waves."""
    if local_world_size <= 0:
        raise ValueError("local_world_size must be positive")
    if wave_size <= 0:
        raise ValueError("wave_size must be positive")
    return tuple(
        (start, min(start + wave_size, local_world_size))
        for start in range(0, local_world_size, wave_size)
    )


def construct_in_local_rank_waves(
    factory: Callable[[], _T],
    *,
    local_rank: int,
    local_world_size: int,
    wave_size: int,
    barrier: Callable[[], None],
    on_wave_start: Callable[[int, int, int], None] | None = None,
    on_wave_end: Callable[[int, int, int], None] | None = None,
) -> _T:
    """Run one rank-local constructor while bounding node concurrency.

    Every rank traverses the same wave slots and calls ``barrier`` before and
    after each slot. Only ranks inside the current node-local slot invoke the
    factory. In a two-node launch with the same ``LOCAL_WORLD_SIZE``, matching
    local-rank slots may construct concurrently on the independent hosts.

    This is a static Track-B admission mechanism. It does not shard or alter
    the constructed model and is intentionally separate from model-swap waves.
    """

    if local_world_size <= 0:
        raise ValueError("local_world_size must be positive")
    if not 0 <= local_rank < local_world_size:
        raise ValueError("local_rank must be in [0, local_world_size)")
    if wave_size <= 0:
        return factory()

    result: _T | None = None
    constructed = False
    for wave_index, (start, end) in enumerate(
        swap_wave_ranges(local_world_size, wave_size)
    ):
        barrier()
        active = start <= local_rank < end
        if active:
            if on_wave_start is not None:
                on_wave_start(wave_index, start, end)
            try:
                result = factory()
                constructed = True
            finally:
                if on_wave_end is not None:
                    on_wave_end(wave_index, start, end)
        barrier()
    if not constructed:
        raise RuntimeError("local rank was not admitted by any construction wave")
    return result  # type: ignore[return-value]


def plan_swap_wave_size(
    *,
    local_world_size: int,
    memory_limit_bytes: int,
    steady_state_bytes: int,
    transient_per_rank_bytes: int,
    reserve_bytes: int,
) -> int:
    """Choose the largest local rank wave admitted by a measured host budget.

    ``steady_state_bytes`` and ``transient_per_rank_bytes`` come from a prior
    model-move trace.  ``reserve_bytes`` protects uninstrumented allocator and
    filesystem-cache growth.  A plan that cannot admit even one rank is
    rejected rather than silently producing an unsafe schedule.
    """
    values = {
        "local_world_size": local_world_size,
        "memory_limit_bytes": memory_limit_bytes,
        "steady_state_bytes": steady_state_bytes,
        "transient_per_rank_bytes": transient_per_rank_bytes,
        "reserve_bytes": reserve_bytes,
    }
    if local_world_size <= 0:
        raise ValueError("local_world_size must be positive")
    for name in ("memory_limit_bytes", "transient_per_rank_bytes"):
        if values[name] <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("steady_state_bytes", "reserve_bytes"):
        if values[name] < 0:
            raise ValueError(f"{name} must be non-negative")
    available = memory_limit_bytes - steady_state_bytes - reserve_bytes
    admitted = available // transient_per_rank_bytes
    if admitted < 1:
        raise MemoryError(
            "host budget cannot admit one model-swap rank: "
            f"available={available}, transient_per_rank={transient_per_rank_bytes}"
        )
    return min(local_world_size, int(admitted))


def stable_block_noise(
    block_number: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample the exact PSD block covariance without a fragile Cholesky.

    The official covariance for gamma=1/3 is singular: diagonal 1 and
    off-diagonal -1/3. Projecting four iid normals onto the zero-sum subspace
    and scaling by sqrt(4/3) has exactly that covariance.
    """

    samples = torch.randn(block_number, 4, device=device, dtype=dtype)
    samples = samples - samples.mean(dim=-1, keepdim=True)
    return samples * math.sqrt(4.0 / 3.0)


@dataclass(frozen=True)
class OfficialBaselineOptions:
    enabled: bool = False
    direct_bf16_load: bool = False
    precast_generator_path: Path | None = None
    precast_score_path: Path | None = None
    allow_precast_fresh_init: bool = False
    staged_zero3_prepare: bool = False
    release_preprocessors: bool = False
    sparse_ema: bool = False
    skip_final_save: bool = False
    skip_validation: bool = False
    skip_reporting: bool = False
    stop_after_prepare: bool = False
    stable_block_noise: bool = False
    split_modulation: bool = False
    b0_compute_path: bool = False
    full_activation_offload: bool = False
    activation_offload_min_bytes: int = 0
    activation_offload_components: tuple[str, ...] = ()
    immutable_model_backing_components: tuple[str, ...] = ()
    immutable_model_mmap_cache: Path | None = None
    frozen_block_streaming_window: int = 0
    frozen_block_streaming_block_count: int = 0
    frozen_block_streaming_indices: tuple[int, ...] = ()
    frozen_block_streaming_score_indices: tuple[int, ...] = ()
    frozen_block_streaming_components: tuple[str, ...] = ()
    frozen_block_host_policy: str = "HOST_PRIVATE"
    frozen_block_mmap_cache: Path | None = None
    frozen_block_pin_host_state: bool = False
    frozen_block_pin_host_component: str = "all"
    frozen_block_shared_generator_indices: tuple[int, ...] = ()
    frozen_block_shared_score_indices: tuple[int, ...] = ()
    frozen_block_memory_snapshots: bool = False
    phase_resident_critic_generator: bool = False
    critic_phase_offload_generator_lp_state: bool = False
    hybrid_residency: bool = False
    expert_static_plan: bool = False
    detailed_hotpath_trace: bool = True
    swap_generator_before_score_prepare: bool = False
    trim_host_after_model_move: bool = False
    swap_wave_size: int = 0
    construction_wave_size: int = 0
    semantic_assertions: bool = False
    logical_start_step: int = 0
    synthetic_ema_optimization_step: int | None = None
    trace_dir: Path | None = None
    operation_trace_dir: Path | None = None

    @classmethod
    def from_env(cls) -> "OfficialBaselineOptions":
        enabled = _env_flag("DISTILLSCHEDULE_OFFICIAL_CPU_OFFLOAD")
        trace = os.environ.get("DISTILLSCHEDULE_PHASE_TRACE_DIR")
        operation_trace = os.environ.get("DISTILLSCHEDULE_OPERATION_TRACE_DIR")
        full_activation_offload = enabled and _env_flag(
            "DISTILLSCHEDULE_FULL_ACTIVATION_OFFLOAD"
        )
        frozen_block_streaming_window = (
            _env_nonnegative_int("DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_WINDOW")
            if enabled
            else 0
        )
        return cls(
            enabled=enabled,
            direct_bf16_load=enabled and _env_flag("DISTILLSCHEDULE_DIRECT_BF16_LOAD"),
            precast_generator_path=(
                Path(os.environ["DISTILLSCHEDULE_PRECAST_GENERATOR_PATH"])
                if enabled
                and os.environ.get(
                    "DISTILLSCHEDULE_PRECAST_GENERATOR_PATH", ""
                ).strip()
                else None
            ),
            precast_score_path=(
                Path(os.environ["DISTILLSCHEDULE_PRECAST_SCORE_PATH"])
                if enabled
                and os.environ.get("DISTILLSCHEDULE_PRECAST_SCORE_PATH", "").strip()
                else None
            ),
            allow_precast_fresh_init=enabled
            and _env_flag("DISTILLSCHEDULE_ALLOW_PRECAST_FRESH_INIT"),
            staged_zero3_prepare=enabled and _env_flag("DISTILLSCHEDULE_STAGED_ZERO3_PREPARE"),
            release_preprocessors=enabled and _env_flag("DISTILLSCHEDULE_RELEASE_PREPROCESSORS"),
            sparse_ema=enabled and _env_flag("DISTILLSCHEDULE_SPARSE_EMA"),
            skip_final_save=enabled and _env_flag("DISTILLSCHEDULE_SKIP_FINAL_SAVE"),
            skip_validation=enabled and _env_flag("DISTILLSCHEDULE_SKIP_VALIDATION"),
            skip_reporting=enabled and _env_flag("DISTILLSCHEDULE_SKIP_REPORTING"),
            stop_after_prepare=enabled and _env_flag("DISTILLSCHEDULE_STOP_AFTER_PREPARE"),
            stable_block_noise=enabled and _env_flag("DISTILLSCHEDULE_STABLE_BLOCK_NOISE"),
            split_modulation=enabled and _env_flag("DISTILLSCHEDULE_SPLIT_MODULATION"),
            b0_compute_path=enabled
            and _env_flag("DISTILLSCHEDULE_B0_COMPUTE_PATH"),
            full_activation_offload=full_activation_offload,
            activation_offload_min_bytes=(
                _env_nonnegative_int("DISTILLSCHEDULE_ACTIVATION_OFFLOAD_MIN_BYTES")
                if enabled
                else 0
            ),
            activation_offload_components=(
                _env_name_tuple(
                    "DISTILLSCHEDULE_ACTIVATION_OFFLOAD_COMPONENTS",
                    default=("generator_selected", "critic_train"),
                )
                if full_activation_offload
                else ()
            ),
            immutable_model_backing_components=(
                _env_name_tuple(
                    "DISTILLSCHEDULE_IMMUTABLE_MODEL_BACKING_COMPONENTS"
                )
                if enabled
                else ()
            ),
            immutable_model_mmap_cache=(
                Path(os.environ["DISTILLSCHEDULE_IMMUTABLE_MODEL_MMAP_CACHE"])
                if enabled
                and os.environ.get(
                    "DISTILLSCHEDULE_IMMUTABLE_MODEL_MMAP_CACHE", ""
                ).strip()
                else None
            ),
            frozen_block_streaming_window=frozen_block_streaming_window,
            frozen_block_streaming_block_count=(
                _env_nonnegative_int(
                    "DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_BLOCK_COUNT"
                )
                if enabled
                else 0
            ),
            frozen_block_streaming_indices=(
                _env_nonnegative_int_tuple(
                    "DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_INDICES"
                )
                if enabled
                else ()
            ),
            frozen_block_streaming_score_indices=(
                _env_nonnegative_int_tuple(
                    "DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_SCORE_INDICES"
                )
                if enabled
                else ()
            ),
            frozen_block_streaming_components=(
                _env_name_tuple(
                    "DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_COMPONENTS",
                    default=("generator", "score"),
                )
                if frozen_block_streaming_window
                else ()
            ),
            frozen_block_host_policy=(
                os.environ.get(
                    "DISTILLSCHEDULE_FROZEN_BLOCK_HOST_POLICY", "HOST_PRIVATE"
                ).strip()
                or "HOST_PRIVATE"
            ),
            frozen_block_mmap_cache=(
                Path(os.environ["DISTILLSCHEDULE_FROZEN_BLOCK_MMAP_CACHE"])
                if enabled
                and os.environ.get("DISTILLSCHEDULE_FROZEN_BLOCK_MMAP_CACHE", "").strip()
                else None
            ),
            frozen_block_pin_host_state=enabled
            and _env_flag("DISTILLSCHEDULE_FROZEN_BLOCK_PIN_HOST_STATE"),
            frozen_block_pin_host_component=(
                os.environ.get(
                    "DISTILLSCHEDULE_FROZEN_BLOCK_PIN_HOST_COMPONENT", "all"
                ).strip()
                or "all"
            ),
            frozen_block_shared_generator_indices=(
                _env_nonnegative_int_tuple(
                    "DISTILLSCHEDULE_FROZEN_BLOCK_SHARED_GENERATOR_INDICES"
                )
                if enabled
                else ()
            ),
            frozen_block_shared_score_indices=(
                _env_nonnegative_int_tuple(
                    "DISTILLSCHEDULE_FROZEN_BLOCK_SHARED_SCORE_INDICES"
                )
                if enabled
                else ()
            ),
            frozen_block_memory_snapshots=enabled
            and _env_flag("DISTILLSCHEDULE_BLOCK_STREAM_MEMORY_SNAPSHOT"),
            phase_resident_critic_generator=enabled
            and _env_flag("DISTILLSCHEDULE_PHASE_RESIDENT_CRITIC_GENERATOR"),
            critic_phase_offload_generator_lp_state=enabled
            and _env_flag(
                "DISTILLSCHEDULE_CRITIC_PHASE_OFFLOAD_GENERATOR_LP_STATE"
            ),
            hybrid_residency=enabled
            and _env_flag("DISTILLSCHEDULE_HYBRID_RESIDENCY"),
            expert_static_plan=enabled
            and _env_flag("DISTILLSCHEDULE_EXPERT_STATIC_PLAN"),
            detailed_hotpath_trace=not _env_flag(
                "DISTILLSCHEDULE_DISABLE_HOTPATH_TRACE"
            ),
            swap_generator_before_score_prepare=enabled
            and _env_flag("DISTILLSCHEDULE_SWAP_GENERATOR_BEFORE_SCORE_PREPARE"),
            trim_host_after_model_move=enabled
            and _env_flag("DISTILLSCHEDULE_TRIM_HOST_AFTER_MODEL_MOVE"),
            swap_wave_size=(
                _env_nonnegative_int("DISTILLSCHEDULE_SWAP_WAVE_SIZE")
                if enabled
                else 0
            ),
            construction_wave_size=(
                _env_nonnegative_int("DISTILLSCHEDULE_CONSTRUCTION_WAVE_SIZE")
                if enabled
                else 0
            ),
            semantic_assertions=enabled
            and _env_flag("DISTILLSCHEDULE_M0_SEMANTIC_ASSERTIONS"),
            logical_start_step=(
                _env_nonnegative_int("DISTILLSCHEDULE_LOGICAL_START_STEP")
                if enabled
                else 0
            ),
            synthetic_ema_optimization_step=(
                _env_optional_nonnegative_int(
                    "DISTILLSCHEDULE_SYNTHETIC_EMA_OPTIMIZATION_STEP"
                )
                if enabled
                else None
            ),
            trace_dir=Path(trace) if trace else None,
            operation_trace_dir=Path(operation_trace) if operation_trace else None,
        )

    def validate(self) -> None:
        if not self.enabled:
            active = [
                self.direct_bf16_load,
                self.precast_generator_path is not None,
                self.precast_score_path is not None,
                self.allow_precast_fresh_init,
                self.staged_zero3_prepare,
                self.release_preprocessors,
                self.sparse_ema,
                self.skip_final_save,
                self.skip_validation,
                self.skip_reporting,
                self.stop_after_prepare,
                self.stable_block_noise,
                self.split_modulation,
                self.b0_compute_path,
                self.full_activation_offload,
                bool(self.activation_offload_min_bytes),
                bool(self.activation_offload_components),
                bool(self.immutable_model_backing_components),
                self.immutable_model_mmap_cache is not None,
                bool(self.frozen_block_streaming_window),
                bool(self.frozen_block_streaming_block_count),
                bool(self.frozen_block_streaming_indices),
                bool(self.frozen_block_streaming_score_indices),
                bool(self.frozen_block_streaming_components),
                self.frozen_block_pin_host_state,
                bool(self.frozen_block_shared_generator_indices),
                bool(self.frozen_block_shared_score_indices),
                self.frozen_block_memory_snapshots,
                self.phase_resident_critic_generator,
                self.critic_phase_offload_generator_lp_state,
                self.hybrid_residency,
                self.expert_static_plan,
                self.swap_generator_before_score_prepare,
                self.trim_host_after_model_move,
                bool(self.swap_wave_size),
                bool(self.construction_wave_size),
                self.semantic_assertions,
                bool(self.logical_start_step),
                self.synthetic_ema_optimization_step is not None,
            ]
            if any(active):
                raise ValueError("official baseline sub-options require the master enable flag")
        precast_paths = [
            path
            for path in (self.precast_generator_path, self.precast_score_path)
            if path is not None
        ]
        if precast_paths and not self.direct_bf16_load:
            raise ValueError("pre-cast checkpoint loading requires direct BF16 loading")
        if any(not path.is_absolute() for path in precast_paths):
            raise ValueError("pre-cast checkpoint paths must be absolute")
        if self.allow_precast_fresh_init and not precast_paths:
            raise ValueError(
                "allowing pre-cast fresh initialization requires a pre-cast checkpoint"
            )
        if self.sparse_ema and not self.staged_zero3_prepare:
            raise ValueError("sparse EMA must be constructed after staged ZeRO-3 prepare")
        if self.construction_wave_size not in range(0, 9):
            raise ValueError("construction wave size must be in [0,8]")
        if self.logical_start_step and self.operation_trace_dir is None:
            raise ValueError("logical start step requires an operation-keyed trace")
        if (
            self.synthetic_ema_optimization_step is not None
            and self.synthetic_ema_optimization_step != self.logical_start_step
        ):
            raise ValueError(
                "synthetic EMA optimization step must equal the logical start step"
            )
        if self.synthetic_ema_optimization_step is not None and not self.semantic_assertions:
            raise ValueError("synthetic EMA boundary state requires semantic assertions")
        if self.b0_compute_path and not self.split_modulation:
            raise ValueError("B0 compute path requires split modulation")
        if self.full_activation_offload and not self.split_modulation:
            raise ValueError("B2 activation offload requires the preceding B1 split-modulation baseline")
        if self.activation_offload_min_bytes and not self.full_activation_offload:
            raise ValueError(
                "activation offload threshold requires activation offload to be enabled"
            )
        valid_activation_components = {"generator_selected", "critic_train"}
        if self.activation_offload_components and not self.full_activation_offload:
            raise ValueError(
                "activation offload components require activation offload"
            )
        if self.full_activation_offload and not self.activation_offload_components:
            raise ValueError("activation offload requires at least one component")
        if not set(self.activation_offload_components).issubset(
            valid_activation_components
        ):
            raise ValueError(
                "invalid activation offload component; expected a subset of "
                f"{sorted(valid_activation_components)}"
            )
        valid_backing_components = {"generator", "score"}
        if not set(self.immutable_model_backing_components).issubset(
            valid_backing_components
        ):
            raise ValueError(
                "invalid immutable model backing component; expected a subset of "
                f"{sorted(valid_backing_components)}"
            )
        if (
            self.immutable_model_backing_components
            and self.immutable_model_mmap_cache is None
        ):
            raise ValueError(
                "immutable model backing requires "
                "DISTILLSCHEDULE_IMMUTABLE_MODEL_MMAP_CACHE"
            )
        if (
            not self.immutable_model_backing_components
            and self.immutable_model_mmap_cache is not None
        ):
            raise ValueError(
                "immutable model mmap cache requires at least one backing component"
            )
        if (
            self.immutable_model_mmap_cache is not None
            and not self.immutable_model_mmap_cache.is_absolute()
        ):
            raise ValueError("immutable model mmap cache must be an absolute path")
        if self.immutable_model_backing_components and self.trace_dir is None:
            raise ValueError("immutable model backing requires a phase trace directory")
        if (
            self.immutable_model_backing_components
            and self.frozen_block_streaming_window
            and not self.hybrid_residency
        ):
            raise ValueError(
                "whole-model immutable backing cannot be combined with "
                "fine-grained frozen-block streaming without the explicit "
                "hybrid residency contract"
            )
        if self.frozen_block_streaming_window not in (0, 1, 2, 4):
            raise ValueError("B3 frozen-block streaming window must be one of 0, 1, 2, or 4")
        if self.frozen_block_streaming_block_count and not self.frozen_block_streaming_window:
            raise ValueError(
                "partial frozen-block streaming requires a non-zero streaming window"
            )
        if self.frozen_block_streaming_indices and not self.frozen_block_streaming_window:
            raise ValueError(
                "explicit frozen-block streaming indices require a non-zero streaming window"
            )
        if self.frozen_block_streaming_indices and (
            self.frozen_block_streaming_block_count
            != len(self.frozen_block_streaming_indices)
        ):
            raise ValueError(
                "streamed block count must match the explicit streamed block indices"
            )
        if (
            self.frozen_block_streaming_score_indices
            and not self.frozen_block_streaming_window
        ):
            raise ValueError(
                "Score-specific frozen-block streaming indices require a non-zero "
                "streaming window"
            )
        if self.frozen_block_memory_snapshots and not self.frozen_block_streaming_window:
            raise ValueError(
                "block-stream memory snapshots require frozen-block streaming"
            )
        if (
            self.phase_resident_critic_generator
            and (
                not self.frozen_block_streaming_window
                or "generator" not in self.frozen_block_streaming_components
            )
        ):
            raise ValueError(
                "phase-resident Critic Generator requires Generator block streaming"
            )
        if (
            self.critic_phase_offload_generator_lp_state
            and not self.semantic_assertions
        ):
            raise ValueError(
                "Critic-phase Generator LP-state offload requires semantic assertions"
            )
        valid_streaming_components = {"generator", "score"}
        if not set(self.frozen_block_streaming_components).issubset(
            valid_streaming_components
        ):
            raise ValueError(
                "invalid frozen-block streaming component; expected a subset of "
                f"{sorted(valid_streaming_components)}"
            )
        if (
            self.frozen_block_streaming_window
            and not self.frozen_block_streaming_components
        ):
            raise ValueError("block streaming requires at least one component")
        if (
            not self.frozen_block_streaming_window
            and self.frozen_block_streaming_components
        ):
            raise ValueError(
                "block streaming components require a non-zero streaming window"
            )
        if (
            self.frozen_block_streaming_score_indices
            and "score" not in self.frozen_block_streaming_components
        ):
            raise ValueError(
                "Score-specific frozen-block streaming indices require the Score "
                "streaming component"
            )
        if self.hybrid_residency:
            if (
                not self.immutable_model_backing_components
                or not self.frozen_block_streaming_window
            ):
                raise ValueError(
                    "hybrid residency requires both immutable model backing and "
                    "partial frozen-block streaming"
                )
            if self.frozen_block_streaming_block_count <= 0:
                raise ValueError(
                    "hybrid residency requires an explicit positive streamed block count"
                )
            if not set(self.frozen_block_streaming_components).issubset(
                set(self.immutable_model_backing_components)
            ):
                raise ValueError(
                    "hybrid streamed components must also use immutable model backing"
                )
            if self.frozen_block_host_policy != "HOST_SHARED_IMMUTABLE":
                raise ValueError(
                    "hybrid residency requires HOST_SHARED_IMMUTABLE for streamed blocks"
                )
        if (
            self.frozen_block_streaming_window
            and not self.full_activation_offload
            and not self.expert_static_plan
        ):
            raise ValueError(
                "streaming without B2 activation offload is reserved for an explicit "
                "B4 expert static plan"
            )
        valid_host_policies = {
            "HOST_PRIVATE",
            "HOST_SHARED_IMMUTABLE",
            "DROP_RELOAD_FROM_MMAP",
        }
        if self.frozen_block_host_policy not in valid_host_policies:
            raise ValueError(
                "invalid frozen-block host policy: "
                f"{self.frozen_block_host_policy}; expected {sorted(valid_host_policies)}"
            )
        if (
            self.frozen_block_streaming_window
            and self.frozen_block_host_policy != "HOST_PRIVATE"
            and self.frozen_block_mmap_cache is None
        ):
            raise ValueError(
                f"{self.frozen_block_host_policy} requires "
                "DISTILLSCHEDULE_FROZEN_BLOCK_MMAP_CACHE"
            )
        if self.frozen_block_pin_host_state and not self.frozen_block_streaming_window:
            raise ValueError("pinned frozen host state requires block streaming")
        if (
            self.frozen_block_pin_host_state
            and self.frozen_block_host_policy != "HOST_PRIVATE"
        ):
            raise ValueError(
                "pinned frozen host state requires HOST_PRIVATE ownership"
            )
        valid_pin_components = {"all", "generator", "score"}
        if self.frozen_block_pin_host_component not in valid_pin_components:
            raise ValueError(
                "invalid pinned frozen host component: "
                f"{self.frozen_block_pin_host_component}; expected "
                f"{sorted(valid_pin_components)}"
            )
        shared_indices = (
            self.frozen_block_shared_generator_indices
            + self.frozen_block_shared_score_indices
        )
        if shared_indices and not self.frozen_block_streaming_window:
            raise ValueError("mixed immutable sharing requires block streaming")
        if shared_indices and self.frozen_block_host_policy != "HOST_PRIVATE":
            raise ValueError(
                "mixed immutable sharing requires HOST_PRIVATE as the base policy"
            )
        if shared_indices and self.frozen_block_mmap_cache is None:
            raise ValueError("mixed immutable sharing requires a frozen-block mmap cache")

    def frozen_block_indices_for(self, logical_component: str) -> tuple[int, ...]:
        """Resolve a component's immutable streamed-block set.

        Existing plans retain the common index set. A Score-specific override
        permits a bounded selective-Score experiment without changing the
        already frozen Generator selection.
        """

        if logical_component == "score" and self.frozen_block_streaming_score_indices:
            return self.frozen_block_streaming_score_indices
        return self.frozen_block_streaming_indices

    def frozen_block_count_for(self, logical_component: str) -> int:
        indices = self.frozen_block_indices_for(logical_component)
        return len(indices) if indices else self.frozen_block_streaming_block_count


def _read_first_int(path: Path) -> int | None:
    try:
        value = path.read_text().strip()
    except OSError:
        return None
    return int(value) if value.isdigit() else None


def _read_memory_events(path: Path = Path("/sys/fs/cgroup/memory.events")) -> dict[str, int]:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return {}
    result: dict[str, int] = {}
    for line in lines:
        fields = line.split()
        if len(fields) == 2 and fields[1].isdigit():
            result[fields[0]] = int(fields[1])
    return result


def current_rss_bytes() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    # Linux reports KiB through ru_maxrss; this is a peak fallback, not current RSS.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def cuda_memory_diagnostics(*, include_allocator_snapshot: bool = False) -> dict[str, Any]:
    """Return scalar CUDA allocator diagnostics without retaining tensors."""
    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return {}
    device = torch.cuda.current_device()
    stats = torch.cuda.memory_stats(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    result: dict[str, Any] = {
        "device": device,
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "active_bytes": int(stats.get("active_bytes.all.current", 0)),
        "inactive_split_bytes": int(stats.get("inactive_split_bytes.all.current", 0)),
        "requested_bytes": int(stats.get("requested_bytes.all.current", 0)),
        "num_ooms": int(stats.get("num_ooms", 0)),
        "num_alloc_retries": int(stats.get("num_alloc_retries", 0)),
        "driver_free_bytes": int(free_bytes),
        "driver_total_bytes": int(total_bytes),
    }
    if include_allocator_snapshot:
        largest_inactive = 0
        segment_count = 0
        try:
            snapshot = torch.cuda.memory._snapshot()
            segments = snapshot.get("segments", []) if isinstance(snapshot, dict) else snapshot
            segment_count = len(segments)
            for segment in segments:
                for block in segment.get("blocks", []):
                    if block.get("state") == "inactive":
                        largest_inactive = max(largest_inactive, int(block.get("size", 0)))
        except BaseException as error:
            result["allocator_snapshot_error"] = f"{type(error).__name__}: {error}"
        result["allocator_segment_count"] = segment_count
        result["largest_allocator_inactive_block_bytes"] = largest_inactive
    return result


def _parse_failed_allocation_bytes(message: str) -> int | None:
    match = re.search(r"(?:Tried to allocate|allocate) ([0-9.]+) (KiB|MiB|GiB)", message)
    if match is None:
        return None
    scale = {"KiB": 2**10, "MiB": 2**20, "GiB": 2**30}[match.group(2)]
    return int(float(match.group(1)) * scale)


def rank_qualified_artifact_path(path: str | os.PathLike[str]) -> Path:
    """Expand optional rank placeholders so distributed failures never race one file."""

    return Path(
        os.fspath(path).format(
            rank=os.environ.get("RANK", "0"),
            local_rank=os.environ.get("LOCAL_RANK", "0"),
        )
    )


def write_oom_report(path: str | os.PathLike[str], error: BaseException) -> None:
    """Persist the failing allocation and a post-failure scalar allocator snapshot."""
    destination = rank_qualified_artifact_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cuda = cuda_memory_diagnostics(include_allocator_snapshot=True)
    failed = _parse_failed_allocation_bytes(str(error))
    free = int(cuda.get("driver_free_bytes", 0))
    required_release = None if failed is None else max(0, failed + 2**30 - free)
    report = {
        "status": "OOM",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "failed_allocation_bytes": failed,
        "release_for_failed_allocation_plus_1gib_safety_bytes": required_release,
        "cuda": cuda,
        "rss_bytes": current_rss_bytes(),
        "cgroup_memory_current_bytes": _read_first_int(Path("/sys/fs/cgroup/memory.current")),
        "cgroup_memory_max_bytes": _read_first_int(Path("/sys/fs/cgroup/memory.max")),
        "cgroup_memory_events": _read_memory_events(),
        "limitations": [
            "largest_allocator_inactive_block is not the same as a guaranteed driver allocation",
            "reserved bytes are not claimed as fully attributed",
            "category-level live tensor attribution is produced by a separate instrumentation run",
        ],
    }
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")


class PhaseTracer:
    """Append rank-local JSONL phase records without retaining tensor references."""

    def __init__(self, output_dir: Path | None):
        self.output_dir = output_dir
        self.rank = int(os.environ.get("RANK", "0"))
        self.timing_only = _env_flag("DISTILLSCHEDULE_TIMING_ONLY_TRACE")
        self.path = output_dir / f"rank_{self.rank:03d}.jsonl" if output_dir else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def record(self, phase: str, **extra: Any) -> None:
        if not self.path:
            return
        record: dict[str, Any] = {
            "phase": phase,
            "time_unix": time.time(),
            "time_monotonic": time.perf_counter(),
            "pid": os.getpid(),
            "rank": self.rank,
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        }
        if not self.timing_only:
            record.update(
                {
                    "rss_bytes": current_rss_bytes(),
                    "rss_peak_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
                    "cgroup_memory_current_bytes": _read_first_int(Path("/sys/fs/cgroup/memory.current")),
                    "cgroup_memory_max_bytes": _read_first_int(Path("/sys/fs/cgroup/memory.max")),
                    "cgroup_memory_events": _read_memory_events(),
                }
            )
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                record["cuda"] = cuda_memory_diagnostics(include_allocator_snapshot=False)
        record.update(extra)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def parameter_summary(module: torch.nn.Module) -> dict[str, Any]:
    logical_numel = 0
    local_numel = 0
    trainable_logical_numel = 0
    logical_bytes_by_dtype: dict[str, int] = {}
    devices: dict[str, int] = {}
    for parameter in module.parameters():
        numel = int(getattr(parameter, "ds_numel", parameter.numel()))
        logical_numel += numel
        if parameter.requires_grad:
            trainable_logical_numel += numel
        dtype = str(parameter.dtype).replace("torch.", "")
        logical_bytes_by_dtype[dtype] = logical_bytes_by_dtype.get(dtype, 0) + numel * parameter.element_size()
        backing = getattr(parameter, "ds_tensor", parameter.data)
        local_numel += backing.numel()
        device = str(backing.device)
        devices[device] = devices.get(device, 0) + backing.numel() * backing.element_size()
    return {
        "logical_numel": logical_numel,
        "local_numel": local_numel,
        "trainable_logical_numel": trainable_logical_numel,
        "logical_bytes_by_dtype": logical_bytes_by_dtype,
        "local_bytes_by_device": devices,
    }


def _backing_tensor(parameter: torch.nn.Parameter) -> torch.Tensor:
    tensor = getattr(parameter, "ds_tensor", None)
    return tensor if tensor is not None else parameter.data


class SparseShardEMA:
    """EMA over trainable local shards; frozen state is reconstructed from the base asset."""

    format_version = 1

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        decay: float,
        update_after_step: int,
        base_model: str,
        initialize_shadows: bool = True,
    ):
        items = list(named_parameters)
        self.parameter_count = len(items)
        self.names: list[str] = []
        self.indices: list[int] = []
        self.shadows: list[torch.Tensor] = []
        self.decay = float(decay)
        self.min_decay = 0.0
        self.update_after_step = int(update_after_step)
        self.optimization_step = 0
        self.cur_decay_value: float | None = None
        self.base_model = base_model
        for index, (name, parameter) in enumerate(items):
            if not parameter.requires_grad:
                continue
            source = _backing_tensor(parameter).detach()
            self.names.append(name)
            self.indices.append(index)
            self.shadows.append(
                source.to(device="cpu", copy=True).contiguous()
                if initialize_shadows
                else torch.empty(0, dtype=source.dtype)
            )
        if not self.names:
            raise ValueError("sparse EMA found no trainable parameters")

    @classmethod
    def from_module(
        cls,
        module: torch.nn.Module,
        *,
        decay: float,
        update_after_step: int,
        base_model: str,
    ) -> "SparseShardEMA":
        return cls(
            module.named_parameters(),
            decay=decay,
            update_after_step=update_after_step,
            base_model=base_model,
        )

    def get_decay(self, optimization_step: int) -> float:
        step = max(0, optimization_step - self.update_after_step - 1)
        if step <= 0:
            return 0.0
        value = (1 + step) / (10 + step)
        return max(self.min_decay, min(value, self.decay))

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        values = list(parameters)
        if len(values) != self.parameter_count:
            raise ValueError(
                f"parameter count changed: initialized with {self.parameter_count}, got {len(values)}"
            )
        self.optimization_step += 1
        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        one_minus_decay = 1.0 - decay
        for shadow, index in zip(self.shadows, self.indices):
            source = _backing_tensor(values[index]).detach()
            if source.shape != shadow.shape:
                raise ValueError(f"EMA shard shape changed at index {index}: {source.shape} != {shadow.shape}")
            source_cpu = source.to(device="cpu", dtype=shadow.dtype)
            # The bootstrap EMA equation has decay == 0 and is semantically an
            # exact assignment.  Evaluating ``shadow - (shadow - source)`` in
            # BF16 can round twice and fail that contract even within one
            # layout, so preserve the equation's exact special case.
            if decay == 0.0:
                shadow.copy_(source_cpu)
            else:
                shadow.sub_(one_minus_decay * (shadow - source_cpu))

    def checksum(self) -> dict[str, Any]:
        total_sum = 0.0
        total_sq = 0.0
        numel = 0
        for tensor in self.shadows:
            value = tensor.double()
            total_sum += value.sum().item()
            total_sq += value.square().sum().item()
            numel += tensor.numel()
        return {
            "num_tensors": len(self.shadows),
            "numel": numel,
            "sum": total_sum,
            "norm": math.sqrt(total_sq),
            "optimization_step": self.optimization_step,
            "decay": self.cur_decay_value,
        }

    @torch.no_grad()
    def compare_to_parameters(self, parameters: Iterable[torch.nn.Parameter]) -> dict[str, Any]:
        """Compare shadows to current trainable shards without retaining copies."""

        values = list(parameters)
        if len(values) != self.parameter_count:
            raise ValueError(
                f"parameter count changed: initialized with {self.parameter_count}, got {len(values)}"
            )
        mismatched_tensors = 0
        mismatched_numel = 0
        for shadow, index in zip(self.shadows, self.indices):
            source = _backing_tensor(values[index]).detach().to(
                device="cpu", dtype=shadow.dtype
            )
            if not torch.equal(shadow, source):
                mismatched_tensors += 1
                mismatched_numel += source.numel()
        return {
            "matches_exact": mismatched_tensors == 0,
            "mismatched_tensors": mismatched_tensors,
            "mismatched_numel": mismatched_numel,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "base_model": self.base_model,
            "frozen_state_policy": "reconstruct_from_base_model",
            "parameter_count": self.parameter_count,
            "tracked_names": self.names,
            "tracked_indices": self.indices,
            "decay": self.decay,
            "update_after_step": self.update_after_step,
            "optimization_step": self.optimization_step,
        }

    def state_dict(self) -> dict[str, Any]:
        return {"metadata": self.metadata(), "shadows": [tensor.clone() for tensor in self.shadows]}

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def save_pretrained(
        self,
        _args: Any,
        output_dir: str | os.PathLike[str],
        _base_model: str | None = None,
        *,
        max_shard_bytes: int = 1024**3,
        **_unused: Any,
    ) -> None:
        """Persist sparse CPU EMA without materializing the frozen backbone.

        Helios' normal EMA checkpoint reconstructs and serializes a complete
        model.  That defeats the A100 substrate's sparse-EMA contract.  The
        M0 checkpoint instead writes bounded CPU tensor shards plus an
        immutable-base identity.  No shadow tensor is cloned by this method.
        """

        if max_shard_bytes <= 0:
            raise ValueError("max_shard_bytes must be positive")
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        shard_specs: list[dict[str, Any]] = []
        current: dict[str, torch.Tensor] = {}
        current_bytes = 0

        def flush() -> None:
            nonlocal current, current_bytes
            if not current:
                return
            index = len(shard_specs)
            filename = f"sparse_ema-{index:05d}.pt"
            destination = root / filename
            temporary = destination.with_suffix(".pt.tmp")
            torch.save(current, temporary)
            os.replace(temporary, destination)
            shard_specs.append(
                {
                    "file": filename,
                    "sha256": self._file_sha256(destination),
                    "storage_bytes": current_bytes,
                    "names": list(current),
                }
            )
            current = {}
            current_bytes = 0

        for name, shadow in zip(self.names, self.shadows):
            if shadow.device.type != "cpu" or not shadow.is_contiguous():
                raise ValueError(f"EMA shadow {name!r} is not contiguous CPU storage")
            storage_bytes = shadow.numel() * shadow.element_size()
            if current and current_bytes + storage_bytes > max_shard_bytes:
                flush()
            current[name] = shadow
            current_bytes += storage_bytes
        flush()

        metadata = self.metadata()
        metadata.update(
            {
                "schema_version": "distillschedule.sparse_ema.v1",
                "cur_decay_value": self.cur_decay_value,
                "shards": shard_specs,
            }
        )
        destination = root / "sparse_ema_manifest.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, destination)

    @classmethod
    def load_pretrained(
        cls,
        module: torch.nn.Module,
        input_dir: str | os.PathLike[str],
    ) -> "SparseShardEMA":
        root = Path(input_dir)
        metadata = json.loads((root / "sparse_ema_manifest.json").read_text())
        if metadata.get("schema_version") != "distillschedule.sparse_ema.v1":
            raise ValueError(f"unsupported sparse EMA checkpoint: {metadata.get('schema_version')!r}")
        ema = cls(
            module.named_parameters(),
            decay=float(metadata["decay"]),
            update_after_step=int(metadata["update_after_step"]),
            base_model=str(metadata["base_model"]),
            initialize_shadows=False,
        )
        if ema.names != metadata["tracked_names"] or ema.indices != metadata["tracked_indices"]:
            raise ValueError("sparse EMA tracked-parameter identity changed")
        if ema.parameter_count != int(metadata["parameter_count"]):
            raise ValueError("sparse EMA model parameter count changed")

        loaded: dict[str, torch.Tensor] = {}
        for shard in metadata["shards"]:
            path = root / shard["file"]
            if cls._file_sha256(path) != shard["sha256"]:
                raise ValueError(f"sparse EMA shard checksum mismatch: {path}")
            try:
                state = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(path, map_location="cpu")
            if list(state) != shard["names"]:
                raise ValueError(f"sparse EMA shard name order changed: {path}")
            overlap = loaded.keys() & state.keys()
            if overlap:
                raise ValueError(f"duplicate sparse EMA tensors: {sorted(overlap)[:3]}")
            loaded.update(state)

        if set(loaded) != set(ema.names):
            missing = sorted(set(ema.names) - set(loaded))
            unexpected = sorted(set(loaded) - set(ema.names))
            raise ValueError(
                f"sparse EMA tensor set changed; missing={missing[:3]}, unexpected={unexpected[:3]}"
            )
        parameters = list(module.parameters())
        ema.shadows = []
        for name, parameter_index in zip(ema.names, ema.indices):
            tensor = loaded[name]
            source = _backing_tensor(parameters[parameter_index])
            if tensor.shape != source.shape or tensor.dtype != source.dtype:
                raise ValueError(
                    f"sparse EMA tensor metadata changed for {name}: "
                    f"{tuple(tensor.shape)}/{tensor.dtype} != {tuple(source.shape)}/{source.dtype}"
                )
            ema.shadows.append(tensor.contiguous())
        ema.optimization_step = int(metadata["optimization_step"])
        ema.cur_decay_value = metadata.get("cur_decay_value")
        return ema


@torch.no_grad()
def trainable_checksum(module: torch.nn.Module) -> dict[str, Any]:
    total_sum = 0.0
    total_sq = 0.0
    numel = 0
    tensors = 0
    for parameter in module.parameters():
        if not parameter.requires_grad:
            continue
        value = _backing_tensor(parameter).detach().float()
        total_sum += value.sum().item()
        total_sq += value.square().sum().item()
        numel += value.numel()
        tensors += 1
    return {"num_tensors": tensors, "numel": numel, "sum": total_sum, "norm": math.sqrt(total_sq)}


@torch.no_grad()
def trainable_sha256(module: torch.nn.Module) -> str:
    """Hash trainable parameter names, metadata, and bytes one tensor at a time."""

    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        value = _backing_tensor(parameter).detach().to(device="cpu").contiguous()
        metadata = {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "stride": list(value.stride()),
        }
        digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def parameter_gradient_diagnostics(module: torch.nn.Module) -> dict[str, int]:
    """Count gradient ownership without retaining or materializing gradients."""

    result = {
        "trainable_parameters": 0,
        "trainable_grad_present": 0,
        "frozen_parameters": 0,
        "frozen_grad_present": 0,
    }
    for parameter in module.parameters():
        prefix = "trainable" if parameter.requires_grad else "frozen"
        result[f"{prefix}_parameters"] += 1
        if parameter.grad is not None:
            result[f"{prefix}_grad_present"] += 1
    return result


class TrainableGradientObserver:
    """Observe autograd delivery without retaining gradient tensors.

    DeepSpeed performs optimizer step and gradient clearing inside
    ``Accelerator.backward``. Parameter hooks therefore provide the only cheap,
    lifetime-neutral evidence that trainable gradients reached the engine.
    """

    def __init__(self, module: torch.nn.Module):
        self._active = False
        self._hook_calls = 0
        self._gradient_numel = 0
        self._parameter_names: set[str] = set()
        self._handles = []
        self.trainable_parameter_count = 0
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            self.trainable_parameter_count += 1

            def observe(gradient: torch.Tensor, *, parameter_name: str = name) -> torch.Tensor:
                if self._active:
                    self._hook_calls += 1
                    self._gradient_numel += gradient.numel()
                    self._parameter_names.add(parameter_name)
                return gradient

            self._handles.append(parameter.register_hook(observe))
        if not self._handles:
            raise ValueError("gradient observer found no trainable parameters")

    def begin(self) -> None:
        if self._active:
            raise RuntimeError("gradient observer scope is already active")
        self._hook_calls = 0
        self._gradient_numel = 0
        self._parameter_names.clear()
        self._active = True

    def end(self) -> dict[str, int]:
        if not self._active:
            raise RuntimeError("gradient observer scope is not active")
        self._active = False
        return {
            "trainable_parameter_count": self.trainable_parameter_count,
            "parameters_with_gradient": len(self._parameter_names),
            "hook_calls": self._hook_calls,
            "gradient_numel": self._gradient_numel,
        }

    def close(self) -> None:
        self._active = False
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def should_restore_dataloader_state(checkpoint_step: int, updates_per_epoch: int) -> bool:
    """Restore a mid-epoch cursor, but reset a checkpoint taken at an epoch boundary."""

    if checkpoint_step < 0:
        raise ValueError("checkpoint_step must be non-negative")
    if updates_per_epoch <= 0:
        raise ValueError("updates_per_epoch must be positive")
    return checkpoint_step % updates_per_epoch != 0


def optimizer_state_checksum(optimizer: Any) -> dict[str, Any]:
    """Summarize Adam moments after a checkpoint load without retaining tensors."""

    current = optimizer
    seen: set[int] = set()
    while not hasattr(current, "state") and hasattr(current, "optimizer"):
        if id(current) in seen:
            raise ValueError("optimizer wrapper cycle")
        seen.add(id(current))
        current = current.optimizer
    state = getattr(current, "state", {})
    result: dict[str, Any] = {
        "optimizer_type": type(current).__name__,
        "state_count": len(state),
        "step_min": None,
        "step_max": None,
    }
    steps: list[int] = []
    for value in state.values():
        step = value.get("step") if isinstance(value, dict) else None
        if isinstance(step, torch.Tensor):
            step = int(step.item())
        if isinstance(step, (int, float)):
            steps.append(int(step))
    if steps:
        result["step_min"] = min(steps)
        result["step_max"] = max(steps)
    for key in ("exp_avg", "exp_avg_sq"):
        total_sum = 0.0
        total_sq = 0.0
        numel = 0
        tensors = 0
        for value in state.values():
            tensor = value.get(key) if isinstance(value, dict) else None
            if not isinstance(tensor, torch.Tensor):
                continue
            item = tensor.detach().float()
            total_sum += item.sum().item()
            total_sq += item.square().sum().item()
            numel += item.numel()
            tensors += 1
        result[key] = {
            "num_tensors": tensors,
            "numel": numel,
            "sum": total_sum,
            "norm": math.sqrt(total_sq),
        }
    return result

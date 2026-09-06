"""Grid-quantized sigma + mmap-backed AdaLN lookup for MiniMax-H3.

This branch deliberately changes the DMD renoise-time distribution from a
continuous uniform base sigma to a uniform 1000-point grid. All consumers use
the same quantized base sigma: noise injection, video/audio flow shifts,
Fake/Teacher score conditioning and AdaLN lookup. There is therefore no
conditioning mismatch; the approximation is only the discrete quadrature over
sigma.

The cold-start builder writes all AdaLN modulation values to one BF16 mmap.
At training startup the original per-block AdaLN modules are replaced by
parameter-free lookup handles before FSDP2 wrapping, so the ~13B frozen AdaLN
parameters are absent from the training model and FSDP units.
"""

from __future__ import annotations

import contextlib
import contextvars
import gc
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
from loguru import logger
from torch import nn

from .adaln_cache import CacheStats
from .fa3_replay_cache import (
    install_fa3_replay_cache,
    parse_block_indices,
    parse_max_d2h_inflight,
    parse_storage,
)

SCHEMA_VERSION = "h3_a100.adaln_grid.v1"
DEFAULT_GRID_SIZE = 1000
GRID_CONTROLLER_ATTR = "_h3_a100_grid_adaln_controller"


def _f32_bits(values: torch.Tensor) -> tuple[int, ...]:
    tensor = values.detach().to(device="cpu", dtype=torch.float32).contiguous().reshape(-1)
    return tuple(int(value) for value in tensor.view(torch.int32).tolist())


@dataclass(frozen=True)
class _Scope:
    key: Any | None
    persistent: bool


class GridAdaLNController:
    """Stage one precomputed 50-block modulation entry at a time from mmap."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        max_dynamic_keys: int = 2,
        pin_memory: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(
                f"unsupported AdaLN grid schema: {payload.get('schema_version')!r}"
            )
        self.payload = payload
        self.grid_size = int(payload["grid_size"])
        self.num_rollout_entries = int(payload["num_rollout_entries"])
        self.num_entries = int(payload["num_entries"])
        self.num_blocks = int(payload["num_blocks"])
        self.hidden_size = int(payload["hidden_size"])
        self.rows_per_entry = int(payload["rows_per_entry"])
        self.modulation_chunks = int(payload["modulation_chunks"])
        self.pin_memory = bool(pin_memory)
        self.max_dynamic_keys = int(max_dynamic_keys)
        if self.grid_size <= 0 or self.num_blocks != 50:
            raise RuntimeError("invalid H3 AdaLN grid manifest dimensions")
        if self.rows_per_entry != 6 or self.modulation_chunks != 6:
            raise RuntimeError(
                "H3 joint AV table must contain 2 unique timesteps x 3 modalities and 6 modulation chunks"
            )

        binary = self.manifest_path.parent / payload["binary_file"]
        expected_numel = (
            self.num_entries
            * self.num_blocks
            * self.modulation_chunks
            * self.rows_per_entry
            * self.hidden_size
        )
        expected_bytes = expected_numel * torch.empty((), dtype=torch.bfloat16).element_size()
        if not binary.is_file() or binary.stat().st_size != expected_bytes:
            raise RuntimeError(
                f"AdaLN grid binary size mismatch path={binary} "
                f"actual={binary.stat().st_size if binary.exists() else None} expected={expected_bytes}"
            )
        self.binary_path = binary
        self._table = torch.from_file(
            str(binary), shared=False, size=expected_numel, dtype=torch.bfloat16
        ).reshape(
            self.num_entries,
            self.num_blocks,
            self.modulation_chunks,
            self.rows_per_entry,
            self.hidden_size,
        )

        timestep_bits = payload.get("timestep_bits")
        if not isinstance(timestep_bits, list) or len(timestep_bits) != self.num_entries:
            raise RuntimeError("AdaLN grid manifest timestep_bits length mismatch")
        self._entry_by_timestep_bits = {
            tuple(int(value) for value in row): index
            for index, row in enumerate(timestep_bits)
        }
        if len(self._entry_by_timestep_bits) != self.num_entries:
            raise RuntimeError("AdaLN grid timestep pairs are not unique")

        self._entries: OrderedDict[Any, torch.Tensor] = OrderedDict()
        self._persistent: set[Any] = set()
        self._scope: contextvars.ContextVar[_Scope] = contextvars.ContextVar(
            "h3_a100_grid_adaln_scope", default=_Scope(None, False)
        )
        self._hits = 0
        self._misses = 0
        self._stores = 0
        self._evictions = 0
        self.dropped_parameter_numel = 0

    @contextlib.contextmanager
    def scope(self, key: Any | None, *, persistent: bool = False) -> Iterator[None]:
        token = self._scope.set(_Scope(key, bool(persistent)))
        if key is not None and persistent:
            self._persistent.add(key)
        try:
            yield
        finally:
            self._scope.reset(token)

    def current_scope(self) -> _Scope:
        return self._scope.get()

    def _entry_index(self, timesteps: torch.Tensor) -> int:
        bits = _f32_bits(timesteps)
        if len(bits) != 2:
            raise RuntimeError(
                f"Grid-1000 H3 expects exactly two unique AV timesteps, got {len(bits)}"
            )
        try:
            return self._entry_by_timestep_bits[bits]
        except KeyError as exc:
            raise RuntimeError(
                "Runtime sigma/timestep is not represented in the frozen AdaLN grid. "
                f"bits={bits}; this would violate the Grid-1000 contract."
            ) from exc

    def ensure_key(self, key: Any, timesteps: torch.Tensor, *, persistent: bool) -> None:
        if key in self._entries:
            if persistent:
                self._persistent.add(key)
            self._entries.move_to_end(key)
            return
        index = self._entry_index(timesteps)
        host = self._table[index]
        # Pin only one selected ~18.5 MiB entry, never the ~18 GiB mmap.
        if self.pin_memory and torch.cuda.is_available():
            host = host.pin_memory()
        device = timesteps.device
        value = host.to(
            device=device,
            non_blocking=self.pin_memory and device.type == "cuda",
        )
        self._entries[key] = value
        if persistent:
            self._persistent.add(key)
        self._stores += 1
        self._entries.move_to_end(key)
        self._evict_dynamic()

    def has_complete_key(self, key: Any, num_blocks: int) -> bool:
        return int(num_blocks) == self.num_blocks and key in self._entries

    def lookup(self, block_index: int) -> tuple[torch.Tensor, ...]:
        scope = self._scope.get()
        if scope.key is None:
            self._misses += 1
            raise RuntimeError(
                "Grid AdaLN lookup has no active key; projection fallback does not exist"
            )
        value = self._entries.get(scope.key)
        if value is None:
            self._misses += 1
            raise RuntimeError(
                f"Grid AdaLN key {scope.key!r} was not staged before transformer execution"
            )
        self._entries.move_to_end(scope.key)
        self._hits += 1
        block = value[int(block_index)]
        return tuple(block[index] for index in range(self.modulation_chunks))

    def drop(self, key: Any) -> None:
        if key not in self._persistent:
            self._entries.pop(key, None)

    def clear_dynamic(self) -> None:
        for key in list(self._entries):
            if key not in self._persistent:
                self._entries.pop(key, None)

    def _evict_dynamic(self) -> None:
        dynamic = [key for key in self._entries if key not in self._persistent]
        while len(dynamic) > self.max_dynamic_keys:
            victim = dynamic.pop(0)
            self._entries.pop(victim, None)
            self._evictions += 1

    def stats(self) -> CacheStats:
        bytes_on_gpu = sum(
            int(value.numel() * value.element_size()) for value in self._entries.values()
        )
        dynamic = sum(1 for key in self._entries if key not in self._persistent)
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            stores=self._stores,
            evictions=self._evictions,
            persistent_keys=len(self._persistent),
            dynamic_keys=dynamic,
            bytes=bytes_on_gpu,
        )


class GridAdaLNProjectionHandle(nn.Module):
    """Parameter-free replacement for one ~260M-parameter AdaLN projection."""

    def __init__(self, controller: GridAdaLNController, block_index: int) -> None:
        super().__init__()
        object.__setattr__(self, "_controller", controller)
        self.block_index = int(block_index)

    def forward(self, _temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self._controller.lookup(self.block_index)


def install_grid_adaln_table(
    transformer: nn.Module,
    manifest_path: str | Path,
    *,
    max_dynamic_keys: int = 2,
    pin_memory: bool = True,
) -> GridAdaLNController:
    existing = getattr(transformer, GRID_CONTROLLER_ATTR, None)
    if isinstance(existing, GridAdaLNController):
        return existing
    blocks = list(getattr(transformer, "transformer_blocks", ()))
    if len(blocks) != 50:
        raise RuntimeError(f"Grid AdaLN expects 50 H3 blocks, got {len(blocks)}")
    controller = GridAdaLNController(
        manifest_path,
        max_dynamic_keys=max_dynamic_keys,
        pin_memory=pin_memory,
    )
    dropped = 0
    for index, block in enumerate(blocks):
        projection = getattr(block, "adaln_proj", None)
        if projection is None:
            raise RuntimeError(f"H3 block {index} has no adaln_proj before Grid-1000 install")
        dropped += sum(parameter.numel() for parameter in projection.parameters())
        block.adaln_proj = GridAdaLNProjectionHandle(controller, index)
    controller.dropped_parameter_numel = int(dropped)
    object.__setattr__(transformer, GRID_CONTROLLER_ATTR, controller)
    gc.collect()
    remaining = [name for name, _ in transformer.named_parameters() if ".adaln_proj." in name]
    if remaining:
        raise RuntimeError(
            "Grid-1000 failed to eliminate AdaLN parameters: " + ", ".join(remaining[:4])
        )
    logger.info(
        "[h3-a100][grid-adaln] installed manifest={} dropped_params={} ({:.3f}B) mmap_gib={:.2f}",
        controller.manifest_path,
        dropped,
        dropped / 1e9,
        controller.binary_path.stat().st_size / 1024**3,
    )
    return controller


def grid_adaln_controller(transformer: nn.Module) -> GridAdaLNController | None:
    value = getattr(transformer, GRID_CONTROLLER_ATTR, None)
    return value if isinstance(value, GridAdaLNController) else None


@dataclass
class GridReplayRegistration:
    transformer: Any
    controller: GridAdaLNController
    original_checkpoint: Callable[..., Any]
    stats: dict[str, int]


def install_grid_checkpoint_replay_scope(
    transformer: Any,
    controller: GridAdaLNController,
    *,
    expected_block_count: int = 50,
) -> GridReplayRegistration:
    """Capture the table key into the function stored by native checkpoint."""
    original = getattr(transformer, "_gradient_checkpointing_func", None)
    if not callable(original):
        raise RuntimeError("Grid replay requires callable native checkpoint function")
    stats = {
        "checkpoint_wrap_count": 0,
        "scoped_execution_count": 0,
        "cache_hit_count": 0,
        "cache_miss_count": 0,
        "missing_key_count": 0,
    }

    def wrapped(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        active = controller.current_scope()
        key = active.key
        if key is None or not controller.has_complete_key(key, expected_block_count):
            stats["missing_key_count"] += 1
            raise RuntimeError("Grid checkpoint created without a staged AdaLN table key")
        persistent = bool(active.persistent)
        stats["checkpoint_wrap_count"] += 1

        def scoped_function(*fn_args: Any, **fn_kwargs: Any) -> Any:
            # PyTorch non-reentrant checkpoint can early-stop recomputation via
            # an internal exception; entry/finally accounting is therefore
            # required for correct replay census.
            before_hits = int(controller._hits)
            before_misses = int(controller._misses)
            stats["scoped_execution_count"] += 1
            try:
                with controller.scope(key, persistent=persistent):
                    return function(*fn_args, **fn_kwargs)
            finally:
                hit_delta = int(controller._hits) - before_hits
                miss_delta = int(controller._misses) - before_misses
                stats["cache_hit_count"] += hit_delta
                stats["cache_miss_count"] += miss_delta
                if miss_delta != 0 or hit_delta != 1:
                    raise RuntimeError(
                        f"Grid checkpoint replay lookup mismatch key={key!r} "
                        f"hits={hit_delta} misses={miss_delta}"
                    )

        return original(scoped_function, *args, **kwargs)

    transformer._gradient_checkpointing_func = wrapped
    logger.info("[h3-a100][grid-adaln] installed early-stop-safe checkpoint replay scope")
    return GridReplayRegistration(transformer, controller, original, stats)


def _shift_sigma(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def install_grid_trainer_patch() -> None:
    """Patch registered model/trainer classes for the dedicated Grid-1000 branch."""
    from .model import FAKE_ADAPTER, STUDENT_ADAPTER, MiniMaxH3A100Model, _lora_config
    from .trainer import MiniMaxH3A100DmdTrainer

    if getattr(MiniMaxH3A100DmdTrainer, "_grid1000_patch_installed", False):
        return

    original_trainer_init = MiniMaxH3A100DmdTrainer.__init__
    original_trainer_setup = MiniMaxH3A100DmdTrainer.setup

    def trainer_init(self, config):
        original_trainer_init(self, config)
        grid = self.training_config.get("a100", {}).get("adaln_grid", {})
        replay_cache = self.training_config.get("a100", {}).get(
            "fa3_replay_cache", {}
        )
        self.adaln_grid_enabled = bool(grid.get("enabled", True))
        self.adaln_grid_size = int(grid.get("grid_size", DEFAULT_GRID_SIZE))
        self.adaln_grid_manifest = str(
            os.environ.get("H3_ADALN_GRID_MANIFEST", grid.get("manifest", ""))
        )
        self.adaln_grid_pin_memory = bool(grid.get("pin_memory", True))
        if not self.adaln_grid_enabled or self.adaln_grid_size != DEFAULT_GRID_SIZE:
            raise RuntimeError("This experiment branch requires enabled Grid-1000 AdaLN")
        if not self.adaln_grid_manifest:
            raise RuntimeError("H3_ADALN_GRID_MANIFEST (or a100.adaln_grid.manifest) is required")
        low = float(self.dmd_config.get("renoise_sigma_min", 0.02))
        high = float(self.dmd_config.get("renoise_sigma_max", 0.98))
        self._grid_base_sigmas_cpu = torch.linspace(
            low, high, self.adaln_grid_size, dtype=torch.float32
        )
        env_blocks = os.environ.get("H3_FA3_REPLAY_CACHE_BLOCKS")
        configured_blocks = replay_cache.get("blocks", ())
        if env_blocks is not None:
            self.fa3_replay_cache_blocks = parse_block_indices(env_blocks)
        elif bool(replay_cache.get("enabled", False)):
            self.fa3_replay_cache_blocks = parse_block_indices(configured_blocks)
        else:
            self.fa3_replay_cache_blocks = ()
        self.fa3_replay_cache_storage = parse_storage(
            os.environ.get(
                "H3_FA3_REPLAY_CACHE_STORAGE", replay_cache.get("storage", "cuda")
            )
        )
        self.fa3_replay_cache_max_d2h_inflight = parse_max_d2h_inflight(
            os.environ.get(
                "H3_FA3_REPLAY_CACHE_MAX_D2H_INFLIGHT",
                replay_cache.get("max_d2h_inflight", 2),
            )
        )
        trim_value = str(
            os.environ.get(
                "H3_FA3_REPLAY_CACHE_TRIM_BEFORE_BACKWARD",
                int(bool(replay_cache.get("trim_before_backward", False))),
            )
        ).strip().lower()
        if trim_value not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
            raise ValueError(
                "H3_FA3_REPLAY_CACHE_TRIM_BEFORE_BACKWARD must be boolean, "
                f"got {trim_value!r}"
            )
        self.fa3_replay_cache_trim_before_backward = trim_value in {
            "1", "true", "yes", "on"
        }
        self.fa3_replay_cache_registration = None

    def model_prepare(
        self,
        *,
        student_lora,
        fake_lora,
        cache_enabled=True,
        max_dynamic_cache_keys=2,
    ):
        del cache_enabled
        if self._shared_backbone_ready:
            return
        manifest = getattr(self, "_grid_adaln_manifest_path", None)
        if not manifest:
            raise RuntimeError("Grid model has no table manifest path before prepare_shared_backbone")
        transformer = self.denoiser_module()
        transformer.requires_grad_(False)
        install_grid_adaln_table(
            transformer,
            manifest,
            max_dynamic_keys=max_dynamic_cache_keys,
            pin_memory=bool(getattr(self, "_grid_adaln_pin_memory", True)),
        )
        self._add_named_adapter(STUDENT_ADAPTER, _lora_config(student_lora))
        self._add_named_adapter(FAKE_ADAPTER, _lora_config(fake_lora))
        counts = self._mark_all_adapters_trainable()
        if not all(counts.values()):
            raise RuntimeError(f"Failed to discover named H3 LoRA parameters: {counts}")
        self._activate_role(STUDENT_ADAPTER)
        self._shared_backbone_ready = True
        logger.info(
            "[h3-a100][grid-adaln] shared non-AdaLN backbone ready student_lora={} fake_lora={}",
            counts[STUDENT_ADAPTER],
            counts[FAKE_ADAPTER],
        )

    def model_adaln_cache(self):
        controller = grid_adaln_controller(self.denoiser_module())
        if controller is None:
            raise RuntimeError("Grid AdaLN controller is not installed")
        return controller

    @torch.no_grad()
    def model_precompute(self, key, timesteps: torch.Tensor, *, persistent: bool):
        self.adaln_cache().ensure_key(key, timesteps, persistent=persistent)

    def model_drop(self, key):
        self.adaln_cache().drop(key)

    def trainer_setup(self, *args: Any, **kwargs: Any):
        self.shared_model._grid_adaln_manifest_path = self.adaln_grid_manifest
        self.shared_model._grid_adaln_pin_memory = self.adaln_grid_pin_memory
        result = original_trainer_setup(self, *args, **kwargs)
        controller = self.shared_model.adaln_cache()
        payload = controller.payload
        if int(payload["grid_size"]) != self.adaln_grid_size:
            raise RuntimeError("training Grid-1000 size does not match table manifest")
        if abs(float(payload["renoise_sigma_min"]) - float(self.dmd_config.get("renoise_sigma_min", 0.02))) > 1e-12:
            raise RuntimeError("grid manifest renoise_sigma_min mismatch")
        if abs(float(payload["renoise_sigma_max"]) - float(self.dmd_config.get("renoise_sigma_max", 0.98))) > 1e-12:
            raise RuntimeError("grid manifest renoise_sigma_max mismatch")
        if abs(float(payload["video_shift"]) - float(self.video_shift)) > 1e-12:
            raise RuntimeError("grid manifest video_shift mismatch")
        if abs(float(payload["audio_shift"]) - float(self.audio_shift)) > 1e-12:
            raise RuntimeError("grid manifest audio_shift mismatch")
        # Install the selective wrapper first. The exact Grid wrapper then sits
        # outside it and supplies the captured AdaLN key to both original and
        # replay executions. This ordering is required because pinned
        # Diffusers' checkpoint closure does not accept ``context_fn``.
        self.fa3_replay_cache_registration = install_fa3_replay_cache(
            self.shared_model.denoiser_module(),
            block_indices=self.fa3_replay_cache_blocks,
            parent_split_registration=self.fa3_nograd_split_registration,
            storage=self.fa3_replay_cache_storage,
            max_d2h_inflight=self.fa3_replay_cache_max_d2h_inflight,
            trim_before_backward=self.fa3_replay_cache_trim_before_backward,
        )
        self.grid_replay_registration = install_grid_checkpoint_replay_scope(
            self.shared_model.denoiser_module(), controller, expected_block_count=50
        )
        logger.info(
            "[h3-a100][fa3-replay-cache] enabled={} blocks={} storage={} "
            "max_d2h_inflight={} trim_before_backward={}",
            self.fa3_replay_cache_registration.enabled,
            list(self.fa3_replay_cache_blocks),
            self.fa3_replay_cache_storage,
            self.fa3_replay_cache_max_d2h_inflight,
            self.fa3_replay_cache_trim_before_backward,
        )
        return result

    def sample_grid_sigmas(self):
        # Uniform over the 1000 discrete quadrature points. Shift is computed
        # in float32 on CPU from the exact frozen grid value, then moved to GPU,
        # making the table timestep pair bitwise reproducible.
        index = int(torch.randint(self.adaln_grid_size, (), device="cpu").item())
        base = self._grid_base_sigmas_cpu[index]
        video = _shift_sigma(base, float(self.video_shift)).to(self.shared_model.device)
        audio = _shift_sigma(base, float(self.audio_shift)).to(self.shared_model.device)
        return video, audio

    MiniMaxH3A100Model.prepare_shared_backbone = model_prepare
    MiniMaxH3A100Model.adaln_cache = model_adaln_cache
    MiniMaxH3A100Model.precompute_adaln = model_precompute
    MiniMaxH3A100Model.drop_adaln_key = model_drop
    MiniMaxH3A100DmdTrainer.__init__ = trainer_init
    MiniMaxH3A100DmdTrainer.setup = trainer_setup
    MiniMaxH3A100DmdTrainer._sample_renoise_sigmas = sample_grid_sigmas
    MiniMaxH3A100DmdTrainer._grid1000_patch_installed = True


install_grid_trainer_patch()

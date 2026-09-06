"""Exact AdaLN extraction and modulation caching for MiniMax-H3.

MiniMax-H3 stores one very large AdaLN projection in every transformer block.
Those projections consume roughly 13B parameters, but their outputs depend only
on the shared timestep embedding. They do not depend on prompt tokens, seed,
video/audio latents, or hidden states.

This module moves the per-block AdaLN projections into a separately shardable
bank and replaces each block's projection with a parameter-free handle. Fixed
rollout timesteps are cached persistently. Dynamic DMD score timesteps can be
cached for the lifetime of one fake/teacher evaluation and then discarded.

The cache is exact only while the timestep MLP and AdaLN projection weights are
frozen. The installer validates this invariant.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import weakref
from collections import OrderedDict
from collections.abc import Hashable, Iterator, Sequence
from typing import Any

import torch
from torch import nn

CacheKey = Hashable
Modulation = tuple[torch.Tensor, ...]


@dataclasses.dataclass(frozen=True)
class CacheStats:
    hits: int
    misses: int
    stores: int
    evictions: int
    persistent_keys: int
    dynamic_keys: int
    bytes: int


@dataclasses.dataclass(frozen=True)
class _Scope:
    key: CacheKey | None
    persistent: bool


class AdaLNCacheController:
    """Request-scoped exact cache shared by all H3 transformer blocks."""

    def __init__(self, *, enabled: bool = True, max_dynamic_keys: int = 2) -> None:
        if max_dynamic_keys < 0:
            raise ValueError(f"max_dynamic_keys must be non-negative, got {max_dynamic_keys}")
        self.enabled = bool(enabled)
        self.max_dynamic_keys = int(max_dynamic_keys)
        self._entries: OrderedDict[CacheKey, dict[int, Modulation]] = OrderedDict()
        self._persistent: set[CacheKey] = set()
        self._scope: contextvars.ContextVar[_Scope] = contextvars.ContextVar(
            "h3_a100_adaln_scope", default=_Scope(None, False)
        )
        self._last_scope = _Scope(None, False)
        self._hits = 0
        self._misses = 0
        self._stores = 0
        self._evictions = 0

    @contextlib.contextmanager
    def scope(self, key: CacheKey | None, *, persistent: bool = False) -> Iterator[None]:
        token = self._scope.set(_Scope(key, bool(persistent)))
        self._last_scope = _Scope(key, bool(persistent))
        if key is not None and persistent:
            self._persistent.add(key)
        try:
            yield
        finally:
            self._scope.reset(token)

    def current_scope(self) -> _Scope:
        """Return the active scope for checkpoint wrappers.

        PyTorch's non-reentrant checkpoint replay preserves RNG/autocast state,
        but not arbitrary ``contextvars``.  A segmented checkpoint wrapper uses
        this read-only snapshot to restore the exact AdaLN cache key during
        replay; it does not alter cache contents or training math.
        """

        return self._scope.get()

    def last_scope(self) -> _Scope:
        """Return the most recently entered scope for checkpoint replay."""

        return self._last_scope

    def get_or_compute(
        self,
        block_index: int,
        temb: torch.Tensor,
        projection: nn.Module,
    ) -> Modulation:
        scope = self._scope.get()
        key = scope.key
        if not self.enabled or key is None:
            output = projection(temb)
            return (output,) if torch.is_tensor(output) else tuple(output)

        block_cache = self._entries.get(key)
        if block_cache is not None and block_index in block_cache:
            self._entries.move_to_end(key)
            self._hits += 1
            return block_cache[block_index]

        self._misses += 1
        with torch.no_grad():
            output = projection(temb.detach())
        modulation = (
            (output.detach(),)
            if torch.is_tensor(output)
            else tuple(tensor.detach() for tensor in output)
        )

        if block_cache is None:
            block_cache = {}
            self._entries[key] = block_cache
        block_cache[block_index] = modulation
        self._stores += 1
        if scope.persistent:
            self._persistent.add(key)
        self._entries.move_to_end(key)
        self._evict_dynamic_keys()
        return modulation

    def has_complete_key(self, key: CacheKey, num_blocks: int) -> bool:
        block_cache = self._entries.get(key)
        return block_cache is not None and len(block_cache) == int(num_blocks)

    def drop(self, key: CacheKey) -> None:
        if key not in self._persistent:
            self._entries.pop(key, None)

    def clear_dynamic(self) -> None:
        for key in list(self._entries):
            if key not in self._persistent:
                del self._entries[key]

    def clear(self) -> None:
        self._entries.clear()
        self._persistent.clear()

    def stats(self) -> CacheStats:
        total_bytes = 0
        dynamic_keys = 0
        for key, block_cache in self._entries.items():
            if key not in self._persistent:
                dynamic_keys += 1
            for modulation in block_cache.values():
                total_bytes += sum(t.numel() * t.element_size() for t in modulation)
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            stores=self._stores,
            evictions=self._evictions,
            persistent_keys=len(self._persistent),
            dynamic_keys=dynamic_keys,
            bytes=total_bytes,
        )

    def _evict_dynamic_keys(self) -> None:
        dynamic = [key for key in self._entries if key not in self._persistent]
        while len(dynamic) > self.max_dynamic_keys:
            victim = dynamic.pop(0)
            self._entries.pop(victim, None)
            self._evictions += 1


class AdaLNProjectionBank(nn.Module):
    """Single owner for all per-block AdaLN projection modules."""

    def __init__(self, projections: Sequence[nn.Module]) -> None:
        super().__init__()
        if not projections:
            raise ValueError("MiniMax-H3 AdaLN bank cannot be empty")
        self.projections = nn.ModuleList(list(projections))


class AdaLNProjectionHandle(nn.Module):
    """Parameter-free handle forwarding into a shared AdaLN bank."""

    def __init__(
        self,
        bank: AdaLNProjectionBank,
        controller: AdaLNCacheController,
        block_index: int,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_bank_ref", weakref.ref(bank))
        object.__setattr__(self, "_controller_ref", weakref.ref(controller))
        self.block_index = int(block_index)

    def forward(self, temb: torch.Tensor) -> Any:
        bank = self._bank_ref()
        controller = self._controller_ref()
        if bank is None or controller is None:
            raise RuntimeError("MiniMax-H3 AdaLN bank/controller was released unexpectedly")
        return controller.get_or_compute(
            self.block_index,
            temb,
            bank.projections[self.block_index],
        )


def _transformer_blocks(transformer: nn.Module) -> list[nn.Module]:
    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None:
        blocks = getattr(transformer, "blocks", None)
    if blocks is None:
        raise AttributeError("MiniMax-H3 transformer has neither transformer_blocks nor blocks")
    return list(blocks)


def _assert_frozen(module: nn.Module, name: str) -> None:
    trainable = [param_name for param_name, param in module.named_parameters() if param.requires_grad]
    if trainable:
        preview = ", ".join(trainable[:6])
        raise RuntimeError(
            f"{name} must be frozen before enabling exact AdaLN caching; "
            f"found {len(trainable)} trainable parameters: {preview}"
        )


def install_adaln_cache(
    transformer: nn.Module,
    *,
    enabled: bool = True,
    max_dynamic_keys: int = 2,
) -> AdaLNCacheController:
    """Extract AdaLN modules and install exact cache handles.

    Call after loading the H3 checkpoint and before FSDP/HSDP wrapping.
    """

    existing = getattr(transformer, "_h3_a100_adaln_controller", None)
    if existing is not None:
        return existing

    blocks = _transformer_blocks(transformer)
    projections: list[nn.Module] = []
    for index, block in enumerate(blocks):
        projection = getattr(block, "adaln_proj", None)
        if projection is None:
            raise AttributeError(f"MiniMax-H3 block {index} has no adaln_proj module")
        projections.append(projection)

    bank = AdaLNProjectionBank(projections)
    controller = AdaLNCacheController(enabled=enabled, max_dynamic_keys=max_dynamic_keys)
    transformer.add_module("h3_a100_adaln_bank", bank)
    object.__setattr__(transformer, "_h3_a100_adaln_controller", controller)

    for index, block in enumerate(blocks):
        block.adaln_proj = AdaLNProjectionHandle(bank, controller, index)

    _assert_frozen(bank, "MiniMax-H3 AdaLN bank")
    time_embedder = getattr(transformer, "time_embedder", None)
    if time_embedder is not None:
        _assert_frozen(time_embedder, "MiniMax-H3 timestep embedder")
    return controller


def adaln_bank(transformer: nn.Module) -> AdaLNProjectionBank | None:
    bank = getattr(transformer, "h3_a100_adaln_bank", None)
    return bank if isinstance(bank, AdaLNProjectionBank) else None


def adaln_controller(transformer: nn.Module) -> AdaLNCacheController | None:
    controller = getattr(transformer, "_h3_a100_adaln_controller", None)
    return controller if isinstance(controller, AdaLNCacheController) else None

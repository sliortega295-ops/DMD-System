"""Exact rank16 trace binding for the static FSDP2 baselines.

Unlike Role-Phase, every logical slot stays on the same world16 rank.  This
module never projects a B16 trace to B12 and never samples new randomness.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from .critic_pool_helios import LoadedPoolEntry
from .operation_trace import (
    OperationTrace,
    load_operation_trace_bundle_manifest,
    replay_stage3_source,
    resolve_rank_trace_root,
    validate_official_rank_stage_schedule_semantics,
)


WORLD_SIZE = 16


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _history_keys(prefix: str) -> dict[str, str]:
    return {
        "text_embedding": "sample/0/text_embedding",
        "attention_mask": "sample/0/attention_mask",
        "continuous_history": f"{prefix}/continuous_history",
        "history_short": f"{prefix}/history_short",
        "history_mid": f"{prefix}/history_mid",
        "history_long": f"{prefix}/history_long",
        "indices_hidden_states": f"{prefix}/indices_hidden_states",
        "indices_history_short": f"{prefix}/indices_history_short",
        "indices_history_mid": f"{prefix}/indices_history_mid",
        "indices_history_long": f"{prefix}/indices_history_long",
        "entry_noise": f"{prefix}/entry_noise",
        "rollout_noise": f"{prefix}/rollout_noise",
        "block_noise_stage1": f"{prefix}/block_noise/1",
        "block_noise_stage2": f"{prefix}/block_noise/2",
    }


def _generator_keys(step: int, *, gan: bool) -> dict[str, str]:
    prefix = f"step/{step}/generator"
    result = {
        **_history_keys(prefix),
        "dmd_timestep": f"{prefix}/dmd_timestep",
        "dmd_noise": f"{prefix}/dmd_noise",
    }
    if gan:
        result.update(
            gan_g_noise=f"{prefix}/gan_g_noise",
            gan_g_crop=f"{prefix}/gan_g_crop",
        )
    return result


def _critic_keys(step: int, *, gan: bool) -> dict[str, str]:
    prefix = f"step/{step}/critic"
    result = {
        **_history_keys(prefix),
        "critic_timestep": f"{prefix}/critic_timestep",
        "critic_noise": f"{prefix}/critic_noise",
    }
    if gan:
        result.update(
            gan_d_fake_crop=f"{prefix}/gan_d_fake_crop",
            gan_d_real_crop=f"{prefix}/gan_d_real_crop",
            gan_d_r1_crop=f"{prefix}/gan_d_r1_crop",
            r1_epsilon=f"{prefix}/r1_epsilon",
        )
    return result


def _descriptor_hashes(trace: OperationTrace, keys: dict[str, str]) -> dict[str, str]:
    descriptors = trace.manifest["tensors"]
    return {role: str(descriptors[key]["sha256"]) for role, key in keys.items()}


@dataclass(frozen=True)
class StaticGeneratorEntry:
    source_id: int
    logical_step: int
    operation_trace: OperationTrace
    tensors: dict[str, torch.Tensor]
    identity_sha256: str

    @property
    def identity(self) -> dict[str, int]:
        return {"source_id": self.source_id, "source_rank": self.source_id, "source_wave": 0}

    @property
    def rollout_operation(self) -> str:
        return f"step/{self.logical_step}/generator/rollout"

    @property
    def dmd_operation(self) -> str:
        return f"step/{self.logical_step}/generator/dmd"


class _AliasedOperationTrace:
    """Read-only step alias over an immutable five-step source trace.

    Fifty-cycle trajectory experiments deliberately replay one fixed,
    rank-qualified five-step cohort while optimizer/scheduler/model versions
    advance continuously.  Keeping this as a view avoids copying the 3.7-GiB
    trace bundle fifty times.  The logical and source step are both exposed in
    receipts so this cannot be mistaken for fifty independently sampled
    batches.
    """

    def __init__(self, source: OperationTrace, *, logical_step: int, source_step: int) -> None:
        self._source = source
        self.logical_step = int(logical_step)
        self.source_step = int(source_step)
        self.root = source.root
        self.tensors = source.tensors

    def _key(self, key: str) -> str:
        prefix = f"step/{self.logical_step}/"
        if key.startswith(prefix):
            return f"step/{self.source_step}/" + key[len(prefix) :]
        return key

    @property
    def trace_hash(self) -> str:
        return self._source.trace_hash

    @property
    def manifest(self) -> dict[str, Any]:
        # Callers that need tensor descriptors use source-keyed descriptors;
        # runtime operation/tensor consumption goes through the methods below.
        return self._source.manifest

    def tensor(self, key: str, *, clone: bool = True) -> torch.Tensor:
        return self._source.tensor(self._key(key), clone=clone)

    def operation(self, key: str) -> dict[str, Any]:
        operation = self._source.operation(self._key(key))
        result = dict(operation)
        result["key"] = key
        control = dict(result.get("control", {}))
        if isinstance(control.get("global_step"), int):
            control["global_step"] = self.logical_step
        if result.get("operation") == "stage3_branch":
            control["train_generator"] = self.logical_step % 5 == 0
        result["control"] = control
        result["tensor_keys"] = [
            key.replace(
                f"step/{self.source_step}/", f"step/{self.logical_step}/", 1
            )
            if key.startswith(f"step/{self.source_step}/")
            else key
            for key in result.get("tensor_keys", [])
        ]
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)


class StaticWorld16Trace:
    def __init__(self, root: str | Path, *, variant: str, cycles: int = 1) -> None:
        if variant not in {"no-gan", "gan"}:
            raise ValueError(variant)
        self.root = Path(root).resolve()
        self.variant = variant
        self.gan = variant == "gan"
        if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles <= 0:
            raise ValueError("trace cycles must be a positive integer")
        self.cycles = cycles
        self.source_steps = (
            tuple(range(1000, 1005)) if self.gan else tuple(range(10, 15))
        )
        self.logical_steps = tuple(
            range(self.source_steps[0], self.source_steps[0] + 5 * cycles)
        )
        bundle = load_operation_trace_bundle_manifest(self.root, verify=True)
        if bundle is None or len(bundle.get("ranks", ())) != WORLD_SIZE:
            raise ValueError("static FSDP2 requires one authoritative rank16 trace bundle")
        self.bundle_hash = str(bundle["bundle_hash"])
        self._control: dict[int, OperationTrace] = {}
        self._loaded: dict[int, OperationTrace] = {}
        for rank in range(WORLD_SIZE):
            trace_root = resolve_rank_trace_root(self.root, rank)
            manifest = json.loads((trace_root / "trace.json").read_text())
            self._control[rank] = OperationTrace(root=trace_root, manifest=manifest, tensors={})
        self.rank_control_validation = validate_official_rank_stage_schedule_semantics(
            self._control
        )
        self._validate()
        binding: dict[str, Any] = {
            "schema": "distillschedule.static_world16_trace.v1",
            "variant": variant,
            "bundle_hash": self.bundle_hash,
            "logical_steps": list(self.logical_steps),
            "rank_slots": list(range(WORLD_SIZE)),
        }
        if self.cycles > 1:
            binding.update(
                source_steps=list(self.source_steps),
                cycle_count=self.cycles,
                cohort_policy="FIXED_FIVE_STEP_COHORT_REPLAY",
            )
        self.binding_hash = hashlib.sha256(_canonical(binding)).hexdigest()

    def _load(self, rank: int) -> OperationTrace:
        if not 0 <= rank < WORLD_SIZE:
            raise IndexError(rank)
        value = self._loaded.get(rank)
        if value is None:
            value = OperationTrace.load(resolve_rank_trace_root(self.root, rank), verify=True)
            if value.trace_hash != self._control[rank].trace_hash:
                raise ValueError("trace payload changed after manifest preflight")
            self._loaded[rank] = value
        return value

    def source_step(self, logical_step: int) -> int:
        if logical_step not in self.logical_steps:
            raise IndexError(logical_step)
        return self.source_steps[
            (int(logical_step) - self.logical_steps[0]) % len(self.source_steps)
        ]

    def _view(
        self, rank: int, logical_step: int
    ) -> OperationTrace | _AliasedOperationTrace:
        if self.cycles == 1:
            return self._load(rank)
        return _AliasedOperationTrace(
            self._load(rank),
            logical_step=logical_step,
            source_step=self.source_step(logical_step),
        )

    @staticmethod
    def _physical_signature(operation: dict[str, Any]) -> tuple[Any, ...]:
        schedule = operation["control"]["pyramid_stage_call_sequence"]
        return tuple(
            (
                int(item["stage"]),
                int(item["call_count"]),
                int(item["denoising_exit"]),
                tuple(int(v) for v in item["latent_shape"]),
                len(item["timestep"]),
            )
            for item in schedule
        )

    def _validate(self) -> None:
        for rank, trace in self._control.items():
            contract = trace.manifest.get("contract", {})
            expected_contract = {
                "resolution": [384, 640],
                "stage_spatial_shapes": [[12, 20], [24, 40], [48, 80]],
                "full_latent_shape": [1, 16, 9, 48, 80],
                "dtype": "bfloat16",
                "dfake_gen_update_ratio": 5,
            }
            for key, expected in expected_contract.items():
                if contract.get(key) != expected:
                    raise ValueError(f"rank {rank} trace violates {key}")
            for step in self.source_steps:
                branch = trace.operation(f"step/{step}/branch")["control"]
                expected = {
                    "global_step": step,
                    "dfake_gen_update_ratio": 5,
                    "train_generator": step == self.logical_steps[0],
                    "train_critic": True,
                    "gan": self.gan,
                    "reward": False,
                    "vae_decode": False,
                }
                mismatch = {
                    key: (branch.get(key), value)
                    for key, value in expected.items()
                    if branch.get(key) != value
                }
                if mismatch:
                    raise ValueError(f"rank {rank} Step{step} branch mismatch: {mismatch}")
                trace.operation(f"step/{step}/critic/rollout")
                trace.operation(f"step/{step}/critic/train")
                if self.gan:
                    trace.operation(f"step/{step}/critic/gan_d")
            first = self.source_steps[0]
            trace.operation(f"step/{first}/generator/rollout")
            trace.operation(f"step/{first}/generator/dmd")
            if self.gan:
                trace.operation(f"step/{first}/generator/gan_g")

        for step in self.source_steps:
            values = [
                self._physical_signature(
                    self._control[rank].operation(f"step/{step}/critic/rollout")
                )
                for rank in range(WORLD_SIZE)
            ]
            if any(value != values[0] for value in values[1:]):
                raise ValueError(f"Step{step} Critic physical call sequence diverges by rank")
        first = self.source_steps[0]
        values = [
            self._physical_signature(
                self._control[rank].operation(f"step/{first}/generator/rollout")
            )
            for rank in range(WORLD_SIZE)
        ]
        if any(value != values[0] for value in values[1:]):
            raise ValueError("Generator physical call sequence diverges by rank")

    def _tensors(
        self, trace: OperationTrace, keys: dict[str, str], device: torch.device | str
    ) -> dict[str, torch.Tensor]:
        missing = set(keys.values()) - set(trace.manifest["tensors"])
        if missing:
            raise ValueError(f"trace misses {sorted(missing)}")
        tensors = {
            role: trace.tensor(key, clone=True).to(device)
            for role, key in keys.items()
        }
        if tensors["text_embedding"].ndim == 2:
            tensors["text_embedding"] = tensors["text_embedding"].unsqueeze(0)
        return tensors

    def load_generator_entry(
        self,
        *,
        rank: int,
        device: torch.device | str,
        step: int | None = None,
    ) -> StaticGeneratorEntry:
        logical_step = self.logical_steps[0] if step is None else int(step)
        if logical_step not in self.logical_steps or logical_step % 5 != 0:
            raise IndexError(logical_step)
        trace = self._load(rank)
        source_step = self.source_step(logical_step)
        source_keys = _generator_keys(source_step, gan=self.gan)
        tensors = self._tensors(trace, source_keys, device)
        view = self._view(rank, logical_step)
        identity_payload = {
            "variant": self.variant,
            "phase": "generator_update",
            "rank": rank,
            "step": logical_step,
            "trace_hash": trace.trace_hash,
            "tensor_hashes": _descriptor_hashes(trace, source_keys),
        }
        if self.cycles > 1:
            identity_payload["source_step"] = source_step
        identity = hashlib.sha256(_canonical(identity_payload)).hexdigest()
        return StaticGeneratorEntry(rank, logical_step, view, tensors, identity)

    def load_critic_entry(
        self, *, rank: int, step: int, device: torch.device | str
    ) -> LoadedPoolEntry:
        if step not in self.logical_steps:
            raise IndexError(step)
        trace = self._load(rank)
        source_step = self.source_step(step)
        source_keys = _critic_keys(source_step, gan=self.gan)
        tensors = self._tensors(trace, source_keys, device)
        view = self._view(rank, step)
        identity_payload = {
            "variant": self.variant,
            "phase": "critic",
            "rank": rank,
            "step": step,
            "trace_hash": trace.trace_hash,
            "tensor_hashes": _descriptor_hashes(trace, source_keys),
        }
        if self.cycles > 1:
            identity_payload["source_step"] = source_step
        identity = hashlib.sha256(_canonical(identity_payload)).hexdigest()
        entry = SimpleNamespace(
            trace_key=f"static-fsdp2/{self.variant}/step{step}/rank{rank}",
            logical_sample_slot=rank,
            update=self.logical_steps.index(step),
            production_round=0,
            producer_pool_rank=rank,
            source_trace_rank=rank,
            logical_step=step,
            rng_stream=(
                f"rank-qualified/{rank}/step/{source_step}"
                if self.cycles > 1
                else f"rank-qualified/{rank}/step/{step}"
            ),
            trace_path=str(trace.root),
            operation_trace_hash=trace.trace_hash,
            rollout_operation=f"step/{step}/critic/rollout",
            train_operation=f"step/{step}/critic/train",
            tensor_keys=_critic_keys(step, gan=self.gan),
            identity_sha256=identity,
        )
        if self.cycles > 1:
            entry.source_trace_step = source_step
        return LoadedPoolEntry(entry, view, tensors)

    def source(
        self, *, rank: int, device: torch.device | str
    ) -> dict[str, torch.Tensor]:
        trace = self._load(rank)
        return replay_stage3_source(trace, device=device, dtype=torch.bfloat16)


__all__ = ["StaticGeneratorEntry", "StaticWorld16Trace", "WORLD_SIZE"]

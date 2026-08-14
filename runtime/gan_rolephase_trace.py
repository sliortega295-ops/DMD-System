"""Rank-qualified trace binding for the real GAN role-phase Full5 worker.

The source rank16 bundle is immutable.  Phase A consumes all sixteen logical
slots in two DP8 waves; Phase B deliberately projects the first twelve source
slots onto the preregistered 4G+12C batch12 topology.  No RNG is sampled here.
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


LOGICAL_STEPS = tuple(range(1000, 1005))
PHASE_A_BATCH = 16
PHASE_B_BATCH = 12

REQUIRED_CONTRACT = {
    "resolution": [384, 640],
    "stage_spatial_shapes": [[12, 20], [24, 40], [48, 80]],
    "full_latent_shape": [1, 16, 9, 48, 80],
    "text_shape": [1, 512, 4096],
    "dtype": "bfloat16",
    "dfake_gen_update_ratio": 5,
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _generator_keys(step: int) -> dict[str, str]:
    prefix = f"step/{step}/generator"
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
        "dmd_timestep": f"{prefix}/dmd_timestep",
        "dmd_noise": f"{prefix}/dmd_noise",
        "gan_g_noise": f"{prefix}/gan_g_noise",
        "gan_g_crop": f"{prefix}/gan_g_crop",
    }


def _critic_keys(step: int) -> dict[str, str]:
    prefix = f"step/{step}/critic"
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
        "critic_timestep": f"{prefix}/critic_timestep",
        "critic_noise": f"{prefix}/critic_noise",
        "gan_d_fake_crop": f"{prefix}/gan_d_fake_crop",
        "gan_d_real_crop": f"{prefix}/gan_d_real_crop",
        "gan_d_r1_crop": f"{prefix}/gan_d_r1_crop",
        "r1_epsilon": f"{prefix}/r1_epsilon",
    }


def _source_keys() -> dict[str, str]:
    return {
        "primary_x0": "sample/0/primary/x0",
        "primary_history": "sample/0/primary/history",
        "primary_target": "sample/0/primary/target",
        "secondary_x0": "sample/0/secondary/x0",
        "secondary_history": "sample/0/secondary/history",
        "secondary_target": "sample/0/secondary/target",
        "text_embedding": "sample/0/text_embedding",
    }


def _identity(trace: OperationTrace, payload: dict[str, Any], keys: dict[str, str]) -> str:
    descriptors = trace.manifest["tensors"]
    return hashlib.sha256(
        _canonical(
            {
                **payload,
                "operation_trace_hash": trace.trace_hash,
                "tensor_sha256": {
                    role: descriptors[key]["sha256"] for role, key in keys.items()
                },
            }
        )
    ).hexdigest()


@dataclass(frozen=True)
class GanGeneratorTraceEntry:
    source_id: int
    logical_step: int
    operation_trace: OperationTrace
    tensors: dict[str, torch.Tensor]
    identity_sha256: str

    @property
    def identity(self) -> dict[str, int]:
        return {
            "source_id": self.source_id,
            "source_wave": self.source_id // 8,
            "source_rank": self.source_id % 8,
        }

    @property
    def rollout_operation(self) -> str:
        return f"step/{self.logical_step}/generator/rollout"

    @property
    def dmd_operation(self) -> str:
        return f"step/{self.logical_step}/generator/dmd"


class GanRolePhaseTrace:
    """Lazy tensor loader plus eager control/fingerprint validation."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        bundle = load_operation_trace_bundle_manifest(self.root, verify=True)
        if bundle is None or len(bundle.get("ranks", [])) != PHASE_A_BATCH:
            raise ValueError("GAN role-phase requires one rank16 trace bundle")
        self.bundle_hash = str(bundle["bundle_hash"])
        systems_manifest_path = self.root / "systems_only_gan_trace_manifest.json"
        systems_manifest = json.loads(systems_manifest_path.read_text())
        if systems_manifest.get("status") != "PASS":
            raise ValueError("systems-only GAN trace manifest is not PASS")
        if systems_manifest.get("output_bundle_hash") != self.bundle_hash:
            raise ValueError("systems-only GAN trace manifest disagrees with bundle")
        if systems_manifest.get("logical_steps") != list(LOGICAL_STEPS):
            raise ValueError("systems-only GAN trace logical window changed")
        if int(systems_manifest.get("world_size", -1)) != PHASE_A_BATCH:
            raise ValueError("systems-only GAN trace world size changed")
        self.source_bundle_hash = str(systems_manifest["source_bundle_hash"])
        self._control: dict[int, OperationTrace] = {}
        self._loaded: dict[int, OperationTrace] = {}
        for rank in range(PHASE_A_BATCH):
            trace_root = resolve_rank_trace_root(self.root, rank)
            manifest = json.loads((trace_root / "trace.json").read_text())
            self._control[rank] = OperationTrace(
                root=trace_root, manifest=manifest, tensors={}
            )
        self.rank_control_validation = validate_official_rank_stage_schedule_semantics(
            self._control
        )
        self._validate()
        self.binding_hash = hashlib.sha256(
            _canonical(
                {
                    "schema": "distillschedule.gan_rolephase_trace.v1",
                    "bundle_hash": self.bundle_hash,
                    "phase_a_slots": list(range(PHASE_A_BATCH)),
                    "phase_b_slots": list(range(PHASE_B_BATCH)),
                    "logical_steps": list(LOGICAL_STEPS),
                }
            )
        ).hexdigest()

    @property
    def loaded_trace_count(self) -> int:
        return len(self._loaded)

    def _load(self, source_rank: int) -> OperationTrace:
        if not 0 <= source_rank < PHASE_A_BATCH:
            raise IndexError(source_rank)
        value = self._loaded.get(source_rank)
        if value is None:
            value = OperationTrace.load(
                resolve_rank_trace_root(self.root, source_rank), verify=True
            )
            if value.trace_hash != self._control[source_rank].trace_hash:
                raise ValueError("trace payload changed after manifest preflight")
            self._loaded[source_rank] = value
        return value

    def _validate(self) -> None:
        for rank, trace in self._control.items():
            contract = trace.manifest.get("contract", {})
            for key, expected in REQUIRED_CONTRACT.items():
                if contract.get(key) != expected:
                    raise ValueError(f"rank {rank} trace violates {key}")
            for step in LOGICAL_STEPS:
                branch = trace.operation(f"step/{step}/branch")["control"]
                expected = {
                    "global_step": step,
                    "dfake_gen_update_ratio": 5,
                    "train_generator": step == 1000,
                    "train_critic": True,
                    "gan": True,
                    "reward": False,
                    "vae_decode": False,
                }
                mismatch = {
                    key: (branch.get(key), value)
                    for key, value in expected.items()
                    if branch.get(key) != value
                }
                if mismatch:
                    raise ValueError(f"rank {rank} Step{step} branch drift: {mismatch}")
                trace.operation(f"step/{step}/critic/rollout")
                trace.operation(f"step/{step}/critic/train")
                trace.operation(f"step/{step}/critic/gan_d")
            trace.operation("step/1000/generator/rollout")
            trace.operation("step/1000/generator/dmd")
            trace.operation("step/1000/generator/gan_g")

        # Normal Helios execution broadcasts the stage exits, so physical
        # stage/call counts must match on all ranks.  Denoising budgets and
        # timestep values remain rank-local stochastic inputs: comparing the
        # entire schedule object would incorrectly force those values equal.
        # This distinction is the corrected official trace contract.
        def physical_signature(schedule: list[dict[str, Any]]) -> tuple[Any, ...]:
            return tuple(
                (
                    int(item["stage"]),
                    int(item["call_count"]),
                    int(item["denoising_exit"]),
                    tuple(item["latent_shape"]),
                    len(item["timestep"]),
                )
                for item in schedule
            )

        for step in LOGICAL_STEPS:
            key = f"step/{step}/critic/rollout"
            schedules = [
                physical_signature(
                    self._control[rank].operation(key)["control"][
                        "pyramid_stage_call_sequence"
                    ]
                )
                for rank in range(PHASE_A_BATCH)
            ]
            if any(value != schedules[0] for value in schedules[1:]):
                raise ValueError(
                    f"Step{step} Critic physical call sequence is rank divergent"
                )
        schedules = [
            physical_signature(
                self._control[rank].operation("step/1000/generator/rollout")[
                    "control"
                ]["pyramid_stage_call_sequence"]
            )
            for rank in range(PHASE_A_BATCH)
        ]
        if any(value != schedules[0] for value in schedules[1:]):
            raise ValueError(
                "Step1000 Generator physical call sequence is rank divergent"
            )

    def load_generator_entry(
        self, *, source_id: int, device: torch.device | str
    ) -> GanGeneratorTraceEntry:
        trace = self._load(source_id)
        step = 1000
        keys = _generator_keys(step)
        missing = set(keys.values()) - set(trace.manifest["tensors"])
        if missing:
            raise ValueError(f"rank {source_id} generator trace misses {sorted(missing)}")
        tensors = {
            role: trace.tensor(key).to(device) for role, key in keys.items()
        }
        if tensors["text_embedding"].ndim == 2:
            tensors["text_embedding"] = tensors["text_embedding"].unsqueeze(0)
        identity = _identity(
            trace,
            {
                "phase": "A",
                "source_id": source_id,
                "logical_step": step,
                "rollout_operation": f"step/{step}/generator/rollout",
                "dmd_operation": f"step/{step}/generator/dmd",
            },
            keys,
        )
        return GanGeneratorTraceEntry(
            source_id=source_id,
            logical_step=step,
            operation_trace=trace,
            tensors=tensors,
            identity_sha256=identity,
        )

    def source(self, *, source_rank: int, device: torch.device | str) -> dict[str, torch.Tensor]:
        trace = self._load(source_rank)
        return replay_stage3_source(
            trace, device=device, dtype=torch.bfloat16
        )

    def load_critic_entry(
        self,
        *,
        update: int,
        production_round: int,
        producer_pool_rank: int,
        device: torch.device | str,
    ) -> LoadedPoolEntry:
        if not 0 <= update < len(LOGICAL_STEPS):
            raise IndexError(update)
        if not 0 <= production_round < 3 or not 0 <= producer_pool_rank < 4:
            raise IndexError((production_round, producer_pool_rank))
        source_rank = production_round * 4 + producer_pool_rank
        logical_slot = source_rank
        step = LOGICAL_STEPS[update]
        trace = self._load(source_rank)
        keys = _critic_keys(step)
        missing = set(keys.values()) - set(trace.manifest["tensors"])
        if missing:
            raise ValueError(f"rank {source_rank} Step{step} misses {sorted(missing)}")
        tensors = {
            role: trace.tensor(key).to(device) for role, key in keys.items()
        }
        if tensors["text_embedding"].ndim == 2:
            tensors["text_embedding"] = tensors["text_embedding"].unsqueeze(0)
        identity = _identity(
            trace,
            {
                "phase": "B",
                "source_trace_rank": source_rank,
                "logical_slot": logical_slot,
                "logical_step": step,
                "update": update,
                "production_round": production_round,
                "producer_pool_rank": producer_pool_rank,
                "rollout_operation": f"step/{step}/critic/rollout",
                "train_operation": f"step/{step}/critic/train",
            },
            keys,
        )
        entry = SimpleNamespace(
            trace_key=f"gan-rolephase/step{step}/slot{logical_slot}",
            logical_sample_slot=logical_slot,
            update=update,
            production_round=production_round,
            producer_pool_rank=producer_pool_rank,
            source_trace_rank=source_rank,
            logical_step=step,
            rng_stream=f"rank-qualified/{source_rank}/step/{step}",
            trace_path=str(trace.root),
            operation_trace_hash=trace.trace_hash,
            rollout_operation=f"step/{step}/critic/rollout",
            train_operation=f"step/{step}/critic/train",
            tensor_keys=keys,
            identity_sha256=identity,
        )
        return LoadedPoolEntry(entry, trace, tensors)


__all__ = [
    "GanGeneratorTraceEntry",
    "GanRolePhaseTrace",
    "LOGICAL_STEPS",
    "PHASE_A_BATCH",
    "PHASE_B_BATCH",
]

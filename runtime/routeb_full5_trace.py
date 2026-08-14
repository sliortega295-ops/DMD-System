"""Rank-qualified trace binding for Route-B Full5.

The canonical rank16 operation-trace bundle remains the tensor authority.  A
Route-B binding only maps physical pool slots to those immutable rank traces;
it never samples RNG and never assumes that two gradient-accumulation
micro-waves belong to different logical steps.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .critic_pool_helios import LoadedPoolEntry
from .operation_trace import (
    OperationTrace,
    load_operation_trace_bundle_manifest,
    resolve_rank_trace_root,
    validate_official_rank_stage_schedule_semantics,
)
from .routeb_full5 import RouteBFull5Contract


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


def _critic_tensor_keys(step: int) -> dict[str, str]:
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
    }


def _generator_tensor_keys(step: int) -> dict[str, str]:
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
    }


def _descriptor_identity(trace: OperationTrace, keys: dict[str, str]) -> dict[str, str]:
    """Return the hashes already bound into the verified trace manifest.

    The 16-rank bundle contains roughly 3.7 GiB of tensors.  Loading all of it
    into every process would multiply immutable input memory by world size and
    can turn an otherwise valid 14B run into a host-OOM.  The bundle/manifest
    hashes are validated eagerly; each rank payload is then loaded and fully
    verified on first use.
    """

    return {
        role: str(trace.manifest["tensors"][key]["sha256"])
        for role, key in keys.items()
    }


@dataclass(frozen=True)
class RouteBTraceEntry:
    trace_key: str
    logical_sample_slot: int
    update: int
    production_round: int
    producer_pool_rank: int
    source_trace_rank: int
    logical_step: int
    rng_stream: str
    trace_path: str
    operation_trace_hash: str
    rollout_operation: str
    train_operation: str
    tensor_keys: dict[str, str]
    identity_sha256: str


@dataclass(frozen=True)
class RouteBStep10AEntry:
    bundle_root: Path
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


@dataclass(frozen=True)
class RouteBStep10ACollectivePlan:
    production_round: int
    source_ids: tuple[int, ...]
    logical_stage_exits: tuple[tuple[int, ...], ...]
    logical_call_counts: tuple[int, ...]
    synchronized_stage_exits: tuple[int, ...]

    @property
    def physical_stage_call_counts(self) -> tuple[int, ...]:
        return tuple(value + 1 for value in self.synchronized_stage_exits)

    @property
    def physical_call_count(self) -> int:
        return sum(self.physical_stage_call_counts)

    @property
    def dummy_call_counts(self) -> tuple[int, ...]:
        return tuple(
            self.physical_call_count - logical
            for logical in self.logical_call_counts
        )


class RouteBFull5Trace:
    def __init__(
        self,
        bundle_root: str | Path,
        contract: RouteBFull5Contract,
    ) -> None:
        self.root = Path(bundle_root).resolve()
        self.contract = contract
        bundle = load_operation_trace_bundle_manifest(self.root, verify=True)
        if bundle is None or len(bundle.get("ranks", [])) != 16:
            raise ValueError("Route-B requires one authoritative rank16 trace bundle")
        self.bundle_hash = str(bundle["bundle_hash"])
        self._control_traces: dict[int, OperationTrace] = {}
        self._loaded_traces: dict[int, OperationTrace] = {}
        for rank in range(16):
            trace_root = resolve_rank_trace_root(self.root, rank)
            manifest = json.loads((trace_root / "trace.json").read_text())
            # load_operation_trace_bundle_manifest already verified both the
            # manifest digest and its binding to bundle.json.  This lightweight
            # object is sufficient for control/descriptor validation.
            self._control_traces[rank] = OperationTrace(
                root=trace_root, manifest=manifest, tensors={}
            )
        self.rank_control_validation = validate_official_rank_stage_schedule_semantics(
            self._control_traces
        )
        self.entries = tuple(self._build_entries())
        payload = {
            "schema_version": "distillschedule.routeb_full5_trace.v1",
            "contract": contract.name,
            "bundle_hash": self.bundle_hash,
            "entries": [entry.__dict__ for entry in self.entries],
        }
        self.binding_hash = hashlib.sha256(_canonical(payload)).hexdigest()
        self._verify_common_contract()

    @property
    def loaded_trace_count(self) -> int:
        return len(self._loaded_traces)

    def _load_trace(self, source_rank: int) -> OperationTrace:
        trace = self._loaded_traces.get(source_rank)
        if trace is None:
            trace = OperationTrace.load(
                resolve_rank_trace_root(self.root, source_rank), verify=True
            )
            expected = self._control_traces[source_rank].trace_hash
            if trace.trace_hash != expected:
                raise ValueError(
                    f"rank {source_rank} payload changed after manifest preflight"
                )
            self._loaded_traces[source_rank] = trace
        return trace

    def _verify_common_contract(self) -> None:
        for source_rank, trace in self._control_traces.items():
            contract = trace.manifest.get("contract", {})
            for key, expected in REQUIRED_CONTRACT.items():
                if contract.get(key) != expected:
                    raise ValueError(
                        f"rank {source_rank} operation trace violates {key}"
                    )
            for step in range(10, 15):
                branch = trace.operation(f"step/{step}/branch").get("control", {})
                expected_generator = step == 10
                expected = {
                    "global_step": step,
                    "dfake_gen_update_ratio": 5,
                    "train_generator": expected_generator,
                    "train_critic": True,
                    "gan": False,
                    "reward": False,
                    "vae_decode": False,
                }
                mismatch = {
                    key: (branch.get(key), value)
                    for key, value in expected.items()
                    if branch.get(key) != value
                }
                if mismatch:
                    raise ValueError(
                        f"rank {source_rank} Step{step} branch mismatch: {mismatch}"
                    )

    def _build_entries(self):
        slot = 0
        c = self.contract
        for update in range(c.logical_critic_updates):
            step = c.logical_step_for_update(update)
            for production_round in range(c.generator_production_rounds_per_update):
                for producer_pool_rank in range(c.generator_pool_width):
                    source_rank = c.source_trace_rank(
                        production_round, producer_pool_rank
                    )
                    trace = self._control_traces[source_rank]
                    keys = _critic_tensor_keys(step)
                    rollout = f"step/{step}/critic/rollout"
                    train = f"step/{step}/critic/train"
                    trace.operation(rollout)
                    trace.operation(train)
                    missing = set(keys.values()) - set(
                        trace.manifest.get("tensors", {})
                    )
                    if missing:
                        raise ValueError(
                            f"rank {source_rank} Step{step} misses {sorted(missing)}"
                        )
                    identity_payload = {
                        "contract": c.name,
                        "logical_sample_slot": slot,
                        "update": update,
                        "production_round": production_round,
                        "producer_pool_rank": producer_pool_rank,
                        "source_trace_rank": source_rank,
                        "logical_step": step,
                        "operation_trace_hash": trace.trace_hash,
                        "rollout_operation": rollout,
                        "train_operation": train,
                        "tensor_hashes": (
                            _descriptor_identity(trace, keys)
                        ),
                    }
                    identity = hashlib.sha256(_canonical(identity_payload)).hexdigest()
                    yield RouteBTraceEntry(
                        trace_key=(
                            f"step/{step}/round/{production_round}/"
                            f"producer/{producer_pool_rank}/source/{source_rank}"
                        ),
                        logical_sample_slot=slot,
                        update=update,
                        production_round=production_round,
                        producer_pool_rank=producer_pool_rank,
                        source_trace_rank=source_rank,
                        logical_step=step,
                        rng_stream=(
                            f"routeb/{c.name}/step/{step}/source/{source_rank}"
                        ),
                        trace_path=str(
                            resolve_rank_trace_root(self.root, source_rank)
                            .relative_to(self.root)
                        ),
                        operation_trace_hash=trace.trace_hash,
                        rollout_operation=rollout,
                        train_operation=train,
                        tensor_keys=keys,
                        identity_sha256=identity,
                    )
                    slot += 1

    def entry(
        self, update: int, production_round: int, producer_pool_rank: int
    ) -> RouteBTraceEntry:
        for entry in self.entries:
            if (
                entry.update,
                entry.production_round,
                entry.producer_pool_rank,
            ) == (update, production_round, producer_pool_rank):
                return entry
        raise KeyError((update, production_round, producer_pool_rank))

    def load_critic_entry(
        self,
        *,
        update: int,
        production_round: int,
        producer_pool_rank: int,
        device: torch.device | str,
    ) -> LoadedPoolEntry:
        entry = self.entry(update, production_round, producer_pool_rank)
        trace = self._load_trace(entry.source_trace_rank)
        tensors = {
            role: trace.tensor(key, clone=True).to(device)
            for role, key in entry.tensor_keys.items()
        }
        if tensors["text_embedding"].ndim == 2:
            tensors["text_embedding"] = tensors["text_embedding"].unsqueeze(0)
        return LoadedPoolEntry(entry, trace, tensors)

    def step10a_collective_plan(
        self, *, production_round: int
    ) -> RouteBStep10ACollectivePlan:
        if production_round not in (0, 1):
            raise IndexError(production_round)
        source_ids = tuple(
            range(production_round * 8, (production_round + 1) * 8)
        )
        exits: list[tuple[int, ...]] = []
        call_counts: list[int] = []
        for source_id in source_ids:
            control = self._control_traces[source_id].operation(
                "step/10/generator/rollout"
            )["control"]
            schedule = control["pyramid_stage_call_sequence"]
            stage_exits = tuple(int(item["denoising_exit"]) for item in schedule)
            if len(stage_exits) != 3:
                raise ValueError("Step10-A collective plan requires three stages")
            exits.append(stage_exits)
            call_count = int(control["call_count"])
            if call_count != sum(value + 1 for value in stage_exits):
                raise ValueError("Step10-A logical call count is inconsistent")
            call_counts.append(call_count)
        if len(set(exits)) != 1:
            raise ValueError(
                "Step10-A strict normal-path trace requires identical stage exits "
                "on every DP8 rank; collective padding is not part of the baseline"
            )
        if len(set(call_counts)) != 1:
            raise ValueError(
                "Step10-A strict normal-path trace requires identical total calls "
                "on every DP8 rank"
            )
        synchronized = exits[0]
        return RouteBStep10ACollectivePlan(
            production_round=production_round,
            source_ids=source_ids,
            logical_stage_exits=tuple(exits),
            logical_call_counts=tuple(call_counts),
            synchronized_stage_exits=synchronized,
        )

    def load_step10a_entry(
        self, *, source_id: int, device: torch.device | str
    ) -> RouteBStep10AEntry:
        if not 0 <= source_id < 16:
            raise IndexError(source_id)
        trace = self._load_trace(source_id)
        step = 10
        keys = _generator_tensor_keys(step)
        rollout = f"step/{step}/generator/rollout"
        dmd = f"step/{step}/generator/dmd"
        trace.operation(rollout)
        trace.operation(dmd)
        missing = set(keys.values()) - set(trace.manifest.get("tensors", {}))
        if missing:
            raise ValueError(f"rank {source_id} Step10-A misses {sorted(missing)}")
        tensors = {
            role: trace.tensor(key, clone=True).to(device)
            for role, key in keys.items()
        }
        if tensors["text_embedding"].ndim == 2:
            tensors["text_embedding"] = tensors["text_embedding"].unsqueeze(0)
        identity = hashlib.sha256(
            _canonical(
                {
                    "contract": self.contract.name,
                    "source_id": source_id,
                    "logical_step": step,
                    "operation_trace_hash": trace.trace_hash,
                    "rollout_operation": rollout,
                    "dmd_operation": dmd,
                    "tensor_hashes": _descriptor_identity(trace, keys),
                }
            )
        ).hexdigest()
        return RouteBStep10AEntry(
            bundle_root=self.root,
            source_id=source_id,
            logical_step=step,
            operation_trace=trace,
            tensors=tensors,
            identity_sha256=identity,
        )

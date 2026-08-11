"""Rank-qualified operation trace for the real-14B Critic-only pool.

The existing operation trace remains the tensor authority.  This module adds a
    small manifest that binds Critic-only logical steps to Generator producer
    ranks.  A pool entry
references immutable tensors by operation key; replay never samples RNG.

The pool trace hash is intentionally distinct from the checkpoint provenance
trace hash.  The former identifies the 16 inputs used by this isolated
subgraph, while the latter records the trace associated with the imported C11
logical state.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .operation_trace import (
    OperationTrace,
    tensor_sha256,
    validate_official_rank_stage_schedule_semantics,
)


CRITIC_POOL_TRACE_SCHEMA = "distillschedule.critic_pool_trace.v4"
REQUIRED_CONTRACT = {
    "resolution": [384, 640],
    "stage_spatial_shapes": [[12, 20], [24, 40], [48, 80]],
    "full_latent_shape": [1, 16, 9, 48, 80],
    "dtype": "bfloat16",
    "dfake_gen_update_ratio": 5,
}
RANK_CONTROL_CONTRACT = {
    "ordinary_stage_exits": "global_rank0_broadcast",
    "rank_local_stage_exit_exception": "forbidden_by_experiment_contract",
    "per_rank_call_count": "identical_per_operation_and_step",
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _pool_hash(manifest: Mapping[str, Any]) -> str:
    unsigned = dict(manifest)
    unsigned.pop("pool_trace_hash", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _entry_tensor_keys(step: int) -> dict[str, str]:
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


@dataclass(frozen=True)
class CriticPoolTraceEntry:
    trace_key: str
    logical_sample_slot: int
    wave: int
    producer_pool_rank: int
    logical_step: int
    rng_stream: str
    trace_path: str
    operation_trace_hash: str
    rollout_operation: str
    train_operation: str
    tensor_keys: dict[str, str]
    identity_sha256: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CriticPoolTraceEntry":
        return cls(
            trace_key=str(value["trace_key"]),
            logical_sample_slot=int(value["logical_sample_slot"]),
            wave=int(value["wave"]),
            producer_pool_rank=int(value["producer_pool_rank"]),
            logical_step=int(value["logical_step"]),
            rng_stream=str(value["rng_stream"]),
            trace_path=str(value["trace_path"]),
            operation_trace_hash=str(value["operation_trace_hash"]),
            rollout_operation=str(value["rollout_operation"]),
            train_operation=str(value["train_operation"]),
            tensor_keys={
                str(key): str(path)
                for key, path in dict(value["tensor_keys"]).items()
            },
            identity_sha256=str(value["identity_sha256"]),
        )


def _entry_identity(
    entry: Mapping[str, Any], trace: OperationTrace
) -> str:
    tensor_hashes = {
        role: tensor_sha256(trace.tensor(key, clone=False))
        for role, key in dict(entry["tensor_keys"]).items()
    }
    identity = {
        "trace_key": entry["trace_key"],
        "logical_sample_slot": entry["logical_sample_slot"],
        "wave": entry["wave"],
        "producer_pool_rank": entry["producer_pool_rank"],
        "logical_step": entry["logical_step"],
        "rng_stream": entry["rng_stream"],
        "operation_trace_hash": entry["operation_trace_hash"],
        "rollout_operation": entry["rollout_operation"],
        "train_operation": entry["train_operation"],
        "tensor_hashes": tensor_hashes,
    }
    return hashlib.sha256(_canonical_json(identity)).hexdigest()


@dataclass(frozen=True)
class CriticPoolTrace:
    root: Path
    manifest: dict[str, Any]
    entries: tuple[CriticPoolTraceEntry, ...]

    @property
    def pool_trace_hash(self) -> str:
        return str(self.manifest["pool_trace_hash"])

    @property
    def state_provenance_trace_hash(self) -> str:
        return str(self.manifest["state_provenance_trace_hash"])

    @classmethod
    def load(cls, root: str | Path, *, verify: bool = True) -> "CriticPoolTrace":
        root = Path(root)
        manifest = json.loads((root / "critic_pool_trace.json").read_text())
        entries = tuple(
            CriticPoolTraceEntry.from_dict(value)
            for value in manifest.get("entries", [])
        )
        result = cls(root=root, manifest=manifest, entries=entries)
        if verify:
            result.verify()
        return result

    def verify(self) -> None:
        manifest = self.manifest
        if manifest.get("schema_version") != CRITIC_POOL_TRACE_SCHEMA:
            raise ValueError("unsupported Critic pool trace schema")
        if manifest.get("pool_trace_hash") != _pool_hash(manifest):
            raise ValueError("Critic pool trace hash mismatch")
        if manifest.get("execution_scope") != "isolated_critic_only_subgraph":
            raise ValueError("pool trace must identify the isolated subgraph")
        if not isinstance(manifest.get("state_provenance_trace_hash"), str):
            raise ValueError("state provenance trace hash is missing")
        if manifest["state_provenance_trace_hash"] == manifest["pool_trace_hash"]:
            raise ValueError("state provenance and execution trace identities are conflated")
        contract = manifest.get("contract")
        if not isinstance(contract, dict):
            raise ValueError("pool trace contract is missing")
        for key, expected in REQUIRED_CONTRACT.items():
            if contract.get(key) != expected:
                raise ValueError(f"pool trace violates official contract field {key}")
        if manifest.get("rank_control_contract") != RANK_CONTROL_CONTRACT:
            raise ValueError("pool trace rank-control contract is missing or changed")

        producer_count = int(manifest.get("producer_count", 4))
        waves = int(manifest.get("waves", 4))
        start_step = int(manifest.get("start_step", 11))
        if producer_count <= 0 or waves <= 0 or start_step < 0:
            raise ValueError("invalid Critic pool trace dimensions")
        expected_entries = producer_count * waves
        if len(self.entries) != expected_entries:
            raise ValueError(
                "Critic pool trace entry count changed: "
                f"{len(self.entries)} != {waves} waves x {producer_count} ranks"
            )
        fields = {
            "trace_key": [entry.trace_key for entry in self.entries],
            "logical_sample_slot": [
                entry.logical_sample_slot for entry in self.entries
            ],
            "rng_stream": [entry.rng_stream for entry in self.entries],
            "identity_sha256": [entry.identity_sha256 for entry in self.entries],
        }
        for field, values in fields.items():
            if len(set(values)) != expected_entries:
                raise ValueError(f"pool trace {field} values must be unique")
        if sorted(fields["logical_sample_slot"]) != list(range(expected_entries)):
            raise ValueError(
                f"logical sample slots must cover [0, {expected_entries})"
            )

        expected_pairs = {
            (wave, rank)
            for wave in range(waves)
            for rank in range(producer_count)
        }
        actual_pairs = {
            (entry.wave, entry.producer_pool_rank) for entry in self.entries
        }
        if actual_pairs != expected_pairs:
            raise ValueError("pool trace does not cover every wave/rank pair")

        rank_traces: dict[int, OperationTrace] = {}
        for entry in self.entries:
            if entry.logical_step != start_step + entry.wave:
                raise ValueError(
                    "Critic pool trace logical step does not match start_step + wave"
                )
            trace_root = (self.root / entry.trace_path).resolve()
            trace = OperationTrace.load(trace_root)
            rank_traces.setdefault(entry.producer_pool_rank, trace)
            if trace.trace_hash != entry.operation_trace_hash:
                raise ValueError("referenced operation trace hash mismatch")
            for operation_key in (entry.rollout_operation, entry.train_operation):
                trace.operation(operation_key)
            if set(entry.tensor_keys) != set(_entry_tensor_keys(entry.logical_step)):
                raise ValueError("pool trace tensor-role inventory is incomplete")
            missing = set(entry.tensor_keys.values()) - set(trace.tensors)
            if missing:
                raise ValueError(f"pool trace references missing tensors: {sorted(missing)}")
            raw_entry = {
                "trace_key": entry.trace_key,
                "logical_sample_slot": entry.logical_sample_slot,
                "wave": entry.wave,
                "producer_pool_rank": entry.producer_pool_rank,
                "logical_step": entry.logical_step,
                "rng_stream": entry.rng_stream,
                "operation_trace_hash": entry.operation_trace_hash,
                "rollout_operation": entry.rollout_operation,
                "train_operation": entry.train_operation,
                "tensor_keys": entry.tensor_keys,
            }
            if _entry_identity(raw_entry, trace) != entry.identity_sha256:
                raise ValueError("pool trace entry identity mismatch")
        rank_control_validation = validate_official_rank_stage_schedule_semantics(
            rank_traces
        )
        if manifest.get("rank_control_validation") != rank_control_validation:
            raise ValueError("pool trace rank-control validation summary mismatch")

    def entry(self, wave: int, producer_pool_rank: int) -> CriticPoolTraceEntry:
        for entry in self.entries:
            if (
                entry.wave == wave
                and entry.producer_pool_rank == producer_pool_rank
            ):
                return entry
        raise KeyError((wave, producer_pool_rank))

    def replay(
        self, wave: int, producer_pool_rank: int
    ) -> dict[str, torch.Tensor]:
        entry = self.entry(wave, producer_pool_rank)
        trace = OperationTrace.load((self.root / entry.trace_path).resolve())
        return {
            role: trace.tensor(key, clone=True)
            for role, key in entry.tensor_keys.items()
        }


def write_critic_pool_trace(
    root: str | Path,
    *,
    rank_trace_paths: Mapping[int, str | Path],
    state_provenance_trace_hash: str,
    waves: int = 4,
    start_step: int = 11,
) -> CriticPoolTrace:
    """Bind rank-qualified traces covering a contiguous Critic-only interval."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    producer_count = len(rank_trace_paths)
    if producer_count <= 0 or set(rank_trace_paths) != set(range(producer_count)):
        raise ValueError(
            "rank_trace_paths must contain contiguous producer pool ranks"
        )
    if waves <= 0 or start_step < 0:
        raise ValueError("waves must be positive and start_step non-negative")
    if len(state_provenance_trace_hash) != 64:
        raise ValueError("state provenance trace hash must be SHA-256")

    loaded_rank_traces: dict[int, OperationTrace] = {}
    for rank, trace_path_value in rank_trace_paths.items():
        trace_path = Path(trace_path_value)
        absolute = trace_path if trace_path.is_absolute() else root / trace_path
        loaded_rank_traces[rank] = OperationTrace.load(absolute.resolve())
    rank_control_validation = validate_official_rank_stage_schedule_semantics(
        loaded_rank_traces
    )

    entries: list[dict[str, Any]] = []
    for wave in range(waves):
        step = start_step + wave
        for rank in range(producer_count):
            trace_path = Path(rank_trace_paths[rank])
            absolute = trace_path if trace_path.is_absolute() else root / trace_path
            trace = loaded_rank_traces[rank]
            operation_prefix = f"step/{step}/critic"
            raw: dict[str, Any] = {
                "trace_key": f"cycle/0/wave/{wave}/producer/{rank}",
                "logical_sample_slot": wave * producer_count + rank,
                "wave": wave,
                "producer_pool_rank": rank,
                "logical_step": step,
                "rng_stream": f"critic_pool_v3/wave/{wave}/rank/{rank}",
                "trace_path": str(trace_path),
                "operation_trace_hash": trace.trace_hash,
                "rollout_operation": f"{operation_prefix}/rollout",
                "train_operation": f"{operation_prefix}/train",
                "tensor_keys": _entry_tensor_keys(step),
            }
            raw["identity_sha256"] = _entry_identity(raw, trace)
            entries.append(raw)

    manifest: dict[str, Any] = {
        "schema_version": CRITIC_POOL_TRACE_SCHEMA,
        "execution_scope": "isolated_critic_only_subgraph",
        "state_provenance_trace_hash": state_provenance_trace_hash,
        "producer_count": producer_count,
        "waves": waves,
        "start_step": start_step,
        "contract": dict(REQUIRED_CONTRACT),
        "rank_control_contract": dict(RANK_CONTROL_CONTRACT),
        "rank_control_validation": rank_control_validation,
        "physical_feature_policy": (
            "SYN-FEATURE files may be reused; logical identities and RNG streams "
            "are rank-qualified and must not be described as distinct real videos"
        ),
        "entries": entries,
    }
    manifest["pool_trace_hash"] = _pool_hash(manifest)
    (root / "critic_pool_trace.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return CriticPoolTrace.load(root)

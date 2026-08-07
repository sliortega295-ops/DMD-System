"""Operation-keyed tensor trace bundles for reproducible DistillSchedule runs.

The bundle stores sampled tensors, control decisions, layout metadata and exact
content hashes.  Replay never invokes a random-number generator: it loads values
by semantic operation key and verifies the full bundle before returning data.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

SCHEMA_VERSION = "distillschedule.operation_trace.v1"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash logical values plus representation metadata.

    Logical bytes are canonicalized through a contiguous CPU copy while shape,
    dtype, stride, storage offset and contiguity remain part of the digest. This
    detects both value drift and layout drift without retaining source storage.
    """
    value = tensor.detach().cpu()
    metadata = {
        "shape": list(value.shape),
        "dtype": _dtype_name(value.dtype),
        "stride": list(value.stride()),
        "storage_offset": int(value.storage_offset()),
        "contiguous": bool(value.is_contiguous()),
    }
    logical = value.contiguous().view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(_canonical_json(metadata))
    digest.update(b"\0")
    digest.update(logical)
    return digest.hexdigest()


def tensor_descriptor(tensor: torch.Tensor, *, role: str, packing: str) -> dict[str, Any]:
    value = tensor.detach()
    element_size = value.element_size()
    return {
        "role": role,
        "shape": list(value.shape),
        "dtype": _dtype_name(value.dtype),
        "layout": str(value.layout).removeprefix("torch."),
        "stride": list(value.stride()),
        "storage_offset": int(value.storage_offset()),
        "contiguous": bool(value.is_contiguous()),
        "zero_stride": any(int(item) == 0 for item in value.stride()),
        "logical_bytes": int(value.numel() * element_size),
        "storage_bytes": int(value.untyped_storage().nbytes()),
        "packing": packing,
        "sha256": tensor_sha256(value),
    }


def _manifest_hash(manifest: Mapping[str, Any]) -> str:
    unsigned = dict(manifest)
    unsigned.pop("trace_hash", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


BUNDLE_SCHEMA_VERSION = "distillschedule.operation_trace_bundle.v1"

GLOBAL_STAGE_SCHEDULE_SCOPE = "global_rank0_broadcast_equivalent"
LOCAL_ZERO_HISTORY_STAGE_SCHEDULE_SCOPE = (
    "rank_local_amplified_gt_zero_history_exception"
)


def _bundle_manifest_hash(manifest: Mapping[str, Any]) -> str:
    unsigned = dict(manifest)
    unsigned.pop("bundle_hash", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def load_operation_trace_bundle_manifest(
    root: str | Path, *, verify: bool = True
) -> dict[str, Any] | None:
    """Load a rank-qualified trace bundle, or return None for a single trace."""

    root = Path(root)
    if (root / "trace.json").is_file():
        return None
    bundle_path = root / "bundle.json"
    if not bundle_path.is_file():
        raise FileNotFoundError(f"no trace.json or bundle.json under {root}")
    manifest = json.loads(bundle_path.read_text())
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"unsupported operation trace bundle schema {manifest.get('schema_version')!r}")
    entries = manifest.get("ranks")
    if not isinstance(entries, list) or not entries:
        raise ValueError("operation trace bundle must contain rank entries")
    ranks = [entry.get("rank") for entry in entries]
    if ranks != list(range(len(entries))):
        raise ValueError(f"operation trace bundle ranks must be contiguous from zero: {ranks}")
    if verify and manifest.get("bundle_hash") != _bundle_manifest_hash(manifest):
        raise ValueError("operation trace bundle hash mismatch")
    for entry in entries:
        trace_root = root / str(entry.get("path"))
        trace_manifest_path = trace_root / "trace.json"
        if not trace_manifest_path.is_file():
            raise FileNotFoundError(f"rank {entry['rank']} trace manifest is missing: {trace_manifest_path}")
        trace_manifest = json.loads(trace_manifest_path.read_text())
        if verify and trace_manifest.get("trace_hash") != _manifest_hash(trace_manifest):
            raise ValueError(f"rank {entry['rank']} trace manifest hash mismatch")
        if trace_manifest.get("trace_hash") != entry.get("trace_hash"):
            raise ValueError(f"rank {entry['rank']} trace hash does not match bundle manifest")
    return manifest


def resolve_rank_trace_root(root: str | Path, rank: int) -> Path:
    """Resolve either a legacy single trace or a rank entry in a trace bundle."""

    root = Path(root)
    manifest = load_operation_trace_bundle_manifest(root)
    if manifest is None:
        return root
    entries = manifest["ranks"]
    if rank < 0 or rank >= len(entries):
        raise ValueError(f"rank {rank} is outside operation trace bundle size {len(entries)}")
    return root / str(entries[rank]["path"])


def write_operation_trace_bundle_manifest(
    root: str | Path, entries: list[Mapping[str, Any]]
) -> dict[str, Any]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "ranks": [dict(entry) for entry in entries],
    }
    manifest["bundle_hash"] = _bundle_manifest_hash(manifest)
    (root / "bundle.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    load_operation_trace_bundle_manifest(root)
    return manifest


def validate_official_rank_stage_schedule_semantics(
    rank_traces: Mapping[int, "OperationTrace"],
) -> dict[str, Any]:
    """Fail closed unless frozen rollout controls match official Helios.

    Ordinary stage exits are sampled once by global rank zero and broadcast.
    Rank-local variation is legal only in the exact amplified, GT-history,
    all-zero-history branch where the official code passes ``is_sync=False``.
    """

    if not rank_traces:
        raise ValueError("rank stage-schedule validation requires traces")
    ranks = sorted(rank_traces)
    if ranks != list(range(len(ranks))):
        raise ValueError(f"rank traces must be contiguous from zero: {ranks}")

    def schedule_operations(trace: "OperationTrace") -> dict[str, dict[str, Any]]:
        return {
            str(operation["key"]): operation
            for operation in trace.manifest.get("operations", [])
            if isinstance(operation.get("control", {}).get("pyramid_stage_call_sequence"), list)
        }

    per_rank = {rank: schedule_operations(trace) for rank, trace in rank_traces.items()}
    reference_keys = set(per_rank[0])
    if not reference_keys:
        raise ValueError("rank traces contain no frozen pyramid stage schedules")
    for rank in ranks[1:]:
        if set(per_rank[rank]) != reference_keys:
            raise ValueError(
                f"rank {rank} frozen rollout operation keys differ from rank 0"
            )

    local_exception_entries = 0
    global_entries = 0
    operations: list[dict[str, Any]] = []
    for operation_key in sorted(reference_keys):
        controls = {
            rank: dict(per_rank[rank][operation_key].get("control", {}))
            for rank in ranks
        }
        initial_by_rank: dict[int, tuple[int, ...]] = {}
        executed_by_rank: dict[int, tuple[int, ...]] = {}
        for rank, control in controls.items():
            initial = control.get("initial_synchronized_pyramid_stage_exit")
            schedule = control.get("pyramid_stage_call_sequence")
            if not isinstance(initial, list) or not initial:
                raise ValueError(
                    f"{operation_key} rank {rank} lacks synchronized initial exits"
                )
            if not isinstance(schedule, list) or len(schedule) != len(initial):
                raise ValueError(
                    f"{operation_key} rank {rank} has malformed stage schedule"
                )
            initial_by_rank[rank] = tuple(int(value) for value in initial)
            executed_by_rank[rank] = tuple(
                int(item["denoising_exit"]) for item in schedule
            )

        if len(set(initial_by_rank.values())) != 1:
            raise ValueError(
                f"{operation_key} violates rank-zero broadcast semantics: "
                f"initial exits differ across ranks"
            )
        synchronized_initial = initial_by_rank[0]
        normal_exits: set[tuple[int, ...]] = set()
        local_ranks: list[int] = []
        for rank, control in controls.items():
            scope = control.get("stage_schedule_scope")
            predicate = control.get("official_local_exception_predicate")
            if scope == GLOBAL_STAGE_SCHEDULE_SCOPE:
                global_entries += 1
                if executed_by_rank[rank] != synchronized_initial:
                    raise ValueError(
                        f"{operation_key} rank {rank} marks a broadcast schedule "
                        "but executes different exits"
                    )
                normal_exits.add(executed_by_rank[rank])
            elif scope == LOCAL_ZERO_HISTORY_STAGE_SCHEDULE_SCOPE:
                local_exception_entries += 1
                local_ranks.append(rank)
                if not isinstance(predicate, dict) or not (
                    predicate.get("is_use_gt_history") is True
                    and predicate.get("is_amplify_first_chunk") is True
                    and predicate.get("all_history_zero") is True
                ):
                    raise ValueError(
                        f"{operation_key} rank {rank} claims an illegal local "
                        "stage-schedule exception"
                    )
                if control.get("local_exception_seed") is None:
                    raise ValueError(
                        f"{operation_key} rank {rank} local exception seed is missing"
                    )
            else:
                raise ValueError(
                    f"{operation_key} rank {rank} has unsupported stage schedule "
                    f"scope {scope!r}"
                )
        if len(normal_exits) > 1:
            raise ValueError(
                f"{operation_key} non-exception ranks execute different exits"
            )
        operations.append(
            {
                "operation": operation_key,
                "synchronized_initial_exit": list(synchronized_initial),
                "global_rank_count": len(ranks) - len(local_ranks),
                "local_exception_ranks": local_ranks,
            }
        )

    return {
        "status": "PASS",
        "semantics": "official_helios_rank0_broadcast_with_gt_zero_history_exception",
        "rank_count": len(ranks),
        "operation_count": len(operations),
        "global_entry_count": global_entries,
        "local_exception_entry_count": local_exception_entries,
        "operations": operations,
    }


def validate_logical_step_coverage(
    manifest: Mapping[str, Any], *, start: int, stop: int
) -> tuple[int, ...]:
    """Require every logical training step before an expensive real-model launch."""

    if start < 0 or stop <= start:
        raise ValueError(f"invalid required logical step range [{start}, {stop})")
    operations = manifest.get("operations")
    if not isinstance(operations, list):
        raise ValueError("operation trace manifest has no operation list")
    observed = sorted(
        int(item.get("control", {}).get("global_step"))
        for item in operations
        if item.get("operation") == "stage3_branch"
        and isinstance(item.get("control", {}).get("global_step"), int)
    )
    required = list(range(start, stop))
    missing = sorted(set(required) - set(observed))
    if missing:
        available = [observed[0], observed[-1]] if observed else None
        raise ValueError(
            f"operation trace does not cover required logical steps {missing}; "
            f"available_range={available}"
        )
    return tuple(required)


STAGE3_SOURCE_KEYS = {
    "primary_x0": "sample/0/primary/x0",
    "primary_history": "sample/0/primary/history",
    "primary_target": "sample/0/primary/target",
    "secondary_x0": "sample/0/secondary/x0",
    "secondary_history": "sample/0/secondary/history",
    "secondary_target": "sample/0/secondary/target",
    "text_embedding": "sample/0/text_embedding",
}


def replay_stage3_source(
    trace: "OperationTrace", *, device: torch.device | str, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Materialize authoritative Stage-3 source tensors without consuming RNG.

    The StatefulDataLoader is still advanced for the official control flow, but
    its sampled payload must not replace the operation-keyed source after a
    checkpoint resume changes sampler position.  ``copy=True`` ensures callers
    cannot mutate the immutable trace even in a CPU-only correctness test.
    """

    contract = trace.manifest.get("contract", {})
    expected_target = tuple(contract.get("full_latent_shape", [])[1:])
    expected_text = tuple(contract.get("text_shape", [])[1:])
    values: dict[str, torch.Tensor] = {}
    for name, key in STAGE3_SOURCE_KEYS.items():
        source = trace.tensor(key, clone=False)
        target_dtype = dtype if source.is_floating_point() else source.dtype
        values[name] = source.to(device=device, dtype=target_dtype, copy=True).unsqueeze(0)

    if expected_target and tuple(values["primary_target"].shape[1:]) != expected_target:
        raise ValueError("primary operation-trace target violates the Stage-3 latent contract")
    if expected_target and tuple(values["secondary_target"].shape[1:]) != expected_target:
        raise ValueError("secondary operation-trace target violates the Stage-3 latent contract")
    if expected_text and tuple(values["text_embedding"].shape[1:]) != expected_text:
        raise ValueError("operation-trace text embedding violates the Stage-3 text contract")
    return values


@dataclass(frozen=True)
class OperationTrace:
    root: Path
    manifest: dict[str, Any]
    tensors: dict[str, torch.Tensor]

    @property
    def trace_hash(self) -> str:
        return str(self.manifest["trace_hash"])

    @classmethod
    def load(cls, root: str | Path, *, verify: bool = True) -> "OperationTrace":
        root = Path(root)
        manifest = json.loads((root / "trace.json").read_text())
        tensors = torch.load(root / "tensors.pt", map_location="cpu", weights_only=False)
        if not isinstance(tensors, dict) or not all(isinstance(k, str) for k in tensors):
            raise TypeError("tensors.pt must contain a string-keyed tensor dictionary")
        result = cls(root=root, manifest=manifest, tensors=tensors)
        if verify:
            result.verify()
        return result

    def verify(self) -> None:
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported operation trace schema {self.manifest.get('schema_version')!r}")
        expected_hash = _manifest_hash(self.manifest)
        if self.trace_hash != expected_hash:
            raise ValueError(f"trace hash mismatch: {self.trace_hash} != {expected_hash}")
        descriptors = self.manifest.get("tensors")
        if not isinstance(descriptors, dict) or set(descriptors) != set(self.tensors):
            raise ValueError("manifest and tensor payload keys differ")
        for key, tensor in self.tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"trace payload {key!r} is not a Tensor")
            descriptor = descriptors[key]
            actual = tensor_descriptor(
                tensor,
                role=str(descriptor["role"]),
                packing=str(descriptor["packing"]),
            )
            if actual != descriptor:
                raise ValueError(f"tensor descriptor mismatch for {key}: {actual} != {descriptor}")
        operations = self.manifest.get("operations")
        if not isinstance(operations, list):
            raise ValueError("operations must be a list")
        keys = [entry.get("key") for entry in operations]
        if len(keys) != len(set(keys)) or not all(isinstance(key, str) and key for key in keys):
            raise ValueError("operation keys must be unique non-empty strings")
        for operation in operations:
            for tensor_key in operation.get("tensor_keys", []):
                if tensor_key not in self.tensors:
                    raise ValueError(f"operation {operation['key']} references missing tensor {tensor_key}")

    def tensor(self, key: str, *, clone: bool = True) -> torch.Tensor:
        value = self.tensors[key]
        return value.clone(memory_format=torch.preserve_format) if clone else value

    def operation(self, key: str) -> dict[str, Any]:
        for operation in self.manifest["operations"]:
            if operation["key"] == key:
                return dict(operation)
        raise KeyError(key)


class OperationTraceWriter:
    def __init__(self, *, classification: str, source: Mapping[str, Any]) -> None:
        self.classification = classification
        self.source = dict(source)
        self._tensors: dict[str, torch.Tensor] = {}
        self._descriptors: dict[str, dict[str, Any]] = {}
        self._operations: list[dict[str, Any]] = []

    def add_tensor(
        self,
        key: str,
        tensor: torch.Tensor,
        *,
        role: str,
        packing: str = "logical",
    ) -> str:
        if not key or key in self._tensors:
            raise ValueError(f"duplicate or empty tensor key {key!r}")
        value = tensor.detach().cpu().clone(memory_format=torch.preserve_format)
        self._tensors[key] = value
        self._descriptors[key] = tensor_descriptor(value, role=role, packing=packing)
        return key

    def add_operation(
        self,
        key: str,
        *,
        component: str,
        operation: str,
        tensor_keys: list[str] | tuple[str, ...] = (),
        control: Mapping[str, Any] | None = None,
    ) -> None:
        if not key or any(item["key"] == key for item in self._operations):
            raise ValueError(f"duplicate or empty operation key {key!r}")
        missing = set(tensor_keys) - set(self._tensors)
        if missing:
            raise KeyError(f"operation {key} references tensors not yet registered: {sorted(missing)}")
        self._operations.append(
            {
                "key": key,
                "component": component,
                "operation": operation,
                "tensor_keys": list(tensor_keys),
                "control": dict(control or {}),
            }
        )

    def write(self, root: str | Path, *, contract: Mapping[str, Any]) -> OperationTrace:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        payload_path = root / "tensors.pt"
        torch.save(self._tensors, payload_path)
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "classification": self.classification,
            "source": self.source,
            "contract": dict(contract),
            "payload": {"path": payload_path.name, "format": "torch.save"},
            "tensors": self._descriptors,
            "operations": self._operations,
        }
        manifest["trace_hash"] = _manifest_hash(manifest)
        (root / "trace.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return OperationTrace.load(root)

"""Rank-qualified Step15-A trace binding for the target G/S/T pool."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .gst_pool_contract import GLOBAL_BATCH, LOGICAL_STEP, source_identity
from .operation_trace import (
    OperationTrace,
    load_operation_trace_bundle_manifest,
    resolve_rank_trace_root,
    validate_official_rank_stage_schedule_semantics,
)


EXPECTED_TRACE_CONTRACT = {
    "resolution": [384, 640],
    "stage_spatial_shapes": [[12, 20], [24, 40], [48, 80]],
    "full_latent_shape": [1, 16, 9, 48, 80],
    "text_shape": [1, 512, 4096],
    "dtype": "bfloat16",
    "dfake_gen_update_ratio": 5,
}


def _keys(step: int = LOGICAL_STEP) -> dict[str, str]:
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


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


@dataclass(frozen=True)
class GSTTraceEntry:
    bundle_root: Path
    source_id: int
    identity: dict[str, int]
    operation_trace: OperationTrace
    tensors: dict[str, torch.Tensor]
    identity_sha256: str

    @property
    def rollout_operation(self) -> str:
        return f"step/{LOGICAL_STEP}/generator/rollout"

    @property
    def dmd_operation(self) -> str:
        return f"step/{LOGICAL_STEP}/generator/dmd"


def validate_gst_trace_bundle(root: str | Path) -> dict[str, Any]:
    """Validate all control manifests without materializing all tensor payloads."""

    root = Path(root).resolve()
    bundle = load_operation_trace_bundle_manifest(root, verify=True)
    if bundle is None:
        raise ValueError("target G/S/T trace must be a rank-qualified bundle")
    entries = bundle["ranks"]
    if len(entries) != GLOBAL_BATCH:
        raise ValueError(f"target trace needs {GLOBAL_BATCH} rank entries")
    traces: dict[int, OperationTrace] = {}
    for source_id in range(GLOBAL_BATCH):
        trace_root = resolve_rank_trace_root(root, source_id)
        manifest = json.loads((trace_root / "trace.json").read_text())
        trace = OperationTrace(root=trace_root, manifest=manifest, tensors={})
        traces[source_id] = trace
        contract = manifest.get("contract", {})
        for key, expected in EXPECTED_TRACE_CONTRACT.items():
            if contract.get(key) != expected:
                raise ValueError(f"rank {source_id} trace violates {key}")
        branch = trace.operation(f"step/{LOGICAL_STEP}/branch")
        expected_branch = {
            "global_step": LOGICAL_STEP,
            "dfake_gen_update_ratio": 5,
            "train_generator": True,
            "train_critic": True,
            "gan": False,
            "reward": False,
            "vae_decode": False,
        }
        control = branch.get("control", {})
        for key, expected in expected_branch.items():
            if control.get(key) != expected:
                raise ValueError(
                    f"rank {source_id} Step15 branch violates {key}"
                )
        rollout = trace.operation(f"step/{LOGICAL_STEP}/generator/rollout")
        dmd = trace.operation(f"step/{LOGICAL_STEP}/generator/dmd")
        if rollout.get("operation") != "selected_rollout":
            raise ValueError("Step15 Generator rollout is not selected")
        if rollout.get("control", {}).get("gradient_enabled") is not True:
            raise ValueError("Step15 Generator trace lost its grad selection")
        if dmd.get("operation") != "detached_dmd_gradient":
            raise ValueError("Step15 DMD boundary changed")
        missing = set(_keys().values()) - set(manifest.get("tensors", {}))
        if missing:
            raise ValueError(
                f"rank {source_id} Step15 trace misses tensors: {sorted(missing)}"
            )
    controls = validate_official_rank_stage_schedule_semantics(traces)
    return {
        "status": "PASS",
        "bundle_hash": str(bundle["bundle_hash"]),
        "rank_count": len(entries),
        "logical_step": LOGICAL_STEP,
        "rank_control_validation": controls,
    }


def load_gst_trace_entry(
    root: str | Path,
    *,
    source_id: int,
    device: torch.device | str,
) -> GSTTraceEntry:
    root = Path(root).resolve()
    identity = source_identity(source_id)
    trace = OperationTrace.load(resolve_rank_trace_root(root, source_id))
    branch = trace.operation(f"step/{LOGICAL_STEP}/branch").get("control", {})
    if branch.get("global_step") != LOGICAL_STEP or branch.get(
        "train_generator"
    ) is not True:
        raise ValueError("rank-qualified trace is not Step15-A")
    keys = _keys()
    tensors = {
        role: trace.tensor(key, clone=True).to(device)
        for role, key in keys.items()
    }
    if tensors["text_embedding"].ndim == 2:
        tensors["text_embedding"] = tensors["text_embedding"].unsqueeze(0)
    descriptor_hashes = {
        role: trace.manifest["tensors"][key]["sha256"]
        for role, key in keys.items()
    }
    digest = hashlib.sha256(
        _canonical(
            {
                **identity,
                "logical_step": LOGICAL_STEP,
                "operation_trace_hash": trace.trace_hash,
                "rollout_operation": f"step/{LOGICAL_STEP}/generator/rollout",
                "dmd_operation": f"step/{LOGICAL_STEP}/generator/dmd",
                "tensor_sha256": descriptor_hashes,
            }
        )
    ).hexdigest()
    return GSTTraceEntry(
        bundle_root=root,
        source_id=source_id,
        identity=identity,
        operation_trace=trace,
        tensors=tensors,
        identity_sha256=digest,
    )

"""Logical-state loading helpers for the Critic-only 4G+4S pool.

The pool intentionally has two different restore paths:

* Generator ranks materialize no optimizer.  They therefore load the logical
  FP32 Generator master tensors directly into the mixed-dtype resident model.
* Student ranks restore the already-retargeted native world4 ZeRO-1 package
  through DeepSpeed.

This module owns only the first path and target-package metadata validation.
It never converts a world16 shard during a measured run.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


LOGICAL_CHECKPOINT_SCHEMA = "distillschedule.logical_checkpoint.v3"
POOL_TARGET_SCHEMA = "distillschedule.critic_pool_target_package.v1"


def _sha256_file(path: Path, *, chunk_bytes: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PoolTargetPackage:
    """Immutable metadata shared by Sequential and Pipeline executors."""

    root: Path
    manifest: dict[str, Any]
    contract: dict[str, Any]

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        expected_execution_trace_hash: str | None = None,
    ) -> "PoolTargetPackage":
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        contract = json.loads(
            (root / "critic_pool_target_contract.json").read_text()
        )
        if manifest.get("schema_version") != LOGICAL_CHECKPOINT_SCHEMA:
            raise ValueError("unsupported target logical-checkpoint schema")
        if contract.get("schema_version") != POOL_TARGET_SCHEMA:
            raise ValueError("unsupported Critic-pool target contract")
        if contract.get("status") != "PASS":
            raise ValueError("target-layout package did not pass its import gate")
        if contract.get("target_manifest_hash") != manifest.get("manifest_hash"):
            raise ValueError("target contract and logical manifest disagree")
        if contract.get("execution_scope") != "isolated_critic_only_subgraph":
            raise ValueError("target package is not the isolated Critic-only subgraph")
        if expected_execution_trace_hash is not None and (
            contract.get("execution_pool_trace_hash")
            != expected_execution_trace_hash
        ):
            raise ValueError("target package is bound to a different execution trace")

        ema = contract.get("ema_contract", {})
        if ema.get("ema_execution") != "skipped_by_experiment_contract":
            raise ValueError("early-step EMA skip is not explicit")
        if ema.get("ema_input_hash") != ema.get("ema_output_hash"):
            raise ValueError("skipped EMA changed its state")
        if (
            int(ema.get("optimization_step_before", -1))
            != int(ema.get("optimization_step_after", -2))
        ):
            raise ValueError("skipped EMA advanced its optimization step")

        components = contract.get("components", {})
        generator = components.get("generator", {})
        critic = components.get("critic", {})
        if (
            int(generator.get("parameter_version", -1)) != 3
            or int(critic.get("parameter_version", -1)) != 11
        ):
            raise ValueError("target package is not G3/C11")
        return cls(root=root, manifest=manifest, contract=contract)

    def component(self, name: str) -> Mapping[str, Any]:
        try:
            component = self.manifest["components"][name]
        except KeyError as error:
            raise KeyError(f"missing logical component {name!r}") from error
        if not isinstance(component, Mapping):
            raise TypeError(f"logical component {name!r} is not a mapping")
        return component


@torch.no_grad()
def load_logical_master_into_trainables(
    model: torch.nn.Module,
    package: PoolTargetPackage,
    *,
    component: str,
    verify_payload_sha256: bool = False,
) -> dict[str, Any]:
    """Restore one logical FP32 master payload without constructing an optimizer.

    The manifest parameter inventory is authoritative.  Every named trainable
    parameter must match exactly; the loader neither accepts a missing LoRA nor
    silently initializes an extra trainable tensor.
    """

    component_manifest = package.component(component)
    payload = component_manifest["payloads"]["master"]
    if payload.get("dtype") != "torch.float32":
        raise ValueError("logical master payload must be FP32")
    payload_path = package.root / str(payload["path"])
    expected_bytes = int(payload["bytes"])
    if payload_path.stat().st_size != expected_bytes:
        raise ValueError("logical master payload byte size changed")
    if verify_payload_sha256 and _sha256_file(payload_path) != payload["sha256"]:
        raise ValueError("logical master payload SHA-256 changed")

    parameter_names = [str(name) for name in component_manifest["parameter_names"]]
    named_trainables = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(named_trainables) != set(parameter_names):
        missing = sorted(set(parameter_names) - set(named_trainables))
        unexpected = sorted(set(named_trainables) - set(parameter_names))
        raise ValueError(
            "logical trainable inventory mismatch; "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )

    raw = torch.from_file(
        str(payload_path),
        shared=False,
        size=expected_bytes // torch.tensor([], dtype=torch.float32).element_size(),
        dtype=torch.float32,
    )
    copied_numel = 0
    for name in parameter_names:
        record = component_manifest["records"][name]
        offset = int(record["offset_elements"])
        numel = int(record["numel"])
        shape = tuple(int(value) for value in record["shape"])
        source = raw.narrow(0, offset, numel).view(shape)
        target = named_trainables[name]
        if tuple(target.shape) != shape:
            raise ValueError(f"logical tensor shape mismatch for {name}")
        if target.dtype != torch.float32:
            target.data = target.data.float()
        target.copy_(source)
        if not torch.equal(target.detach(), source):
            raise AssertionError(f"logical master copy was not exact for {name}")
        copied_numel += numel
    if copied_numel != int(component_manifest["logical_numel"]):
        raise AssertionError("logical master copy did not cover the component")
    del raw

    return {
        "component": component,
        "parameter_version": int(component_manifest["parameter_version"]),
        "logical_numel": copied_numel,
        "tensor_count": len(parameter_names),
        "payload_path": str(payload_path),
        "payload_sha256": str(payload["sha256"]),
        "payload_sha256_verified_now": bool(verify_payload_sha256),
        "optimizer_materialized": False,
        "scheduler_materialized": False,
    }

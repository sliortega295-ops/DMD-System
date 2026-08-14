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
CANDIDATE_B_TRANSITION_SCHEMA = (
    "distillschedule.routeb_candidate_b_transition_logical.v1"
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


@dataclass(frozen=True)
class LogicalCheckpointPackage:
    """Layout-independent logical state used to seed the live Full5 run.

    Unlike :class:`PoolTargetPackage`, this is not the old C11 isolated-pool
    contract and therefore does not encode an EMA skip.  It validates the
    authoritative G2/C10/EMA10 state before any target-layout native shard is
    restored.
    """

    root: Path
    manifest: dict[str, Any]

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        expected_trace_hash: str | None = None,
        expected_global_step: int = 10,
    ) -> "LogicalCheckpointPackage":
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("schema_version") != LOGICAL_CHECKPOINT_SCHEMA:
            raise ValueError("unsupported Full5 logical-checkpoint schema")
        unsigned = dict(manifest)
        expected_hash = str(unsigned.pop("manifest_hash", ""))
        actual_hash = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError("Full5 logical-checkpoint manifest hash mismatch")
        if int(manifest.get("global_step", -1)) != expected_global_step:
            raise ValueError("Full5 logical checkpoint is not step 10")
        if expected_trace_hash is not None and (
            manifest.get("input_trace_hash") != expected_trace_hash
        ):
            raise ValueError("Full5 checkpoint and operation trace disagree")
        components = manifest.get("components", {})
        versions = {
            name: int(components.get(name, {}).get("parameter_version", -1))
            for name in ("generator", "critic")
        }
        if versions != {"generator": 2, "critic": 10}:
            raise ValueError(f"Full5 logical state is not G2/C10: {versions}")
        ema = manifest.get("ema", {})
        if int(ema.get("optimization_step", -1)) != 10:
            raise ValueError("Full5 logical state is not EMA10")
        for name, expected_version in versions.items():
            scheduler = components[name].get("scheduler_state", {})
            if int(scheduler.get("last_epoch", -1)) != expected_version:
                raise ValueError(
                    f"{name} logical scheduler does not match its parameter version"
                )
        return cls(root=root, manifest=manifest)

    @classmethod
    def load_systems_only_gan(
        cls,
        root: str | Path,
    ) -> "LogicalCheckpointPackage":
        """Load the immutable GAN systems bootstrap without relabelling state.

        The physical optimizer/scheduler state is the source checkpoint's
        G2/C10 state.  The systems experiment separately maps it to logical
        schedule versions G200/C1000/EMA800 for the 1000--1004 control window.
        Keeping both namespaces explicit prevents a logical version label from
        being written into Adam's physical step counter.
        """

        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("schema_version") != LOGICAL_CHECKPOINT_SCHEMA:
            raise ValueError("unsupported GAN logical-checkpoint schema")
        unsigned = dict(manifest)
        expected_hash = str(unsigned.pop("manifest_hash", ""))
        actual_hash = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError("GAN logical-checkpoint manifest hash mismatch")
        if int(manifest.get("global_step", -1)) != 10:
            raise ValueError("GAN systems bootstrap source is not step 10")
        components = manifest.get("components", {})
        physical_versions = {
            name: int(components.get(name, {}).get("parameter_version", -1))
            for name in ("generator", "critic")
        }
        if physical_versions != {"generator": 2, "critic": 10}:
            raise ValueError(
                "GAN systems bootstrap physical versions changed: "
                f"{physical_versions}"
            )
        for name, expected in physical_versions.items():
            component = components[name]
            if int(component.get("optimizer_step", -1)) != expected:
                raise ValueError(f"GAN {name} optimizer step changed")
            if int(component.get("scheduler_state", {}).get("last_epoch", -1)) != expected:
                raise ValueError(f"GAN {name} scheduler step changed")
        if int(manifest.get("ema", {}).get("optimization_step", -1)) != 800:
            raise ValueError("GAN systems bootstrap is not EMA800")
        control = manifest.get("systems_only_control", {})
        expected_control = {
            "generator": 200,
            "critic": 1000,
            "ema": 800,
        }
        if control.get("classification") != "systems_only_gan_bootstrap":
            raise ValueError("checkpoint is not marked systems_only_gan_bootstrap")
        if control.get("logical_schedule_versions") != expected_control:
            raise ValueError("GAN logical schedule version mapping changed")
        if control.get("logical_control_window") != list(range(1000, 1005)):
            raise ValueError("GAN logical control window changed")
        if bool(control.get("official_checkpoint_1000", True)):
            raise ValueError("systems bootstrap was mislabeled as official checkpoint1000")
        return cls(root=root, manifest=manifest)

    @property
    def manifest_hash(self) -> str:
        return str(self.manifest["manifest_hash"])

    def component(self, name: str) -> Mapping[str, Any]:
        try:
            component = self.manifest["components"][name]
        except KeyError as error:
            raise KeyError(f"missing logical component {name!r}") from error
        if not isinstance(component, Mapping):
            raise TypeError(f"logical component {name!r} is not a mapping")
        return component


def _sha256_file(path: Path, *, chunk_bytes: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_equal_on_source_device(
    target: torch.Tensor, source: torch.Tensor
) -> bool:
    """Compare a restored tensor exactly without creating another CUDA copy.

    Logical checkpoint payloads are CPU mmaps.  During a live role transition
    the trainable target may already be resident on CUDA.  ``torch.equal``
    requires a common device, so verification must move the (usually much
    smaller) target view back to the authoritative source device rather than
    materializing a duplicate tensor on an already capacity-bound GPU.
    """

    candidate = target.detach()
    if candidate.device != source.device:
        candidate = candidate.to(device=source.device)
    return bool(torch.equal(candidate, source))


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


@dataclass(frozen=True)
class CandidateBTransitionPackage:
    """Lightweight G3/EMA10 boundary state for the 4G+12C candidate.

    Critic C10 master/moments continue to come from the already validated
    layout-independent M1 logical package.  This package adds only the G3
    BF16 trainable module state produced by Candidate-A Step10-A, plus the
    exact sparse EMA10 and rank-qualified RNG files from that transition.
    """

    root: Path
    manifest: dict[str, Any]

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        expected_transition_manifest_sha256: str | None = None,
        expected_critic_logical_manifest_hash: str | None = None,
        verify_generator_payload_sha256: bool = False,
    ) -> "CandidateBTransitionPackage":
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("schema_version") != CANDIDATE_B_TRANSITION_SCHEMA:
            raise ValueError("unsupported Candidate-B transition package")
        unsigned = dict(manifest)
        expected_hash = str(unsigned.pop("manifest_hash", ""))
        actual_hash = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError("Candidate-B transition manifest hash mismatch")
        if manifest.get("status") != "PASS":
            raise ValueError("Candidate-B transition package did not pass export")
        versions = manifest.get("versions", {})
        if versions != {
            "generator": 3,
            "critic": 10,
            "ema_optimization_step": 10,
        }:
            raise ValueError(f"Candidate-B transition is not G3/C10/EMA10: {versions}")
        if (
            expected_transition_manifest_sha256 is not None
            and manifest.get("source_transition_manifest_sha256")
            != expected_transition_manifest_sha256
        ):
            raise ValueError("Candidate-B source transition identity changed")
        if (
            expected_critic_logical_manifest_hash is not None
            and manifest.get("critic_logical_manifest_hash")
            != expected_critic_logical_manifest_hash
        ):
            raise ValueError("Candidate-B C10 logical state identity changed")
        payload = manifest.get("generator", {}).get("payload", {})
        payload_path = root / str(payload.get("path", ""))
        if not payload_path.is_file() or payload_path.stat().st_size != int(
            payload.get("bytes", -1)
        ):
            raise ValueError("Candidate-B G3 payload size changed")
        if verify_generator_payload_sha256 and _sha256_file(payload_path) != payload.get(
            "sha256"
        ):
            raise ValueError("Candidate-B G3 payload SHA-256 changed")
        ema_root = root / "model_ema"
        if not (ema_root / "sparse_ema_manifest.json").is_file():
            raise ValueError("Candidate-B sparse EMA10 is missing")
        rng = manifest.get("rng", {})
        if set(rng) != {str(rank) for rank in range(16)}:
            raise ValueError("Candidate-B transition RNG inventory changed")
        return cls(root=root, manifest=manifest)

    @property
    def manifest_hash(self) -> str:
        return str(self.manifest["manifest_hash"])

    @property
    def ema_root(self) -> Path:
        return self.root / "model_ema"

    def rng_path(self, rank: int) -> Path:
        record = self.manifest["rng"][str(int(rank))]
        path = self.root / str(record["path"])
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"Candidate-B RNG payload changed for rank {rank}")
        return path


@torch.no_grad()
def load_logical_master_into_trainables(
    model: torch.nn.Module,
    package: PoolTargetPackage | LogicalCheckpointPackage,
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
        if not _exact_equal_on_source_device(target, source):
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


@torch.no_grad()
def load_candidate_b_generator_module_into_trainables(
    model: torch.nn.Module,
    package: CandidateBTransitionPackage,
) -> dict[str, Any]:
    """Restore the exact post-Step10-A BF16 Generator LoRA inventory."""

    component = package.manifest["generator"]
    names = [str(name) for name in component["parameter_names"]]
    trainables = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if list(trainables) != names:
        raise ValueError("Candidate-B G3 trainable order/inventory changed")
    payload = component["payload"]
    total_numel = int(component["logical_numel"])
    raw = torch.from_file(
        str(package.root / payload["path"]),
        shared=False,
        size=total_numel,
        dtype=torch.uint16,
    )
    copied = 0
    for name in names:
        record = component["records"][name]
        offset = int(record["offset_elements"])
        numel = int(record["numel"])
        shape = tuple(int(value) for value in record["shape"])
        source = raw.narrow(0, offset, numel).view(torch.bfloat16).view(shape)
        target = trainables[name]
        if tuple(target.shape) != shape:
            raise ValueError(f"Candidate-B G3 shape changed for {name}")
        if target.dtype != torch.bfloat16:
            target.data = target.data.to(dtype=torch.bfloat16)
        target.copy_(source)
        if not torch.equal(target.detach().cpu(), source):
            raise AssertionError(f"Candidate-B G3 copy was not exact for {name}")
        copied += numel
    del raw
    if copied != total_numel:
        raise AssertionError("Candidate-B G3 payload coverage changed")
    return {
        "component": "generator",
        "parameter_version": 3,
        "logical_numel": copied,
        "tensor_count": len(names),
        "dtype": "torch.bfloat16",
        "payload_sha256": str(payload["sha256"]),
        "optimizer_policy": "BACKING_ONLY",
    }


def _unwrap_optimizer(optimizer: Any) -> Any:
    current = optimizer
    seen: set[int] = set()
    while not hasattr(current, "state") and hasattr(current, "optimizer"):
        if id(current) in seen:
            raise RuntimeError("optimizer wrapper cycle")
        seen.add(id(current))
        current = current.optimizer
    if not hasattr(current, "state"):
        raise TypeError("optimizer has no state mapping")
    return current


def _concrete_nested_optimizer(optimizer: Any) -> tuple[Any, list[str]]:
    """Return the deepest optimizer implementation behind runtime wrappers.

    DeepSpeed ZeRO-3 deliberately exposes the same ``state`` mapping as its
    nested client optimizer.  Stopping at the first object with ``state`` is
    therefore insufficient when the state representation is owned by the
    nested PyTorch optimizer (for example AdamW's singleton Tensor step).
    """

    current = optimizer
    seen: set[int] = set()
    chain: list[str] = []
    while True:
        if id(current) in seen:
            raise RuntimeError("optimizer wrapper cycle")
        seen.add(id(current))
        chain.append(f"{type(current).__module__}.{type(current).__name__}")
        nested = getattr(current, "optimizer", None)
        if nested is None or nested is current:
            break
        current = nested
    if not hasattr(current, "state"):
        raise TypeError("concrete optimizer has no state mapping")
    return current, chain


@torch.no_grad()
def restore_zero1_optimizer_partition_from_logical(
    engine: Any,
    package: LogicalCheckpointPackage,
    *,
    component: str,
    partition_rank: int,
    dp_world_size: int,
) -> dict[str, Any]:
    """Inject one padded ZeRO-1 FP32 partition from logical flat tensors.

    This is a target-layout construction primitive, not part of the measured
    Critic cycle.  It supports the DP12 padding that the older M1 import tool
    intentionally rejected during its unpadded-only gate.
    """

    if int(engine.zero_optimization_stage()) != 1:
        raise ValueError("logical partition restore requires ZeRO-1")
    if bool(getattr(engine.optimizer, "partition_gradients", False)):
        raise ValueError("logical partition restore refuses ZeRO-2")
    component_manifest = package.component(component)
    names = [str(name) for name in component_manifest["parameter_names"]]
    observed_names = [
        name
        for name, parameter in engine.module.named_parameters()
        if parameter.requires_grad
    ]
    if observed_names != names:
        raise ValueError("logical flat order differs from the live module")
    if not 0 <= partition_rank < dp_world_size:
        raise ValueError("partition rank is outside target DP world")

    masters = getattr(engine.optimizer, "single_partition_of_fp32_groups", None)
    if not isinstance(masters, (list, tuple)) or len(masters) != 1:
        raise ValueError("expected one ZeRO-1 FP32 master partition")
    master = masters[0]
    if master.dtype != torch.float32:
        raise ValueError("ZeRO-1 master partition is not FP32")
    partition_numel = int(master.numel())
    total_numel = int(component_manifest["logical_numel"])
    padded_numel = partition_numel * int(dp_world_size)
    if padded_numel < total_numel:
        raise ValueError("target ZeRO-1 partitions do not cover logical state")
    lower = int(partition_rank) * partition_numel
    valid = max(0, min(partition_numel, total_numel - lower))

    payload_keys = {
        "master": "master",
        "exp_avg": "first_moment",
        "exp_avg_sq": "second_moment",
    }
    logical: dict[str, torch.Tensor] = {}
    for target_name, payload_name in payload_keys.items():
        payload = component_manifest["payloads"][payload_name]
        if payload.get("dtype") != "torch.float32":
            raise ValueError(f"logical {payload_name} is not FP32")
        path = package.root / str(payload["path"])
        if path.stat().st_size != int(payload["bytes"]):
            raise ValueError(f"logical {payload_name} byte size changed")
        logical[target_name] = torch.from_file(
            str(path), shared=False, size=total_numel, dtype=torch.float32
        )

    master.zero_()
    if valid:
        master.narrow(0, 0, valid).copy_(
            logical["master"].narrow(0, lower, valid).to(master.device)
        )
    base = _unwrap_optimizer(engine.optimizer)
    params = list(base.param_groups[0].get("params", ()))
    if len(params) != 1 or params[0] is not master:
        raise ValueError("base optimizer is not bound to the ZeRO-1 master partition")
    state = base.state[master]
    state.clear()
    state["step"] = torch.tensor(
        float(component_manifest["optimizer_step"]),
        dtype=torch.float32,
        device=master.device,
    )
    for key in ("exp_avg", "exp_avg_sq"):
        value = torch.zeros_like(master, dtype=torch.float32, device=master.device)
        if valid:
            value.narrow(0, 0, valid).copy_(
                logical[key].narrow(0, lower, valid).to(value.device)
            )
        state[key] = value

    engine.global_steps = int(component_manifest["parameter_version"])
    if hasattr(engine, "global_samples"):
        engine.global_samples = int(
            component_manifest["model_state"]["logical_samples"]
        ) * int(dp_world_size)
    engine.lr_scheduler.load_state_dict(dict(component_manifest["scheduler_state"]))
    linker = getattr(engine.optimizer, "_link_all_hp_params", None)
    if callable(linker):
        linker()

    del logical
    return {
        "component": component,
        "parameter_version": int(component_manifest["parameter_version"]),
        "optimizer_step": int(component_manifest["optimizer_step"]),
        "scheduler_step": int(component_manifest["scheduler_state"]["last_epoch"]),
        "target_dp_world_size": int(dp_world_size),
        "partition_rank": int(partition_rank),
        "partition_numel": partition_numel,
        "valid_numel": valid,
        "padding_numel": partition_numel - valid,
        "logical_numel": total_numel,
        "padded_numel": padded_numel,
    }


@torch.no_grad()
def restore_zero3_optimizer_partition_from_logical(
    engine: Any,
    package: LogicalCheckpointPackage,
    *,
    component: str,
    partition_rank: int,
    dp_world_size: int,
) -> dict[str, Any]:
    """Restore one logical optimizer state into a ZeRO-3 target layout.

    The logical checkpoint is flat in named-parameter order, whereas ZeRO-3
    partitions every parameter independently (including per-parameter tail
    padding).  A flat ``total_numel / world_size`` slice is therefore wrong.
    This loader maps every named logical tensor to that parameter's local
    ZeRO-3 shard and initializes the CPU Adam master/moment state before the
    first resumed optimizer step.

    This is a construction/restore primitive.  It performs no collectives and
    must run on every member of the already-created ZeRO-3 DP group.
    """

    if int(engine.zero_optimization_stage()) != 3:
        raise ValueError("logical ZeRO-3 restore requires ZeRO-3")
    if not 0 <= partition_rank < dp_world_size:
        raise ValueError("partition rank is outside target ZeRO-3 DP world")

    component_manifest = package.component(component)
    names = [str(name) for name in component_manifest["parameter_names"]]
    trainables = {
        name: parameter
        for name, parameter in engine.module.named_parameters()
        if parameter.requires_grad
    }
    if list(trainables) != names:
        raise ValueError("logical flat order differs from the ZeRO-3 module")

    total_numel = int(component_manifest["logical_numel"])
    payload_names = {
        "master": "master",
        "exp_avg": "first_moment",
        "exp_avg_sq": "second_moment",
    }
    logical: dict[str, torch.Tensor] = {}
    for target_name, payload_name in payload_names.items():
        payload = component_manifest["payloads"][payload_name]
        if payload.get("dtype") != "torch.float32":
            raise ValueError(f"logical {payload_name} is not FP32")
        path = package.root / str(payload["path"])
        if path.stat().st_size != int(payload["bytes"]):
            raise ValueError(f"logical {payload_name} byte size changed")
        logical[target_name] = torch.from_file(
            str(path), shared=False, size=total_numel, dtype=torch.float32
        )

    zero = engine.optimizer
    masters = getattr(zero, "fp32_partitioned_groups_flat", None)
    if not isinstance(masters, (list, tuple)) or not masters:
        raise ValueError("ZeRO-3 optimizer exposes no FP32 master partitions")
    base, optimizer_chain = _concrete_nested_optimizer(zero)
    optimizer_step = int(component_manifest["optimizer_step"])
    if not isinstance(base, (torch.optim.Adam, torch.optim.AdamW)):
        raise TypeError(
            "logical ZeRO-3 restore has no native step-state contract for "
            f"{optimizer_chain[-1]}"
        )
    for master in masters:
        state = base.state[master]
        state.clear()
        # Current PyTorch Adam-family implementations require a singleton
        # Tensor.  The logical value remains unchanged; only the target
        # optimizer's native state representation is reconstructed.
        state["step"] = torch.tensor(
            float(optimizer_step), dtype=torch.float32, device=master.device
        )
        state["exp_avg"] = torch.zeros_like(master, dtype=torch.float32)
        state["exp_avg_sq"] = torch.zeros_like(master, dtype=torch.float32)

    copied = 0
    padding = 0
    shard_numel = 0
    for name in names:
        parameter = trainables[name]
        record = component_manifest["records"][name]
        offset = int(record["offset_elements"])
        numel = int(record["numel"])
        if int(getattr(parameter, "ds_numel", -1)) != numel:
            raise ValueError(f"ZeRO-3 logical size mismatch for {name}")
        partition = getattr(parameter, "ds_tensor", None)
        if not isinstance(partition, torch.Tensor):
            raise ValueError(f"ZeRO-3 parameter {name} has no local shard")
        local_numel = int(partition.numel())
        lower = int(partition_rank) * local_numel
        valid = max(0, min(local_numel, numel - lower))
        shard_numel += local_numel
        padding += local_numel - valid

        for state_name in ("master", "exp_avg", "exp_avg_sq"):
            local = torch.zeros(local_numel, dtype=torch.float32)
            if valid:
                local.narrow(0, 0, valid).copy_(
                    logical[state_name].narrow(0, offset + lower, valid)
                )
            zero.set_local_hp_param(
                local,
                parameter,
                None if state_name == "master" else state_name,
            )
            del local

        # DeepSpeed built the low-precision shard from the same logical master
        # before initialization.  Refresh it explicitly so the target module
        # and restored FP32 master cannot diverge through a construction path.
        low = torch.zeros(
            local_numel, dtype=partition.dtype, device=partition.device
        )
        if valid:
            low.narrow(0, 0, valid).copy_(
                logical["master"]
                .narrow(0, offset + lower, valid)
                .to(device=partition.device, dtype=partition.dtype)
            )
        partition.copy_(low)
        del low
        copied += valid

    engine.global_steps = int(component_manifest["parameter_version"])
    if hasattr(engine, "global_samples"):
        engine.global_samples = int(
            component_manifest["model_state"]["logical_samples"]
        ) * int(dp_world_size)
    engine.lr_scheduler.load_state_dict(dict(component_manifest["scheduler_state"]))

    del logical
    return {
        "component": component,
        "parameter_version": int(component_manifest["parameter_version"]),
        "optimizer_step": optimizer_step,
        "scheduler_step": int(component_manifest["scheduler_state"]["last_epoch"]),
        "target_dp_world_size": int(dp_world_size),
        "partition_rank": int(partition_rank),
        "logical_numel": total_numel,
        "local_valid_numel": copied,
        "local_shard_numel": shard_numel,
        "local_padding_numel": padding,
        "partitioning": "per_parameter_zero3",
        "optimizer_wrapper_chain": optimizer_chain,
        "optimizer_step_representation": "torch_singleton_tensor",
    }

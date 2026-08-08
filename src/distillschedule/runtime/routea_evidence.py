"""Fail-closed preregistration and evidence contracts for Route-A world16.

The module is deliberately model-free.  It binds the typed plan, rendered Helios
configuration, Accelerate/DeepSpeed launcher, rank-qualified operation trace and
input checkpoint before a 14B process is constructed.  Runtime evidence is then
checked against the immutable contract by the Route-A summarizer.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from distillschedule.core.config import load_plan
from distillschedule.core.contracts import ContractViolation
from distillschedule.runtime.launch_config import inspect_accelerate_config

ROUTEA_SCHEMA = "distillschedule.routea_world16.evidence_contract.v1"
NODE_SCHEMA = "distillschedule.routea_world16.node_contract.v1"
TRACE_BUNDLE_SCHEMA = "distillschedule.operation_trace_bundle.v1"
TRACE_SCHEMA = "distillschedule.operation_trace.v1"
EXPECTED_STEPS = (10, 11, 12, 13, 14)
EXPECTED_WORLD_SIZE = 16
EXPECTED_NODES = 2
EXPECTED_LOCAL_WORLD_SIZE = 8

# Only these fields may differ without changing training.  Everything else in the
# rendered YAML participates in normalized_config_digest.
NON_TRAINING_CONFIG_ALLOWLIST: tuple[str, ...] = (
    "output_dir",
    "logging_dir",
    "report_to",
)


@dataclass(frozen=True)
class Inventory:
    path: Path
    file_count: int
    total_bytes: int
    structural_digest: str
    entries: tuple[Mapping[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "structural_digest": self.structural_digest,
            "entries": [dict(item) for item in self.entries],
        }


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def object_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise ContractViolation("file does not exist", path=str(candidate))
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)
    return destination


def load_json_mapping(path: str | Path) -> dict[str, Any]:
    candidate = Path(path).expanduser().resolve()
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractViolation(f"invalid JSON: {exc}", path=str(candidate)) from exc
    if not isinstance(value, Mapping):
        raise ContractViolation("expected a JSON object", path=str(candidate))
    return dict(value)


def _unsigned_digest(value: Mapping[str, Any], signature_key: str) -> str:
    unsigned = dict(value)
    unsigned.pop(signature_key, None)
    return object_digest(unsigned)


def directory_inventory(
    path: str | Path,
    *,
    max_entries: int | None = None,
) -> Inventory:
    """Create a deterministic names-and-sizes inventory without hashing huge payloads.

    This is intentionally called a structural digest, not a content digest.  Trace
    manifests and small configuration files are content-hashed separately.
    """

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ContractViolation("directory does not exist", path=str(root))
    files = sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    if max_entries is not None and len(files) > max_entries:
        raise ContractViolation(
            f"inventory contains {len(files)} files, limit is {max_entries}",
            path=str(root),
        )
    entries = tuple(
        {
            "path": candidate.relative_to(root).as_posix(),
            "bytes": int(candidate.stat().st_size),
        }
        for candidate in files
    )
    return Inventory(
        path=root,
        file_count=len(entries),
        total_bytes=sum(int(item["bytes"]) for item in entries),
        structural_digest=object_digest(entries),
        entries=entries,
    )


def asset_identity(path: str | Path) -> dict[str, object]:
    """Record a bounded, cheap identity for a model/data/runtime directory."""

    root = Path(path).expanduser().resolve()
    if root.is_file():
        return {
            "path": str(root),
            "kind": "file",
            "bytes": int(root.stat().st_size),
            "sha256": sha256_file(root),
        }
    if not root.is_dir():
        raise ContractViolation("asset does not exist", path=str(root))

    marker_names = {
        "config.json",
        "model_index.json",
        "bundle.json",
        "trace.json",
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
        "manifest.json",
    }
    markers: list[dict[str, object]] = []
    top_level: list[dict[str, object]] = []
    for candidate in sorted(root.iterdir(), key=lambda item: item.name):
        top_level.append(
            {
                "name": candidate.name,
                "kind": "dir" if candidate.is_dir() else "file",
                "bytes": int(candidate.stat().st_size) if candidate.is_file() else None,
            }
        )
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.name in marker_names or candidate.name.endswith(".index.json"):
            relative = candidate.relative_to(root)
            if len(relative.parts) <= 3:
                markers.append(
                    {
                        "path": relative.as_posix(),
                        "bytes": int(candidate.stat().st_size),
                        "sha256": sha256_file(candidate),
                    }
                )
    payload = {
        "path": str(root),
        "kind": "directory_metadata",
        "top_level": top_level,
        "markers": markers,
    }
    payload["identity_digest"] = object_digest(payload)
    return payload


def inspect_trace_bundle(path: str | Path, *, expected_world_size: int = 16) -> dict[str, object]:
    root = Path(path).expanduser().resolve()
    bundle_path = root / "bundle.json"
    bundle = load_json_mapping(bundle_path)
    if bundle.get("schema_version") != TRACE_BUNDLE_SCHEMA:
        raise ContractViolation(
            f"unsupported trace bundle schema {bundle.get('schema_version')!r}",
            path=str(bundle_path),
        )
    if bundle.get("bundle_hash") != _unsigned_digest(bundle, "bundle_hash"):
        raise ContractViolation("trace bundle hash mismatch", path=str(bundle_path))
    entries = bundle.get("ranks")
    if not isinstance(entries, list) or len(entries) != expected_world_size:
        raise ContractViolation(
            f"trace bundle must contain {expected_world_size} rank entries",
            path=f"{bundle_path}.ranks",
        )
    observed_ranks = [entry.get("rank") for entry in entries if isinstance(entry, Mapping)]
    if observed_ranks != list(range(expected_world_size)):
        raise ContractViolation(
            f"trace ranks must be contiguous; observed={observed_ranks}",
            path=f"{bundle_path}.ranks",
        )

    ranks: list[dict[str, object]] = []
    for entry in entries:
        assert isinstance(entry, Mapping)
        rank = int(entry["rank"])
        relative_path = entry.get("path")
        expected_trace_hash = entry.get("trace_hash")
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ContractViolation("rank trace path is missing", path=f"trace.ranks.{rank}")
        if not isinstance(expected_trace_hash, str) or len(expected_trace_hash) != 64:
            raise ContractViolation("rank trace hash is invalid", path=f"trace.ranks.{rank}")
        trace_root = (root / relative_path).resolve()
        trace_path = trace_root / "trace.json"
        trace = load_json_mapping(trace_path)
        if trace.get("schema_version") != TRACE_SCHEMA:
            raise ContractViolation(
                f"unsupported rank trace schema {trace.get('schema_version')!r}",
                path=str(trace_path),
            )
        actual_trace_hash = _unsigned_digest(trace, "trace_hash")
        if trace.get("trace_hash") != actual_trace_hash:
            raise ContractViolation("rank trace manifest hash mismatch", path=str(trace_path))
        if actual_trace_hash != expected_trace_hash:
            raise ContractViolation(
                "rank trace does not match bundle entry", path=str(trace_path)
            )
        source = trace.get("source")
        operations = trace.get("operations")
        if not isinstance(source, Mapping):
            raise ContractViolation("trace source must be a mapping", path=str(trace_path))
        if not isinstance(operations, list):
            raise ContractViolation("trace operations must be a list", path=str(trace_path))
        aliases: list[str] = []
        for key in ("source_authority", "authority", "trace_root", "sample_id"):
            value = source.get(key)
            if isinstance(value, (str, int)) and str(value):
                aliases.append(str(value))
        aliases.extend(
            [
                relative_path,
                str(trace_root),
                str(trace_path),
            ]
        )
        aliases = list(dict.fromkeys(aliases))
        ranks.append(
            {
                "rank": rank,
                "relative_path": relative_path,
                "trace_manifest_path": str(trace_path),
                "trace_manifest_sha256": sha256_file(trace_path),
                "trace_hash": actual_trace_hash,
                "source": dict(source),
                "source_fingerprint": object_digest(source),
                "source_authority_aliases": aliases,
                "operation_fingerprint": object_digest(operations),
                "operation_count": len(operations),
                "mapping_fingerprint": object_digest(
                    {
                        "rank": rank,
                        "path": relative_path,
                        "trace_hash": actual_trace_hash,
                        "source": source,
                    }
                ),
            }
        )
    payload: dict[str, object] = {
        "path": str(root),
        "bundle_manifest_path": str(bundle_path),
        "bundle_manifest_sha256": sha256_file(bundle_path),
        "bundle_hash": bundle["bundle_hash"],
        "rank_count": len(ranks),
        "ranks": ranks,
    }
    payload["inventory_digest"] = object_digest(ranks)
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:  # pragma: no cover - test/Helios environments provide it
        raise ContractViolation(
            "YAML contract validation requires OmegaConf", path=str(path)
        ) from exc
    try:
        value = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception as exc:  # pragma: no cover - OmegaConf errors are version-specific
        raise ContractViolation(f"invalid YAML: {exc}", path=str(path)) from exc
    if not isinstance(value, Mapping):
        raise ContractViolation("YAML root must be a mapping", path=str(path))
    return copy.deepcopy(dict(value))


def _remove_path(value: MutableMapping[str, Any], dotted_path: str) -> None:
    parts = dotted_path.split(".")
    current: MutableMapping[str, Any] = value
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, MutableMapping):
            return
        current = child
    current.pop(parts[-1], None)


def normalize_helios_config(
    path: str | Path,
    *,
    allowlist: Sequence[str] = NON_TRAINING_CONFIG_ALLOWLIST,
) -> dict[str, object]:
    candidate = Path(path).expanduser().resolve()
    payload = _load_yaml(candidate)
    for dotted_path in allowlist:
        _remove_path(payload, dotted_path)
    return {
        "path": str(candidate),
        "file_sha256": sha256_file(candidate),
        "allowlisted_non_training_fields": list(allowlist),
        "normalized_digest": object_digest(payload),
        "normalized": payload,
    }


def _lookup(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ContractViolation("required field is missing", path=dotted_path)
        current = current[part]
    return current


def _expect(value: Mapping[str, Any], dotted_path: str, expected: Any) -> Any:
    observed = _lookup(value, dotted_path)
    if observed != expected:
        raise ContractViolation(
            f"observed {observed!r}, expected {expected!r}", path=dotted_path
        )
    return observed


def inspect_rendered_helios_config(path: str | Path) -> dict[str, object]:
    normalized = normalize_helios_config(path)
    config = _load_yaml(Path(path).expanduser().resolve())
    expected = {
        "data_config.single_height": 384,
        "data_config.single_width": 640,
        "training_config.mixed_precision": "bf16",
        "training_config.max_train_steps": 15,
        "training_config.train_batch_size": 1,
        "training_config.gradient_accumulation_steps": 1,
        "training_config.checkpointing_steps": 15,
        "training_config.gradient_checkpointing": True,
        "training_config.dfake_gen_update_ratio": 5,
        "training_config.latent_window_size": [9],
        "training_config.use_ema": True,
        "training_config.is_train_dmd": True,
        "training_config.is_dmd_vae_decode": False,
        "training_config.is_use_gan": False,
        "training_config.is_use_reward_model": False,
        "training_config.ema_start_step": 750,
        "validation_config.validation_height": 384,
        "validation_config.validation_width": 640,
        "validation_config.validation_latent_window_size": [9],
    }
    for dotted_path, expected_value in expected.items():
        _expect(config, dotted_path, expected_value)
    resume = _lookup(config, "training_config.resume_from_checkpoint")
    if not isinstance(resume, str) or not resume.endswith("checkpoint-10"):
        raise ContractViolation(
            "Route-A must resume canonical checkpoint-10",
            path="training_config.resume_from_checkpoint",
        )
    generator_ds = _lookup(config, "training_config.dmd_generator_deepspeed_config")
    critic_ds = _lookup(config, "training_config.dmd_critic_deepspeed_config")
    if generator_ds != critic_ds:
        raise ContractViolation(
            "Generator and Score must reference the same Route-A DeepSpeed contract",
            path="training_config.dmd_*_deepspeed_config",
        )
    contract = {
        "resolution": [384, 640],
        "micro_batch_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "global_batch": 16,
        "max_train_steps": 15,
        "logical_steps": list(EXPECTED_STEPS),
        "checkpointing_steps": 15,
        "resume_from_checkpoint": resume,
        "mixed_precision": "bf16",
        "gradient_checkpointing": True,
        "dfake_gen_update_ratio": 5,
        "latent_window_size": [9],
        "branches": {
            "dmd": True,
            "ema": True,
            "gan": False,
            "reward": False,
            "vae_decode": False,
        },
        "ema_decay_start_step": 750,
        "deepspeed_config": str(generator_ds),
        "generator_checkpoint": _lookup(
            config, "model_config.transformer_model_name_or_path"
        ),
        "score_checkpoint": _lookup(
            config, "model_config.real_score_model_name_or_path"
        ),
    }
    return {
        "file_sha256": normalized["file_sha256"],
        "normalized_digest": normalized["normalized_digest"],
        "allowlisted_non_training_fields": normalized[
            "allowlisted_non_training_fields"
        ],
        "training_contract": contract,
    }


def _require_cli_value(command: Sequence[str], flag: str, expected: str) -> None:
    try:
        index = command.index(flag)
    except ValueError as exc:
        raise ContractViolation("required launcher flag is missing", path=flag) from exc
    if index + 1 >= len(command) or command[index + 1] != expected:
        observed = command[index + 1] if index + 1 < len(command) else None
        raise ContractViolation(
            f"observed {observed!r}, expected {expected!r}", path=flag
        )


def inspect_compiled_preflight(path: str | Path) -> dict[str, object]:
    payload = load_json_mapping(path)
    invocation = payload.get("invocation")
    if not isinstance(invocation, Mapping):
        raise ContractViolation("compiled invocation is missing", path=str(path))
    command = invocation.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise ContractViolation("compiled command is invalid", path=str(path))
    _require_cli_value(command, "--num_machines", "2")
    _require_cli_value(command, "--num_processes", "16")
    _require_cli_value(command, "--mixed_precision", "bf16")
    _require_cli_value(command, "--gradient_accumulation_steps", "1")
    if "--use_deepspeed" in command or "--zero_stage" in command:
        raise ContractViolation(
            "Route-A must use the explicit Accelerate config without CLI ZeRO fallback",
            path="compiled.command",
        )
    return {
        "plan_id": payload.get("plan_id"),
        "plan_digest": payload.get("plan_digest"),
        "command_contract": {
            "num_machines": 2,
            "num_processes": 16,
            "mixed_precision": "bf16",
            "gradient_accumulation_steps": 1,
        },
        "invocation_digest": object_digest(invocation),
    }


def inspect_typed_plan(path: str | Path, *, repo_root: str | Path) -> dict[str, object]:
    loaded = load_plan(path)
    plan = loaded.plan
    shape = plan.workload.shape
    branches = plan.workload.branches
    schedule = plan.workload.schedule
    expected_checks = {
        "hardware.nodes": (plan.hardware.nodes, 2),
        "hardware.local_world_size": (plan.hardware.local_world_size, 8),
        "hardware.world_size": (plan.hardware.world_size, 16),
        "workload.shape.batch_size": (shape.batch_size, 1),
        "workload.shape.frames": (shape.frames, 81),
        "workload.shape.resolution": (
            [shape.resolution.height, shape.resolution.width],
            [384, 640],
        ),
        "workload.shape.latent": (
            [shape.latent.channels, shape.latent.frames, shape.latent.height, shape.latent.width],
            [16, 9, 48, 80],
        ),
        "evidence.start_step": (plan.evidence.start_step, 10),
        "evidence.steps": (plan.evidence.steps, 5),
        "schedule.generator_update_every": (schedule.generator_update_every, 5),
        "schedule.critic_update_every": (schedule.critic_update_every, 1),
        "schedule.ema_update_every": (schedule.ema_update_every, 1),
        "schedule.ema_bootstrap_before_start": (
            schedule.ema_bootstrap_before_start,
            True,
        ),
        "schedule.ema_after_critic": (schedule.ema_after_critic, True),
        "branches.dmd": (branches.dmd, True),
        "branches.ema": (branches.ema, True),
        "branches.gan": (branches.gan, False),
        "branches.reward": (branches.reward, False),
        "branches.vae_decode": (branches.vae_decode, False),
    }
    for field, (observed, expected) in expected_checks.items():
        if observed != expected:
            raise ContractViolation(
                f"observed {observed!r}, expected {expected!r}", path=field
            )
    backends: dict[str, object] = {}
    for component in ("generator", "score"):
        backend = plan.training_backends[component]
        observed = {
            "kind": backend.kind.value,
            "zero_stage": backend.zero_stage,
            "optimizer_offload": backend.optimizer_offload,
            "parameter_offload": backend.parameter_offload,
            "gradient_checkpointing": backend.gradient_checkpointing,
        }
        expected = {
            "kind": "deepspeed",
            "zero_stage": 1,
            "optimizer_offload": "cpu",
            "parameter_offload": "none",
            "gradient_checkpointing": True,
        }
        if observed != expected:
            raise ContractViolation(
                f"backend drift: observed={observed}, expected={expected}",
                path=f"training_backends.{component}",
            )
        backends[component] = observed
    return {
        "path": str(Path(path).expanduser().resolve()),
        "top_level_sha256": sha256_file(path),
        "resolved_digest": loaded.digest,
        "plan_id": plan.plan_id,
        "source_files": [
            {"path": str(source), "sha256": sha256_file(source)}
            for source in loaded.source_files
        ],
        "hardware": {
            "nodes": 2,
            "local_world_size": 8,
            "world_size": 16,
        },
        "shape": {
            "micro_batch_per_gpu": 1,
            "source_frames": 81,
            "resolution": [384, 640],
            "latent": [1, 16, 9, 48, 80],
            "global_batch": 16,
        },
        "logical_steps": list(EXPECTED_STEPS),
        "branches": {
            "dmd": True,
            "ema": True,
            "gan": False,
            "reward": False,
            "vae_decode": False,
        },
        "schedule": {
            "generator_update_every": 5,
            "critic_update_every": 1,
            "ema_update_every": 1,
            "ema_decay_start_step": schedule.ema_start_step,
            "ema_bootstrap_before_decay_start": True,
            "ema_after_critic": True,
        },
        "backends": backends,
        "repo_root": str(Path(repo_root).expanduser().resolve()),
    }


def build_node_contract(
    *,
    node_rank: int,
    plan_path: str | Path,
    repo_root: str | Path,
    compiled_preflight_path: str | Path,
    rendered_config_path: str | Path,
    accelerate_config_path: str | Path,
    trace_bundle_path: str | Path,
    checkpoint_path: str | Path,
    assets: Mapping[str, str | Path] | None = None,
) -> dict[str, object]:
    if node_rank not in (0, 1):
        raise ContractViolation("node rank must be 0 or 1", path="node_rank")
    plan = inspect_typed_plan(plan_path, repo_root=repo_root)
    compiled = inspect_compiled_preflight(compiled_preflight_path)
    if compiled["plan_digest"] != plan["resolved_digest"]:
        raise ContractViolation(
            "compiled and independently resolved plan digests differ",
            path="plan_digest",
        )
    rendered = inspect_rendered_helios_config(rendered_config_path)
    accelerate = inspect_accelerate_config(
        accelerate_config_path,
        expected_num_machines=2,
        expected_num_processes=16,
        expected_machine_rank=node_rank,
    )
    if accelerate.deepspeed.zero_stage != 1:
        raise ContractViolation("ZeRO stage drift", path="launcher.deepspeed")
    trace = inspect_trace_bundle(trace_bundle_path, expected_world_size=16)
    checkpoint = directory_inventory(checkpoint_path)
    if checkpoint.file_count == 0:
        raise ContractViolation("input checkpoint is empty", path=str(checkpoint.path))
    asset_payload = {
        name: asset_identity(asset_path)
        for name, asset_path in sorted((assets or {}).items())
    }
    shared: dict[str, object] = {
        "plan": plan,
        "compiled": compiled,
        "rendered_helios": rendered,
        "launcher": {
            "distributed_type": accelerate.distributed_type,
            "num_machines": accelerate.num_machines,
            "num_processes": accelerate.num_processes,
            "local_world_size": accelerate.num_processes // accelerate.num_machines,
            "multinode_launcher": accelerate.multinode_launcher,
            "deepspeed": accelerate.deepspeed.to_dict(),
        },
        "batch_contract": {
            "micro_batch_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "global_batch": 16,
        },
        "trace": trace,
        "input_checkpoint": checkpoint.to_dict(),
        "assets": asset_payload,
        "memory_acceptance": {
            "policy": "actual_oom_only",
            "reject_on": [
                "cuda_oom",
                "cgroup_oom_delta",
                "cgroup_oom_kill_delta",
                "memory_attributed_nonzero_exit",
            ],
            "host_reserve_gate": False,
        },
    }
    shared_digest = object_digest(shared)
    payload: dict[str, object] = {
        "schema_version": NODE_SCHEMA,
        "node_rank": node_rank,
        "shared_digest": shared_digest,
        "shared": shared,
        "node": {
            "machine_rank": accelerate.machine_rank,
            "accelerate_config_path": str(accelerate.path),
            "accelerate_config_sha256": accelerate.sha256,
            "rendered_config_path": str(Path(rendered_config_path).resolve()),
            "rendered_config_file_sha256": rendered["file_sha256"],
            "compiled_preflight_sha256": sha256_file(compiled_preflight_path),
        },
    }
    payload["node_contract_digest"] = object_digest(payload)
    return payload


def finalize_node_contracts(
    node_contract_paths: Iterable[str | Path],
) -> dict[str, object]:
    paths = [Path(path).expanduser().resolve() for path in node_contract_paths]
    if len(paths) != 2:
        raise ContractViolation("exactly two node contracts are required", path="nodes")
    contracts = [load_json_mapping(path) for path in paths]
    ranks = sorted(contract.get("node_rank") for contract in contracts)
    if ranks != [0, 1]:
        raise ContractViolation(
            f"node contracts must cover ranks 0 and 1; observed={ranks}", path="nodes"
        )
    for contract, path in zip(contracts, paths):
        if contract.get("schema_version") != NODE_SCHEMA:
            raise ContractViolation("unsupported node contract schema", path=str(path))
        recorded = contract.get("node_contract_digest")
        unsigned = dict(contract)
        unsigned.pop("node_contract_digest", None)
        if recorded != object_digest(unsigned):
            raise ContractViolation("node contract digest mismatch", path=str(path))
    shared_digests = {str(contract.get("shared_digest")) for contract in contracts}
    if len(shared_digests) != 1:
        raise ContractViolation(
            f"node normalized contracts differ: {sorted(shared_digests)}",
            path="nodes.shared_digest",
        )
    shared = contracts[0].get("shared")
    if not isinstance(shared, Mapping):
        raise ContractViolation("shared contract is missing", path="nodes.shared")
    if contracts[1].get("shared") != shared:
        raise ContractViolation(
            "shared contract payload differs despite digest", path="nodes.shared"
        )
    payload: dict[str, object] = {
        "schema_version": ROUTEA_SCHEMA,
        "status": "PASS",
        "shared_digest": next(iter(shared_digests)),
        "shared": dict(shared),
        "nodes": [
            {
                "node_rank": int(contract["node_rank"]),
                "path": str(path),
                "file_sha256": sha256_file(path),
                "node_contract_digest": contract["node_contract_digest"],
                "node": contract["node"],
            }
            for contract, path in sorted(
                zip(contracts, paths), key=lambda pair: int(pair[0]["node_rank"])
            )
        ],
    }
    payload["contract_digest"] = object_digest(payload)
    return payload


def validate_final_contract(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != ROUTEA_SCHEMA or value.get("status") != "PASS":
        raise ContractViolation("Route-A evidence contract is not PASS", path="contract")
    recorded = value.get("contract_digest")
    unsigned = dict(value)
    unsigned.pop("contract_digest", None)
    if recorded != object_digest(unsigned):
        raise ContractViolation("Route-A evidence contract digest mismatch", path="contract")
    shared = value.get("shared")
    if not isinstance(shared, Mapping):
        raise ContractViolation("shared contract is missing", path="contract.shared")
    trace = shared.get("trace")
    if not isinstance(trace, Mapping) or trace.get("rank_count") != 16:
        raise ContractViolation("rank-qualified trace contract is incomplete", path="contract.trace")


__all__ = [
    "EXPECTED_STEPS",
    "NODE_SCHEMA",
    "NON_TRAINING_CONFIG_ALLOWLIST",
    "ROUTEA_SCHEMA",
    "asset_identity",
    "atomic_json",
    "build_node_contract",
    "canonical_json",
    "directory_inventory",
    "finalize_node_contracts",
    "inspect_rendered_helios_config",
    "inspect_trace_bundle",
    "load_json_mapping",
    "normalize_helios_config",
    "object_digest",
    "sha256_file",
    "validate_final_contract",
]

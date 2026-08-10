"""Fail-closed contracts for the official-semantics Helios GAN Route-A baseline.

GAN math remains in a pinned Helios checkout.  This module only freezes source,
config, trace, checkpoint, world16 backend, and the conservative full-offload
lifecycle used as the later GAN-pipeline denominator.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from distillschedule.core.config import load_plan
from distillschedule.core.contracts import ContractViolation
from distillschedule.runtime.launch_config import inspect_accelerate_config
from distillschedule.runtime.routea_evidence import (
    asset_identity,
    atomic_json,
    directory_inventory,
    inspect_trace_bundle,
    load_json_mapping,
    normalize_helios_config,
    object_digest,
    sha256_file,
)

GAN_SCHEMA = "distillschedule.routea_gan_world16.evidence_contract.v1"
GAN_NODE_SCHEMA = "distillschedule.routea_gan_world16.node_contract.v1"
OFFICIAL_HELIOS_REFERENCE_COMMIT = "47219a07860f158ce56a3b1d1ee2e012aab5c39b"
OFFICIAL_GAN_CONFIG_SHA = "a98abf36df8b0179deefb5b1df2de028087822f6"
EXPECTED_STEPS = (1000, 1001, 1002, 1003, 1004)
EXPECTED_INITIAL = {"generator": 200, "critic": 1000, "ema": 1000}
EXPECTED_FINAL = {"generator": 201, "critic": 1005, "ema": 1005}
EXPECTED_TRAJECTORY = {
    step: {"generator": 201, "critic": step + 1, "ema": step + 1}
    for step in EXPECTED_STEPS
}

OFFICIAL_GAN_FIELDS: Mapping[str, Any] = {
    "training_config.dmd_is_low_vram_mode": True,
    "training_config.is_gan_low_vram_mode": True,
    "training_config.is_train_dmd": True,
    "training_config.is_dmd_vae_decode": False,
    "training_config.is_use_gan": True,
    "training_config.gan_start_step": 1000,
    "training_config.is_separate_gan_grad": False,
    "training_config.is_use_gan_hooks": True,
    "training_config.is_use_gan_final": True,
    "training_config.gan_cond_map_dim": 768,
    "training_config.gan_hooks": [5, 15, 25, 35],
    "training_config.gan_g_weight": 5e-2,
    "training_config.gan_d_weight": 1e-2,
    "training_config.aprox_r1": True,
    "training_config.aprox_r2": True,
    "training_config.r1_weight": 100.0,
    "training_config.r2_weight": 0.0,
    "training_config.r1_sigma": 0.1,
    "training_config.r2_sigma": 0.1,
    "training_config.is_use_gt_history": True,
    "training_config.dfake_gen_update_ratio": 5,
    "training_config.train_batch_size": 1,
    "training_config.gradient_accumulation_steps": 1,
    "training_config.mixed_precision": "bf16",
    "training_config.use_ema": True,
    "training_config.is_use_reward_model": False,
    "data_config.single_height": 384,
    "data_config.single_width": 640,
}


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:  # pragma: no cover
        raise ContractViolation("GAN baseline validation requires OmegaConf", path=str(path)) from exc
    value = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(value, Mapping):
        raise ContractViolation("rendered Helios config must be a mapping", path=str(path))
    return copy.deepcopy(dict(value))


def _lookup(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ContractViolation("required field is missing", path=dotted)
        current = current[part]
    return current


def _expect(value: Mapping[str, Any], dotted: str, expected: Any) -> Any:
    observed = _lookup(value, dotted)
    if observed != expected:
        raise ContractViolation(f"observed {observed!r}, expected {expected!r}", path=dotted)
    return observed


def expected_gan_schedule() -> dict[str, Any]:
    per_step: dict[str, Any] = {}
    for step in EXPECTED_STEPS:
        train_g = step % 5 == 0
        per_step[str(step)] = {
            "train_generator": train_g,
            "gan_active": True,
            "generator_path": (
                ["generator_rollout", "dmd_generator_loss", "gan_G_loss", "generator_optimizer_step"]
                if train_g else []
            ),
            "critic_path": [
                "critic_denoising_loss",
                "gan_D_real_loss",
                "gan_D_fake_loss",
                "r1_r2_regularization",
                "single_joint_critic_optimizer_step",
            ],
            "ema_after_critic": True,
        }
    return {
        "source": "official_helios_train_helios_and_utils_helios_post",
        "gan_start_step": 1000,
        "generator_update_every": 5,
        "critic_update_every": 1,
        "steps": per_step,
        "score_ownership": "GAN heads live on the Helios-Base Score backbone and share the Critic optimizer.",
        "low_vram_critic_rule": "denoising and GAN/R1/R2 backwards feed one Critic optimizer update",
    }


def validate_official_gan_source(helios_root: str | Path) -> dict[str, Any]:
    root = Path(helios_root).expanduser().resolve()
    train = root / "train_helios.py"
    helper = root / "helios/utils/utils_helios_post.py"
    if not train.is_file() or not helper.is_file():
        raise ContractViolation("Helios GAN source files are missing", path=str(root))
    train_text = train.read_text(encoding="utf-8", errors="replace")
    helper_text = helper.read_text(encoding="utf-8", errors="replace")
    required_train = (
        "TRAIN_GENERATOR = global_step % args.training_config.dfake_gen_update_ratio == 0",
        "USE_GAN = args.training_config.is_use_gan and global_step >= args.training_config.gan_start_step",
        "is_use_gan=USE_GAN",
        "critic_optimizer.step()",
        "gan_critic_trainable_params",
    )
    required_helper = (
        "class Gan_D_Loss_With_Cached_Grad",
        "prepare_real_latents_for_gan",
        "gan_D_fake_loss",
        "gan_D_real_loss",
        "set_gradient_accumulation_boundary(is_boundary=False)",
        "set_gradient_accumulation_boundary(is_boundary=True)",
        "critic_engine.step()",
    )
    missing = [s for s in required_train if s not in train_text]
    missing += [s for s in required_helper if s not in helper_text]
    if missing:
        raise ContractViolation(
            "pinned Helios source no longer matches GAN scheduling signatures: " + ", ".join(missing),
            path=str(root),
        )
    return {
        "root": str(root),
        "train_helios_sha256": sha256_file(train),
        "utils_helios_post_sha256": sha256_file(helper),
        "reference_public_commit": OFFICIAL_HELIOS_REFERENCE_COMMIT,
        "reference_gan_config_sha": OFFICIAL_GAN_CONFIG_SHA,
        "schedule_signature_status": "PASS",
    }


def validate_gan_plan(plan_path: str | Path) -> dict[str, Any]:
    loaded = load_plan(plan_path)
    raw = loaded.plan.to_dict()
    for dotted, expected in {
        "hardware.nodes": 2,
        "hardware.local_world_size": 8,
        "hardware.world_size": 16,
        "hardware.reserve_gib": 0.0,
        "workload.shape.batch_size": 1,
        "workload.shape.resolution.height": 384,
        "workload.shape.resolution.width": 640,
        "workload.schedule.generator_update_every": 5,
        "workload.schedule.critic_update_every": 1,
        "workload.schedule.ema_update_every": 1,
        "workload.schedule.ema_after_critic": True,
        "workload.branches.dmd": True,
        "workload.branches.gan": True,
        "workload.branches.reward": False,
        "workload.branches.vae_decode": False,
        "workload.branches.ema": True,
        "evidence.start_step": 1000,
        "evidence.steps": 5,
    }.items():
        _expect(raw, dotted, expected)
    initial = _lookup(raw, "evidence.metadata.initial_versions")
    if initial != EXPECTED_INITIAL:
        raise ContractViolation(f"initial versions={initial}, expected={EXPECTED_INITIAL}", path="evidence.metadata.initial_versions")
    for component in ("generator", "score"):
        backend = _lookup(raw, f"training_backends.{component}")
        for key, expected in {
            "kind": "deepspeed",
            "zero_stage": 1,
            "optimizer_offload": "cpu",
            "parameter_offload": "none",
            "gradient_checkpointing": True,
        }.items():
            if backend.get(key) != expected:
                raise ContractViolation(f"observed {backend.get(key)!r}, expected {expected!r}", path=f"training_backends.{component}.{key}")
    lifecycle = _lookup(raw, "lifecycle")
    streaming = lifecycle.get("frozen_block_streaming") if isinstance(lifecycle, Mapping) else None
    if not isinstance(streaming, Mapping):
        raise ContractViolation("full frozen-block streaming is required", path="lifecycle")
    if streaming.get("enabled") is not True or streaming.get("window") != 1:
        raise ContractViolation("streaming window must be 1", path="lifecycle.frozen_block_streaming")
    if streaming.get("components") != ["generator", "score"] or streaming.get("block_indices") != list(range(40)):
        raise ContractViolation("all 40 Generator/Score blocks must stream", path="lifecycle.frozen_block_streaming")
    if lifecycle.get("prefetch_distance") != 0:
        raise ContractViolation("baseline forbids prefetch overlap", path="lifecycle.prefetch_distance")
    return {
        "plan_id": loaded.plan.plan_id,
        "plan_digest": loaded.digest,
        "source_files": [str(path) for path in loaded.source_files],
        "expected_steps": list(EXPECTED_STEPS),
        "initial_versions": dict(EXPECTED_INITIAL),
        "final_versions": dict(EXPECTED_FINAL),
    }


def validate_rendered_gan_config(path: str | Path, *, checkpoint: str | Path) -> dict[str, Any]:
    candidate = Path(path).expanduser().resolve()
    config = _load_yaml(candidate)
    for dotted, expected in OFFICIAL_GAN_FIELDS.items():
        _expect(config, dotted, expected)
    _expect(config, "training_config.max_train_steps", 1005)
    _expect(config, "training_config.checkpointing_steps", 1005)
    resume = str(_lookup(config, "training_config.resume_from_checkpoint"))
    expected_checkpoint = str(Path(checkpoint).expanduser().resolve())
    if str(Path(resume).expanduser().resolve()) != expected_checkpoint or Path(resume).name != "checkpoint-1000":
        raise ContractViolation("GAN baseline must resume the canonical checkpoint-1000", path="training_config.resume_from_checkpoint")
    if _lookup(config, "training_config.dmd_generator_deepspeed_config") != _lookup(config, "training_config.dmd_critic_deepspeed_config"):
        raise ContractViolation("Generator/Score DeepSpeed configs differ", path="training_config.dmd_*_deepspeed_config")
    normalized = normalize_helios_config(candidate)
    return {
        "path": str(candidate),
        "sha256": sha256_file(candidate),
        "normalized_digest": normalized["normalized_digest"],
        "official_gan_fields": dict(OFFICIAL_GAN_FIELDS),
        "resume_checkpoint": expected_checkpoint,
    }


def _trace_steps(trace: Mapping[str, Any]) -> set[int]:
    return {
        int(op["control"]["global_step"])
        for op in trace.get("operations", [])
        if isinstance(op, Mapping)
        and isinstance(op.get("control"), Mapping)
        and isinstance(op["control"].get("global_step"), int)
    }


def validate_common_rank_synchronized_trace(trace_root: str | Path) -> dict[str, Any]:
    inventory = inspect_trace_bundle(Path(trace_root).expanduser().resolve(), expected_world_size=16)
    rank_controls: dict[int, dict[str, Any]] = {}
    for rank_entry in inventory["ranks"]:
        rank = int(rank_entry["rank"])
        trace_path = Path(str(rank_entry["trace_manifest_path"]))
        trace = load_json_mapping(trace_path)
        missing = set(EXPECTED_STEPS) - _trace_steps(trace)
        if missing:
            raise ContractViolation(f"rank {rank} trace misses steps {sorted(missing)}", path=str(trace_path))
        controls: dict[str, Any] = {}
        for operation in trace.get("operations", []):
            if not isinstance(operation, Mapping) or not isinstance(operation.get("control"), Mapping):
                continue
            control = operation["control"]
            schedule = control.get("pyramid_stage_call_sequence")
            if not isinstance(schedule, list):
                continue
            scope = control.get("stage_schedule_scope")
            if scope is not None and scope != "global_rank0_broadcast_equivalent":
                raise ContractViolation(f"rank {rank} uses out-of-scope rank-local stage schedule {scope!r}", path=str(trace_path))
            initial = control.get("initial_synchronized_pyramid_stage_exit")
            if not isinstance(initial, list):
                raise ContractViolation("trace lacks synchronized initial exits", path=str(trace_path))
            controls[str(operation.get("key"))] = {
                "initial": [int(value) for value in initial],
                "executed": [int(item["denoising_exit"]) for item in schedule],
            }
        if not controls:
            raise ContractViolation("trace has no frozen pyramid-stage schedule evidence", path=str(trace_path))
        rank_controls[rank] = controls
    reference = rank_controls[0]
    for rank, controls in rank_controls.items():
        if controls != reference:
            raise ContractViolation(f"rank {rank} DiT denoising/stage exit schedule differs from rank 0", path=f"trace.rank.{rank}")
    return {
        **inventory,
        "common_rank_schedule_status": "PASS",
        "rank_local_exception_supported": False,
        "schedule_fingerprint": object_digest(reference),
    }


def build_node_contract(
    *,
    node_rank: int,
    plan_path: str | Path,
    compiled_preflight_path: str | Path,
    rendered_config_path: str | Path,
    accelerate_config_path: str | Path,
    trace_bundle_path: str | Path,
    checkpoint_path: str | Path,
    runtime_identity_path: str | Path,
    helios_root: str | Path,
    assets: Mapping[str, str | Path],
) -> dict[str, Any]:
    if node_rank not in (0, 1):
        raise ContractViolation("node_rank must be 0 or 1", path="node_rank")
    plan = validate_gan_plan(plan_path)
    rendered = validate_rendered_gan_config(rendered_config_path, checkpoint=checkpoint_path)
    launch = inspect_accelerate_config(
        accelerate_config_path,
        expected_num_machines=2,
        expected_num_processes=16,
        expected_machine_rank=node_rank,
    ).to_dict()
    trace = validate_common_rank_synchronized_trace(trace_bundle_path)
    checkpoint = directory_inventory(checkpoint_path)
    runtime_identity = load_json_mapping(runtime_identity_path)
    source = validate_official_gan_source(helios_root)
    compiled = load_json_mapping(compiled_preflight_path)
    if compiled.get("plan_digest") != plan["plan_digest"]:
        raise ContractViolation("compiled plan digest mismatch", path="compiled_preflight.plan_digest")
    shared = {
        "plan": plan,
        "rendered_config": {
            "normalized_digest": rendered["normalized_digest"],
            "official_gan_fields": rendered["official_gan_fields"],
        },
        "launcher": {
            "distributed_type": launch["distributed_type"],
            "num_machines": launch["num_machines"],
            "num_processes": launch["num_processes"],
            "multinode_launcher": launch["multinode_launcher"],
            "deepspeed": launch["deepspeed"],
        },
        "trace": {
            "bundle_hash": trace["bundle_hash"],
            "inventory_digest": trace["inventory_digest"],
            "schedule_fingerprint": trace["schedule_fingerprint"],
            "common_rank_schedule_status": trace["common_rank_schedule_status"],
            "ranks": trace["ranks"],
        },
        "checkpoint": checkpoint.to_dict(),
        "source": source,
        "runtime_identity": runtime_identity,
        "assets": {name: asset_identity(path) for name, path in sorted(assets.items())},
        "official_schedule": expected_gan_schedule(),
    }
    return {
        "schema_version": GAN_NODE_SCHEMA,
        "node_rank": node_rank,
        "shared": shared,
        "shared_digest": object_digest(shared),
        "node": {
            "rendered_config_path": rendered["path"],
            "rendered_config_sha256": rendered["sha256"],
            "accelerate_config_path": str(Path(accelerate_config_path).resolve()),
            "runtime_identity_path": str(Path(runtime_identity_path).resolve()),
        },
    }


def finalize_node_contracts(paths: list[str | Path]) -> dict[str, Any]:
    if len(paths) != 2:
        raise ContractViolation("exactly two node contracts are required", path="nodes")
    nodes = [load_json_mapping(path) for path in paths]
    if sorted(node.get("node_rank") for node in nodes) != [0, 1]:
        raise ContractViolation("node contracts must contain ranks 0 and 1", path="nodes")
    if any(node.get("schema_version") != GAN_NODE_SCHEMA for node in nodes):
        raise ContractViolation("unsupported GAN node contract schema", path="nodes")
    digests = {node.get("shared_digest") for node in nodes}
    if len(digests) != 1:
        raise ContractViolation("node normalized GAN contracts differ", path="nodes")
    shared = nodes[0].get("shared")
    if not isinstance(shared, Mapping):
        raise ContractViolation("node shared contract missing", path="nodes.0.shared")
    payload: dict[str, Any] = {
        "schema_version": GAN_SCHEMA,
        "status": "PASS",
        "shared_digest": next(iter(digests)),
        "shared": dict(shared),
        "nodes": [{"node_rank": node["node_rank"], "node": node["node"]} for node in nodes],
    }
    payload["contract_digest"] = object_digest(payload)
    return payload


def validate_final_contract(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != GAN_SCHEMA or value.get("status") != "PASS":
        raise ContractViolation("GAN evidence contract did not PASS", path="evidence_contract")
    unsigned = dict(value)
    stored = unsigned.pop("contract_digest", None)
    if stored != object_digest(unsigned):
        raise ContractViolation("GAN evidence contract digest mismatch", path="evidence_contract")
    shared = value.get("shared")
    if not isinstance(shared, Mapping):
        raise ContractViolation("GAN evidence shared payload missing", path="evidence_contract.shared")
    trace = shared.get("trace")
    if not isinstance(trace, Mapping) or trace.get("common_rank_schedule_status") != "PASS":
        raise ContractViolation("rank-synchronized trace did not PASS", path="evidence_contract.shared.trace")
    plan = shared.get("plan")
    if not isinstance(plan, Mapping) or plan.get("expected_steps") != list(EXPECTED_STEPS):
        raise ContractViolation("GAN Full5 logical window drift", path="evidence_contract.shared.plan")


__all__ = [
    "EXPECTED_FINAL", "EXPECTED_INITIAL", "EXPECTED_STEPS", "EXPECTED_TRAJECTORY",
    "GAN_NODE_SCHEMA", "GAN_SCHEMA", "atomic_json", "build_node_contract",
    "expected_gan_schedule", "finalize_node_contracts", "load_json_mapping",
    "sha256_file", "validate_common_rank_synchronized_trace", "validate_final_contract",
    "validate_gan_plan", "validate_official_gan_source", "validate_rendered_gan_config",
]

#!/usr/bin/env python3
"""H100 Helios launch preparation and physical/logical evidence admission.

This is a stdlib/CPU-only control plane around the unchanged P0 math worker.
Eight physical receipts remain eight receipts: actual child records must prove
the sixteen B1 contributions. No A100 summarizer, reconstructed leaf, inferred
version or numeric cross-arm PASS is used here. Timing admission is diagnostic
only until a separate full numerical comparison has actually been implemented.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from distillschedule.runtime.fsdp2_p0 import StaticFSDP2P0Policy
from distillschedule.runtime.fsdp2_static import contract_for_variant
from distillschedule.workloads.best_known.registry import (
    HELIOS_SHA, _object, _require, file_sha256, fingerprint, load_profile,
)

LAUNCH_SCHEMA = "distillschedule.helios_h100_launch.v1"
HEALTH_SCHEMA = "distillschedule.helios_h100_health.v1"
SUMMARY_SCHEMA = "distillschedule.helios_h100_summary.v1"
ASSET_SCHEMA = "distillschedule.helios_h100_asset_gate.v1"
HARDWARE = "h100-8-b16"
FLASH_SHA = "c2acd487c5d1d77b7f6721eabc0fee4f8c5c8f9e682aadf43bfa99b394eecba7"
FP32_KEEP = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3", "history_key_scale"]
digest = fingerprint


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _new_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _env_key(field: str) -> str:
    prefix = "DISTILLSCHEDULE_" if field.startswith("elide_") or field in ("optimizer_dispatch_warmup", "cuda_allocator_expandable_segments") or field.endswith("residual_storage") else "DISTILLSCHEDULE_HELIOS_"
    return prefix + field.upper()


def freeze_policy(root: Path, *, variant: str, arm: str) -> dict[str, Any]:
    """Read the locked winner and render a complete, explicit worker policy."""
    _require(arm in ("baseline", "best"), "arm must be baseline or best")
    profile = load_profile("helios14b_nogan_best" if variant == "no-gan" else "helios14b_gan_best", root=root)
    contract = _jsonable(dataclasses.asdict(contract_for_variant(variant)))
    _require(profile["contract"]["variant"] == variant, "variant/profile mismatch")
    _require(profile["contract"]["initial_versions"] == contract["initial_versions"] and profile["contract"]["final_versions"] == contract["final_versions"], "frozen version contract changed")
    defaults = dataclasses.asdict(StaticFSDP2P0Policy())
    env = dict(profile["runtime"]["environment"])
    if arm == "best":
        values = dict(defaults, mesh_policy=profile["topology"]["placement"])
        for field, default in defaults.items():
            if field in ("mesh_policy", "rollout_batch_size", "pack_real_score_pair", "rollout_backend"):
                continue
            key = _env_key(field)
            _require(key in env, f"winner does not bind {key}")
            raw = env[key]
            if type(default) is bool:
                _require(raw in ("on", "off"), f"non-explicit boolean {key}")
                values[field] = raw == "on"
            elif type(default) is int:
                values[field] = int(raw)
            else:
                values[field] = raw
        values["rollout_batch_size"] = profile["topology"]["rollout_batch_size"]
        values["pack_real_score_pair"] = profile["contract"]["score_packing"]
        p0 = StaticFSDP2P0Policy(**values)
        extra = {"attention_backend": env[_env_key("attention_backend")],
                 "generator_nograd_lora_fused_epilogue": env[_env_key("generator_nograd_lora_fused_epilogue")] == "on",
                 "all_role_lora_scale1_add": env[_env_key("all_role_lora_scale1_add")] == "on"}
    else:
        p0 = StaticFSDP2P0Policy()
        extra = {"attention_backend": "sdpa", "generator_nograd_lora_fused_epilogue": False, "all_role_lora_scale1_add": False}
    policy = dict(dataclasses.asdict(p0), **extra, generator_residual_storage="cuda_out_of_place", critic_residual_storage="cuda_out_of_place", id=p0.id)
    _require(policy["rollout_batch_size"] == 1 and policy["pack_real_score_pair"] is False, "H100 migration must preserve B1 and packing OFF")
    _require(profile["memory"]["ema_device"] == "cuda", "winner EMA contract changed")
    for field, value in policy.items():
        if field in ("id", "mesh_policy", "rollout_batch_size", "pack_real_score_pair", "rollout_backend"):
            continue
        env[_env_key(field)] = "on" if value is True else "off" if value is False else str(value)
    # Profiling/diagnostic switches cannot leak in from the caller. Correctness
    # retains scalar/fingerprint diagnostics; timing omits them in the worker.
    env.update(DISTILLSCHEDULE_HARDWARE_PROFILE=HARDWARE, DISTILLSCHEDULE_EMA_DEVICE="cuda",
               DISTILLSCHEDULE_NODE_RANK="0", DISTILLSCHEDULE_PROFILE_DIAGNOSTICS="off",
               DISTILLSCHEDULE_CRITIC_ONLY_STEP="", DISTILLSCHEDULE_OPERATION_SOURCE="off")
    trace_key = "trace"
    logical_key = "logical_checkpoint"
    return {"arm": arm, "variant": variant, "winner": profile["contract"]["winner"] if arm == "best" else "P0_all_performance_options_off",
            "profile_id": profile["id"], "profile_sha256": digest(profile), "policy": policy,
            "policy_fingerprint": digest(policy), "contract": contract,
            "trace_hash": profile["required_assets"][trace_key]["logical_hash"],
            "logical_hash": profile["required_assets"][logical_key]["logical_hash"],
            "mesh_shape": [2, 4] if policy["mesh_policy"] == "node-hsdp" else [8],
            "environment": env, "unset_environment": profile["runtime"]["unset_environment"]}


def policy_arguments(policy: dict[str, Any]) -> list[str]:
    result = []
    for field, value in policy.items():
        if field == "id":
            continue
        flag = field.replace("_", "-")
        if type(value) is bool:
            result.append("--" + ("" if value else "no-") + flag)
        else:
            result.extend(("--" + flag, str(value)))
    return result


def _confined(root: Path, path: Path) -> Path:
    root, path = root.resolve(), path.resolve()
    _require(path.is_relative_to(root), f"asset/evidence path escapes owned root: {path}")
    return path


def verify_asset_gate(path: Path, expected_sha256: str, *, expected_head: str, asset_root: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    _require(bool(re.fullmatch(r"[0-9a-f]{64}", expected_sha256)), "asset gate SHA must be explicit")
    _require(path.is_file() and file_sha256(path) == expected_sha256, "asset gate receipt SHA mismatch")
    gate = _object(path)
    _require(gate.get("schema_version") == ASSET_SCHEMA and gate.get("status") == "PASS", "complete H100 full-hash asset gate must be PASS")
    _require(gate.get("exact_head") == expected_head, "asset gate HEAD mismatch")
    _require(Path(gate.get("artifact_root", "/__missing__")).resolve() == asset_root.resolve(), "asset gate owned artifact_root mismatch")
    groups = gate.get("groups", [])
    _require(isinstance(groups, list) and groups, "asset gate groups missing")
    by_name = {}
    for group in groups:
        name = group.get("name", group.get("role"))
        _require(isinstance(name, str) and name not in by_name, "asset gate duplicate or missing group name")
        actual_root = _confined(asset_root, Path(group["actual_root"]))
        _require(actual_root.is_dir(), f"asset group directory missing: {name}")
        files = group.get("files", [])
        _require(isinstance(files, list) and files, f"full-hash file inventory missing: {name}")
        records = {}
        for record in files:
            relative = record.get("relative_path", "")
            _require(isinstance(relative, str) and relative and not Path(relative).is_absolute() and ".." not in Path(relative).parts and relative not in records, f"invalid/duplicate asset relative_path: {name}")
            expected = record.get("sha256")
            _require(isinstance(expected, str) and bool(re.fullmatch(r"[0-9a-f]{64}", expected)), f"asset file SHA missing: {name}/{relative}")
            _require(record.get("status") == "PASS" and record.get("observed_sha256") == expected, f"full-hash asset file receipt failed: {name}/{relative}")
            file = _confined(actual_root, actual_root / relative)
            _require(file.is_file() and file.stat().st_size == record.get("bytes") and file_sha256(file) == expected, f"live asset file bytes/SHA mismatch: {name}/{relative}")
            records[relative] = record
        by_name[name] = dict(group, actual_root=str(actual_root), file_records=records)
    required = {"generator_precast", "score_precast", "trace", "logical_checkpoint", "negative_prompt"}
    if frozen["policy"]["attention_backend"] == "local-fa2":
        required.add("flash_attn_kernel")
    _require(required <= set(by_name), f"required full-hash groups missing: {sorted(required - set(by_name))}")
    for name, subfolder in (("generator_precast", "transformer_ode"), ("score_precast", "transformer")):
        group = by_name[name]
        manifest_path = Path(group["actual_root"]) / "distillschedule_precast_manifest.json"
        _require(manifest_path.name in group["file_records"], f"precast manifest not fully hashed: {name}")
        _require(file_sha256(manifest_path) == group.get("precast_manifest_sha256"), f"precast manifest SHA mismatch: {name}")
        manifest = _object(manifest_path)
        source_path = group.get("source_path")
        _require(manifest.get("schema_version") == "distillschedule.precast_safetensors.v1", "precast manifest schema mismatch")
        _require(isinstance(source_path, str) and Path(source_path).is_absolute() and manifest.get("source", {}).get("path") == source_path and Path(source_path).name == subfolder, f"unchanged precast source.path/provenance mismatch: {name}")
        artifact = manifest.get("artifact", {})
        _require(artifact.get("target_dtype") == "bfloat16" and artifact.get("keep_in_fp32_modules") == FP32_KEEP, f"precast mixed-dtype contract changed: {name}")
        shard_records = artifact.get("shards", [])
        _require(isinstance(shard_records, list) and shard_records, f"precast shard inventory missing: {name}")
        seen = set()
        for shard in shard_records:
            relative = shard.get("file")
            _require(isinstance(relative, str) and relative not in seen and Path(relative).name == relative, f"invalid precast shard: {name}")
            seen.add(relative)
            row = group["file_records"].get(relative, {})
            _require(row.get("sha256") == shard.get("sha256") and row.get("bytes") == shard.get("size_bytes") and bool(row), f"full-hash gate omits/changes original precast shard: {name}/{relative}")
        for prefix in ("config", "index"):
            relative = artifact.get(prefix + "_file")
            _require(relative in group["file_records"] and group["file_records"][relative]["sha256"] == artifact.get(prefix + "_sha256"), f"precast {prefix} inventory mismatch: {name}")
        index = _object(Path(group["actual_root"]) / artifact["index_file"])
        weights = index.get("weight_map", {})
        _require(isinstance(weights, dict) and set(weights.values()) == seen and len(weights) == artifact.get("tensor_count"), f"precast index/shard tensor inventory mismatch: {name}")
        _require(index.get("metadata", {}).get("total_size") == artifact.get("tensor_bytes"), f"precast tensor byte inventory mismatch: {name}")
        group["provenance_root"] = str(Path(source_path).parent)
    trace_root, logical_root = Path(by_name["trace"]["actual_root"]), Path(by_name["logical_checkpoint"]["actual_root"])
    bundle = _object(trace_root / "bundle.json")
    _require(bundle.get("bundle_hash") == frozen["trace_hash"], "asset trace frozen hash mismatch")
    _require("bundle.json" in by_name["trace"]["file_records"], "trace bundle not full-hash verified")
    for rank in bundle.get("ranks", []):
        for filename in ("trace.json", "tensors.pt"):
            _require(str(Path(rank["path"]) / filename) in by_name["trace"]["file_records"], f"full-hash gate omits rank trace payload: {rank.get('rank')}/{filename}")
    logical = _object(logical_root / "manifest.json")
    _require(logical.get("manifest_hash") == frozen["logical_hash"] and "manifest.json" in by_name["logical_checkpoint"]["file_records"], "asset logical checkpoint frozen hash mismatch")
    if "flash_attn_kernel" in required:
        _require(by_name["flash_attn_kernel"]["file_records"].get("_flash_attn2_cuda_a511a4c.abi3.so", {}).get("sha256") == FLASH_SHA, "pinned FlashAttention-2 extension SHA mismatch")
    return {"path": str(path.resolve()), "sha256": expected_sha256, "artifact_root": str(asset_root.resolve()), "groups": by_name,
            "fingerprint": digest({"sha256": expected_sha256, "groups": by_name})}


def _operations(trace: dict[str, Any]) -> dict[str, Any]:
    rows = trace.get("operations", [])
    _require(isinstance(rows, list), "trace operation inventory missing")
    result = {row["key"]: row for row in rows}
    _require(len(result) == len(rows), "trace operation keys duplicated")
    return result


def entry_identity(trace: dict[str, Any], *, variant: str, slot: int, step: int, role: str) -> str:
    """Audit the existing StaticWorld16Trace binding, not a new input/trace.

    Field names and identity preimage are the unchanged source contract from
    runtime/static_world16_trace.py. Only descriptor SHA metadata is read.
    """
    prefix = f"step/{step}/{role}"
    keys = {"text_embedding": "sample/0/text_embedding", "attention_mask": "sample/0/attention_mask"}
    for name in ("continuous_history", "history_short", "history_mid", "history_long", "indices_hidden_states",
                 "indices_history_short", "indices_history_mid", "indices_history_long", "entry_noise", "rollout_noise"):
        keys[name] = prefix + "/" + name
    keys.update({f"block_noise_stage{s}": f"{prefix}/block_noise/{s}" for s in (1, 2)})
    extra = ["dmd_timestep", "dmd_noise"] if role == "generator" else ["critic_timestep", "critic_noise"]
    if variant == "gan":
        extra += ["gan_g_noise", "gan_g_crop"] if role == "generator" else ["gan_d_fake_crop", "gan_d_real_crop", "gan_d_r1_crop", "r1_epsilon"]
    keys.update({name: prefix + "/" + name for name in extra})
    tensor_hashes = {name: trace["tensors"][key]["sha256"] for name, key in keys.items()}
    _require(all(isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha) for sha in tensor_hashes.values()), "frozen input/noise SHA descriptor missing")
    return digest({"variant": variant, "phase": "generator_update" if role == "generator" else "critic",
                   "rank": slot, "step": step, "trace_hash": trace["trace_hash"], "tensor_hashes": tensor_hashes})


def trace_controls(root: Path, variant: str, expected_hash: str) -> dict[int, dict[str, Any]]:
    """Read, hash and validate all 16 immutable control rows without Torch."""
    bundle = _object(root / "bundle.json")
    unsigned = dict(bundle)
    _require(unsigned.pop("bundle_hash", None) == expected_hash == digest(unsigned), "trace bundle hash mismatch")
    _require(bundle.get("schema_version") == "distillschedule.operation_trace_bundle.v1", "trace bundle schema mismatch")
    _require([r.get("rank") for r in bundle.get("ranks", [])] == list(range(16)), "trace must retain exact sixteen-slot bijection")
    contract = _jsonable(dataclasses.asdict(contract_for_variant(variant)))
    traces = {}
    signatures = {}
    for entry in bundle["ranks"]:
        slot = entry["rank"]
        manifest = _object(_confined(root, root / entry["path"]) / "trace.json")
        unsigned = dict(manifest)
        observed = unsigned.pop("trace_hash", None)
        _require(observed == entry.get("trace_hash") == digest(unsigned), f"slot{slot} immutable trace hash mismatch")
        expected_contract = {"resolution": [384, 640], "dtype": "bfloat16", "stage_spatial_shapes": [[12, 20], [24, 40], [48, 80]],
                             "full_latent_shape": [1, 16, 9, 48, 80], "dfake_gen_update_ratio": 5}
        _require(all(manifest.get("contract", {}).get(k) == v for k, v in expected_contract.items()), f"slot{slot} frozen trace contract changed")
        operations = _operations(manifest)
        for step in contract["logical_steps"]:
            branch = operations[f"step/{step}/branch"]["control"]
            expected_branch = {"global_step": step, "dfake_gen_update_ratio": 5, "train_generator": step == contract["logical_steps"][0],
                               "train_critic": True, "gan": variant == "gan", "reward": False, "vae_decode": False}
            _require(all(branch.get(k) == v for k, v in expected_branch.items()), f"slot{slot} frozen branch changed")
            for role in ("generator", "critic"):
                if role == "generator" and step != contract["logical_steps"][0]:
                    continue
                key = f"step/{step}/{role}/rollout"
                control = operations[key]["control"]
                stages = control["pyramid_stage_call_sequence"]
                _require(control.get("stage_schedule_scope") == "global_rank0_broadcast_equivalent", "rank-local rollout exit is outside frozen source semantics")
                _require(not {"local_exception_seed", "official_local_exception_predicate"} & set(control), "rank-local exception trace forbidden")
                _require([s["stage"] for s in stages] == [0, 1, 2] and [s["denoising_exit"] for s in stages] == control.get("initial_synchronized_pyramid_stage_exit"), "synchronized exits/stages changed")
                _require(all(s["call_count"] == s["denoising_exit"] + 1 for s in stages) and sum(s["call_count"] for s in stages) == control["call_count"], "frozen rollout count inconsistent with exits")
                signature = [[s["stage"], s["call_count"], s["denoising_exit"], s["latent_shape"], len(s["timestep"])] for s in stages]
                _require(key not in signatures or signatures[key] == signature, "frozen physical call sequence diverges across logical slots")
                signatures[key] = signature
        traces[slot] = manifest
    return traces


def _check_rollout(row: dict[str, Any], trace: dict[str, Any], *, variant: str, slot: int, step: int, role: str, label: str, errors: list[str]) -> None:
    control = _operations(trace)[f"step/{step}/{role}/rollout"]["control"]
    stages = control["pyramid_stage_call_sequence"]
    counts = {str(s["stage"]): s["call_count"] for s in stages}
    for key, expected in (("call_count", control["call_count"]), ("logical_call_count", control["call_count"]),
                          ("stage_call_count", counts), ("logical_stage_call_count", counts), ("dummy_call_count", 0)):
        if row.get(key) != expected:
            errors.append(f"{label}: actual call inventory {key} changed/missing")
    calls = row.get("calls", [])
    if not isinstance(calls, list) or len(calls) != control["call_count"]:
        errors.append(f"{label}: actual call inventory missing")
        calls = []
    index, selected = 0, []
    for stage in stages:
        for denoise in range(stage["call_count"]):
            call = calls[index] if index < len(calls) else {}
            grad = role == "generator" and denoise == stage["denoising_exit"]
            expected = {"call_index": index, "stage": stage["stage"], "denoise_call_in_stage": denoise,
                        "hidden_states_shape": stage["latent_shape"], "status": "success",
                        "grad_enabled": grad, "output_requires_grad": grad}
            if any(call.get(k) != v for k, v in expected.items()):
                errors.append(f"{label}: actual call inventory/source exit mismatch at call{index}")
            if grad:
                selected.append(index)
            index += 1
    if role == "generator" and row.get("selected_call_indices") != selected:
        errors.append(f"{label}: selected exit call inventory changed")
    if row.get("output_dtype") != "torch.bfloat16" or row.get("output_shape") != [1, 16, 9, 48, 80] or row.get("output_requires_grad") is not (role == "generator"):
        errors.append(f"{label}: actual output shape/dtype/autograd contract changed")
    if role == "generator":
        if row.get("operation_trace_hash") != trace["trace_hash"] or row.get("source_rank") != slot or row.get("trace_identity_sha256") != entry_identity(trace, variant=variant, slot=slot, step=step, role=role):
            errors.append(f"{label}: rank-qualified source/trace identity changed")
    elif row.get("batch_size") != 1 or row.get("logical_steps") != [step]:
        errors.append(f"{label}: B1 rollout step identity changed")
    elif row.get("member_trace_identities") != [entry_identity(trace, variant=variant, slot=slot, step=step, role=role)] or row.get("trace_key") != f"static-fsdp2-p0/batched/{step}":
        errors.append(f"{label}: original B1 input/noise/trace identity changed")


def _finite_group(group: dict[str, Any], label: str, errors: list[str]) -> None:
    if group.get("finite") is not True or not re.fullmatch(r"[0-9a-f]{64}", str(group.get("sha256", ""))) or group.get("tensor_count", 0) < 1:
        errors.append(f"{label}: finite/state SHA audit missing or failed")
    tensors = group.get("tensors", [])
    if len(tensors) != group.get("tensor_count"):
        errors.append(f"{label}: actual tensor audit inventory missing")
    for tensor in tensors:
        if tensor.get("finite") is not True or not re.fullmatch(r"[0-9a-f]{64}", str(tensor.get("sha256", ""))) or any(not _finite(tensor.get(k)) for k in ("sum", "l2", "max_abs")):
            errors.append(f"{label}: actual tensor finite/SHA audit failed")


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _audit_persistence(row: dict[str, Any], *, slots: list[int], policy: dict[str, Any], label: str, errors: list[str]) -> None:
    receipt = row.get("generator_nograd_persistent_unshard", {})
    children = receipt.get("microbatches", [])
    if receipt.get("logical_b1_microbatch_count") != 2 or receipt.get("rollout_batch_count") != 10 or [c.get("logical_slot") for c in children] != slots:
        errors.append(f"{label}: actual per-slot reshard persistence inventory missing")
    for child in children:
        p = child.get("persistence", {})
        if p.get("enabled") is not policy["generator_nograd_persistent_unshard"] or p.get("rollout_batch_count") != 5:
            errors.append(f"{label}: persistent unshard policy/rollout coverage changed")
        if policy["generator_nograd_persistent_unshard"]:
            expected = {"scope": "GENERATOR_NOGRAD_ROLLOUTS_ONLY", "expected_fsdp_module_count": 41, "fsdp_module_count": 41,
                        "parameter_group_count": 41, "set_no_reshard_count": 41, "restore_reshard_policy_count": 41,
                        "explicit_reshard_count": 41, "fully_unsharded_before_exit_count": 41, "unsharded_after_exit_count": 0,
                        "restored": True, "flush_allocator_cache": policy["generator_nograd_persistent_unshard_flush_allocator_cache"],
                        "pre_empty_cache_count": int(policy["generator_nograd_persistent_unshard_flush_allocator_cache"]),
                        "post_empty_cache_count": int(policy["generator_nograd_persistent_unshard_flush_allocator_cache"])}
            if any(p.get(k) != v for k, v in expected.items()) or len(p.get("rollout_memory", [])) != 5:
                errors.append(f"{label}: persistent unshard failed final reshard/restore")


def _audit_parity(row: dict[str, Any], *, required: bool, label: str, errors: list[str]) -> None:
    parity = row.get("batch_parity", {})
    if required:
        samples = parity.get("samples", [])
        if parity.get("enabled") is not True or parity.get("status") != "PASS" or len(samples) != 1:
            errors.append(f"{label}: runtime B1 reference parity missing or failed")
        for sample in samples:
            expected = {"status": "PASS", "finite": True, "same_shape": True, "same_dtype": True}
            if any(sample.get(k) != v for k, v in expected.items()) or any(not _finite(sample.get(k)) for k in ("mean_relative", "normalized_l2", "cosine", "max_abs", "max_abs_limit")):
                errors.append(f"{label}: strict runtime parity metrics missing or failed")
            elif sample["mean_relative"] > .005 or sample["normalized_l2"] > .005 or sample["cosine"] < .9999 or sample["max_abs"] > sample["max_abs_limit"]:
                errors.append(f"{label}: strict runtime parity threshold failed")
    elif parity.get("enabled"):
        errors.append(f"{label}: unexpected runtime reference parity work")


def summarize(run_root: Path, *, expected_head: str, frozen_policy: dict[str, Any] | None = None) -> dict[str, Any]:
    errors: list[str] = []
    loss_evidence = []
    evidence: list[dict[str, str]] = []
    manifest_path, health_path = run_root / "manifests/launch.json", run_root / "run_status.json"
    def read(path: Path, label: str) -> dict[str, Any]:
        try:
            value = _object(path)
            evidence.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
            return value
        except (OSError, ValueError, TypeError) as error:
            errors.append(f"{label} missing/malformed: {error}")
            return {}
    manifest = read(manifest_path, "launcher manifest")
    health = read(health_path, "launcher health/teardown")
    frozen = frozen_policy or freeze_policy(ROOT, variant=manifest.get("variant", "no-gan"), arm=manifest.get("arm", "baseline"))
    policy, contract = frozen["policy"], frozen["contract"]
    variant, mode = frozen["variant"], manifest.get("mode")
    if manifest.get("schema_version") != LAUNCH_SCHEMA or manifest.get("frozen") != frozen:
        errors.append("launcher frozen policy binding changed/missing")
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("asset_fingerprint", ""))):
        errors.append("launcher full-hash asset binding missing")
    gate_binding = manifest.get("asset_gate", {})
    if gate_binding:
        gate_path = Path(gate_binding.get("path", "/__missing__"))
        gate = read(gate_path, "full-hash asset gate")
        if gate_path.is_file() and file_sha256(gate_path) != gate_binding.get("sha256"):
            errors.append("full-hash asset gate receipt SHA changed")
        if gate.get("status") != "PASS" or gate.get("exact_head") != expected_head or gate.get("schema_version") != ASSET_SCHEMA:
            errors.append("full-hash asset gate PASS/HEAD identity failed")
    else:
        errors.append("full-hash asset gate receipt binding missing")
    if manifest.get("expected_head") != expected_head or health.get("expected_head") != expected_head or health.get("observed_head_after") != expected_head or health.get("clean_source_after") is not True:
        errors.append("launcher exact clean HEAD binding failed")
    if health.get("schema_version") != HEALTH_SCHEMA or health.get("status") != "PASS" or health.get("exit_code") != 0:
        errors.append("launcher construction/run/teardown receipt failed")
    if health.get("cgroup_oom_delta") != {"oom": 0, "oom_kill": 0} or health.get("gpu_processes_before") != [] or health.get("gpu_processes_after") != []:
        errors.append("launcher CPU/cgroup OOM or GPU ownership teardown failed")
    if manifest_path.is_file() and health.get("launch_manifest_sha256") != file_sha256(manifest_path):
        errors.append("launcher manifest SHA changed")
    inventory = health.get("gpu_inventory", [])
    if [gpu.get("index") for gpu in inventory] != list(range(8)) or len({gpu.get("uuid") for gpu in inventory}) != 8 or any("H100" not in str(gpu.get("name")) for gpu in inventory):
        errors.append("launcher requires actual eight idle/owned H100 GPUs")
    for receipt in health.get("control_files", []):
        try:
            path = _confined(run_root, run_root / receipt["path"])
            if not path.is_file() or file_sha256(path) != receipt["sha256"]:
                errors.append(f"launcher CPU/environment receipt SHA changed: {path}")
            else:
                evidence.append({"path": str(path.resolve()), "sha256": receipt["sha256"]})
        except (KeyError, OSError, ValueError) as error:
            errors.append(f"launcher CPU/environment receipt failed: {error}")
    paths = sorted((run_root / "execution/ranks").glob("rank_*.json"))
    rows = [read(path, "physical rank receipt") for path in paths]
    if sorted((row.get("rank", -1) for row in rows)) != list(range(8)) or len(rows) != 8:
        errors.append("incomplete/duplicate physical rank coverage; expected eight, never sixteen")
    expected_files = {str(path.relative_to(run_root)): file_sha256(path) for path in paths}
    sealed = health.get("rank_receipts", [])
    if len(sealed) != 8 or {r.get("path"): r.get("sha256") for r in sealed} != expected_files:
        errors.append("physical receipt SHA index missing or changed")
    if len({row.get("hostname") for row in rows}) != 1 or not rows or not rows[0].get("hostname"):
        errors.append("physical workers are not on exactly one host")
    try:
        trace_root = Path(manifest["assets"]["trace"])
        traces = trace_controls(trace_root, variant, frozen["trace_hash"])
        for path in [trace_root / "bundle.json"] + [trace_root / entry["path"] / "trace.json" for entry in _object(trace_root / "bundle.json")["ranks"]]:
            evidence.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
        checkpoint_path = Path(manifest["assets"]["logical_checkpoint"]) / "manifest.json"
        checkpoint = read(checkpoint_path, "logical checkpoint")
        unsigned = dict(checkpoint)
        if unsigned.pop("manifest_hash", None) != frozen["logical_hash"] or digest(unsigned) != frozen["logical_hash"]:
            errors.append("immutable initial checkpoint hash changed")
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(f"immutable sixteen-slot trace/checkpoint evidence failed: {error}")
        traces, checkpoint = {}, {}
    all_slots = []
    initial, final = contract["initial_versions"], contract["final_versions"]
    # Canonical GAN systems-only bootstrap restores earlier Adam counters,
    # while its immutable metadata maps the official logical window to 1000.
    # Audit both namespaces independently; never relabel or mutate Adam state.
    physical_initial = {
        role: checkpoint.get("components", {}).get(role, {}).get("optimizer_step")
        for role in ("generator", "critic")
    }
    physical_initial["ema"] = checkpoint.get("ema", {}).get("optimization_step")
    if any(type(value) is not int or value < 0 for value in physical_initial.values()):
        errors.append("immutable checkpoint physical optimizer/EMA counters invalid")
    physical_offset = any(physical_initial[role] != initial[role] for role in ("generator", "critic"))
    if physical_offset:
        control = checkpoint.get("systems_only_control", {})
        if (variant != "gan" or control.get("classification") != "systems_only_gan_bootstrap"
                or control.get("logical_schedule_versions") != initial
                or control.get("logical_control_window") != contract["logical_steps"]
                or control.get("official_checkpoint_1000") is not False
                or control.get("source_faithful_continuation") is not False):
            errors.append("physical/logical offset lacks immutable systems-only GAN control binding")
    if physical_initial["ema"] != initial["ema"]:
        errors.append("immutable checkpoint actual EMA start changed")
    physical_final = {
        role: start + final[role] - initial[role] if type(start) is int else None
        for role, start in physical_initial.items()
    }
    binding_hash = digest({"schema": "distillschedule.static_world16_trace.v1", "variant": variant,
                           "bundle_hash": frozen["trace_hash"], "logical_steps": contract["logical_steps"], "rank_slots": list(range(16))})
    reference_required = bool(policy["generator_nograd_cross_kv_cache"] or policy["generator_nograd_static_context_cache"])
    for row in rows:
        rank = row.get("rank", -1)
        label = f"physical rank{rank}"
        if type(rank) is not int or rank not in range(8):
            continue
        slots = [rank, rank + 8]
        hardware = row.get("hardware_profile", {})
        expected_hardware = {"name": HARDWARE, "physical_world_size": 8, "logical_global_batch": 16, "logical_b1_microbatches_per_rank": 2,
                             "logical_slots": slots, "loss_scale_per_logical_b1": .5, "rank_mapping": "slot = physical_rank + microbatch * physical_world_size"}
        if hardware != expected_hardware or row.get("world_size") != 8 or row.get("local_rank") != rank:
            errors.append(f"{label}: physical/logical hardware slot contract changed")
        all_slots.extend(hardware.get("logical_slots", []))
        if row.get("schema_version") != "distillschedule.static_fsdp2_p0_rank.v1" or row.get("status") != "PASS" or row.get("run_id") != manifest.get("run_id") or row.get("variant") != variant:
            errors.append(f"{label}: actual worker status/run/variant failed")
        if row.get("p0_policy") != policy or row.get("contract") != contract:
            errors.append(f"{label}: frozen source math/policy contract changed")
        if row.get("profile_mode") is not False or row.get("timing_mode") is not (mode == "timing") or (mode == "validate" and row.get("validate_only") is not True):
            errors.append(f"{label}: unprofiled worker mode binding changed")
        identities = row.get("identities", {})
        if identities.get("logical_manifest_hash") != frozen["logical_hash"] or identities.get("trace_bundle_hash") != frozen["trace_hash"] or identities.get("trace_binding_hash") != binding_hash or identities.get("cross_step_rollout_batchability") != "PASS":
            errors.append(f"{label}: immutable trace/initial checkpoint identity changed")
        validation = identities.get("rank_control_validation", {})
        if validation.get("status") != "PASS" or validation.get("rank_count") != 16 or validation.get("cross_rank_call_count_parity") is not True:
            errors.append(f"{label}: all sixteen source control rows not validated")
        attention = row.get("attention_backend", {})
        if attention.get("status") != "PASS" or attention.get("requested") != policy["attention_backend"] or attention.get("runtime") != policy["attention_backend"]:
            errors.append(f"{label}: attention source identity failed")
        if policy["attention_backend"] == "local-fa2":
            if attention.get("extension_sha256") != FLASH_SHA or attention.get("flash_attn_func_available") is not True or "flash_attn2" not in str(attention.get("flash_attn_func_module")):
                errors.append(f"{label}: pinned FlashAttention source SHA failed")
        elif attention.get("local_kernels") or attention.get("flash_attn_func_available") is not False:
            errors.append(f"{label}: SDPA baseline inherited optimization")
        if mode == "validate":
            if any(key in row for key in ("build", "generator", "critic", "full5_seconds")):
                errors.append(f"{label}: validate-only unexpectedly ran model work")
            continue
        if mode not in ("correctness", "timing"):
            errors.append("launcher mode must be validate, correctness or timing")
        for field, role in (("generator_physical_step", "generator"), ("critic_physical_step", "critic"), ("ema_step", "ema")):
            expected_initial = checkpoint.get("components", {}).get(role, {}).get("optimizer_step") if role != "ema" else checkpoint.get("ema", {}).get("optimization_step")
            if row.get("initial", {}).get(field) != expected_initial or row.get("final", {}).get(field) != physical_final[role]:
                errors.append(f"{label}: initial/final actual {role} version changed")
        if row.get("final", {}).get("generator_logical_version") != final["generator"] or row.get("final", {}).get("critic_logical_version") != final["critic"]:
            errors.append(f"{label}: final logical versions changed")
        if row.get("updates") != {"generator": 1, "critic": 5, "ema": len(contract["ema_update_steps"])} or row.get("ema_device") != "cuda":
            errors.append(f"{label}: G1/C5/GPU-EMA update inventory changed")
        for role in ("generator", "critic"):
            aggregate = row.get(role, {}) if role == "generator" else None
            records = [aggregate] if role == "generator" else row.get("critic", [])
            if not isinstance(records, list) or len(records) != (1 if role == "generator" else 5):
                errors.append(f"{label}: actual {role} update records missing")
                records = []
            for offset, record in enumerate(records):
                children = record.get("microbatches", [])
                if not isinstance(children, list) or len(children) != 2:
                    errors.append(f"{label}: {role} children must be two actual B1 receipts")
                    children = []
                step = contract["logical_steps"][offset if role == "critic" else 0]
                score_count = (3 if variant == "no-gan" else 4) if role == "generator" else (1 if variant == "no-gan" else 2)
                score_key = "physical_score_forwards" if role == "generator" else "score_forwards"
                if record.get("logical_b1_microbatch_count") != 2 or record.get("source_ranks") != slots or record.get(score_key) != 2 * score_count:
                    errors.append(f"{label}: actual {role} aggregate contribution/call census changed")
                for index, child in enumerate(children):
                    slot = slots[index]
                    if child.get("logical_slot") != slot or child.get("source_rank") != slot or child.get("physical_rank") != rank or child.get("logical_step") != step:
                        errors.append(f"{label}: {role} child slot/source/step changed")
                    if child.get("optimizer_committed") is not bool(index):
                        errors.append(f"{label}: {role} child commit before outer boundary")
                    expected_child_step = (physical_initial[role] + (offset if role == "critic" else 0) + index
                                           if type(physical_initial[role]) is int else None)
                    if child.get(f"physical_{role}_step_after") != expected_child_step:
                        errors.append(f"{label}: {role} child actual physical version drift")
                    if child.get(score_key) != score_count:
                        errors.append(f"{label}: {role} actual Score call inventory changed")
                    if role == "generator":
                        if child.get("real_score_pair_packed") is not False or child.get("logical_score_requests") != score_count:
                            errors.append(f"{label}: Score packing/request math changed")
                        if slot in traces:
                            _check_rollout(child.get("rollout", {}), traces[slot], variant=variant, slot=slot, step=step, role=role, label=f"{label} G/slot{slot}", errors=errors)
                    elif variant == "gan" and (child.get("critic_optimizer_committed") is not bool(index) or child.get("critic_logical_microbatch_boundary_reached") is not True):
                        errors.append(f"{label}: GAN inner backward/outer commit boundary changed")
                    if mode == "correctness":
                        loss = child.get("surrogate_loss") if role == "generator" and variant == "no-gan" else child.get("losses") if role == "generator" else child.get("loss")
                        values = list(loss.values()) if isinstance(loss, dict) else [loss]
                        recorded = bool(values) and all(_finite(v) for v in values)
                        if any(v is not None and not _finite(v) for v in values):
                            errors.append(f"{label}: {role} child recorded loss nonfinite/malformed")
                        loss_evidence.append({"physical_rank": rank, "logical_slot": slot, "role": role, "logical_step": step,
                                              "status": "RECORDED_FINITE" if recorded else "NOT_RECORDED_SCALAR_DIAGNOSTICS_DISABLED"})
                if role == "generator" and len(children) == 2:
                    rollout = record.get("rollout", {})
                    expected_calls = sum(child.get("rollout", {}).get("call_count", 0) for child in children)
                    expected_stages = {str(stage): sum(child.get("rollout", {}).get("stage_call_count", {}).get(str(stage), 0) for child in children) for stage in range(3)}
                    if rollout.get("call_count") != expected_calls or rollout.get("logical_call_count") != expected_calls or rollout.get("stage_call_count") != expected_stages or rollout.get("logical_stage_call_count") != expected_stages or rollout.get("dummy_call_count") != 0 or rollout.get("microbatches") != [child.get("rollout") for child in children]:
                        errors.append(f"{label}: Generator aggregate actual call inventory changed")
        batches = row.get("rollout_batches", [])
        coverage = [(batch.get("logical_slot"), batch.get("logical_steps")) for batch in batches]
        expected_coverage = [(slot, [step]) for slot in slots for step in contract["logical_steps"]]
        if len(batches) != 10 or coverage != expected_coverage:
            errors.append(f"{label}: actual B1 rollout coverage must be two slots x five steps")
        for batch in batches:
            slot, steps = batch.get("logical_slot"), batch.get("logical_steps", [])
            if slot not in slots or len(steps) != 1 or steps[0] not in contract["logical_steps"] or batch.get("physical_rank") != rank:
                errors.append(f"{label}: rollout slot/source/step changed")
                continue
            if slot in traces:
                _check_rollout(batch, traces[slot], variant=variant, slot=slot, step=steps[0], role="critic", label=f"{label} rollout/slot{slot}/step{steps[0]}", errors=errors)
            _audit_parity(batch, required=mode == "correctness" and reference_required, label=label, errors=errors)
        ema = row.get("ema", [])
        if [r.get("logical_step") for r in ema] != contract["ema_update_steps"] or [r.get("optimization_step") for r in ema] != list(range(initial["ema"] + 1, final["ema"] + 1)) or any(not _finite(r.get("decay")) for r in ema):
            errors.append(f"{label}: EMA recurrence/update/version inventory changed")
        if mode == "correctness":
            checksum = row.get("final", {}).get("ema_local_checksum", {})
            if not isinstance(checksum, dict) or checksum.get("device") != "cuda" or checksum.get("optimization_step") != final["ema"] or checksum.get("communication_per_update") is not False or not _finite(checksum.get("local_sum")) or not _finite(checksum.get("local_norm")) or checksum.get("tracked_tensor_count", 0) < 1:
                errors.append(f"{label}: actual GPU EMA finite checksum/version evidence missing")
        build = row.get("build", {})
        residency = build.get("static_residency", {})
        if any(residency.get(k) is not v for k, v in {"both_backbones_sharded_for_full5": True, "parameter_offload": False,
                                                   "frozen_parameter_streaming": False, "role_transition": False, "process_group_rebuild_in_full5": False}.items()):
            errors.append(f"{label}: offload/streaming/role transition is not zero")
        for role in ("generator", "critic"):
            model = build.get(role, {})
            metadata, fsdp = model.get("metadata", {}), model.get("fsdp2", {})
            if metadata.get("loading_mode") != "mixed_bf16_precast_direct" or metadata.get("block_count") != 40 or metadata.get("trainable_dtype_inventory") != ["torch.float32"] or metadata.get("chunked_residual_storage") != "cuda_out_of_place" or not re.fullmatch(r"[0-9a-f]{64}", str(metadata.get("precast_manifest_sha256", ""))):
                errors.append(f"{label}/{role}: precast model SHA/dtype/GPU residual contract failed")
            if metadata.get("precast_manifest_sha256") != manifest.get("precast_manifest_sha256s", {}).get(role):
                errors.append(f"{label}/{role}: exact relocated precast model manifest SHA changed")
            asset_name = "generator_precast" if role == "generator" else "score_precast"
            subfolder = "transformer_ode" if role == "generator" else "transformer"
            source_path = manifest.get("source_provenance", {}).get(asset_name)
            if metadata.get("load_root") != manifest.get("assets", {}).get(asset_name) or metadata.get("subfolder") != subfolder or not source_path or metadata.get("backbone_root") != str(Path(source_path).parent):
                errors.append(f"{label}/{role}: actual relocated-only model read/provenance root changed")
            expected_fsdp = {"mesh_policy": policy["mesh_policy"], "mesh_shape": frozen["mesh_shape"], "world_size": 8, "hardware_profile": HARDWARE,
                             "logical_global_batch": 16, "logical_b1_microbatches_per_rank": 2, "block_count": 40, "reshard_after_forward": True,
                             "param_compute_dtype": "torch.bfloat16", "reduce_dtype": "torch.float32", "cpu_parameter_offload": False,
                             "parameter_streaming": False, "transition": False}
            if any(fsdp.get(k) != v for k, v in expected_fsdp.items()):
                errors.append(f"{label}/{role}: FSDP actual topology/dtype/reshard/offload contract changed")
            if mode == "correctness":
                audit = row.get("final", {}).get("training_state_audit", {}).get(role, {})
                if audit.get("physical_step") != physical_final[role]:
                    errors.append(f"{label}/{role}: final training state physical version changed")
                for group in ("parameters", "optimizer"):
                    _finite_group(audit.get(group, {}), f"{label}/{role}/{group}", errors)
                gradients = audit.get("gradient_steps", [])
                if len(gradients) != (1 if role == "generator" else 5):
                    errors.append(f"{label}/{role}: actual gradient/optimizer commit census changed")
                for index, gradient in enumerate(gradients):
                    expected_gradient_step = (physical_initial[role] + index
                                              if type(physical_initial[role]) is int else None)
                    if gradient.get("physical_step_before") != expected_gradient_step:
                        errors.append(f"{label}/{role}: gradient audit version drift")
                    _finite_group(gradient.get("gradient", {}), f"{label}/{role}/gradient{index}", errors)
        _audit_persistence(row, slots=slots, policy=policy, label=label, errors=errors)
        if any(row.get(k) != 0 for k in ("transition_count", "model_swap_count", "parameter_streaming_count")) or row.get("external_rollout_engine_used") is not False:
            errors.append(f"{label}: offload/transition/model swap inventory changed")
        if not _finite(row.get("full5_seconds")) or row.get("full5_seconds", 0) <= 0 or type(row.get("timing_start_unix_ns")) is not int or type(row.get("timing_stop_unix_ns")) is not int or row.get("timing_stop_unix_ns", 0) <= row.get("timing_start_unix_ns", 0):
            errors.append(f"{label}: actual Full5 timer missing or invalid")
    if sorted(all_slots) != list(range(16)):
        errors.append("logical slot coverage is not exact sixteen-slot bijection")
    seconds = [row.get("full5_seconds") for row in rows if _finite(row.get("full5_seconds"))]
    return {"schema_version": SUMMARY_SCHEMA, "status": "FAIL" if errors else "PASS", "errors": errors,
            "run_id": manifest.get("run_id"), "run_root": str(run_root.resolve()), "variant": variant, "mode": mode,
            "arm": frozen["arm"], "expected_head": expected_head, "hardware_profile": HARDWARE,
            "physical_rank_count": len(rows), "logical_global_batch": 16, "logical_slots": sorted(all_slots),
            "physical_initial_versions": physical_initial, "physical_final_versions": physical_final,
            "logical_initial_versions": initial, "logical_final_versions": final,
            "initialization_class": "systems_only_gan_bootstrap" if physical_offset else "logical_and_physical_versions_aligned",
            "policy": policy, "policy_fingerprint": frozen["policy_fingerprint"], "asset_fingerprint": manifest.get("asset_fingerprint"),
            "numeric_comparison": "NOT_RUN", "required_numeric_receipts": "NOT_RUN_FULL_CROSS_ARM_LOSS_GRADIENT_MODEL_OPTIMIZER_EMA_COMPARISON",
            "loss_numeric_evidence": "NOT_RECORDED" if any(r["status"] != "RECORDED_FINITE" for r in loss_evidence) else "RECORDED_FINITE" if loss_evidence else "NOT_RUN",
            "loss_receipts": loss_evidence, "gan_inner_backward_census": "NOT_RECORDED_BY_CURRENT_WORKER" if variant == "gan" else "NOT_APPLICABLE",
            "strict_claimable": False, "claimable_correctness": False, "claimable_timing": False,
            "execution_correctness_admitted": not errors and mode == "correctness",
            "runtime_reference_parity": "PASS" if mode == "correctness" and reference_required and not errors else "NOT_REQUIRED" if not reference_required else "NOT_RUN",
            "diagnostic_only": True, "full5_seconds_max_rank": max(seconds) if len(seconds) == 8 else None,
            "completed_unix_ns": health.get("completed_unix_ns"), "created_unix_ns": time.time_ns(),
            "evidence_index": evidence, "evidence_fingerprint": digest(evidence), "launcher_health": health,
            "physical_receipts": rows}


def validate_timing_gate(path: Path, expected_sha256: str, *, expected_head: str, frozen: dict[str, Any], asset_fingerprint: str, max_age_seconds: int = 86400) -> dict[str, Any]:
    _require(path.is_file() and file_sha256(path) == expected_sha256, "correctness summary SHA mismatch")
    value = _object(path)
    _require(value.get("schema_version") == SUMMARY_SCHEMA and value.get("hardware_profile") == HARDWARE and value.get("physical_rank_count") == 8, "fresh H100 correctness summary required; A100 receipts cannot admit timing")
    _require(value.get("status") == "PASS" and value.get("mode") == "correctness" and value.get("execution_correctness_admitted") is True, "H100 execution correctness gate did not pass")
    _require(value.get("expected_head") == expected_head and value.get("policy") == frozen["policy"] and value.get("variant") == frozen["variant"] and value.get("arm") == frozen["arm"] and value.get("asset_fingerprint") == asset_fingerprint, "correctness/timing HEAD/hardware/policy/assets mismatch")
    completed = value.get("completed_unix_ns")
    _require(type(completed) is int and 0 <= (time.time_ns() - completed) / 1e9 <= max_age_seconds, "correctness receipt stale or from future")
    evidence = value.get("evidence_index", [])
    _require(len(evidence) >= 10 and digest(evidence) == value.get("evidence_fingerprint"), "correctness evidence SHA index missing")
    for receipt in evidence:
        item = Path(receipt["path"])
        _require(item.is_file() and file_sha256(item) == receipt["sha256"], f"correctness evidence SHA changed: {item}")
    _require(value.get("numeric_comparison") == "NOT_RUN" and value.get("strict_claimable") is False, "unsupported full numeric correctness claim; do not promote diagnostics")
    return {"path": str(path.resolve()), "sha256": expected_sha256, "strict_claimable": False, "measurement_scope": "diagnostic_only_numeric_comparison_NOT_RUN"}


def prepare_launch(args: argparse.Namespace) -> None:
    frozen = freeze_policy(ROOT, variant=args.variant, arm=args.arm)
    gate = verify_asset_gate(args.asset_gate, args.asset_gate_sha256, expected_head=args.expected_head, asset_root=args.asset_root, frozen=frozen)
    groups = gate["groups"]
    assets = {name: group["actual_root"] for name, group in groups.items()}
    helios = args.helios_root.resolve()
    _require(file_sha256(helios / "helios/modules/transformer_helios.py") == HELIOS_SHA, "frozen Helios Transformer source SHA mismatch")
    trace_controls(Path(assets["trace"]), args.variant, frozen["trace_hash"])
    worker = [str(ROOT / "scripts/distillschedule/probe_static_fsdp2_p0_world16_hardened.py"), "--variant", args.variant,
              "--run-id", args.run_id, "--output-dir", str(args.run_root / "execution"),
              "--logical-package", assets["logical_checkpoint"], "--expected-logical-hash", frozen["logical_hash"],
              "--trace-bundle", assets["trace"], "--expected-trace-hash", frozen["trace_hash"],
              "--generator-root", groups["generator_precast"]["provenance_root"], "--precast-generator-root", assets["generator_precast"],
              "--score-root", groups["score_precast"]["provenance_root"], "--precast-score-root", assets["score_precast"],
              "--negative-prompt", assets["negative_prompt"], "--ema-device", "cuda", *policy_arguments(frozen["policy"])]
    if args.variant == "gan":
        worker.extend(("--gan-config", str(ROOT / "configs/distillschedule/templates/helios_stage3_gan_384x640_systems.yaml")))
    if args.mode == "validate":
        worker.append("--validate-only")
    elif args.mode == "timing":
        worker.extend(("--timing-mode", "--no-profile-diagnostics"))
    else:
        worker.append("--profile-diagnostics")
    environment = dict(frozen["environment"], CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1", HF_HOME=str(_confined(args.asset_root, args.hf_home)),
                       PYTHONPATH=f"{ROOT / 'src'}:{ROOT}:{helios}")
    _require(args.hf_home.is_dir(), "explicit relocated HF_HOME missing")
    if frozen["policy"]["attention_backend"] == "local-fa2":
        environment["LOCAL_KERNELS"] = "kernels-community/flash-attn2=" + assets["flash_attn_kernel"]
    unset = list(set(frozen["unset_environment"] + ["DISTILLSCHEDULE_STATIC_FSDP2_NSYS_EXACT_RANGE", "DISTILLSCHEDULE_STATIC_FSDP2_NSYS_RECEIPT_DIR"]))
    if "LOCAL_KERNELS" in environment:
        unset.remove("LOCAL_KERNELS")
    correctness = None
    if args.mode == "timing":
        _require(args.correctness_summary is not None and args.correctness_sha256, "timing requires fresh correctness summary and exact SHA")
        correctness = validate_timing_gate(args.correctness_summary, args.correctness_sha256, expected_head=args.expected_head, frozen=frozen, asset_fingerprint=gate["fingerprint"])
    manifest = {"schema_version": LAUNCH_SCHEMA, "run_id": args.run_id, "variant": args.variant, "mode": args.mode,
                "arm": args.arm, "expected_head": args.expected_head, "frozen": frozen, "hardware_profile": {"name": HARDWARE, "physical_world_size": 8,
                "logical_global_batch": 16, "logical_b1_microbatches_per_rank": 2, "loss_scale_per_logical_b1": .5,
                "rank_mapping": "slot = physical_rank + microbatch * physical_world_size"},
                "asset_gate": {"path": gate["path"], "sha256": gate["sha256"]}, "asset_fingerprint": gate["fingerprint"], "assets": assets,
                "precast_manifest_sha256s": {"generator": groups["generator_precast"]["precast_manifest_sha256"], "critic": groups["score_precast"]["precast_manifest_sha256"]},
                "source_provenance": {name: groups[name]["source_path"] for name in ("generator_precast", "score_precast")},
                "environment": environment, "unset_environment": sorted(unset), "worker_argv": worker, "correctness_gate": correctness,
                "started_unix_ns": time.time_ns(), "gpu_claim": "GPU_NOT_RUN_UNTIL_EXPLICIT_LAUNCH"}
    _new_json(args.run_root / "manifests/launch.json", manifest)
    # NUL records avoid shell eval, quoting bugs, and arbitrary argv overrides.
    (args.run_root / "manifests/worker_argv.nul").write_bytes(b"\0".join(v.encode() for v in worker) + b"\0")
    (args.run_root / "manifests/environment.nul").write_bytes(b"\0".join(f"{k}={v}".encode() for k, v in sorted(environment.items())) + b"\0")
    (args.run_root / "manifests/unset_environment.nul").write_bytes(b"\0".join(k.encode() for k in sorted(unset)) + b"\0")


def seal_run(args: argparse.Namespace) -> None:
    logs = args.run_root / "logs"
    def events(name):
        return {k: int(v) for k, v in (line.split() for line in (logs / f"cgroup_{name}.txt").read_text().splitlines())}
    before, after = events("before"), events("after")
    delta = {key: after[key] - before[key] for key in ("oom", "oom_kill")}
    inventory = _object(logs / "gpu_inventory.json")["gpus"]
    processes = (logs / "gpu_processes_after.txt").read_text().strip().splitlines()
    manifest = args.run_root / "manifests/launch.json"
    actual_head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    # Source status is recorded by the shell before this CPU-only seal step.
    clean = not (logs / "source_status_after.txt").read_text().strip()
    healthy = args.exit_code == 0 and delta == {"oom": 0, "oom_kill": 0} and not processes and clean and actual_head == args.expected_head
    value = {"schema_version": HEALTH_SCHEMA, "status": "PASS" if healthy else "FAIL", "exit_code": args.exit_code,
             "expected_head": args.expected_head, "observed_head_after": actual_head, "clean_source_after": clean,
             "cgroup_before": before, "cgroup_after": after, "cgroup_oom_delta": delta, "gpu_inventory": inventory,
             "gpu_processes_before": (logs / "gpu_processes_before.txt").read_text().strip().splitlines(),
             "gpu_processes_after": processes, "completed_unix_ns": time.time_ns(),
             "launch_manifest_sha256": file_sha256(manifest) if manifest.exists() else None,
             "rank_receipts": [{"path": str(path.relative_to(args.run_root)), "sha256": file_sha256(path)} for path in sorted((args.run_root / "execution/ranks").glob("rank_*.json"))],
             "control_files": [{"path": str(path.relative_to(args.run_root)), "sha256": file_sha256(path)}
                               for path in sorted(logs.glob("*.txt"))] + [{"path": str(path.relative_to(args.run_root)), "sha256": file_sha256(path)}
                               for path in sorted((args.run_root / "manifests").glob("*")) if path.is_file()],
             "host_memory_scope": "shared node/cgroup; memory peak is not exclusively attributable to this run"}
    _new_json(args.run_root / "run_status.json", value)
    _require(healthy, "launcher health/owned teardown failed; retained failure receipt")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prepare-launch", action="store_true")
    action.add_argument("--seal-run", action="store_true")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--hardware-profile", choices=(HARDWARE,), default=HARDWARE)
    parser.add_argument("--variant", choices=("no-gan", "gan"))
    parser.add_argument("--arm", choices=("baseline", "best"))
    parser.add_argument("--mode", choices=("validate", "correctness", "timing"))
    parser.add_argument("--run-id")
    parser.add_argument("--asset-gate", type=Path)
    parser.add_argument("--asset-gate-sha256")
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--helios-root", type=Path)
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--correctness-summary", type=Path)
    parser.add_argument("--correctness-sha256")
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    _require(bool(re.fullmatch(r"[0-9a-f]{40}", args.expected_head)), "exact 40-digit HEAD required")
    if args.prepare_launch:
        _require(all(getattr(args, name) is not None for name in ("variant", "arm", "mode", "run_id", "asset_gate", "asset_gate_sha256", "asset_root", "helios_root", "hf_home")), "prepare-launch requires all explicit source/asset/run bindings")
        _require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", args.run_id)), "unsafe immutable run ID")
        prepare_launch(args)
        return 0
    if args.seal_run:
        _require(args.exit_code is not None, "seal-run requires observed worker exit code")
        seal_run(args)
        return 0
    _require(args.output is not None, "summary output required")
    value = summarize(args.run_root, expected_head=args.expected_head)
    _new_json(args.output, value)
    print(json.dumps({k: v for k, v in value.items() if k not in ("physical_receipts", "launcher_health", "evidence_index")}, indent=2))
    return 0 if value["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

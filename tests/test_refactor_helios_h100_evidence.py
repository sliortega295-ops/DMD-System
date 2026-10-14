"""CPU/source-only fixtures; these are not H100 or model-parity evidence."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "helios_h100_summary", ROOT / "scripts/distillschedule/summarize_helios_hardware_p0.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
HEAD = "a" * 40


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signed(value, key):
    result = copy.deepcopy(value)
    result[key] = module.digest(result)
    return result


def trace_fixture(root, variant):
    steps = list(range(10, 15)) if variant == "no-gan" else list(range(1000, 1005))
    manifests, ranks = {}, []
    for slot in range(16):
        tensors = {}
        history = {"text_embedding": "sample/0/text_embedding", "attention_mask": "sample/0/attention_mask"}
        for role, role_steps in (("generator", steps[:1]), ("critic", steps)):
            for step in role_steps:
                prefix = f"step/{step}/{role}"
                keys = dict(history)
                for name in ("continuous_history", "history_short", "history_mid", "history_long",
                             "indices_hidden_states", "indices_history_short", "indices_history_mid",
                             "indices_history_long", "entry_noise", "rollout_noise"):
                    keys[name] = f"{prefix}/{name}"
                keys.update({f"block_noise_stage{s}": f"{prefix}/block_noise/{s}" for s in (1, 2)})
                extras = ["dmd_timestep", "dmd_noise"] if role == "generator" else ["critic_timestep", "critic_noise"]
                if variant == "gan":
                    extras += ["gan_g_noise", "gan_g_crop"] if role == "generator" else ["gan_d_fake_crop", "gan_d_real_crop", "gan_d_r1_crop", "r1_epsilon"]
                keys.update({name: f"{prefix}/{name}" for name in extras})
                for key in keys.values():
                    tensors[key] = {"sha256": hashlib.sha256(f"fixture:{slot}:{key}".encode()).hexdigest()}
        operations = []
        for step in steps:
            operations.append({"key": f"step/{step}/branch", "control": {
                "global_step": step, "dfake_gen_update_ratio": 5, "train_generator": step == steps[0],
                "train_critic": True, "gan": variant == "gan", "reward": False, "vae_decode": False,
            }})
            for role in ("generator", "critic"):
                if role == "generator" and step != steps[0]:
                    continue
                operations.append({"key": f"step/{step}/{role}/rollout", "control": {
                    "call_count": 6, "stage_schedule_scope": "global_rank0_broadcast_equivalent",
                    "initial_synchronized_pyramid_stage_exit": [1, 1, 1],
                    "pyramid_stage_call_sequence": [
                        {"stage": stage, "call_count": 2, "denoising_exit": 1,
                         "latent_shape": [1, 16, 9, h, w], "timestep": [999, 500]}
                        for stage, (h, w) in enumerate(((12, 20), (24, 40), (48, 80)))
                    ],
                }})
        manifest = signed({"schema_version": "distillschedule.operation_trace.v1", "source": {"fixture_only": True},
                           "contract": {"resolution": [384, 640], "dtype": "bfloat16",
                                        "stage_spatial_shapes": [[12, 20], [24, 40], [48, 80]],
                                        "full_latent_shape": [1, 16, 9, 48, 80], "dfake_gen_update_ratio": 5},
                           "tensors": tensors, "operations": operations}, "trace_hash")
        manifests[slot] = manifest
        path = f"rank{slot}"
        write(root / path / "trace.json", manifest)
        ranks.append({"rank": slot, "path": path, "trace_hash": manifest["trace_hash"]})
    bundle = signed({"schema_version": "distillschedule.operation_trace_bundle.v1", "ranks": ranks}, "bundle_hash")
    write(root / "bundle.json", bundle)
    return bundle, manifests


def actual_rollout(trace, slot, step, role):
    calls = []
    for stage, (h, w) in enumerate(((12, 20), (24, 40), (48, 80))):
        for denoise in range(2):
            selected = role == "generator" and denoise == 1
            calls.append({"call_index": len(calls), "stage": stage, "denoise_call_in_stage": denoise,
                          "hidden_states_shape": [1, 16, 9, h, w], "output_shape": [1, 16, 9, h, w],
                          "grad_enabled": selected, "output_requires_grad": selected, "status": "success"})
    identity = module.entry_identity(trace, variant="gan" if step >= 1000 else "no-gan", slot=slot, step=step, role=role)
    return {"call_count": 6, "logical_call_count": 6, "stage_call_count": {"0": 2, "1": 2, "2": 2},
            "logical_stage_call_count": {"0": 2, "1": 2, "2": 2}, "dummy_call_count": 0,
            "calls": calls, "output_dtype": "torch.bfloat16", "output_shape": [1, 16, 9, 48, 80],
            "output_requires_grad": role == "generator", "operation_trace_hash": trace["trace_hash"],
            "source_rank": slot, "selected_call_indices": [1, 3, 5] if role == "generator" else [],
            "logical_steps": [step], "batch_size": 1, "synchronize_after_rollout": True,
            "empty_cache_before_rollout": True, "trace_identity_sha256": identity,
            "member_trace_identities": [identity], "trace_key": f"static-fsdp2-p0/batched/{step}"}


def audit_group():
    return {"tensor_count": 1, "numel": 1, "finite": True, "sha256": "b" * 64,
            "tensors": [{"name": "fixture", "finite": True, "sha256": "c" * 64, "sum": 1., "l2": 1., "max_abs": 1.}]}


def fixture(root, variant="no-gan", arm="baseline", mode="correctness"):
    frozen = module.freeze_policy(ROOT, variant=variant, arm=arm)
    policy = frozen["policy"]
    contract = frozen["contract"]
    # Synthetic asset/control hashes intentionally replace production pins only
    # in this fixture's manifest. Summarize compares to its separately supplied
    # expected frozen policy; no launcher accepts these synthetic production pins.
    bundle, traces = trace_fixture(root / "assets/trace", variant)
    initial, final = contract["initial_versions"], contract["final_versions"]
    checkpoint = signed({"components": {role: {"optimizer_step": initial[role], "parameter_version": initial[role]}
                                          for role in ("generator", "critic")},
                         "ema": {"optimization_step": initial["ema"]}}, "manifest_hash")
    write(root / "assets/checkpoint/manifest.json", checkpoint)
    frozen = copy.deepcopy(frozen)
    frozen["trace_hash"] = bundle["bundle_hash"]
    frozen["logical_hash"] = checkpoint["manifest_hash"]
    binding = {"name": "h100-8-b16", "physical_world_size": 8, "logical_global_batch": 16,
               "logical_b1_microbatches_per_rank": 2, "loss_scale_per_logical_b1": .5,
               "rank_mapping": "slot = physical_rank + microbatch * physical_world_size"}
    manifest = {"schema_version": module.LAUNCH_SCHEMA, "run_id": "fixture", "expected_head": HEAD,
                "variant": variant, "mode": mode, "arm": arm, "frozen": frozen,
                "hardware_profile": binding, "asset_fingerprint": "d" * 64,
                "assets": {"trace": str(root / "assets/trace"), "logical_checkpoint": str(root / "assets/checkpoint"),
                           "generator_precast": str(root / "assets/generator_precast"), "score_precast": str(root / "assets/score_precast")},
                "source_provenance": {"generator_precast": "/original-share/model/transformer_ode", "score_precast": "/original-share/model/transformer"},
                "precast_manifest_sha256s": {"generator": "e" * 64, "critic": "e" * 64},
                "started_unix_ns": time.time_ns(), "fixture_only": True}
    gate_path = root / "assets/fixture_asset_gate.json"
    write(gate_path, {"schema_version": module.ASSET_SCHEMA, "status": "PASS", "exact_head": HEAD,
                      "fixture_only": True, "groups": []})
    manifest["asset_gate"] = {"path": str(gate_path), "sha256": sha(gate_path)}
    write(root / "manifests/launch.json", manifest)
    for rank in range(8):
        slots = [rank, rank + 8]
        generator_children = []
        for index, slot in enumerate(slots):
            generator_children.append({"logical_slot": slot, "physical_rank": rank, "source_rank": slot,
                                       "logical_step": contract["logical_steps"][0], "optimizer_committed": bool(index),
                                       "physical_generator_step_after": initial["generator"] + index,
                                       "logical_score_requests": 3 if variant == "no-gan" else 4,
                                       "physical_score_forwards": 3 if variant == "no-gan" else 4,
                                       "real_score_pair_packed": False, "surrogate_loss": 1., "losses": {"dmd": 1., "gan": 2.},
                                       "rollout": actual_rollout(traces[slot], slot, contract["logical_steps"][0], "generator")})
        generator = {**generator_children[-1], "source_rank": rank, "source_ranks": slots,
                     "microbatches": generator_children, "logical_b1_microbatch_count": 2,
                     "logical_score_requests": 6 if variant == "no-gan" else 8,
                     "physical_score_forwards": 6 if variant == "no-gan" else 8}
        generator["rollout"] = {"call_count": 12, "logical_call_count": 12, "dummy_call_count": 0,
                                "stage_call_count": {"0": 4, "1": 4, "2": 4}, "logical_stage_call_count": {"0": 4, "1": 4, "2": 4},
                                "logical_b1_microbatch_count": 2, "microbatches": [child["rollout"] for child in generator_children]}
        critic = []
        for offset, step in enumerate(contract["logical_steps"]):
            children = [{"logical_step": step, "logical_slot": slot, "physical_rank": rank, "source_rank": slot,
                         "optimizer_committed": bool(index), "physical_critic_step_after": initial["critic"] + offset + index,
                         "score_forwards": 1 if variant == "no-gan" else 2, "loss": 1.,
                         "critic_optimizer_committed": bool(index), "critic_logical_microbatch_boundary_reached": True}
                        for index, slot in enumerate(slots)]
            critic.append({**children[-1], "source_rank": rank, "source_ranks": slots, "microbatches": children,
                           "logical_b1_microbatch_count": 2, "score_forwards": 2 if variant == "no-gan" else 4})
        batches = []
        for slot in slots:
            for step in contract["logical_steps"]:
                batches.append({**actual_rollout(traces[slot], slot, step, "critic"), "logical_slot": slot, "physical_rank": rank,
                                "batch_index": contract["logical_steps"].index(step), "batch_parity": {"enabled": False, "status": "NOT_RUN", "samples": []}})
        builds = {}
        for role in ("generator", "critic"):
            builds[role] = {"metadata": {"loading_mode": "mixed_bf16_precast_direct", "block_count": 40,
                                         "load_root": str(root / ("assets/generator_precast" if role == "generator" else "assets/score_precast")),
                                         "backbone_root": "/original-share/model", "subfolder": "transformer_ode" if role == "generator" else "transformer",
                                         "precast_manifest_sha256": "e" * 64, "trainable_dtype_inventory": ["torch.float32"],
                                         "chunked_residual_storage": "cuda_out_of_place"},
                            "fsdp2": {"backend": "FSDP2_COMPOSABLE_FULL_SHARD", "mesh_policy": policy["mesh_policy"],
                                      "mesh_shape": [8], "world_size": 8, "hardware_profile": "h100-8-b16",
                                      "logical_global_batch": 16, "logical_b1_microbatches_per_rank": 2, "block_count": 40,
                                      "reshard_after_forward": True, "param_compute_dtype": "torch.bfloat16",
                                      "reduce_dtype": "torch.float32", "cpu_parameter_offload": False, "parameter_streaming": False, "transition": False}}
        builds["static_residency"] = {"both_backbones_sharded_for_full5": True, "mesh_policy": policy["mesh_policy"],
                                      "parameter_offload": False, "frozen_parameter_streaming": False, "role_transition": False,
                                      "process_group_rebuild_in_full5": False}
        row = {"schema_version": "distillschedule.static_fsdp2_p0_rank.v1", "status": "PASS", "run_id": "fixture",
               "rank": rank, "local_rank": rank, "hostname": "one-host", "variant": variant, "world_size": 8,
               "hardware_profile": {**binding, "logical_slots": slots}, "p0_policy": policy, "contract": contract,
               "timing_mode": mode == "timing", "profile_mode": False, "validate_only": mode == "validate",
               "identities": {"logical_manifest_hash": checkpoint["manifest_hash"], "trace_bundle_hash": bundle["bundle_hash"],
                              "trace_binding_hash": module.digest({"schema": "distillschedule.static_world16_trace.v1", "variant": variant,
                                                                   "bundle_hash": bundle["bundle_hash"], "logical_steps": contract["logical_steps"], "rank_slots": list(range(16))}),
                              "cross_step_rollout_batchability": "PASS", "rank_control_validation": {"status": "PASS", "rank_count": 16,
                                                                                                       "cross_rank_call_count_parity": True}},
               "attention_backend": {"status": "PASS", "requested": "sdpa", "runtime": "sdpa",
                                     "flash_attn_func_available": False, "local_kernels": ""},
               "initial": {"generator_physical_step": initial["generator"], "critic_physical_step": initial["critic"], "ema_step": initial["ema"]},
               "final": {"generator_physical_step": final["generator"], "critic_physical_step": final["critic"],
                         "generator_logical_version": final["generator"], "critic_logical_version": final["critic"], "ema_step": final["ema"],
                         "ema_local_checksum": {"device": "cuda", "optimization_step": final["ema"], "communication_per_update": False,
                                                "local_sum": 1., "local_norm": 1., "tracked_tensor_count": 1},
                         "training_state_audit": {role: {"physical_step": final[role], "parameters": audit_group(), "optimizer": audit_group(),
                                                        "gradient_steps": [{"physical_step_before": initial[role] + i, "gradient": audit_group()}
                                                                           for i in range(1 if role == "generator" else 5)]}
                                                  for role in ("generator", "critic")}},
               "ema_device": "cuda", "updates": {"generator": 1, "critic": 5, "ema": len(contract["ema_update_steps"])},
               "generator": generator, "critic": critic, "rollout_batches": batches,
               "ema": [{"logical_step": step, "optimization_step": initial["ema"] + i + 1, "decay": .999}
                       for i, step in enumerate(contract["ema_update_steps"])], "build": builds,
               "transition_count": 0, "model_swap_count": 0, "parameter_streaming_count": 0, "external_rollout_engine_used": False,
               "generator_nograd_persistent_unshard": {"enabled": False, "logical_b1_microbatch_count": 2,
                                                      "rollout_batch_count": 10, "microbatches": [{"logical_slot": slot, "persistence": {"enabled": False, "rollout_batch_count": 5}} for slot in slots]},
               "timing_start_unix_ns": 1000000000, "timing_stop_unix_ns": 3000000000, "full5_seconds": 2.}
        write(root / f"execution/ranks/rank_{rank:02d}.json", row)
    seal(root)
    return frozen


def seal(root):
    write(root / "run_status.json", {"schema_version": module.HEALTH_SCHEMA, "status": "PASS", "exit_code": 0,
                                    "expected_head": HEAD, "observed_head_after": HEAD, "clean_source_after": True,
                                    "cgroup_oom_delta": {"oom": 0, "oom_kill": 0}, "gpu_processes_before": [], "gpu_processes_after": [],
                                    "gpu_inventory": [{"index": i, "name": "NVIDIA H100 80GB HBM3", "uuid": f"GPU-{i}"} for i in range(8)],
                                    "launch_manifest_sha256": sha(root / "manifests/launch.json"),
                                    "rank_receipts": [{"path": f"execution/ranks/rank_{i:02d}.json", "sha256": sha(root / f"execution/ranks/rank_{i:02d}.json")} for i in range(8)],
                                    "completed_unix_ns": time.time_ns()})


def result(root, frozen):
    return module.summarize(root, expected_head=HEAD, frozen_policy=frozen)


def test_resident_requires_actual_full_gpu_models_and_zero1_state(tmp_path):
    """Schema fixture only; it must not pass by pretending residency is FSDP."""
    frozen = fixture(tmp_path, arm="resident")
    assert frozen["policy"]["parameter_placement"] == "gpu-resident-zero1"
    assert frozen["mesh_shape"] == []
    assert result(tmp_path, frozen)["status"] == "FAIL"
    for rank in range(8):
        p = tmp_path / f"execution/ranks/rank_{rank:02d}.json"
        row = json.loads(p.read_text())
        row["build"]["static_residency"].update(
            both_backbones_sharded_for_full5=False,
            both_backbones_full_parameter_gpu_resident=True,
            optimizer_state_placement="GPU_ZERO1_PARTITIONS")
        for role in ("generator", "critic"):
            model = row["build"][role]
            model["fsdp2"] = {"enabled": False}
            model["resident"] = {"backend": "FULL_PARAMETER_GPU_RESIDENT_ZERO1", "world_size": 8,
                "logical_global_batch": 16, "physical_b1_rows_per_rank": 2,
                "full_parameters_on_gpu": True, "parameter_offload": False,
                "activation_offload": False, "optimizer_cpu_offload": False,
                "parameter_streaming": False, "model_swap": False,
                "model_compute_dtype": "torch.bfloat16", "gradient_communication_dtype": "torch.float32"}
            model["optimizer_restore"] = {"config": {"zero_optimization": {"stage": 1}},
                "optimizer_cpu_offload": False, "trainable_master_dtype": "torch.float32"}
        write(p, row)
    seal(tmp_path)
    assert result(tmp_path, frozen)["status"] == "PASS"
    mutate(tmp_path, 0, lambda r: r["build"]["critic"]["optimizer_restore"].update(optimizer_cpu_offload=True))
    assert result(tmp_path, frozen)["status"] == "FAIL"


def mutate(root, rank, action, *, reseal=True):
    path = root / f"execution/ranks/rank_{rank:02d}.json"
    row = json.loads(path.read_text())
    action(row)
    write(path, row)
    if reseal:
        seal(root)


@pytest.mark.parametrize("variant", ["no-gan", "gan"])
def test_actual_eight_physical_receipts_sixteen_child_rows_are_diagnostic_pass(tmp_path, variant):
    frozen = fixture(tmp_path, variant)
    value = result(tmp_path, frozen)
    assert value["status"] == "PASS", value["errors"]
    assert value["physical_rank_count"] == 8
    assert value["logical_slots"] == list(range(16))
    assert value["numeric_comparison"] == "NOT_RUN"
    assert value["strict_claimable"] is False
    assert value["claimable_correctness"] is False
    assert value["execution_correctness_admitted"] is True
    assert value["full5_seconds_max_rank"] == 2.


def systems_gan_fixture(root):
    """Actual canonical bootstrap uses early Adam state, not Adam step1000."""
    frozen = fixture(root, "gan")
    checkpoint_path = root / "assets/checkpoint/manifest.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint.pop("manifest_hash")
    checkpoint["systems_only_control"] = {
        "classification": "systems_only_gan_bootstrap", "official_checkpoint_1000": False,
        "source_faithful_continuation": False,
        "logical_schedule_versions": frozen["contract"]["initial_versions"],
        "logical_control_window": frozen["contract"]["logical_steps"],
    }
    physical = {"generator": 2, "critic": 10}
    for role, step in physical.items():
        checkpoint["components"][role].update(optimizer_step=step, parameter_version=step)
    checkpoint = signed(checkpoint, "manifest_hash")
    write(checkpoint_path, checkpoint)
    frozen["logical_hash"] = checkpoint["manifest_hash"]
    launch = json.loads((root / "manifests/launch.json").read_text())
    launch["frozen"] = frozen
    write(root / "manifests/launch.json", launch)
    for rank in range(8):
        path = root / f"execution/ranks/rank_{rank:02d}.json"
        row = json.loads(path.read_text())
        row["identities"]["logical_manifest_hash"] = checkpoint["manifest_hash"]
        for role in physical:
            delta = frozen["contract"]["initial_versions"][role] - physical[role]
            field = role + "_physical_step"
            row["initial"][field] -= delta; row["final"][field] -= delta
            records = [row[role]] if role == "generator" else row[role]
            for record in records:
                for child in record["microbatches"]:
                    child[f"physical_{role}_step_after"] -= delta
            audit = row["final"]["training_state_audit"][role]
            audit["physical_step"] -= delta
            for gradient in audit["gradient_steps"]:
                gradient["physical_step_before"] -= delta
        write(path, row)
    seal(root)
    return frozen


def test_systems_only_gan_keeps_actual_adam_and_logical_versions_separate(tmp_path):
    frozen = systems_gan_fixture(tmp_path)
    value = result(tmp_path, frozen)
    assert value["status"] == "PASS", value["errors"]
    assert value["physical_initial_versions"] == {"generator": 2, "critic": 10, "ema": 800}
    assert value["physical_final_versions"] == {"generator": 3, "critic": 15, "ema": 801}
    assert value["logical_final_versions"] == {"generator": 201, "critic": 1005, "ema": 801}
    assert value["initialization_class"] == "systems_only_gan_bootstrap"
    assert not value["strict_claimable"]


@pytest.mark.parametrize("change,message", [
    (lambda row: row["final"].update(generator_logical_version=3), "logical versions"),
    (lambda row: row["critic"][0]["microbatches"][1].update(physical_critic_step_after=12), "physical version"),
    (lambda row: row["final"]["training_state_audit"]["generator"].update(physical_step=201), "physical version"),
    (lambda row: row["initial"].update(generator_physical_step=200), "actual generator version"),
])
def test_systems_only_gan_still_rejects_commit_and_version_drift(tmp_path, change, message):
    frozen = systems_gan_fixture(tmp_path)
    mutate(tmp_path, 0, change)
    value = result(tmp_path, frozen)
    assert value["status"] == "FAIL"
    assert any(message in error for error in value["errors"]), value["errors"]


@pytest.mark.parametrize("change, message", [
    (lambda r: r["generator"]["microbatches"].pop(), "generator children"),
    (lambda r: r["generator"]["microbatches"][0].update(optimizer_committed=True), "commit"),
    (lambda r: r["generator"]["microbatches"][0].update(physical_generator_step_after=3), "version"),
    (lambda r: r["critic"][0]["microbatches"][1].update(logical_slot=0), "slot"),
    (lambda r: r["rollout_batches"].pop(), "rollout coverage"),
    (lambda r: r["rollout_batches"][0]["calls"][0].update(stage=2), "call inventory"),
    (lambda r: r["rollout_batches"][0].update(output_dtype="torch.float32"), "dtype"),
    (lambda r: r["final"]["training_state_audit"]["critic"]["parameters"].update(finite=False), "finite"),
    (lambda r: r["build"]["critic"]["fsdp2"].update(reshard_after_forward=False), "reshard"),
    (lambda r: r["build"]["critic"]["metadata"].update(precast_manifest_sha256="f" * 64), "model manifest SHA"),
    (lambda r: r["build"]["critic"]["metadata"].update(load_root="/original-share/model/transformer"), "relocated-only"),
    (lambda r: r["rollout_batches"][0].update(member_trace_identities=["f" * 64]), "input/noise"),
    (lambda r: r["ema"][0].update(optimization_step=99), "EMA"),
])
def test_missing_or_modified_actual_evidence_fails_closed(tmp_path, change, message):
    frozen = fixture(tmp_path)
    mutate(tmp_path, 0, change)
    value = result(tmp_path, frozen)
    assert value["status"] == "FAIL"
    assert any(message in error for error in value["errors"]), value["errors"]
    assert not value["strict_claimable"]


def test_receipt_bytes_and_expected_head_are_bound_not_invented(tmp_path):
    frozen = fixture(tmp_path)
    mutate(tmp_path, 0, lambda row: row.update(hostname="other-host"), reseal=False)
    value = result(tmp_path, frozen)
    assert any("receipt SHA" in error for error in value["errors"])
    value = module.summarize(tmp_path, expected_head="f" * 40, frozen_policy=frozen)
    assert any("HEAD" in error for error in value["errors"])


def test_missing_rank_and_oom_receipt_cannot_pass(tmp_path):
    frozen = fixture(tmp_path)
    (tmp_path / "execution/ranks/rank_07.json").unlink()
    (tmp_path / "run_status.json").unlink()
    value = result(tmp_path, frozen)
    assert value["status"] == "FAIL"
    assert any("physical rank coverage" in error for error in value["errors"])
    assert any("launcher" in error for error in value["errors"])


def test_frozen_registry_policy_rendering_is_not_environment_selected():
    baseline = module.freeze_policy(ROOT, variant="no-gan", arm="baseline")
    best = module.freeze_policy(ROOT, variant="no-gan", arm="best")
    gan = module.freeze_policy(ROOT, variant="gan", arm="best")
    assert baseline["policy"]["mesh_policy"] == "world16-fsdp"
    assert baseline["policy"]["attention_backend"] == "sdpa"
    assert baseline["policy"]["selective_flash_checkpoint_stride"] == 0
    assert not baseline["policy"]["all_role_lora_scale1_add"]
    assert best["mesh_shape"] == [2, 4]
    assert best["winner"] == "iteration380"
    assert best["policy"]["exact_modulated_norm_materialization"]
    assert best["policy"]["rollout_batch_size"] == 1
    assert best["policy"]["pack_real_score_pair"] is False
    assert gan["mesh_shape"] == [8]
    assert gan["winner"] == "C25"
    assert gan["policy"]["generator_nograd_persistent_unshard"]
    assert gan["policy"]["exact_history_current_fusion"]
    assert gan["contract"]["final_versions"] == {"generator": 201, "critic": 1005, "ema": 801}
    args = module.policy_arguments(gan["policy"])
    assert "--no-pack-real-score-pair" in args
    assert "--exact-history-current-fusion" in args
    assert "--profile-mode" not in args


def test_old_a100_correctness_summary_cannot_admit_h100_timing(tmp_path):
    path = tmp_path / "old.json"
    write(path, {"status": "PASS", "claimable_correctness": True,
                 "schema_version": "distillschedule.static_fsdp2_p0_summary.v3"})
    with pytest.raises(ValueError, match="H100"):
        module.validate_timing_gate(path, sha(path), expected_head=HEAD,
                                    frozen=module.freeze_policy(ROOT, variant="no-gan", arm="best"), asset_fingerprint="d" * 64)


def test_fresh_correctness_receipt_and_sha_gate_diagnostic_timing_only(tmp_path):
    frozen = fixture(tmp_path)
    value = result(tmp_path, frozen)
    path = tmp_path / "summary.json"
    write(path, value)
    admitted = module.validate_timing_gate(path, sha(path), expected_head=HEAD, frozen=frozen, asset_fingerprint="d" * 64)
    assert admitted["strict_claimable"] is False
    mutate(tmp_path, 0, lambda row: row.update(hostname="changed"), reseal=False)
    with pytest.raises(ValueError, match="SHA"):
        module.validate_timing_gate(path, sha(path), expected_head=HEAD, frozen=frozen, asset_fingerprint="d" * 64)


def test_launcher_has_only_owned_single_host_gpu_path():
    path = ROOT / "scripts/distillschedule/launch_helios_hardware_p0_node.sh"
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0
    source = path.read_text()
    assert "--nnodes=1" in source and "--nproc-per-node=8" in source
    assert "flock" in source and "memory.events" in source
    assert "pkill" not in source and "--profile-mode" not in source
    assert "--prepare-launch" in source and "--seal-run" in source


def gate_fixture(root):
    """Tiny fully hashed files, explicitly not real model/production assets."""
    frozen = copy.deepcopy(module.freeze_policy(ROOT, variant="no-gan", arm="baseline"))
    groups = []
    for name, subfolder in (("generator_precast", "transformer_ode"), ("score_precast", "transformer")):
        actual = root / name
        actual.mkdir()
        (actual / "model-00001.safetensors").write_bytes(b"CPU test fixture, not model weights")
        write(actual / "config.json", {"fixture_only": True})
        write(actual / "index.json", {"metadata": {"total_size": 2}, "weight_map": {"x": "model-00001.safetensors"}})
        manifest = {"schema_version": "distillschedule.precast_safetensors.v1",
                    "source": {"path": "/absent-original-share/model/" + subfolder},
                    "artifact": {"target_dtype": "bfloat16", "keep_in_fp32_modules": module.FP32_KEEP,
                                 "config_file": "config.json", "config_sha256": sha(actual / "config.json"),
                                 "index_file": "index.json", "index_sha256": sha(actual / "index.json"),
                                 "tensor_count": 1, "tensor_bytes": 2,
                                 "shards": [{"file": "model-00001.safetensors", "size_bytes": (actual / "model-00001.safetensors").stat().st_size,
                                             "sha256": sha(actual / "model-00001.safetensors")}]}}
        write(actual / "distillschedule_precast_manifest.json", manifest)
        groups.append({"name": name, "actual_root": str(actual), "source_path": manifest["source"]["path"],
                       "precast_manifest_sha256": sha(actual / "distillschedule_precast_manifest.json")})
    bundle, _ = trace_fixture(root / "trace", "no-gan")
    for rank in range(16):
        (root / "trace" / f"rank{rank}" / "tensors.pt").write_bytes(b"not a tensor payload; control-plane fixture only")
    frozen["trace_hash"] = bundle["bundle_hash"]
    checkpoint = signed({"fixture_only": True}, "manifest_hash")
    write(root / "logical_checkpoint/manifest.json", checkpoint)
    frozen["logical_hash"] = checkpoint["manifest_hash"]
    (root / "logical_checkpoint/state.pt").write_bytes(b"not an optimizer; fixture only")
    write(root / "negative_prompt/manifest.json", {"fixture_only": True})
    for name in ("trace", "logical_checkpoint", "negative_prompt"):
        groups.append({"name": name, "actual_root": str(root / name)})
    for group in groups:
        actual = Path(group["actual_root"])
        group["files"] = [{"relative_path": str(file.relative_to(actual)), "bytes": file.stat().st_size,
                           "sha256": sha(file), "observed_sha256": sha(file), "status": "PASS"}
                          for file in sorted(actual.rglob("*")) if file.is_file()]
    gate = {"schema_version": module.ASSET_SCHEMA, "status": "PASS", "exact_head": HEAD,
            "artifact_root": str(root), "groups": groups, "fixture_only": True}
    path = root / "gate.json"
    write(path, gate)
    return path, frozen


def test_full_precast_hash_gate_allows_absent_original_provenance_without_alias(tmp_path):
    path, frozen = gate_fixture(tmp_path)
    admitted = module.verify_asset_gate(path, sha(path), expected_head=HEAD, asset_root=tmp_path, frozen=frozen)
    assert admitted["groups"]["generator_precast"]["provenance_root"] == "/absent-original-share/model"
    assert not Path("/absent-original-share/model").exists()
    assert not (tmp_path / "share").exists()


@pytest.mark.parametrize("change, message", [
    (lambda gate: gate.update(status="PENDING"), "PASS"),
    (lambda gate: gate.update(exact_head="f" * 40), "HEAD"),
    (lambda gate: gate["groups"][0].update(source_path="/changed/model/transformer_ode"), "source.path"),
    (lambda gate: gate["groups"][0]["files"].pop(), "shard"),
    (lambda gate: gate["groups"][0]["files"][0].update(observed_sha256="f" * 64), "full-hash"),
    (lambda gate: gate["groups"][0]["files"][0].update(relative_path="../escape"), "relative_path"),
    (lambda gate: gate["groups"][0].update(precast_manifest_sha256="f" * 64), "manifest SHA"),
    (lambda gate: gate["groups"][0].update(actual_root="/outside-owned-root"), "escapes"),
])
def test_asset_gate_fails_closed_before_any_model_or_gpu_work(tmp_path, change, message):
    path, frozen = gate_fixture(tmp_path)
    gate = json.loads(path.read_text())
    change(gate)
    write(path, gate)
    with pytest.raises(ValueError, match=message):
        module.verify_asset_gate(path, sha(path), expected_head=HEAD, asset_root=tmp_path, frozen=frozen)


def test_changed_live_precast_file_cannot_reuse_pass_gate(tmp_path):
    path, frozen = gate_fixture(tmp_path)
    (tmp_path / "generator_precast/model-00001.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="live asset file"):
        module.verify_asset_gate(path, sha(path), expected_head=HEAD, asset_root=tmp_path, frozen=frozen)


def test_missing_scalar_diagnostics_is_reported_not_fabricated_as_loss_parity(tmp_path):
    frozen = fixture(tmp_path)
    def no_scalars(row):
        for child in row["generator"]["microbatches"]:
            child["surrogate_loss"] = None
        for update in row["critic"]:
            for child in update["microbatches"]:
                child["loss"] = None
    for rank in range(8):
        mutate(tmp_path, rank, no_scalars)
    value = result(tmp_path, frozen)
    assert value["status"] == "PASS", value["errors"]
    assert value["loss_numeric_evidence"] == "NOT_RECORDED"
    assert value["strict_claimable"] is False


def test_policy_argv_is_accepted_by_the_actual_worker_parser_without_torch():
    # Extract only the two argparse factory functions from the frozen source.
    # This tests the real CLI surface without importing models, CUDA or Torch.
    def factory(path, name, namespace):
        tree = ast.parse(path.read_text())
        selected = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[selected], type_ignores=[]), str(path), "exec"), namespace)
        return namespace[name]
    base = factory(ROOT / "scripts/distillschedule/probe_static_fsdp2_full5_world16.py", "parser",
                   {"argparse": argparse, "Path": Path})
    parser = factory(ROOT / "scripts/distillschedule/probe_static_fsdp2_p0_world16.py", "_parser",
                     {"base": SimpleNamespace(parser=base), "DEFAULT_TARGET_BYTES": 113049600})
    for variant in ("no-gan", "gan"):
        for arm in ("baseline", "best"):
            policy = module.freeze_policy(ROOT, variant=variant, arm=arm)["policy"]
            argv = ["--variant", variant, "--run-id", "fixture", "--output-dir", "/unused",
                    "--logical-package", "/unused", "--trace-bundle", "/unused",
                    "--generator-root", "/provenance", "--precast-generator-root", "/relocated",
                    "--score-root", "/provenance", "--precast-score-root", "/relocated",
                    "--negative-prompt", "/unused", *module.policy_arguments(policy)]
            parsed = parser().parse_args(argv)
            assert parsed.rollout_batch_size == 1
            assert parsed.pack_real_score_pair is False


def test_validate_receipts_do_not_claim_or_contain_model_work(tmp_path):
    frozen = fixture(tmp_path, mode="validate")
    def control_only(row):
        for key in ("build", "generator", "critic", "ema", "full5_seconds"):
            row.pop(key, None)
    for rank in range(8):
        mutate(tmp_path, rank, control_only)
    value = result(tmp_path, frozen)
    assert value["status"] == "PASS", value["errors"]
    assert value["execution_correctness_admitted"] is False
    assert value["full5_seconds_max_rank"] is None


@pytest.mark.parametrize("field, replacement", [
    ("expected_head", "f" * 40),
    ("asset_fingerprint", "f" * 64),
    ("hardware_profile", "a100-16-b16"),
    ("completed_unix_ns", 1),
])
def test_timing_gate_rejects_mismatched_or_stale_correctness(tmp_path, field, replacement):
    frozen = fixture(tmp_path)
    value = result(tmp_path, frozen)
    value[field] = replacement
    path = tmp_path / "summary.json"
    write(path, value)
    with pytest.raises(ValueError):
        module.validate_timing_gate(path, sha(path), expected_head=HEAD, frozen=frozen, asset_fingerprint="d" * 64)


def test_native_parent_and_child_physical_step_receipts_are_not_normalized(tmp_path):
    frozen = fixture(tmp_path)
    value = result(tmp_path, frozen)
    parent = value["physical_receipts"][0]
    assert [child["physical_generator_step_after"] for child in parent["generator"]["microbatches"]] == [2, 3]
    assert parent["generator"]["source_ranks"] == [0, 8]
    assert len(value["physical_receipts"]) == 8
    assert "normalized_ranks" not in value

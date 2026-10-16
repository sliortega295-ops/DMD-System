"""CPU-only adversarial protocol tests, never H100 execution or measured costs.

Actual 8fc outer two-slot helpers run with the existing explicit model fixtures.
Source derivation/inventory and public planner are real. Raw fixture health and
asset reads are controlled test doubles; their scope cannot pass native costs.
"""
import copy
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from distillschedule.runtime import h100_helios_c2_control as c
from distillschedule.search import hardware_helios_c2 as h
from distillschedule.search.model_search import search_plans
from test_refactor_hardware_helios import deployment as legacy_deployment
import test_refactor_h100_helios_two_cycle as protocol
from test_refactor_h100_helios_two_cycle import source_functions

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("DMD_HELIOS_H100_C2_ROOT", str(ROOT / "sources/native8fc_c2_r1")))
EVIDENCE = Path(os.environ.get("DMD_HELIOS_H100_C2_EVIDENCE", str(ROOT / "control_evidence")))
PROOF = EVIDENCE / "source-proof.json"
NATIVE = os.environ.get("DMD_HELIOS_H100_NATIVE_ROOT")


class PrecisionMetadataTests(unittest.TestCase):
    @staticmethod
    def deployment():
        # Pure identity fixture; it is never source-inspected or launched.
        d = legacy_deployment()
        d.update(schema_version=h.DEPLOYMENT,
            native_source=dict(d["native_source"], head="1" * 40),
            source_proof={"path": "/fixture/source-proof.json", "sha256": "c" * 64},
            evidence_scope="CPU_FIXTURE_ONLY")
        d["binding"] = h._binding(d)
        return h.validate_deployment(d)

    def test_precision_separates_master_storage_and_unsharded_compute(self):
        expected = {"frozen_base": "torch.bfloat16", "trainable_lora_master_storage": "torch.float32",
            "fsdp_unsharded_param_compute_dtype": "torch.bfloat16", "reduce_dtype": "torch.float32",
            "loss_scale_per_logical_b1": .5, "ema": "NATIVE_GPU_LOCAL_SHARD_REFERENCE_RECURRENCE"}
        self.assertEqual(h.PRECISION, expected)
        d = self.deployment()
        self.assertEqual(d["binding"]["precision_fingerprint"], c.digest(expected))
        for arm in h.runtime.POLICIES:
            for cycle in (1, 2):
                self.assertEqual(h.component_contract(d, arm, cycle).precision, expected)
        for key in ("trainable_lora_master_storage", "fsdp_unsharded_param_compute_dtype", "reduce_dtype"):
            with self.subTest(field=key):
                changed = dict(expected, **{key: "torch.bfloat16" if expected[key] == "torch.float32" else "torch.float32"})
                with patch.object(h, "PRECISION", changed):
                    self.assertNotEqual(h._binding(d)["precision_fingerprint"], d["binding"]["precision_fingerprint"])
                    with self.assertRaisesRegex(ValueError, "binding changed"):
                        h.validate_deployment(d)

    def test_ambiguous_legacy_precision_identity_cannot_reuse_costs(self):
        legacy = {"frozen_base": "torch.bfloat16", "trainable_lora": "torch.float32", "reduce_dtype": "torch.float32",
            "loss_scale_per_logical_b1": .5, "ema": "NATIVE_GPU_LOCAL_SHARD_REFERENCE_RECURRENCE"}
        with patch.object(h, "PRECISION", legacy):
            old = self.deployment()
            old_problem = h.planning_problem(old)
            old_contract = h.component_contract(old, "best", 2)
            record = h.OperationCost(old_contract.digest, (1,), old["allocation"]["devices"][0]["capacity_bytes"],
                {"component_contract": old_contract.to_dict(), "fixture_scope": "CPU_FIXTURE_ONLY"})
            old_costs = h.CostModel(old["binding"], (record,), "CPU_FIXTURE_ONLY")
        fresh = self.deployment()
        self.assertNotEqual(old["binding"]["precision_fingerprint"], fresh["binding"]["precision_fingerprint"])
        self.assertNotEqual(old_contract.digest, h.component_contract(fresh, "best", 2).digest)
        with self.assertRaisesRegex(ValueError, "binding changed"):
            h.validate_deployment(old)
        with self.assertRaisesRegex(ValueError, "precision mismatch"):
            old_costs.bind(fresh["binding"], problem_scope="CPU_FIXTURE_ONLY")
        with self.assertRaisesRegex(ValueError, "registered native C2 graph"):
            h.validate_model(old_problem, old_costs, fresh)


@unittest.skipUnless(NATIVE and SOURCE.is_dir() and PROOF.is_file(), "explicit native 8fc and prepared C2 source required")
class SourceAndPlanTests(unittest.TestCase):
    def deployment(self, scope="MEASURED"):
        return h.bind_deployment(legacy_deployment(), c.reference(PROOF), repo_root=SOURCE, evidence_scope=scope)

    def test_exact_derived_diff_old_head_and_source_clean(self):
        proof = c.inspect_source(SOURCE)
        self.assertEqual(proof, c.read(PROOF))
        self.assertEqual(proof["starting_head"], h.runtime.PARENT_HEAD)
        self.assertTrue(proof["source_contract"]["hook_present"])
        self.assertIn(h.runtime.HOOK, (SOURCE / h.runtime.WORKER).read_text())
        self.assertEqual(c.git(Path(NATIVE), "rev-parse", "HEAD").strip(), h.runtime.PARENT_HEAD)
        self.assertEqual(c.git(Path(NATIVE), "status", "--porcelain"), "")
        self.assertEqual(c.sha(SOURCE / c.OLD_LAUNCHER), c.OLD_LAUNCHER_SHA)
        self.assertEqual(c.sha(SOURCE / h.runtime.AUDITOR), h.runtime.AUDITOR_SHA)

    def test_source_prepare_requires_explicit_commit_permission(self):
        with self.assertRaisesRegex(ValueError, "explicit permission"):
            h.prepare_source(NATIVE, SOURCE / "must-not-exist")
        self.assertFalse((SOURCE / "must-not-exist").exists())

    def test_default_off_and_old_source_cannot_compile(self):
        d = self.deployment()
        with self.assertRaisesRegex(ValueError, "default-off"):
            h.compile_plan(d, arm="best", mode="correctness", run_id="test")
        for arm, mode in (("fa3", "timing"), ("best", "validate"), ("best", "two-cycle")):
            with self.assertRaises(ValueError):
                h.compile_plan(d, arm=arm, mode=mode, run_id="test", explicit_opt_in=True)
        d["native_source"]["head"] = h.runtime.PARENT_HEAD
        with self.assertRaises(ValueError): h.validate_deployment(d)

    def test_source_inventory_and_two_cycle_compile_not_old_full5(self):
        d = self.deployment()
        p = h.compile_plan(d, arm="h100-nograd-retain-nosync", mode="correctness", run_id="cpu-source", explicit_opt_in=True)
        self.assertEqual(p["argv"][1], str(SOURCE / c.LAUNCHER))
        self.assertEqual(p["contract"]["continuous_cycles"], 2)
        self.assertEqual(p["execution"], "NOT_RUN")
        self.assertEqual(p["purpose"], "EXPLICIT_CALIBRATION")
        self.assertFalse(p["claimable_speedup"])
        self.assertEqual(h.validate_plan(p), p)
        with tempfile.TemporaryDirectory(dir=EVIDENCE) as tmp:
            f = Path(tmp) / "plan.json"; c.write_new(f, p)
            dry = h.dry_run(c.reference(f))
            self.assertEqual(dry["gpu_execution"], "NOT_RUN")
            self.assertEqual(dry["environment"][c.PREFIX + "PLAN_SHA256"], c.sha(f))
            self.assertIn("NOT_CHECKED", dry["live_assets"])

    def test_source_launcher_retains_real_safety_and_restores_mode(self):
        original = (SOURCE / c.OLD_LAUNCHER).read_text()
        derived = (SOURCE / c.LAUNCHER).read_text()
        self.assertEqual(derived, c.launcher_text(original))
        subprocess.run(["bash", "-n", str(SOURCE / c.LAUNCHER)], check=True)
        for marker in ("flock -n -E 75", "cleanup_owned_workers", "setsid timeout", "nproc-per-node=8", "environment.nul"):
            self.assertIn(marker, derived)
        self.assertIn(c.PREFIX + "OPT_IN", derived)
        self.assertNotIn("probe_static_fsdp2_p0_world16.py", derived)  # hardened actual worker selected by prepare

    def test_compile_timing_requires_c2_ref_and_fixture_cannot_launch(self):
        d = self.deployment("CPU_FIXTURE_ONLY")
        with self.assertRaises(ValueError):
            h.compile_plan(d, arm="best", mode="timing", run_id="test", explicit_opt_in=True)
        p = h.compile_plan(d, arm="best", mode="correctness", run_id="test", explicit_opt_in=True)
        with tempfile.TemporaryDirectory(dir=EVIDENCE) as tmp:
            f = Path(tmp) / "plan.json"; c.write_new(f, p)
            with self.assertRaisesRegex(ValueError, "fixture"):
                h.dry_run(c.reference(f))

    def test_model_search_is_same_public_ir_and_fresh_native_lowering(self):
        d = self.deployment("CPU_FIXTURE_ONLY")
        problem = h.planning_problem(d)
        records = []
        for arm, seconds in zip(h.runtime.POLICIES, (30, 20, 10)):
            for cycle in (1, 2):
                contract = h.component_contract(d, arm, cycle)
                records.append(h.OperationCost(contract.digest, (seconds * 10**9,), d["allocation"]["devices"][0]["capacity_bytes"],
                    {"component_contract": contract.to_dict(), "fixture_scope": "CPU_FIXTURE_ONLY"}))
        costs = h.CostModel(d["binding"], tuple(records), "CPU_FIXTURE_ONLY")
        result = search_plans(problem, costs, algorithm="hierarchical")
        self.assertEqual(result["predicted_best"]["assignment"], {"arm": "h100-nograd-retain-nosync"})
        p = h.lower_selection(result, problem, costs, d, run_id="fresh-selected", correctness={"path": "/fixture/correct.json", "sha256": "a"*64}, explicit_opt_in=True)
        self.assertEqual(p["purpose"], "PLANNER_SELECTED")
        self.assertEqual(p["model_plan"]["search_result"], result)
        self.assertEqual(p["argv"][6], "h100-nograd-retain-nosync")
        self.assertEqual(p["evidence_scope"], "CPU_FIXTURE_ONLY")
        tampered = copy.deepcopy(result); tampered["predicted_best"]["predicted_cycle2_ns"] = 1
        with self.assertRaisesRegex(ValueError, "altered"):
            h.lower_selection(tampered, problem, costs, d, run_id="fake", correctness={"path": "/fixture/correct.json", "sha256": "a"*64}, explicit_opt_in=True)

    def test_uncalibrated_never_imports_old_full5_costs(self):
        d = self.deployment()
        with self.assertRaisesRegex(ValueError, "UNCALIBRATED"):
            h.assemble_problem_and_costs(d, [], output_dir=EVIDENCE / "must-not-create")
        self.assertFalse((EVIDENCE / "must-not-create").exists())
        result = search_plans(h.planning_problem(d), h.CostModel(d["binding"], (), "MEASURED"), algorithm="hierarchical")
        self.assertIsNone(result["predicted_best"])
        self.assertFalse(result["claimable_speedup"])


@unittest.skipUnless(NATIVE and SOURCE.is_dir() and PROOF.is_file(), "explicit native 8fc and prepared C2 source required")
class RawC2AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        protocol.NativeProtocolTests.setUpClass()
        cls.protocol = protocol.NativeProtocolTests()
        cls.source = c.inspect_source(SOURCE)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=EVIDENCE)
        self.addCleanup(self.temp.cleanup)
        base = legacy_deployment(); base["site"]["project_root"] = self.temp.name
        self.d = h.bind_deployment(base, c.reference(PROOF), repo_root=SOURCE, evidence_scope="CPU_FIXTURE_ONLY")
        self.plan = h.compile_plan(self.d, arm="h100-nograd-retain-nosync", mode="correctness", run_id="raw-c2-fixture", explicit_opt_in=True)
        self.frozen = self.plan["frozen_policy"]
        self.rows, self.controls = [], None
        for rank in range(8):
            context = self.protocol.context(rank=rank, mode="correctness")
            row = self.protocol.run_context(context)
            self.native = context[1]
            self.controls = self.native.trace_controls(None, None, None)
            row.update(run_id=self.plan["run_id"], variant="no-gan", local_rank=rank, hostname="CPU_FIXTURE_ONLY_HOST", source_contract=self.source["source_contract"],
                profile_mode=False, timing_mode=False, profile_diagnostics=True, ema_device="cuda", diagnostic_single_critic=False, operation_source={"enabled": False},
                initial={"generator_physical_step": 2, "critic_physical_step": 10, "ema_step": 10})
            row["identities"].update(logical_manifest_hash=self.frozen["logical_hash"], cross_step_rollout_batchability="PASS",
                trace_binding_hash=c.digest({"schema": "distillschedule.static_world16_trace.v1", "variant": "no-gan", "bundle_hash": self.frozen["trace_hash"], "logical_steps": list(h.runtime.STEPS), "rank_slots": list(range(16))}),
                rank_control_validation={"status": "PASS", "rank_count": 16, "cross_rank_call_count_parity": True})
            row["attention_backend"].update(flash_attn_func_available=True, flash_attn_func_module="CPU_FIXTURE_ONLY.flash_attn2")
            for cycle in row["cycles"]:
                cycle["source_identity"] = copy.deepcopy(row["identities"])
                cycle["ema_checksum_outside_timer"] = {"device": "cuda", "optimization_step": cycle["final_versions"]["ema"],
                    "communication_per_update": False, "tracked_tensor_count": 1, "local_sum": 0.1, "local_norm": .2, "fixture_scope": "CPU_FIXTURE_ONLY"}
                for batch in cycle["rollout_batches"]:
                    batch["batch_parity"] = {"enabled": bool(self.frozen["policy"]["generator_nograd_cross_kv_cache"] or self.frozen["policy"]["generator_nograd_static_context_cache"]), "status": "PASS", "samples": [{"status": "PASS", "finite": True, "same_shape": True,
                        "same_dtype": True, "mean_relative": 0., "normalized_l2": 0., "cosine": 1., "max_abs": 0., "max_abs_limit": .01}]}
            self.rows.append(json.loads(json.dumps(row)))
        namespace = {"_require": c.require, "Path": Path, "_finite": lambda x: type(x) in (int, float) and __import__('math').isfinite(x)}
        source_functions(Path(NATIVE) / h.runtime.AUDITOR, ["policy_arguments", "_confined", "_audit_parity"], namespace)
        for key in ("policy_arguments", "_confined", "_audit_parity"): setattr(self.native, key, namespace[key])
        self.native.HEALTH_SCHEMA = "distillschedule.helios_h100_health.v1"
        self.native.freeze_policy = lambda *a, **kw: self.frozen
        self.gate = {"fingerprint": "f"*64, "groups": {name: {"actual_root": f"/fixture/assets/{name}", "provenance_root": "/original",
            "precast_manifest_sha256": c.digest(name), "source_path": "/original/transformer_ode" if name == "generator_precast" else "/original/transformer"}
            for name in ("logical_checkpoint", "trace", "generator_precast", "score_precast", "negative_prompt", "flash_attn_kernel")}}
        self.retention = {"path": self.d["retention_api"]["path"], "sha256": self.d["retention_api"]["sha256"], "status": "PASS_LIBRARY_API_ONLY", "real_Helios_capacity": "NOT_RUN_BY_PRIMITIVE"}
        self.rendered = c._render(self.plan, self.native, self.gate, self.retention)
        for row in self.rows:
            row["build"] = {"static_residency": {"both_backbones_sharded_for_full5": True, "parameter_offload": False, "frozen_parameter_streaming": False,
                "role_transition": False, "process_group_rebuild_in_full5": False}}
            for role, asset, sub in (("generator", "generator_precast", "transformer_ode"), ("critic", "score_precast", "transformer")):
                row["build"][role] = {"metadata": {"loading_mode": "mixed_bf16_precast_direct", "block_count": 40, "trainable_dtype_inventory": ["torch.float32"],
                    "chunked_residual_storage": "cuda_out_of_place", "precast_manifest_sha256": self.rendered["precast_manifest_sha256s"][role],
                    "load_root": self.rendered["assets"][asset], "subfolder": sub, "backbone_root": "/original"},
                    "fsdp2": {"mesh_policy": self.frozen["policy"]["mesh_policy"], "mesh_shape": self.frozen["mesh_shape"], "world_size": 8,
                        "hardware_profile": "h100-8-b16", "logical_global_batch": 16, "logical_b1_microbatches_per_rank": 2, "block_count": 40,
                        "reshard_after_forward": True, "param_compute_dtype": "torch.bfloat16", "reduce_dtype": "torch.float32", "cpu_parameter_offload": False,
                        "parameter_streaming": False, "transition": False}}
        self.start = min(r["cycles"][0]["start_unix_ns"] for r in self.rows) - 100
        self.stop = max(r["cycles"][1]["stop_unix_ns"] for r in self.rows) + 100

    def audit(self, rows=None, **kwargs):
        return c.audit_rank_rows(self.rows if rows is None else rows, plan=self.plan, source=self.source, native=self.native,
            frozen=self.frozen, controls=self.controls, manifest=self.rendered, started_ns=self.start, completed_ns=self.stop,
            evidence_scope=kwargs.get("scope", "CPU_FIXTURE_ONLY"))

    def test_real_helpers_generate_two_census_cycles_and_all_slots(self):
        result = self.audit()
        self.assertEqual(result["logical_slots"], list(range(16)))
        self.assertEqual(result["final_versions"], {"generator": 4, "critic": 20, "ema": 20})
        self.assertEqual(result["cycle_ns"], [max(round(r["cycles"][i]["wall_seconds"] * 1e9) for r in self.rows) for i in range(2)])
        self.assertEqual(self.rendered["environment"][h.runtime.MODE_ENV], "correctness")
        self.assertIn("--profile-diagnostics", self.rendered["worker_argv"])
        self.assertEqual(self.rendered["worker_argv"][0], str(SOURCE / c.WORKER))

    def test_legacy_full5_and_missing_cycle_and_duplicate_physical_rank_rejected(self):
        for mutate in (lambda r: r[0].update(schema_version="distillschedule.static_fsdp2_p0_rank.v1", full5_seconds=94.311432633),
                       lambda r: r[0].update(cycles=r[0]["cycles"][:1]), lambda r: r[1].update(rank=0)):
            rows = copy.deepcopy(self.rows); mutate(rows)
            with self.assertRaises((ValueError, KeyError)): self.audit(rows)

    def test_slot_version_census_identity_and_object_rebuild_fail_closed(self):
        changes = [lambda r: r[0]["cycles"][1]["initial_versions"].update(critic=10),
            lambda r: r[0]["hardware_profile"]["logical_slots"].__setitem__(1, 0),
            lambda r: r[0]["cycles"][1]["census"].update(ema=4),
            lambda r: r[0]["cycles"][1]["object_identity"]["generator"].__setitem__(0, 777),
            lambda r: r[0]["cycles"][1]["generator"]["microbatches"][0].update(optimizer_committed=True),
            lambda r: r[0]["cycles"][1]["rollout_batches"][0]["member_trace_identities"].__setitem__(0, "0"*64),
            lambda r: r[0]["source_contract"].update(actual_source_head=h.runtime.PARENT_HEAD)]
        for mutate in changes:
            rows = copy.deepcopy(self.rows); mutate(rows)
            with self.assertRaises((ValueError, KeyError)): self.audit(rows)

    def test_missing_actual_fields_shape_dtype_and_FA3_never_fallback(self):
        changes = [lambda r: r[0]["cycles"][1].pop("census"), lambda r: r[0]["cycles"][0]["rollout_batches"][0].update(output_dtype="torch.float32"),
            lambda r: r[0]["cycles"][0]["generator"]["microbatches"][0]["rollout"].update(output_shape=[1,16,9,24,40]),
            lambda r: r[0]["attention_backend"].update(runtime="fa3"), lambda r: r[0]["build"]["generator"]["fsdp2"].update(world_size=16),
            lambda r: r[0]["build"]["generator"]["metadata"].update(trainable_dtype_inventory=["torch.bfloat16"]),
            lambda r: r[0]["build"]["generator"]["fsdp2"].update(param_compute_dtype="torch.float32"),
            lambda r: r[0]["build"]["generator"]["fsdp2"].update(reduce_dtype="torch.bfloat16")]
        for mutate in changes:
            rows = copy.deepcopy(self.rows); mutate(rows)
            with self.assertRaises((ValueError, KeyError)): self.audit(rows)

    def test_retention_fence_cache_and_external_bookkeeping_timers_checked(self):
        changes = [lambda r: r[0]["cycles"][1]["generator_nograd_persistent_unshard"]["microbatches"][0]["persistence"].update(restored=False),
            lambda r: r[0]["cycles"][1]["outer_boundary_fences"].update(post_cuda_sync=0),
            lambda r: r[0]["cycles"][1]["trace_cache"].update(critic_entries_still_cached=1),
            lambda r: r[0]["cycles"][1].update(wall_seconds=94.3114),
            lambda r: r[0]["cycles"][1].update(stop_unix_ns=self.stop+1),
            lambda r: r[0].update(timer_scope="INCLUDES_MODEL_INITIALIZATION"),
            lambda r: r[0]["cycles"][1]["rollout_fences"].update(critic_rollout_sync_count=10)]
        for mutate in changes:
            rows = copy.deepcopy(self.rows); mutate(rows)
            with self.assertRaises((ValueError, KeyError)): self.audit(rows)

    def test_cpu_fixture_cannot_become_measured_or_cost(self):
        with self.assertRaisesRegex(ValueError, "fixture"):
            self.audit(scope="MEASURED")
        with self.assertRaisesRegex(ValueError, "fixtures"):
            h.validate_observation(self.d, {"summary": {}, "plan": {}})
        with self.assertRaises(ValueError):
            h.assemble_problem_and_costs(self.d, [], output_dir=Path(self.temp.name) / "costs")

    def bundle(self):
        run = Path(self.plan["run_root"])
        manifest = dict(self.rendered, schema_version=c.LAUNCH_SCHEMA, run_id=self.plan["run_id"], arm=self.plan["arm"], mode=self.plan["mode"], variant="no-gan",
            expected_head=self.d["native_source"]["head"], plan=self.plan, source_proof=self.source, asset_gate=self.d["asset_gate"], asset_fingerprint=self.gate["fingerprint"],
            correctness_gate=None, started_unix_ns=self.start, evidence_scope="CPU_FIXTURE_ONLY")
        c.write_new(run / "manifests/launch.json", manifest)
        for name, rows in (("worker_argv", self.rendered["worker_argv"]), ("environment", [f"{k}={v}" for k,v in sorted(self.rendered["environment"].items())]), ("unset_environment", self.rendered["unset_environment"])):
            (run / f"manifests/{name}.nul").write_bytes(b''.join(v.encode()+b'\0' for v in rows))
        prefix = [self.d["site"]["python"], "-m", "torch.distributed.run", "--nnodes=1", "--nproc-per-node=8", "--node-rank=0", "--master-addr=127.0.0.1", f"--master-port={self.d['master_port']}"]
        (run / "manifests/command.sh").write_text(shlex.join(prefix+self.rendered["worker_argv"]))
        env = {k:v for k,v in self.rendered["environment"].items() if k not in ("TOKENIZERS_PARALLELISM", "PYTHONDONTWRITEBYTECODE")}
        env["LD_LIBRARY_PATH"] = str(Path(self.d["site"]["python"]).parent.parent / "lib")
        (run / "manifests/environment.txt").write_text(''.join(f'{k}={v}\n' for k,v in sorted(env.items())))
        c.write_new(run / "manifests/environment_versions.json", {"python": self.d["runtime_versions"]["python"], "executable": self.d["site"]["python"], "packages": self.d["runtime_versions"]["packages"]})
        (run / "logs").mkdir()
        for name in ("source_status_after", "cgroup_before", "cgroup_after", "gpu_processes_before", "gpu_processes_after"):
            (run / f"logs/{name}.txt").write_text("" if "cgroup" not in name else "oom 0\noom_kill 0\n")
        for row in self.rows: c.write_new(run / f"execution/ranks/rank_{row['rank']:02d}.json", row)
        health = {"schema_version": self.native.HEALTH_SCHEMA, "status": "PASS", "exit_code": 0, "expected_head": self.d["native_source"]["head"],
            "observed_head_after": self.d["native_source"]["head"], "clean_source_after": True, "cgroup_oom_delta": {"oom": 0, "oom_kill": 0},
            "cgroup_before": {"oom": 0, "oom_kill": 0}, "cgroup_after": {"oom": 0, "oom_kill": 0},
            "gpu_processes_before": [], "gpu_processes_after": [], "launch_manifest_sha256": c.sha(run / "manifests/launch.json"),
            "gpu_inventory": [{"index": g["rank"], "uuid": g["uuid"], "name": "CPU_FIXTURE_ONLY H100", "memory_mib": g["capacity_bytes"] // 2**20,
                "driver": self.d["runtime_versions"]["driver"]} for g in self.d["allocation"]["devices"]],
            "completed_unix_ns": self.stop, "control_files": [{"path": str(p.relative_to(run)), "sha256": c.sha(p)} for folder in ("logs", "manifests") for p in sorted((run / folder).iterdir())],
            "rank_receipts": [{"path": str(p.relative_to(run)), "sha256": c.sha(p)} for p in sorted((run / "execution/ranks").glob("*.json"))]}
        c.write_new(run / "logs/gpu_inventory.json", {"gpus": health["gpu_inventory"]})
        c.write_new(run / "run_status.json", health)
        return run

    def audit_bundle(self, run):
        with patch.object(c, "native_module", return_value=self.native), patch.object(c, "_assets", return_value=(self.gate, self.retention, self.controls)):
            return c.audit_run(SOURCE, run, evidence_scope="CPU_FIXTURE_ONLY")

    def test_raw_eight_rank_file_seals_and_saved_summary_match(self):
        run = self.bundle(); result = self.audit_bundle(run)
        self.assertEqual(result["status"], "PASS_C2_PROTOCOL_ONLY")
        self.assertEqual(result["evidence_scope"], "CPU_FIXTURE_ONLY")
        self.assertEqual(result["cycle_ns"], self.audit()["cycle_ns"])
        self.assertFalse(result["claimable_speedup"])
        with patch.object(c, "native_module", return_value=self.native), patch.object(c, "_assets", return_value=(self.gate, self.retention, self.controls)), self.assertRaisesRegex(ValueError, "fixture"):
            c.audit_run(SOURCE, run)
        p = run / "execution/ranks/rank_00.json"; altered = c.read(p); altered["cycles"][1]["census"]["ema"] = 4
        p.write_text(json.dumps(altered))
        with self.assertRaisesRegex(ValueError, "seal"):
            self.audit_bundle(run)

    def test_resealed_rank_tamper_still_fails_semantics(self):
        run = self.bundle(); p = run / "execution/ranks/rank_00.json"
        altered = c.read(p); altered["cycles"][1]["rollout_batches"][0]["member_trace_identities"] = ["0"*64]; p.write_text(json.dumps(altered))
        health = c.read(run / "run_status.json"); health["rank_receipts"][0]["sha256"] = c.sha(p); (run / "run_status.json").write_text(json.dumps(health))
        with self.assertRaisesRegex(ValueError, "identity"):
            self.audit_bundle(run)

    def test_missing_sealed_control_and_forged_argv_rejected(self):
        run = self.bundle(); health = c.read(run / "run_status.json")
        health["control_files"] = [r for r in health["control_files"] if r["path"] != "manifests/command.sh"]
        (run / "run_status.json").write_text(json.dumps(health))
        with self.assertRaisesRegex(ValueError, "omitted"):
            self.audit_bundle(run)

    def test_fresh_c2_correctness_is_reaudited_and_expires(self):
        run = self.bundle(); result = self.audit_bundle(run)
        f = run / "summary.json"; c.write_new(f, result)
        plan = dict(self.plan, mode="timing", run_id="fresh-target", correctness=c.reference(f))
        with patch.object(c, "native_module", return_value=self.native), patch.object(c, "_assets", return_value=(self.gate, self.retention, self.controls)):
            gate = c.correctness_gate(plan, SOURCE, at_ns=self.stop+100)
            self.assertEqual(gate["admission"], "PASS_C2_PROTOCOL_ONLY_NUMERICAL_REVIEW")
            with self.assertRaisesRegex(ValueError, "expired"):
                c.correctness_gate(plan, SOURCE, at_ns=self.stop + 86401 * 10**9)
        rendered = c._render(plan, self.native, self.gate, self.retention)
        self.assertEqual(rendered["environment"][h.runtime.MODE_ENV], "timing")
        self.assertEqual(rendered["worker_argv"][-2:], ["--timing-mode", "--no-profile-diagnostics"])
        self.assertNotIn("--profile-diagnostics", rendered["worker_argv"])

    def test_resealed_nul_override_is_rejected(self):
        run = self.bundle(); path = run / "manifests/worker_argv.nul"
        path.write_bytes(path.read_bytes() + b"--attention-backend\0fa3\0")
        health = c.read(run / "run_status.json")
        for row in health["control_files"]:
            if row["path"] == str(path.relative_to(run)): row["sha256"] = c.sha(path)
        (run / "run_status.json").write_text(json.dumps(health))
        with self.assertRaisesRegex(ValueError, "NUL-rendered"):
            self.audit_bundle(run)

    def test_resealed_oom_log_cannot_hide_behind_health_status(self):
        run = self.bundle(); path = run / "logs/cgroup_after.txt"
        path.write_text("oom 1\noom_kill 1\n")
        health = c.read(run / "run_status.json")
        for row in health["control_files"]:
            if row["path"] == str(path.relative_to(run)): row["sha256"] = c.sha(path)
        (run / "run_status.json").write_text(json.dumps(health))
        with self.assertRaisesRegex(ValueError, "OOM"):
            self.audit_bundle(run)

    def test_actual_counts_cannot_be_booleans(self):
        rows = copy.deepcopy(self.rows); rows[0]["cycles"][0]["census"]["generator_commit"] = True
        with self.assertRaisesRegex(ValueError, "integers"):
            self.audit(rows)

    def test_old_correctness_and_wrong_arm_reject_before_timing(self):
        run = self.bundle(); result = self.audit_bundle(run)
        old = dict(result, schema_version="distillschedule.helios_h100_summary.v1", full5_seconds_max_rank=94.3114)
        f = Path(self.temp.name) / "old.json"; c.write_new(f, old)
        p = dict(self.plan, mode="timing", correctness=c.reference(f))
        with self.assertRaisesRegex(ValueError, "old Full5"):
            c.correctness_gate(p, SOURCE, at_ns=self.stop+100)
        old["schema_version"] = c.SCHEMA; old["arm"] = "best"; f.write_text(json.dumps(old)); p["correctness"] = c.reference(f)
        with self.assertRaisesRegex(ValueError, "same-arm"):
            c.correctness_gate(p, SOURCE, at_ns=self.stop+100)


if __name__ == "__main__":
    unittest.main()

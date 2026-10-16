"""CPU_FIXTURE_ONLY: synthetic receipts; no CUDA import, launch, or GPU evidence."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from distillschedule.search import hardware_wan as old
from distillschedule.search.allocation import ExecutionAllocation, PhysicalDevice
from distillschedule.search.cost_model import CostModel
from distillschedule.search.model_search import search_plans
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint

MODULE = Path(__file__).parents[1] / "src/distillschedule/search/a100_allocation.py"
spec = importlib.util.spec_from_file_location("distillschedule.search.a100_allocation", MODULE)
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)
NATIVE = Path(os.environ.get("A100_ALLOCATION_NATIVE_SOURCE", Path(__file__).parents[3] / "allocation_native_source"))
FIXTURE = "CPU_FIXTURE_ONLY"
SCRATCH = Path(tempfile.gettempdir()) / "lyy-experiments/DMD-System/allocation-planner-bridge-cpu"


class ImportContractTests(unittest.TestCase):
    def test_native_nccl_default_is_unset_not_the_invalid_auto_token(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=SCRATCH) as root:
            for world in (8, 16):
                d = deployment(Path(root), world)
                for row in bridge.command_lowering(d, run_id="CPU_FIXTURE_env", placement="resident"):
                    self.assertNotIn("NCCL_ALGO", row["environment"])
                    self.assertIn("NCCL_", row["clear_environment_prefixes"])

    def test_import_does_not_import_torch_in_fresh_interpreter(self):
        code = ("import sys; assert 'torch' not in sys.modules; "
                "import distillschedule.search.a100_allocation; "
                "assert 'torch' not in sys.modules")
        root = Path(__file__).resolve().parents[1]
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True,
                       env={**os.environ, "PYTHONPATH": str(root / "src")})


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return {"path": str(path), "sha256": file_sha256(path)}


def deployment(root, world=8):
    allocation = ExecutionAllocation(tuple(PhysicalDevice(r, r // 8, r % 8,
        f"GPU-CPU-FIXTURE-{r}", "sm80", 40 * 1024**3) for r in range(world)), 16,
        {role: tuple(tuple(range(r, 16, world)) for r in range(world)) for role in ("generator", "critic")})
    return bridge.make_deployment(profile=f"a100-{world}-b16", allocation=allocation.to_dict(),
        native_root=str(NATIVE), environment={"python": "3.10.20", "torch": "2.10.0+cu128", "driver_version": "535.183.06"},
        assets={k: old.SEMANTICS[k] for k in ("bootstrap_hash", "trace_bundle_hash", "model_identity_sha256", "negative_prompt_manifest_hash")},
        site={"python": "/CPU_FIXTURE/python", "helios_root": "/CPU_FIXTURE/helios", "asset_root": "/CPU_FIXTURE/assets",
              "run_root": str(root / "native_runs"), "master_addr": "fixture.invalid", "master_port": 29671,
              "node_hosts": [f"cpu-fixture-node-{i}" for i in range(world // 8)]}, evidence_scope=FIXTURE)


def trace_contract():
    result = []
    for i, counts in enumerate(((2, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 1), (2, 1, 1), (2, 2, 2))):
        stages = {str(j): c for j, c in enumerate(counts)}
        result.append({"operation": "step/10/generator/rollout" if i == 0 else f"step/{i+9}/critic/rollout",
            "inventory": {"call_count": sum(counts), "dummy_call_count": 0, "logical_call_count": sum(counts),
                          "logical_stage_call_count": stages, "stage_call_count": stages}})
    assert fingerprint(result) == old.SEMANTICS["trace_rollout_contract_sha256"]
    return result


def observation(d, placement, run_id, seconds=(12., 10.)):
    allocation = ExecutionAllocation.from_dict(d["allocation"])
    world = allocation.world_size
    root = Path(d["site"]["run_root"]) / run_id
    commands = bridge.command_lowering(d, run_id=run_id, placement=placement)
    health = {"status": "PASS", "exit_code": 0, "expected_head": bridge.NATIVE_HEAD,
        "cgroup_oom_delta": {"oom": 0, "oom_kill": 0}, "cgroup_before": {"oom": 0, "oom_kill": 0},
        "cgroup_after": {"oom": 0, "oom_kill": 0}, "gpu_processes_after": [], "evidence_scope": FIXTURE}
    rows = []
    for device in allocation.devices:
        rank = device.rank
        slots = list(range(rank, 16, world))
        cycles, censuses = [], []
        start = 1_000_000_000
        for cycle, duration in enumerate(seconds, 1):
            cycles.append({"cycle": cycle, "role": old._CYCLE_CONTRACT[f"cycle_{cycle}_role"],
                "local_elapsed_seconds": duration, "start_unix_ns": start, "stop_unix_ns": start + round(duration * 1e9)})
            start += round(duration * 1e9) + 1000
            samples = []
            for slot in slots:
                ids = [fingerprint(["CPU_FIXTURE_ONLY", slot, i]) for i in range(6)]
                samples.append({"generator_updates": 1, "critic_updates": 5, "ema_updates": 5,
                    "generator_score_forwards": 3, "critic_score_forwards": 5, "physical_rollout_batches": 5,
                    "rollout_batch_size": 1, "generator_trace_payload_identity": ids[0], "critic_trace_payload_identities": ids[1:],
                    "sample_payload_identities": ids,
                    "sample_occurrence_identities": [f"cycle={cycle}/slot={i}/payload={p}" for i, p in enumerate(ids)],
                    "rollout_operation_keys": [x["operation"] for x in trace_contract()],
                    "rollout_call_inventory": [x["inventory"] for x in trace_contract()]})
            censuses.append({"generator_updates": 1, "critic_updates": 5, "ema_updates": 5,
                "physical_rollout_batches": 10, "rollout_batch_size": 1, "logical_slots": slots,
                "logical_sample_census": samples} if world == 8 else samples[0])
        arm = old._arm(placement)
        rows.append({"schema_version": "distillschedule.wan13b_two_cycle_timing_rank.v1", "status": "PASS", "evidence_scope": FIXTURE,
            "rank": rank, "local_rank": device.local_rank, "hostname": d["site"]["node_hosts"][device.node],
            "world_size": world, "logical_world_size": 16, "hardware_profile": d["profile"], "logical_slots": slots,
            "microbatch_count": 16 // world, "run_id": run_id, "mode": "correctness", "placement": placement,
            "expected_head": bridge.NATIVE_HEAD, "profiled": False, "profiler_status": "NOT_RUN",
            "profiler_environment": {"exact_range": None, "lora_operator_attribution": "0"},
            "finite_status": [True, True], "cycle_contract": deepcopy(old._CYCLE_CONTRACT),
            "initial_versions": {"generator": 2, "critic": 10, "ema": 10},
            "cycle_versions": [{"generator": 3, "critic": 15, "ema": 15}, {"generator": 4, "critic": 20, "ema": 20}],
            "final_versions": {"generator": 4, "critic": 20, "ema": 20}, "updates": {"generator": 2, "critic": 10, "ema": 10},
            "matched_contract": deepcopy(old.SEMANTICS), "workload_fingerprint": fingerprint(old.SEMANTICS),
            "arm_contract": arm, "implementation_fingerprint": fingerprint(arm), "runtime_contract": old._runtime(placement),
            "configuration_fingerprint": fingerprint({"workload_fingerprint": fingerprint(old.SEMANTICS), "implementation_fingerprint": fingerprint(arm)}),
            "trace_rollout_contract_sha256": old.SEMANTICS["trace_rollout_contract_sha256"], "trace_rollout_contract": trace_contract(),
            "cycle_census": censuses, "cycles": cycles, "allocation_receipt": bridge._expected_receipt(d, rank),
            "formal_timing_eligible": False, "timing_seed": old.SEMANTICS["seed"], "rank_runtime_seed": old.SEMANTICS["seed"] + rank,
            "timing_arm_id": f"{d['profile']}-{placement}", "evidence_extraction": "AFTER_BOTH_TIMED_WINDOWS",
            "memory": {"max_allocated_bytes": 8 * 1024**3, "max_reserved_bytes": 10 * 1024**3, "driver_total_bytes": 40 * 1024**3}})
    impl = {**old._arm(placement), "hardware_profile": d["profile"], "microbatch_count": 16 // world,
            "trajectory_arm_id": f"{d['profile']}-{placement}"}
    summary = {"schema_version": "distillschedule.wan13b_trajectory_summary.v1", "status": "PASS", "errors": [],
        "evidence_scope": FIXTURE, "run_id": run_id, "mode": "correctness", "placement": placement, "expected_head": bridge.NATIVE_HEAD,
        "cycles": 2, "rank_count": world, "logical_world_size": 16, "hardware_profile": d["profile"],
        "final_versions": {"generator": 4, "critic": 20, "ema": 20}, "matched_contract": deepcopy(old.SEMANTICS),
        "workload_fingerprint": fingerprint(old.SEMANTICS), "implementation_contract": impl, "implementation_fingerprint": fingerprint(impl),
        "diagnostic_only": True, "formal_timing_eligible": False, "launcher_health": health,
        "logical_slots": [s for row in rows for s in row["logical_slots"]], "ranks": rows, "cycle_seconds_max_rank": list(seconds)}
    launch = {"schema_version": "distillschedule.a100_allocation_launch.v1", "evidence_scope": FIXTURE,
        "deployment_sha256": fingerprint(d), "profile": d["profile"], "source_head": bridge.NATIVE_HEAD,
        "source_tree": bridge.NATIVE_TREE, "source_clean": True, "native_files": bridge.NATIVE_FILES,
        "allocation": d["allocation"], "observed_environment": d["environment"], "asset_identities": d["assets"],
        "run_id": run_id, "placement": placement, "mode": "correctness", "started_unix_ns": 1,
        "completed_unix_ns": 100_000_000_000, "nodes": [{**c, "exit_code": 0, "owned_cleanup": True,
            "cgroup_before": {"oom": 0, "oom_kill": 0}, "cgroup_after": {"oom": 0, "oom_kill": 0}, "gpu_processes_after": []}
            for c in commands]}
    return {"placement": placement, "summary": write(root / "summary.json", summary),
        "rank_receipts": [write(root / f"execution/ranks/rank_{r['rank']:02d}.json", r) for r in rows],
        "launcher_health": write(root / "run_status.json", health),
        "allocation_manifest": write(root / "allocation-manifest.json", launch)}


def rewrite_native(obs, mutate):
    summary = json.loads(Path(obs["summary"]["path"]).read_text())
    mutate(summary)
    obs["summary"] = write(Path(obs["summary"]["path"]), summary)
    obs["rank_receipts"] = [write(Path(obs["summary"]["path"]).parent / f"execution/ranks/rank_{r['rank']:02d}.json", r)
                            for r in summary["ranks"]]


@unittest.skipUnless(NATIVE.is_dir(), "set A100_ALLOCATION_NATIVE_SOURCE to the exact pinned native snapshot")
class AllocationPlannerTests(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def build(self, world=8, seconds=((12., 10.), (14., 8.))):
        d = deployment(self.root, world)
        obs = [observation(d, p, f"CPU_FIXTURE_{i}", seconds[i]) for i, p in enumerate(bridge.PLACEMENTS)]
        problem, costs = bridge.assemble_problem_and_costs(d, obs, output_dir=self.root / "calibration")
        return d, obs, problem, costs

    def test_real_common_search_hierarchical_matches_exhaustive_both_allocations(self):
        for world in (8, 16):
            with self.subTest(world=world), tempfile.TemporaryDirectory(dir=SCRATCH) as root:
                self.root = Path(root)
                d, obs, problem, costs = self.build(world)
                hierarchical = search_plans(problem, costs, algorithm="hierarchical", max_proposals=10)
                exhaustive = search_plans(problem, costs, exhaustive=True, max_proposals=10)
                self.assertEqual(hierarchical["predicted_best"], exhaustive["predicted_best"])
                self.assertEqual(hierarchical["predicted_best"]["assignment"], {"placement": "resident"})
                plan = bridge.lower_selection(hierarchical, problem, costs, d, run_id="CPU_FIXTURE_selected")
                self.assertEqual(plan, bridge.validate_lowered_plan(json.loads(json.dumps(plan))))
                self.assertFalse(plan["formal_timing_eligible"])
                self.assertFalse(plan["claimable_speedup"])
                self.assertFalse(plan["execution_permitted"])

    def test_sparse_inventory_remains_uncalibrated_and_costs_choose_other_winner(self):
        d = deployment(self.root)
        obs = [observation(d, "world16-fsdp", "CPU_FIXTURE_baseline", (5., 4.))]
        p, c = bridge.assemble_problem_and_costs(d, obs, output_dir=self.root / "sparse")
        result = search_plans(p, c, exhaustive=True, max_proposals=10)
        self.assertEqual(result["predicted_best"]["assignment"], {"placement": "world16-fsdp"})
        self.assertTrue(result["calibration_requests"])
        obs.append(observation(d, "resident", "CPU_FIXTURE_resident", (14., 8.)))
        p, c = bridge.assemble_problem_and_costs(d, obs, output_dir=self.root / "complete")
        result = search_plans(p, c, algorithm="hierarchical", max_proposals=10)
        self.assertEqual(result["predicted_best"]["assignment"], {"placement": "world16-fsdp"})

    def test_raw_identity_slots_commits_census_and_timer_fail_closed(self):
        mutations = {
            "missing_rank": lambda s: s["ranks"].pop(),
            "source": lambda s: s["ranks"][0].update(expected_head="a" * 40),
            "slot": lambda s: s["ranks"][0]["allocation_receipt"].update(logical_slots=[0, 9]),
            "commit": lambda s: s["ranks"][0]["allocation_receipt"]["cycles"][0]["logical_samples"][0].update(generator_version_after=3),
            "census": lambda s: s["ranks"][0]["cycle_census"][0]["logical_sample_census"][0].update(critic_score_forwards=4),
            "clock": lambda s: s["ranks"][0]["cycles"][1].update(local_elapsed_seconds=2.),
            "seed": lambda s: s["ranks"][0].update(rank_runtime_seed=1),
            "profiled": lambda s: s["ranks"][0].update(profiled=True),
            "mode": lambda s: s.update(mode="timing", formal_timing_eligible=True),
            "duplicate_payload": lambda s: s["ranks"][1]["cycle_census"].__setitem__(0, deepcopy(s["ranks"][0]["cycle_census"][0])),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                d = deployment(self.root)
                obs = observation(d, "world16-fsdp", f"CPU_FIXTURE_{name}")
                rewrite_native(obs, mutate)
                with self.assertRaises(ValueError):
                    bridge.validate_native_observation(d, obs)

    def test_actual_command_env_and_lifecycle_are_validated(self):
        for key in ("environment", "argv", "owned_cleanup", "cgroup_after", "source_head", "observed_environment"):
            with self.subTest(key=key):
                d = deployment(self.root)
                obs = observation(d, "resident", f"CPU_FIXTURE_{key}")
                path = Path(obs["allocation_manifest"]["path"])
                launch = json.loads(path.read_text())
                if key == "environment": launch["nodes"][0][key]["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
                elif key == "argv": launch["nodes"][0][key].append("--profile-mode")
                elif key == "owned_cleanup": launch["nodes"][0][key] = False
                elif key == "cgroup_after": launch["nodes"][0][key]["oom_kill"] = 1
                elif key == "source_head": launch[key] = old.NATIVE_HEAD
                else: launch[key]["torch"] = "changed"
                obs["allocation_manifest"] = write(path, launch)
                with self.assertRaises(ValueError): bridge.validate_native_observation(d, obs)

    def test_missing_or_changed_raw_ref_and_stale_hashed_source_rejected(self):
        d, obs, p, c = self.build()
        path = Path(obs[0]["rank_receipts"][0]["path"])
        path.write_text("{}")
        with self.assertRaises(ValueError): bridge.validate_native_observation(d, obs[0])
        for field, value in (("profile", "h100-8-b16"), ("native_source", {"head": old.NATIVE_HEAD}),
                             ("semantics", {**old.SEMANTICS, "global_batch": 8})):
            edited = deepcopy(d)
            edited[field] = value
            with self.assertRaises(ValueError): bridge.validate_deployment(edited)

    def test_saved_prediction_cost_and_calibration_bytes_are_revalidated(self):
        d, obs, p, c = self.build()
        result = search_plans(p, c, exhaustive=True, max_proposals=10)
        result["predicted_best"]["predicted_cycle2_ns"] += 1
        with self.assertRaises(ValueError): bridge.lower_selection(result, p, c, d, run_id="selected")
        result = search_plans(p, c, exhaustive=True, max_proposals=10)
        value = c.to_dict()
        value["records"][0]["samples_ns"][0] += 1
        with self.assertRaises(ValueError): bridge.lower_selection(result, p, CostModel.from_dict(value), d, run_id="selected")
        Path(c.records[0].evidence["source_receipt"]).write_text("{}")
        with self.assertRaises(ValueError): bridge.lower_selection(result, p, c, d, run_id="selected")

    def test_fixture_cannot_be_promoted_even_after_metadata_rebinding(self):
        d = deployment(self.root)
        obs = observation(d, "resident", "CPU_FIXTURE_fixture")
        d["evidence_scope"] = "MEASURED"
        with self.assertRaises(ValueError): bridge.validate_native_observation(d, obs)

    def test_native_source_bytes_not_only_manifest_are_pinned(self):
        d = deployment(self.root)
        native = self.root / "native_copy"
        for path in bridge.NATIVE_FILES:
            target = native / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(NATIVE / path, target)
        d["native_source"]["repo_root"] = str(native)
        obs = observation(d, "resident", "CPU_FIXTURE_source_bytes")
        with (native / bridge.WORKER).open("a") as output:
            output.write("\n# changed after frozen source observation\n")
        with self.assertRaises(ValueError): bridge.validate_native_observation(d, obs)

    def test_edited_plan_rejects_even_after_outer_rehash_and_cross_allocation_reuse(self):
        d, obs, p, c = self.build()
        result = search_plans(p, c, algorithm="hierarchical", max_proposals=10)
        plan = bridge.lower_selection(result, p, c, d, run_id="CPU_FIXTURE_selected")
        edited = deepcopy(plan)
        edited["commands"][0]["environment"]["DISTILLSCHEDULE_HARDWARE_PROFILE"] = "h100-8-b16"
        edited["plan_sha256"] = fingerprint({k: v for k, v in edited.items() if k != "plan_sha256"})
        with self.assertRaises(ValueError): bridge.validate_lowered_plan(edited)
        d16 = deployment(self.root, 16)
        with self.assertRaises(ValueError): bridge.lower_selection(result, p, c, d16, run_id="CPU_FIXTURE_selected16")
        with self.assertRaises(ValueError): bridge.lower_selection(result, p, c, d, run_id="CPU_FIXTURE_0")

    def test_old_h100_admission_and_default_are_not_widened(self):
        self.assertEqual(old.NATIVE_HEAD, "7a668bcb12ad37d831dced3aca1be4ec7be0a2fe")
        with self.assertRaises(ValueError): old._allocation(deployment(self.root)["allocation"])
        helper = bridge._native_helper(deployment(self.root))
        profile = sys.modules[helper.__package__ + ".hardware_profile"].hardware_profile
        self.assertEqual(profile("h100-8-b16").logical_slots(3), (3, 11))
        self.assertEqual(profile("a100-16-b16").logical_slots(3), (3,))


if __name__ == "__main__":
    unittest.main()

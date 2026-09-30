"""No-GPU execution tests for LiveWan's explicit eight-rank evidence path."""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from distillschedule.evidence.trajectory_coarse import (
    AppendOnlyTrajectoryWriter, TrajectoryCycleReceipt,
)
from distillschedule.runtime.livewan_hardware import livewan_hardware


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts/distillschedule"
HEAD = "a" * 40
C34 = "c34-continuous-b16-residual-layout"


def module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def scoped_receipts(rank, *, cycles, hardware):
    """Synthetic test inputs, validated by the unchanged production gates."""
    from distillschedule.runtime import livewan_score_rmsnorm as score
    from distillschedule.runtime import livewan_generator_rms_epilogue as rms
    from distillschedule.runtime import livewan_rollout_cast_cache as casts
    from distillschedule.runtime import livewan_generator_modulation as modulation
    from distillschedule.runtime import livewan_generator_gate as gates
    from distillschedule.runtime import livewan_attention_checkpoint as sac

    count = cycles * hardware.scope_multiplicity()
    if hardware.is_generator(rank):
        calls = [1260] * (10 * count)
        rms_calls = [1920] * (10 * count)
        mod_calls = [1281] * (10 * count)
        return {
            "score_rmsnorm_kernel": {"execution_status": "NOT_APPLICABLE", "forward_calls": 0, "backward_calls": 0},
            "generator_rms_epilogue": dict(rms.CAST_SQUARE_PLAN_CONTRACT, execution_status="PASS", enabled=True,
                scope_entries=10*count, scope_exits=10*count, scope_failures=0, active=False,
                contract_failures=0, scope_forward_calls=rms_calls, calls=sum(rms_calls), square_calls=sum(rms_calls),
                fused_cast_square=True, native_mean_preserved=True, fp32_square_preserved=True),
            "generator_rollout_cast_cache": dict(casts.PLAN_CONTRACT, execution_status="PASS",
                scope_entries=10*count, scope_exits=10*count, active=False, scope_failures=0,
                misses=0, hits=20*count, temporary_input_casts=0, contract_failures=0,
                effective_dtype_changed=False, cache_cross_optimizer=False),
            "generator_modulation": dict(modulation.PLAN_CONTRACT, execution_status="PASS",
                scope_entries=10*count, scope_exits=10*count, active=False, scope_failures=0,
                contract_failures=0, scope_forward_calls=mod_calls, calls=sum(mod_calls)),
            "generator_gate": dict(gates.PLAN_CONTRACT, execution_status="PASS", output_layout="row_major",
                scope_entries=10*count, scope_exits=10*count, active=False, scope_failures=0,
                contract_failures=0, scope_forward_calls=calls, calls=sum(calls)),
            "critic_attention_checkpoint": {"execution_status": "NOT_APPLICABLE", "checkpoint_calls": 0},
        }
    expected_calls = cycles * hardware.critic_accumulations * 5 * 30
    return {
        "score_rmsnorm_kernel": dict(score.PLAN_CONTRACT, execution_status="PASS", install_status="PASS",
            contract_failures=0, forward_calls=1, backward_calls=1, shape_histogram={"2x1536": 1, "2x5120": 1}),
        "generator_rms_epilogue": {"execution_status": "NOT_APPLICABLE"},
        "generator_rollout_cast_cache": {"execution_status": "NOT_APPLICABLE"},
        "generator_modulation": {"execution_status": "NOT_APPLICABLE"},
        "generator_gate": {"execution_status": "NOT_APPLICABLE"},
        "critic_attention_checkpoint": dict(sac.PLAN_CONTRACT, execution_status="PASS", production_installed=True,
            source={"interface_sha256": sac.INTERFACE_SHA256}, self_attention_only=True, contract_failures=0,
            checkpoint_calls=expected_calls, original_fa_calls=expected_calls,
            replay_fa_cache_reads=expected_calls, expected_checkpoint_calls=expected_calls,
            allow_cache_entry_mutation=False, cpu_offload=False),
    }


def trajectory_fixture(root, *, name, mode="baseline"):
    hardware = livewan_hardware(name)
    variant = "b1" if mode == "baseline" else C34
    identity = {"trace_sha256": "1"*64, "bootstrap_sha256": "2"*64, "source_receipt_sha256": "3"*64}
    for node in range(hardware.physical_world_size // 8):
        path = root / "launcher" / f"node_{node}" / "launch_manifest.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        extra = (f"hardware_profile={name}\nphysical_world_size=8\nlogical_world_size=16\n" if hardware.is_h100 else "")
        path.write_text(f"run_id=test\nhead={HEAD}\nexecution_mode={mode}\npipeline_variant={variant}\n"
            "trajectory_cycles=1\ntrajectory_seed=17\ntrajectory_arm_id=test-arm\nclaim_boundary=DIAGNOSTIC_ONLY\n" + extra)
    from distillschedule.evidence.trajectory_state_sketch import SCHEMA_VERSION
    components = {key: {"schema_version":SCHEMA_VERSION,
        "coordinate_schema":"canonical_tensor_name/global_flat_index/value_fp32",
        "inventory_tensor_count":1,"inventory_sha256":"9"*64,"tensor_count":1,"samples_per_tensor":1,"max_tensors":1,
        "coordinate_count": 1, "coordinates": [{"tensor":key,"flat_index":0,"value_fp32": 1.0}]}
        for key in ("generator_trainable", "generator_adam", "critic_trainable", "critic_adam")}
    anchor = {"schema_version":"distillschedule.livewan_trajectory_state_anchor.v1", "cycle": 1, "components": components}
    anchor_sha = digest(anchor)
    write_json(root / "trajectory/anchors/cycle_01_global.json", dict(anchor, anchor_sha256=anchor_sha))
    for rank in range(hardware.physical_world_size):
        samples = tuple(f"cycle=0/step={step}/slot={slot}" for step in range(5) for slot in hardware.baseline_slots(rank))
        is_generator = hardware.is_generator(rank)
        writer = AppendOnlyTrajectoryWriter(root / "trajectory" / f"rank_{rank:03d}.jsonl", run_id="test", rank=rank)
        writer.append(TrajectoryCycleReceipt(
            run_id="test", workload="livewan", variant="test-arm", rank=rank,
            world_size=hardware.physical_world_size, seed=17, cycle=1, expected_total_cycles=1, logical_step=0,
            matched_contract={"global_batch": "16"}, implementation={"execution_mode": mode,
                "critic_physical_batch_size": "1" if mode == "baseline" else "2",
                "critic_physical_backward_calls": str(5*len(hardware.baseline_slots(rank))) if mode == "baseline" else ("0" if is_generator else str(5*hardware.critic_accumulations)),
                "critic_optimizer_commits": "5" if mode == "baseline" or not is_generator else "0"},
            sample_identities=samples, rng_identities={key: "4"*64 for key in samples},
            versions={"generator": 1, "critic": 5, "ema": 0}, cycle_updates={"generator": 1, "critic": 5, "ema": 0},
            cumulative_updates={"generator": 1, "critic": 5, "ema": 0}, call_census={"all/application": 80},
            losses={"generator_mean": 1.0, "critic_mean": 2.0}, gradient_norms={},
            tensor_contracts={"latent": {"shape": [1,16,3,8,8], "dtype": "torch.float32"}},
            state_fingerprints={**{key: "5"*64 for key in components}, "anchor": anchor_sha}, finite=True))
        final = dict(identity, status="PASS", run_kind="trajectory", final_versions={"generator": 1, "critic": 5, "ema": 0},
            physical_generator_commits=1 if mode == "baseline" or is_generator else 0,
            physical_critic_commits=5 if mode == "baseline" or not is_generator else 0,
            trajectory={"formal_timing_eligible": False, "arm_id": "test-arm"})
        if hardware.is_h100:
            final.update(hardware_profile=hardware.to_dict(), physical_world_size=8, logical_world_size=16)
        if mode == "pipeline":
            final.update(scoped_receipts(rank, cycles=1, hardware=hardware))
        write_json(root / "receipts" / f"rank_{rank:03d}.json", final)


def extend_trajectory_to_two_cycles(root, *, name, mode):
    hardware = livewan_hardware(name)
    anchor1 = json.loads((root/"trajectory/anchors/cycle_01_global.json").read_text())
    anchor2 = dict(anchor1, cycle=2); anchor2.pop("anchor_sha256")
    anchor_sha = digest(anchor2)
    write_json(root/"trajectory/anchors/cycle_02_global.json", dict(anchor2,anchor_sha256=anchor_sha))
    for node in range(hardware.node_count):
        path = root/"launcher"/f"node_{node}"/"launch_manifest.txt"
        path.write_text(path.read_text().replace("trajectory_cycles=1", "trajectory_cycles=2")+"run_kind=trajectory\n")
    for rank in range(hardware.physical_world_size):
        path = root/"trajectory"/f"rank_{rank:03d}.jsonl"
        first = json.loads(path.read_text()); first["expected_total_cycles"] = 2
        second = json.loads(json.dumps(first)); second["cycle"] = 2; second["logical_step"] = 5
        second["versions"] = second["cumulative_updates"] = {"generator":2,"critic":10,"ema":0}
        second["sample_identities"] = [key.replace("cycle=0/", "cycle=1/") for key in first["sample_identities"]]
        second["rng_identities"] = {key:"4"*64 for key in second["sample_identities"]}
        second["state_fingerprints"]["anchor"] = anchor_sha
        # Exercise the same typed receipt parser as production, not ad hoc rows.
        path.write_text("\n".join(json.dumps(TrajectoryCycleReceipt.from_mapping(value).to_mapping()) for value in (first,second))+"\n")
        final_path = root/"receipts"/f"rank_{rank:03d}.json"
        final = json.loads(final_path.read_text()); final["final_versions"] = {"generator":2,"critic":10,"ema":0}
        for key in ("physical_generator_commits", "physical_critic_commits"):
            final[key] *= 2
        if mode=="pipeline":
            final.update(scoped_receipts(rank,cycles=2,hardware=hardware))
        write_json(final_path,final)


def timing_fixture(root, *, name, mode):
    from distillschedule.runtime.livewan_crossscale_worker import FormalTrace, LIVEWAN_COMMIT, WAN21_COMMIT, TWO_CYCLE_TIMING_CONTINUOUS_NAME
    hardware = livewan_hardware(name)
    variant = "b1" if mode == "baseline" else C34
    prompt = root / "prompts.txt"
    prompt.write_text("test prompt\n")
    rows = []
    for step in range(10):
        for slot in range(16):
            rows.append(dict(logical_step=step, logical_slot=slot, dataset_index=(step%5)*16+slot,
                sample_path="/fixture/not_loaded.pt", sample_sha256=None,
                operation_key=f"cycle={step//5}/step={step%5}/global_step={step}/slot={slot}",
                cycle_index=step//5, cycle_step=step%5, world_frames=3, rope_gap=1,
                seeds={key:step*100000+slot*100+i for i,key in enumerate(("rollout", "generator_dmd_t", "generator_dmd_noise", "generator_grad_blocks", "critic_t", "critic_noise"))},
                generator_update=step%5==0, critic_update=True))
    body = dict(livewan_commit=LIVEWAN_COMMIT, wan21_commit=WAN21_COMMIT, world_size=16,
        global_batch=16, cycles=2, cycle_length=5, steps=10, source_reuse_across_cycles=True,
        prompts={"path":str(prompt), "sha256":hashlib.sha256(prompt.read_bytes()).hexdigest()}, rows=rows)
    trace_path = root / "trace.json"
    write_json(trace_path, dict(body, trace_sha256=digest(body)))
    trace = FormalTrace.load(trace_path, expected_cycles=2)
    summary_module = module("summarize_livewan_two_cycle_timing_world16")
    logical, physical = summary_module._expected_inventory(trace, mode=mode)
    plan = {"metadata": {"execution_mode": mode}, "evidence": {"steps": 10}}
    plan_sha = digest(plan)
    for node in range(hardware.node_count):
        path = root / "launcher" / f"node_{node}" / "launch_manifest.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        extra = (f"hardware_profile={name}\nphysical_world_size=8\nlogical_world_size=16\n" if hardware.is_h100 else "")
        path.write_text(f"run_id=test\nnode_rank={node}\nhead={HEAD}\nexecution_mode={mode}\npipeline_variant={variant}\n"
            "run_kind=two-cycle-timing\ntwo_cycle_count=2\nprofiled=false\nprofiler_status=NOT_RUN\n"
            "claim_boundary=STEADY_STATE_CYCLE2_SCORE_COMPLETION_ENDPOINT_DELTA\ntwo_cycle_seed=17\ntwo_cycle_arm_id=test-arm\n"
            "two_cycle_correctness_sha256="+"7"*64+"\n"+extra)
        (path.parent / "teardown_status.txt").write_text("status=PASS\ngpu_process_count=0\nowner_run_id=test\n")
        for suffix in ("before", "after"):
            (path.parent/f"cgroup_memory_events_{suffix}.txt").write_text("oom 0\noom_kill 0\n")
        write_json(root/f"effective_plan.node{node}.json", dict(plan, effective_plan_sha256=plan_sha))
    for rank in range(hardware.physical_world_size):
        g = hardware.is_generator(rank)
        if mode == "baseline":
            logical_local = {key:value//hardware.physical_world_size for key,value in logical.items()}
            physical_local = {key:value//hardware.physical_world_size for key,value in physical.items()}
        else:
            logical_local = {key:value//hardware.generator_pool_size for key,value in logical.items() if key.startswith("student/") == g}
            physical_local = {key:value//hardware.generator_pool_size for key,value in physical.items() if key.startswith("student/") == g}
        binding = {"sha256":"7"*64, "trajectory_comparison_sha256":"8"*64, "arm_id":"test-arm", "pipeline_variant":variant, "status":"PASS"}
        if hardware.is_h100:
            binding["hardware_profile"] = hardware.to_dict()
        contracts = []
        for cycle in (1,2):
            versions = {"generator":cycle if mode=="baseline" or g else 0,
                        "critic":5*cycle if mode=="baseline" or not g else 0, "ema":0}
            contracts.append(dict(cycle=cycle, local_versions=versions,
                logical_call_inventory_delta={key:value//2 for key,value in logical_local.items()},
                physical_call_inventory_delta={key:value//2 for key,value in physical_local.items()}))
        row = dict(status="PASS", rank=rank, run_kind="two-cycle-timing", mode=mode, profiled=False,
            trace_sha256=trace.sha256, bootstrap_sha256="2"*64, source_receipt_sha256="3"*64,
            logical_generator_updates=2, logical_critic_updates=10, final_versions={"generator":2,"critic":10,"ema":0},
            physical_generator_commits=2 if mode=="baseline" or g else 0,
            physical_critic_commits=10 if mode=="baseline" or not g else 0,
            effective_plan_sha256=plan_sha, logical_call_inventory=logical_local, physical_call_inventory=physical_local,
            cgroup_oom_before={"oom":0,"oom_kill":0}, cgroup_oom_after={"oom":0,"oom_kill":0},
            two_cycle_timing={"schema_version":"distillschedule.livewan_two_cycle_timing_rank.v1", "cycles_in_same_process":2,
                "profiled":False, "profiler_status":"NOT_RUN", "state_anchors":"NOT_COLLECTED_IN_TIMING_MODE",
                "evidence_extraction":"AFTER_BOTH_TIMED_CYCLE_ENDPOINTS", "correctness_binding":binding,
                "cycle_1":{"role":"WARMUP_DIAGNOSTIC", "claimable":False,"completion_offset_seconds":100.0+rank},
                "cycle_2":{"role":"STEADY_STATE_CLAIMABLE","claimable":True,"completion_offset_seconds":150.0+rank,"local_endpoint_delta_seconds":50.0},
                "per_cycle_contract":contracts})
        if hardware.is_h100:
            row.update(hardware_profile=hardware.to_dict(), physical_world_size=8, logical_world_size=16)
        if mode == "pipeline":
            row.update(scoped_receipts(rank, cycles=2, hardware=hardware),
                continuous_pipeline={"name":TWO_CYCLE_TIMING_CONTINUOUS_NAME,"cycles":2,"execution_status":"PASS",
                    "critic_physical_batch_size":2,"critic_physical_backward_calls_per_step":hardware.critic_accumulations,
                    "critic_step1_single_commit":True,"source_reuse_across_cycles":True},
                generator_runahead_transport={"execution_status":"PASS","role":"generator" if g else "score",
                    "cycles":2,"cross_cycle_lookahead":True},
                critic_shadow_hot_sync={"execution_status":"PASS","role":"generator_receiver" if g else "score_source",
                    "events":[{"version":5}]})
        write_json(root / "receipts" / f"rank_{rank:03d}.json", row)
    return trace_path


class LiveWanH100ScriptTests(unittest.TestCase):
    def trajectory_summary(self, root, *, profile, cycles=1):
        summarizer = module("summarize_livewan_trajectory_world16")
        output = root / "summary.json"
        argv = ["summarizer", "--run-root", str(root), "--cycles", str(cycles), "--expected-head", HEAD, "--output", str(output)]
        with patch.dict(os.environ, {"DISTILLSCHEDULE_HARDWARE_PROFILE": profile}), patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
            status = summarizer.main()
        return status, json.loads(output.read_text())

    def test_real_trajectory_summarizer_accepts_eight_actual_receipts_and_all_80_slots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trajectory_fixture(root, name="h100-8-b16")
            status, value = self.trajectory_summary(root, profile="h100-8-b16")
            self.assertEqual(status, 0, value.get("errors"))
            self.assertEqual(value["rank_cycle_receipts"], 8)
            self.assertEqual(value["expected_rank_cycle_receipts"], 8)
            self.assertEqual(value["hardware_profile"], livewan_hardware("h100-8-b16").to_dict())
            self.assertFalse(value["formal_timing_eligible"])

    def test_h100_trajectory_requires_new_hardware_receipts_and_all_physical_ranks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trajectory_fixture(root, name="h100-8-b16")
            path = root / "receipts/rank_007.json"
            row = json.loads(path.read_text()); row.pop("hardware_profile"); write_json(path, row)
            status, value = self.trajectory_summary(root, profile="h100-8-b16")
            self.assertEqual(status, 1)
            self.assertTrue(any("hardware" in error for error in value["errors"]))
            (root / "trajectory/rank_007.jsonl").unlink()
            self.assertEqual(self.trajectory_summary(root, profile="h100-8-b16")[0], 1)

    def test_a100_trajectory_default_stays_sixteen_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trajectory_fixture(root, name="a100-16-b16")
            status, value = self.trajectory_summary(root, profile="a100-16-b16")
            self.assertEqual(status, 0, value.get("errors"))
            self.assertEqual(value["rank_cycle_receipts"], 16)
            self.assertNotIn("hardware_profile", value)

    def test_h100_trajectory_score_ranks_require_ten_physical_b2_backwards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trajectory_fixture(root, name="h100-8-b16", mode="pipeline")
            status, value = self.trajectory_summary(root, profile="h100-8-b16")
            self.assertEqual(status, 0, value.get("errors"))
            path = root / "trajectory/rank_004.jsonl"
            row = json.loads(path.read_text()); row["implementation"]["critic_physical_backward_calls"] = "5"
            path.write_text(json.dumps(row)+"\n")
            status, value = self.trajectory_summary(root, profile="h100-8-b16")
            self.assertEqual(status, 1)
            self.assertTrue(any("physical Critic" in error for error in value["errors"]))

    def test_launcher_explicit_one_node_and_default_two_node_argument_gate(self):
        source = (SCRIPTS / "launch_livewan_crossscale_world16_node.sh").read_text()
        prefix = source.split('\nif [[ "${EXECUTION_MODE}" == baseline ]]; then\n')[0]
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, DMD_SYSTEM_ROOT=tmp, DISTILLSCHEDULE_PYENV="/unused", LIVEWAN_PYTHON_OVERLAY="/unused",
                DISTILLSCHEDULE_OUTPUT_DIR="/unused", DISTILLSCHEDULE_NODE_LOCK="/unused", LIVEWAN_WAN21_SOURCE="/unused",
                LIVEWAN_SOURCE_RECEIPT="/unused", LIVEWAN_PIPELINE_VARIANT=C34, LIVEWAN_PROFILE_TWO_CYCLES="0",
                LIVEWAN_TRAJECTORY_CYCLES="2", LIVEWAN_TRAJECTORY_SEED="17", LIVEWAN_TRAJECTORY_ARM_ID="test")
            def run(profile, node, variant=C34):
                return subprocess.run(["bash", "-c", prefix+'\necho "ARGUMENT_GATE_PASS/${PHYSICAL_NNODES:-2}/${PHYSICAL_WORLD_SIZE:-16}"\n',
                    "gate", str(node), "test", "trajectory", "pipeline", HEAD, "127.0.0.1", "32111"],
                    env=dict(env, DISTILLSCHEDULE_HARDWARE_PROFILE=profile, LIVEWAN_PIPELINE_VARIANT=variant), capture_output=True, text=True)
            self.assertEqual(run("a100-16-b16", 1).returncode, 0)
            h100 = run("h100-8-b16", 0)
            self.assertEqual(h100.returncode, 0, h100.stderr)
            self.assertIn("ARGUMENT_GATE_PASS/1/8", h100.stdout)
            self.assertNotEqual(run("h100-8-b16", 1).returncode, 0)
            self.assertNotEqual(run("h100-8-b16", 0, "c23-continuous-b32").returncode, 0)
            self.assertNotEqual(run("unregistered", 0).returncode, 0)

    def test_model_free_transport_uses_four_h100_lanes_and_four_independent_waves(self):
        probe = module("probe_livewan_crossscale_transport")
        h, g, s, waves = probe._transport_layout(C34, profile="h100-8-b16")
        self.assertEqual((g, s, waves), ((0,1,2,3), (4,5,6,7), (0,1,2,3)))
        self.assertEqual(h.pipeline_slots(1), (1,5,9,13))
        _, g, s, waves = probe._transport_layout(C34, profile="a100-16-b16")
        self.assertEqual((g, s, waves), (tuple(range(8)), tuple(range(8,16)), (0,1)))
        self.assertEqual(probe._transport_layout("c23-continuous-b32", profile="a100-16-b16")[3], (0,1,2,3))
        with self.assertRaises(ValueError):
            probe._transport_layout("c23-continuous-b32", profile="h100-8-b16")

    def test_validate_only_rejects_non_c34_before_assets_or_gpu(self):
        from distillschedule.core.config import load_plan
        from distillschedule.runtime import livewan_crossscale_worker as worker
        assets = {key:"/not-loaded/"+key for key in (
            "LIVEWAN_STUDENT_CHECKPOINT","LIVEWAN_CRITIC_CHECKPOINT","LIVEWAN_TEACHER_CHECKPOINT",
            "LIVEWAN_OPERATION_TRACE","LIVEWAN_SYSTEMS_BOOTSTRAP","DISTILLSCHEDULE_OUTPUT_DIR")}
        for variant in ("c24_continuous_b16_score_modulation",):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/"plan.json"
                with patch.dict(os.environ,dict(assets,DISTILLSCHEDULE_HARDWARE_PROFILE="h100-8-b16")):
                    plan = load_plan(ROOT/f"configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_{variant}.json").plan.to_dict()
                    write_json(path,plan)
                    with patch.object(worker.FormalTrace,"load",side_effect=AssertionError("must reject before reading trace")):
                        with self.assertRaisesRegex(RuntimeError,"H100"):
                            worker.run_worker(plan_path=path,mode="pipeline",run_kind="validate",repo_root=ROOT)

    def timing_summary(self, root, trace, *, profile, mode):
        summarizer = module("summarize_livewan_two_cycle_timing_world16")
        output = root / "summary.json"
        argv = ["summarizer", "--run-root", str(root), "--execution-mode", mode,
            "--pipeline-variant", "b1" if mode=="baseline" else C34, "--trace", str(trace),
            "--expected-head", HEAD, "--output", str(output)]
        with patch.dict(os.environ, {"DISTILLSCHEDULE_HARDWARE_PROFILE":profile}), patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
            status = summarizer.main()
        return status, json.loads(output.read_text())

    def test_two_cycle_summarizer_accepts_all_h100_baseline_and_pipeline_calls(self):
        for mode in ("baseline", "pipeline"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                trace = timing_fixture(root, name="h100-8-b16", mode=mode)
                status, value = self.timing_summary(root, trace, profile="h100-8-b16", mode=mode)
                self.assertEqual(status, 0, value.get("errors"))
                self.assertEqual(value["rank_count"], 8)
                self.assertEqual(value["cycle_2"]["seconds"], 50.0)
                self.assertEqual(value["cycle_2"]["critical_rank_local_endpoint_deltas"]["count"], 8 if mode=="baseline" else 4)
                self.assertEqual(value["logical_call_inventory"]["critic/grad/checkpoint"], 160)
                self.assertEqual(value["physical_call_inventory"]["critic/grad/checkpoint"], 160 if mode=="baseline" else 80)
                path = root / "receipts/rank_004.json"
                row = json.loads(path.read_text()); row["two_cycle_timing"]["correctness_binding"].pop("hardware_profile"); write_json(path,row)
                status, value = self.timing_summary(root, trace, profile="h100-8-b16", mode=mode)
                self.assertEqual(status, 1)
                self.assertTrue(any("hardware binding" in error for error in value["errors"]))

    def test_a100_two_cycle_summarizer_preserves_sixteen_rank_denominator(self):
        for mode in ("baseline", "pipeline"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                trace = timing_fixture(root, name="a100-16-b16", mode=mode)
                status, value = self.timing_summary(root, trace, profile="a100-16-b16", mode=mode)
                self.assertEqual(status, 0, value.get("errors"))
                self.assertEqual(value["rank_count"], 16)
                self.assertEqual(value["matched_workload_contract"]["world_size"], 16)
                self.assertNotIn("hardware_profile", value)

    def test_h100_comparison_and_gate_bind_fresh_eight_rank_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for mode in ("baseline", "pipeline"):
                arm = root/mode; arm.mkdir()
                trajectory_fixture(arm, name="h100-8-b16",mode=mode)
                extend_trajectory_to_two_cycles(arm,name="h100-8-b16",mode=mode)
                status,value = self.trajectory_summary(arm,profile="h100-8-b16",cycles=2)
                self.assertEqual(status,0,value.get("errors"))
            comparison = root/"comparison.json"
            gate = root/"gate.json"
            env = dict(os.environ,DISTILLSCHEDULE_HARDWARE_PROFILE="h100-8-b16", PYTHONDONTWRITEBYTECODE="1")
            command = [sys.executable,str(SCRIPTS/"compare_livewan_two_cycle_trajectories.py"),
                "--baseline-run-root",str(root/"baseline"),"--pipeline-run-root",str(root/"pipeline"),
                "--baseline-summary",str(root/"baseline/summary.json"),"--pipeline-summary",str(root/"pipeline/summary.json"),
                "--pipeline-variant",C34,"--expected-head",HEAD,"--output",str(comparison)]
            result = subprocess.run(command,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            value = json.loads(comparison.read_text())
            self.assertEqual(value["expected_world_size"],8)
            self.assertEqual(value["hardware_profile"],livewan_hardware("h100-8-b16").to_dict())
            command = [sys.executable,str(SCRIPTS/"build_livewan_two_cycle_timing_gate.py"),
                "--baseline-trajectory-summary",str(root/"baseline/summary.json"),"--pipeline-trajectory-summary",str(root/"pipeline/summary.json"),
                "--trajectory-comparison",str(comparison),"--pipeline-variant",C34,"--expected-head",HEAD,"--output",str(gate)]
            result = subprocess.run(command,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            value = json.loads(gate.read_text())
            self.assertEqual(value["hardware_profile"],livewan_hardware("h100-8-b16").to_dict())
            # A100-style summaries/old source evidence cannot authorize H100.
            path = root/"baseline/summary.json"
            value = json.loads(path.read_text()); value.pop("hardware_profile"); write_json(path,value)
            result = subprocess.run(command,env=env,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("hardware",result.stderr)


if __name__ == "__main__":
    unittest.main()

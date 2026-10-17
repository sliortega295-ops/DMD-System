"""SYNTHETIC_TEST_ONLY: real producer/consumer interfaces, never GPU evidence."""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest

from distillschedule.core.config import load_plan
from distillschedule.runtime import livewan_crossscale_worker as worker
from distillschedule.runtime import livewan_replay_ablation as ablation
from distillschedule.runtime import livewan_c51_ffn_retention as retention
from test_refactor_livewan_replay_ablation import _script, _node_timing_header

ROOT = Path(__file__).resolve().parents[1]
SCOPE = "SYNTHETIC_TEST_ONLY"
HEAD = "a" * 40
SEED = 17
IDENTITY = {"bootstrap_sha256": "2" * 64, "source_receipt_sha256": "3" * 64}


def _plan(run, variant, kind, trace="/SYNTHETIC_TEST_ONLY/trace.json", *, shape_override=None):
    with pytest.MonkeyPatch.context() as mp:
        for name in ("LIVEWAN_STUDENT_CHECKPOINT", "LIVEWAN_CRITIC_CHECKPOINT",
                     "LIVEWAN_TEACHER_CHECKPOINT", "LIVEWAN_SYSTEMS_BOOTSTRAP"):
            mp.setenv(name, "/SYNTHETIC_TEST_ONLY/" + name)
        mp.setenv("LIVEWAN_OPERATION_TRACE", str(trace))
        mp.setenv("DISTILLSCHEDULE_OUTPUT_DIR", str(run))
        filename = retention.PLAN_FILE if variant == retention.VARIANT else (ablation.PLAN_FILE if variant == ablation.VARIANT else ablation.PARENT_FILE)
        value = load_plan(ROOT / "configs/distillschedule/plans" / filename).plan.to_dict()
        if kind == "trajectory":
            worker._apply_trajectory_contract(value, cycles=2, mode="pipeline")
        elif kind == "two-cycle-timing":
            worker._apply_two_cycle_timing_contract(value, mode="pipeline")
        if shape_override is not None:
            field, replacement = shape_override
            value["workload"]["shape"][field] = replacement
        worker._validate_plan_identity(value, mode="pipeline")
        return value


def _native_writer_statement(plan, run, *, kind, node, local_rank=0,
                             profile_two_cycles=False):
    """Execute the unchanged actual run_worker's producer site, not a replica.

    Only the metadata statement immediately after its actual hash assignment is
    executed; GPU/model setup is intentionally not emulated by this CPU test.
    """
    function = ast.parse(inspect.getsource(worker.run_worker)).body[0]
    index = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "effective_plan_sha256"
                         for t in n.targets))
    statements = function.body[index:index + 2]
    module = ast.Module(body=statements, type_ignores=[])
    ast.fix_missing_locations(module)
    env = dict(vars(worker), plan=plan, output_dir=run, run_kind=kind,
               two_cycle_timing=kind == "two-cycle-timing",
               profile_two_cycles=profile_two_cycles, local_rank=local_rank)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DISTILLSCHEDULE_NODE_RANK", str(node))
        exec(compile(module, inspect.getsourcefile(worker), "exec"), env)
    return env["effective_plan_sha256"]


@pytest.mark.parametrize("variant", [ablation.PARENT_VARIANT, ablation.VARIANT])
@pytest.mark.parametrize("kind", ["trajectory", "two-cycle-timing"])
def test_native_worker_produces_lowered_plan_before_consumers(tmp_path, variant, kind):
    run = tmp_path / SCOPE
    plan = _plan(run, variant, kind)
    before = copy.deepcopy(plan)
    for node in (0, 1):
        digest = _native_writer_statement(plan, run, kind=kind, node=node)
        path = run / f"effective_plan.node{node}.json"
        assert path.is_file(), "native worker omitted required trajectory effective plan"
        saved = json.loads(path.read_text())
        assert saved.pop("effective_plan_sha256") == digest
        assert saved == plan and digest == worker._effective_plan_sha256(plan)
        if variant == ablation.VARIANT:
            ablation.validate_effective_policy(saved)
    assert plan == before
    assert not list(run.glob(".*.tmp"))


def test_native_writer_uses_unique_local_rank_zero_and_atomic_publication(tmp_path, monkeypatch):
    plan = _plan(tmp_path, ablation.VARIANT, "trajectory")
    replacements = []
    replace = worker.os.replace
    def observed_replace(src, dst):
        replacements.append((Path(src), Path(dst)))
        return replace(src, dst)
    monkeypatch.setattr(worker.os, "replace", observed_replace)
    for node in (0, 1):
        for local in range(8):
            _native_writer_statement(plan, tmp_path, kind="trajectory", node=node, local_rank=local)
    assert [dst.name for _, dst in replacements] == ["effective_plan.node0.json", "effective_plan.node1.json"]
    assert all(src != dst and not src.exists() and dst.is_file() for src, dst in replacements)


@pytest.mark.parametrize("kind", ["validate", "capacity", "correctness", "timing", "profile"])
def test_unrelated_canonical_modes_do_not_add_metadata_writes(tmp_path, kind):
    plan = _plan(tmp_path, ablation.PARENT_VARIANT, "trajectory")
    _native_writer_statement(plan, tmp_path, kind=kind, node=0)
    assert not list(tmp_path.glob("effective_plan.node*.json"))


def test_profile_two_cycle_existing_producer_is_preserved(tmp_path):
    plan = _plan(tmp_path, ablation.PARENT_VARIANT, "two-cycle-timing")
    _native_writer_statement(plan, tmp_path, kind="profile", node=0, profile_two_cycles=True)
    assert (tmp_path / "effective_plan.node0.json").is_file()


def test_writer_call_site_is_outside_models_and_cycle_timer():
    function = ast.parse(inspect.getsource(worker.run_worker)).body[0]
    calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)]
    writer = [n for n in calls if n.func.id == "_publish_effective_plan"]
    assert len(writer) == 1
    assert any(isinstance(n, ast.Expr) and n.value is writer[0] for n in function.body)
    assert writer[0].lineno < min(n.lineno for n in calls if n.func.id == "build_models")
    timer = next(n for n in ast.walk(function) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "timing_start_ns" for t in n.targets))
    assert writer[0].lineno < timer.lineno


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _invoke(name, *arguments):
    return subprocess.run([sys.executable, str(ROOT / "scripts/distillschedule" / name),
                           *map(str, arguments)], capture_output=True, text=True, timeout=60)


def _trace(root):
    from test_refactor_livewan_trajectory import _source_trace
    source = root / "synthetic_source_trace.json"
    path = root / "synthetic_trace_two_cycles.json"
    prompts = root / "SYNTHETIC_TEST_ONLY_prompts"
    prompts.write_bytes(b"SYNTHETIC_TEST_ONLY_NOT_MODEL_CONDITIONS")
    _source_trace(source)
    value = json.loads(source.read_text())
    value["seed"] = SEED
    value["prompts"] = {"path": str(prompts), "sha256": _sha(prompts)}
    for row in value["rows"]:
        row["world_frames"] = 6  # Five new blocks; same 81-frame latent contract.
        row["sample_path"] = "/SYNTHETIC_TEST_ONLY/not_training_data.pt"
    source.write_text(json.dumps(value))
    result = _invoke("build_livewan_continuous_trace.py", "--source-trace", source,
                     "--output", path, "--cycles", 2)
    assert result.returncode == 0, result.stderr
    return worker.FormalTrace.load(path, expected_cycles=2, expected_global_batch=16)


def _anchors(run):
    from test_refactor_livewan_two_cycle_timing import _trajectory
    from distillschedule.evidence.trajectory_state_sketch import SCHEMA_VERSION
    seed = run / "SYNTHETIC_TEST_ONLY_fixture_seed.json"
    _trajectory(seed, mode="pipeline", variant=ablation.PARENT_VARIANT, run_id=SCOPE)
    anchors = {}
    for row in json.loads(seed.read_text())["numeric_state_anchors"]:
        path = Path(row["path"])
        body = json.loads(path.read_text())
        body.pop("anchor_sha256")
        body["evidence_scope"] = SCOPE
        body["components"] = {name: dict(
            schema_version=SCHEMA_VERSION,
            coordinate_schema="canonical_tensor_name/global_flat_index/value_fp32",
            inventory_tensor_count=1, inventory_sha256="9" * 64, tensor_count=1,
            coordinate_count=1, samples_per_tensor=1, max_tensors=1,
            coordinates=[dict(tensor="SYNTHETIC_TEST_ONLY_lora", flat_index=0,
                              value_fp32=float(row["cycle"]))])
            for name in ("generator_trainable", "generator_adam", "critic_trainable", "critic_adam")}
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        path.write_text(json.dumps(dict(body, anchor_sha256=digest)))
        anchors[row["cycle"]] = digest
    seed.unlink()  # Only real native summarizer outputs enter the chain.
    return anchors


def _policy_row(rank, variant):
    value = json.loads((ROOT / "tests/fixtures/livewan_c51_policy_receipts_synthetic.json").read_text())
    assert value["evidence_scope"] == SCOPE
    row = copy.deepcopy(value["generator" if rank < 8 else "score"])
    if rank >= 8 and variant == ablation.VARIANT:
        row["critic_attention_checkpoint"].update(
            name=ablation.RECOMPUTE_CONTRACT["name"], replay_fa_cache_reads=0,
            replay_fa_recompute_calls=300)
    if variant == retention.VARIANT:
        from distillschedule.runtime.livewan_ffn_down_checkpoint import CONTRACT
        row['critic_ffn_down_checkpoint'] = (dict(enabled=True,execution_status='NOT_APPLICABLE',checkpoint_calls=0) if rank<8 else
            dict(name=CONTRACT['name'],contract=dict(CONTRACT),enabled=True,production_installed=True,
                 execution_status='PASS',checkpoint_calls=300,expected_checkpoint_calls=300,
                 original_ffn_down_calls=300,replay_ffn_down_cache_reads=300,contract_failures=0,
                 all_checkpoint_ledgers_one_to_one=True,cumulative_cached_output_bytes=300*38640*1536*2,
                 cache_location='cuda_per_checkpoint',cpu_offload=False,allow_cache_entry_mutation=False))
    return dict(row, evidence_scope=SCOPE, rank=rank, status="PASS", mode="pipeline",
                logical_generator_updates=2, logical_critic_updates=10,
                physical_generator_commits=2 if rank < 8 else 0,
                physical_critic_commits=0 if rank < 8 else 10,
                initial_versions=dict(generator=0, critic=0, ema=0),
                final_versions=dict(generator=2, critic=10, ema=0),
                profiled=False, cgroup_oom_before=dict(oom=0, oom_kill=0),
                cgroup_oom_after=dict(oom=0, oom_kill=0),
                peak_memory=dict(max_allocated_bytes=0, max_reserved_bytes=0),
                host_memory=dict(rss_bytes=0), construction_wall_seconds=0)


def _manifest(run, node, variant, kind, arm, *, gate_sha=None):
    root = run / "launcher" / f"node_{node}"
    root.mkdir(parents=True)
    fields = dict(evidence_scope=SCOPE, head=HEAD, node_rank=node, run_id=run.name,
                  execution_mode="pipeline", pipeline_variant=variant, run_kind=kind)
    if kind == "trajectory":
        fields.update(trajectory_cycles=2, trajectory_seed=SEED, trajectory_arm_id=arm,
                      claim_boundary="DIAGNOSTIC_ONLY")
    else:
        fields.update(two_cycle_count=2, two_cycle_seed=SEED, two_cycle_arm_id=arm,
                      two_cycle_correctness_sha256=gate_sha, profiled="false",
                      profiler_status="NOT_RUN",
                      claim_boundary="STEADY_STATE_CYCLE2_SCORE_COMPLETION_ENDPOINT_DELTA")
        (root / "teardown_status.txt").write_text(
            f"status=PASS\ngpu_process_count=0\nowner_run_id={run.name}\n")
    (root / "launch_manifest.txt").write_text("".join(f"{k}={v}\n" for k, v in fields.items()))
    for suffix in ("before", "after"):
        (root / f"cgroup_memory_events_{suffix}.txt").write_text("oom 0\noom_kill 0\n")
    # Synthetic values exist solely to exercise the small receipt collector.
    (root / "resource_monitor.csv").write_text(
        "unix_time,cgroup_bytes,host_available_bytes,gpu_used_mib,gpu_free_mib,gpu_util_percent\n"
        "0,0,0,0:0:0:0:0:0:0:0,0:0:0:0:0:0:0:0,0:0:0:0:0:0:0:0\n")


def _qualification(root, variant, label, trace):
    from distillschedule.evidence.trajectory_coarse import AppendOnlyTrajectoryWriter, TrajectoryCycleReceipt
    run = root / (SCOPE + "_" + label)
    run.mkdir()
    arm = SCOPE + "_" + label + "_arm"
    plan = _plan(run, variant, "trajectory", trace.path)
    anchors = _anchors(run)
    for node in (0, 1):
        _native_writer_statement(plan, run, kind="trajectory", node=node)
        _manifest(run, node, variant, "trajectory", arm)
    digest = worker._effective_plan_sha256(plan)
    for rank in range(16):
        raw = _policy_row(rank, variant)
        raw.update(run_kind="trajectory", trace_sha256=trace.sha256, **IDENTITY,
                   effective_plan_sha256=digest, trajectory=dict(formal_timing_eligible=False, arm_id=arm))
        path = run / "receipts" / f"rank_{rank:03d}.json"
        path.parent.mkdir(exist_ok=True); path.write_text(json.dumps(raw))
        writer = AppendOnlyTrajectoryWriter(run / "trajectory" / f"rank_{rank:03d}.jsonl", run_id=run.name, rank=rank)
        for cycle in (1, 2):
            samples = tuple(f"cycle={cycle}/slot={rank}/step={step}" for step in range(5))
            writer.append(TrajectoryCycleReceipt(
                run_id=run.name, workload="livewan", variant=arm, rank=rank, world_size=16,
                seed=SEED, cycle=cycle, expected_total_cycles=2, logical_step=(cycle - 1) * 5,
                matched_contract=dict(evidence_scope=SCOPE, width="640", height="368", frames="81", global_batch="16"),
                implementation=dict(execution_mode="pipeline", critic_physical_batch_size="2",
                                    critic_physical_backward_calls="0" if rank < 8 else "5",
                                    critic_optimizer_commits="0" if rank < 8 else "5"),
                sample_identities=samples, rng_identities={k:"b" * 64 for k in samples},
                versions=dict(generator=cycle, critic=cycle * 5, ema=0),
                cycle_updates=dict(generator=1, critic=5, ema=0),
                cumulative_updates=dict(generator=cycle, critic=cycle * 5, ema=0),
                call_census={"application/student": 2080},
                losses=dict(generator_mean=1. / cycle, critic_mean=2. / cycle),
                gradient_norms=dict(generator_mean=3., critic_mean=4.),
                tensor_contracts=dict(latent=dict(shape=list(worker.LATENT_SHAPE), dtype="torch.bfloat16")),
                state_fingerprints={**{k:"c" * 64 for k in
                    ("generator_trainable", "generator_adam", "critic_trainable", "critic_adam")},
                    "anchor":anchors[cycle]}, finite=True))
    summary = run / "summary.json"
    result = _invoke("summarize_livewan_trajectory_world16.py", "--run-root", run,
                     "--cycles", 2, "--expected-head", HEAD, "--output", summary)
    assert result.returncode == 0, result.stdout + result.stderr
    return dict(run=run, summary=summary, variant=variant, arm=arm, plan=plan)


def _load_gate(gate, arm, trace, **overrides):
    args = dict(path=gate, expected_sha256=_sha(gate), expected_head=HEAD, mode="pipeline",
                pipeline_variant=arm["variant"], arm_id=arm["arm"], seed=SEED, trace=trace,
                bootstrap=type("Bootstrap", (), {"sha256":IDENTITY["bootstrap_sha256"]})(),
                source_receipt={"source_receipt_sha256":IDENTITY["source_receipt_sha256"]})
    args.update(overrides)
    return worker._load_two_cycle_timing_gate(**args)


@pytest.fixture(scope="module", params=[ablation.VARIANT, retention.VARIANT])
def chain(tmp_path_factory, request):
    root = tmp_path_factory.mktemp(SCOPE + "_producer_consumer_chain")
    trace = _trace(root)
    runs = root / "runs"
    runs.mkdir()
    control = _qualification(runs, ablation.PARENT_VARIANT, "control", trace)
    child = _qualification(runs, request.param, "child", trace)
    comparison, gate = root / "synthetic_comparison.json", root / "synthetic_gate.json"
    result = _invoke("compare_livewan_two_cycle_trajectories.py",
                     "--baseline-run-root", control["run"], "--pipeline-run-root", child["run"],
                     "--baseline-summary", control["summary"], "--pipeline-summary", child["summary"],
                     "--expected-head", HEAD, "--reference-variant", control["variant"],
                     "--pipeline-variant", child["variant"], "--output", comparison)
    assert result.returncode == 0, result.stdout + result.stderr
    result = _invoke("build_livewan_two_cycle_timing_gate.py",
                     "--baseline-trajectory-summary", control["summary"],
                     "--pipeline-trajectory-summary", child["summary"],
                     "--trajectory-comparison", comparison, "--expected-head", HEAD,
                     "--reference-variant", control["variant"], "--pipeline-variant", child["variant"],
                     "--output", gate)
    assert result.returncode == 0, result.stdout + result.stderr
    result = dict(root=root, trace=trace, control=control, child=child, comparison=comparison, gate=gate)
    for role in ("control", "child"):
        result[role + "_timing"] = _timing_run(runs, result[role], trace, gate, role)
    (root / "SYNTHETIC_TEST_ONLY_artifact_map.json").write_text(json.dumps(
        dict(evidence_scope=SCOPE, head=HEAD, trace=str(trace.path),
             control=str(control["run"]), child=str(child["run"]), gate=str(gate),
             control_timing=str(result["control_timing"]), child_timing=str(result["child_timing"]))))
    return result


def test_real_producer_through_summary_comparator_header_worker(chain, record_property):
    record_property("evidence_scope", SCOPE)
    value = json.loads(chain["gate"].read_text())
    assert value["schema_version"] == "distillschedule.livewan_conditional_replay_timing_gate.v1"
    assert value["expected_head"] == HEAD and value["seed"] == SEED
    assert value["final_versions"] == dict(generator=2, critic=10, ema=0)
    assert value["identity"]["trace_sha256"] == chain["trace"].sha256
    for role in ("control", "child"):
        arm = chain[role]
        summary = json.loads(arm["summary"].read_text())
        assert summary["status"] == "PASS" and summary["rank_cycle_receipts"] == 32
        result = _node_timing_header(chain["gate"], variant=arm["variant"], mode="pipeline", head=HEAD)
        assert result.returncode == 0, result.stderr
        assert _load_gate(chain["gate"], arm, chain["trace"])["status"] == "PASS"
        for node in (0, 1):
            path = arm["run"] / f"effective_plan.node{node}.json"
            saved = json.loads(path.read_text())
            assert saved.pop("effective_plan_sha256") == worker._effective_plan_sha256(arm["plan"])
            assert saved == arm["plan"]


def _timing_run(root, arm, trace, gate, role):
    run = root / (SCOPE + "_" + role + "_timing")
    run.mkdir()
    plan = _plan(run, arm["variant"], "two-cycle-timing", trace.path)
    binding = _load_gate(gate, arm, trace)
    for node in (0, 1):
        _native_writer_statement(plan, run, kind="two-cycle-timing", node=node)
        _manifest(run, node, arm["variant"], "two-cycle-timing", arm["arm"], gate_sha=_sha(gate))
    for rank in range(16):
        row = _policy_row(rank, arm["variant"])
        row.update(run_kind="two-cycle-timing", trace_sha256=trace.sha256, **IDENTITY,
                   effective_plan_sha256=worker._effective_plan_sha256(plan))
        contracts = []
        for cycle in (1, 2):
            contracts.append(dict(cycle=cycle,
                local_versions=dict(generator=cycle if rank < 8 else 0,
                                    critic=0 if rank < 8 else cycle * 5, ema=0),
                logical_call_inventory_delta={k:v // 2 for k, v in row["logical_call_inventory"].items()},
                physical_call_inventory_delta={k:v // 2 for k, v in row["physical_call_inventory"].items()}))
        row["two_cycle_timing"] = dict(
            evidence_scope=SCOPE, schema_version="distillschedule.livewan_two_cycle_timing_rank.v1",
            cycles_in_same_process=2, profiled=False, profiler_status="NOT_RUN",
            state_anchors="NOT_COLLECTED_IN_TIMING_MODE", evidence_extraction="AFTER_BOTH_TIMED_CYCLE_ENDPOINTS",
            cycle_1=dict(role="WARMUP_DIAGNOSTIC", claimable=False, completion_offset_seconds=100. + rank * .001),
            cycle_2=dict(role="STEADY_STATE_CLAIMABLE", claimable=True,
                         completion_offset_seconds=160. + rank * .001, local_endpoint_delta_seconds=60.),
            correctness_binding=binding, per_cycle_contract=contracts)
        row["continuous_pipeline"] = dict(name=worker.TWO_CYCLE_TIMING_CONTINUOUS_NAME, cycles=2,
            execution_status="PASS", critic_physical_batch_size=2, critic_physical_backward_calls_per_step=1,
            critic_step1_single_commit=True, source_reuse_across_cycles=True)
        row["generator_runahead_transport"] = dict(execution_status="PASS",
            role="generator" if rank < 8 else "score", cycles=2, cross_cycle_lookahead=True)
        row["critic_shadow_hot_sync"] = dict(execution_status="PASS",
            role="generator_receiver" if rank < 8 else "score_source", events=[dict(version=5)])
        path = run / "receipts" / f"rank_{rank:03d}.json"
        path.parent.mkdir(exist_ok=True); path.write_text(json.dumps(row))
    result = _invoke("summarize_livewan_two_cycle_timing_world16.py", "--run-root", run,
        "--execution-mode", "pipeline", "--pipeline-variant", arm["variant"], "--trace", trace.path,
        "--expected-head", HEAD, "--output", run / "summary.json")
    assert result.returncode == 0, result.stdout + result.stderr
    return run


@pytest.mark.parametrize("role", ["control", "child"])
def test_native_timing_summary_and_collector_file_contract(chain, role, record_property):
    record_property("evidence_scope", SCOPE)
    record_property("synthetic_artifact_root", str(chain["root"]))
    run = chain[role + "_timing"]
    summary = json.loads((run / "summary.json").read_text())
    assert summary["status"] == "PASS" and summary["rank_count"] == 16
    assert summary["cycle_2"]["seconds"] == pytest.approx(60.)  # Invented test endpoints, not timing.
    assert summary["correctness_binding"]["gate_sha256"] == _sha(chain["gate"])
    for node in (0, 1):
        for name in ("launch_manifest.txt", "teardown_status.txt", "resource_monitor.csv",
                     "cgroup_memory_events_before.txt", "cgroup_memory_events_after.txt"):
            assert (run / "launcher" / f"node_{node}" / name).is_file()
        assert (run / f"effective_plan.node{node}.json").is_file()
    for rank in range(16):
        row = json.loads((run / "receipts" / f"rank_{rank:03d}.json").read_text())
        assert row["evidence_scope"] == SCOPE
        assert {"physical_call_inventory", "logical_call_inventory", "peak_memory", "host_memory",
                "construction_wall_seconds", "critic_attention_checkpoint"} <= set(row)
        if rank >= 8:
            checkpoint = row["critic_attention_checkpoint"]
            assert checkpoint["original_fa_calls"] == 300 and checkpoint["contract_failures"] == 0
            assert checkpoint["replay_fa_cache_reads"] == (0 if chain[role]["variant"] == ablation.VARIANT else 300)
            if chain[role]["variant"] == ablation.VARIANT:
                assert checkpoint["replay_fa_recompute_calls"] == 300
            elif chain[role]["variant"] == retention.VARIANT:
                assert row["critic_ffn_down_checkpoint"]["replay_ffn_down_cache_reads"] == 300


@pytest.mark.parametrize("mutation", [
    "missing_file", "hash_drift", "node_disagreement", "wrong_head", "wrong_seed",
    "missing_rank", "missing_cycle", "bad_census", "wrong_versions", "wrong_arm",
])
def test_real_trajectory_consumer_rejects_broken_producer_artifacts(chain, tmp_path, mutation):
    import shutil
    run = tmp_path / SCOPE
    shutil.copytree(chain["child"]["run"], run)
    path = run / "effective_plan.node1.json"
    if mutation == "missing_file":
        path.unlink()
    elif mutation in {"hash_drift", "node_disagreement"}:
        value = json.loads(path.read_text())
        value["evidence"]["output_dir"] += "_drift"
        if mutation == "node_disagreement":
            value.pop("effective_plan_sha256")
            value["effective_plan_sha256"] = worker._effective_plan_sha256(value)
        path.write_text(json.dumps(value))
    elif mutation in {"wrong_head", "wrong_seed", "wrong_arm"}:
        path = run / "launcher/node_1/launch_manifest.txt"
        text = path.read_text()
        old, new = (HEAD, "b" * 40) if mutation == "wrong_head" else (
            ("trajectory_seed=17", "trajectory_seed=18") if mutation == "wrong_seed"
            else (chain["child"]["arm"], "SYNTHETIC_TEST_ONLY_wrong_arm"))
        path.write_text(text.replace(old, new))
    elif mutation == "missing_rank":
        (run / "receipts/rank_015.json").unlink()
    elif mutation == "missing_cycle":
        path = run / "trajectory/rank_015.jsonl"
        path.write_text(path.read_text().splitlines()[0] + "\n")
    else:
        path = run / "receipts/rank_008.json"
        value = json.loads(path.read_text())
        if mutation == "bad_census":
            if chain['child']['variant'] == retention.VARIANT:
                value['critic_ffn_down_checkpoint']['replay_ffn_down_cache_reads'] = 299
            else:
                value["critic_attention_checkpoint"]["replay_fa_recompute_calls"] = 299
        else:
            value["final_versions"]["critic"] = 9
        path.write_text(json.dumps(value))
    result = _invoke("summarize_livewan_trajectory_world16.py", "--run-root", run,
                     "--cycles", 2, "--expected-head", HEAD, "--output", run / "rejected_summary.json")
    assert result.returncode != 0, mutation


@pytest.mark.parametrize("changed", ["expected_head", "arm_id", "seed", "pipeline_variant", "expected_sha256"])
def test_real_worker_gate_loader_rejects_identity_drift(chain, changed):
    wrong = {"expected_head":"b" * 40, "arm_id":"wrong", "seed":18,
             "pipeline_variant":ablation.PARENT_VARIANT, "expected_sha256":"0" * 64}[changed]
    with pytest.raises(RuntimeError):
        _load_gate(chain["gate"], chain["child"], chain["trace"], **{changed:wrong})


def test_child_cannot_use_legacy_gate_at_either_real_entry(chain, tmp_path):
    from test_refactor_livewan_replay_ablation import _header_fixture
    path = tmp_path / "legacy_gate.json"
    path.write_text(json.dumps(_header_fixture(False)))
    result = _node_timing_header(path, variant=ablation.VARIANT, mode="pipeline", head=HEAD)
    assert result.returncode != 0
    with pytest.raises(RuntimeError):
        _load_gate(path, chain["child"], chain["trace"])


@pytest.mark.parametrize("field,value", [("width", 704), ("height", 480), ("num_frames", 69), ("batch_size", 8)])
def test_wrong_shape_is_rejected_before_metadata_producer(tmp_path, field, value):
    with pytest.raises(RuntimeError):
        _plan(tmp_path, ablation.VARIANT, "trajectory", shape_override=(field, value))


@pytest.mark.parametrize("node", [-1, 2])
def test_writer_rejects_missing_or_invalid_node_rank(tmp_path, node):
    plan = _plan(tmp_path, ablation.VARIANT, "trajectory")
    with pytest.raises(RuntimeError, match="node rank"):
        _native_writer_statement(plan, tmp_path, kind="trajectory", node=node)
    assert not list(tmp_path.glob("effective_plan.node*.json"))


@pytest.mark.parametrize("mutation", [
    "missing_plan", "hash_drift", "node_disagreement", "missing_rank", "bad_census",
    "missing_cycle", "wrong_head", "bad_endpoint", "failed_teardown",
])
def test_real_timing_consumer_rejects_incomplete_or_wrong_evidence(chain, tmp_path, mutation):
    import shutil
    run = tmp_path / SCOPE
    shutil.copytree(chain["child_timing"], run)
    if mutation == "missing_plan":
        (run / "effective_plan.node1.json").unlink()
    elif mutation in {"hash_drift", "node_disagreement"}:
        path = run / "effective_plan.node1.json"; value = json.loads(path.read_text())
        value["evidence"]["output_dir"] += "_drift"
        if mutation == "node_disagreement":
            value.pop("effective_plan_sha256")
            value["effective_plan_sha256"] = worker._effective_plan_sha256(value)
        path.write_text(json.dumps(value))
    elif mutation == "missing_rank":
        (run / "receipts/rank_015.json").unlink()
    elif mutation == "wrong_head":
        path = run / "launcher/node_1/launch_manifest.txt"
        path.write_text(path.read_text().replace(HEAD, "b" * 40))
    elif mutation == "failed_teardown":
        path = run / "launcher/node_1/teardown_status.txt"
        path.write_text(path.read_text().replace("status=PASS", "status=FAIL"))
    else:
        path = run / "receipts/rank_008.json"; value = json.loads(path.read_text())
        if mutation == "bad_census":
            if chain['child']['variant'] == retention.VARIANT:
                value['critic_ffn_down_checkpoint']['replay_ffn_down_cache_reads'] = 299
            else:
                value["critic_attention_checkpoint"]["replay_fa_recompute_calls"] = 299
        elif mutation == "missing_cycle":
            value["two_cycle_timing"]["per_cycle_contract"].pop()
        else:
            value["two_cycle_timing"]["cycle_2"]["local_endpoint_delta_seconds"] = 0.
        path.write_text(json.dumps(value))
    result = _invoke("summarize_livewan_two_cycle_timing_world16.py", "--run-root", run,
        "--execution-mode", "pipeline", "--pipeline-variant", ablation.VARIANT, "--trace", chain["trace"].path,
        "--expected-head", HEAD, "--output", run / "rejected_summary.json")
    assert result.returncode != 0, mutation

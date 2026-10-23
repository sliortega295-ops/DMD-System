"""CPU synthetic artifacts through real native producer/consumers; never GPU evidence."""
import ast
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from distillschedule.runtime import livewan_crossscale_worker as worker
from distillschedule.runtime.livewan_c51_bundle import VARIANT, receipt_contract
from distillschedule.search.livewan_implementation import LIVEWAN, implementations
from distillschedule.search.execution import resolve_candidate
from distillschedule.core.plan import ExecutionPlan

ROOT = Path(__file__).resolve().parents[1]
SCOPE = "SYNTHETIC_TEST_ONLY"
HEAD = "a" * 40
SEED = 20260830
IDENTITY = {"bootstrap_sha256": "2" * 64, "source_receipt_sha256": "3" * 64}


def _plan(run, mode, kind, trace):
    impl = implementations(include_c51=True)[0 if mode == "baseline" else -1]
    resolved = resolve_candidate(LIVEWAN, impl.id, run_id="cpu-producer-chain")
    def expand(value):
        if isinstance(value, str) and "${ENV:" in value:
            return str(trace) if "LIVEWAN_OPERATION_TRACE" in value else str(run / "CPU_ONLY_ASSET")
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [expand(v) for v in value]
        return value
    plan = ExecutionPlan.from_mapping(expand(resolved["native_plan"]["template"])).to_dict()
    if kind == "trajectory":
        worker._apply_trajectory_contract(plan, cycles=2, mode=mode)
    else:
        worker._apply_two_cycle_timing_contract(plan, mode=mode)
    worker._validate_plan_identity(plan, mode=mode)
    return plan


@pytest.mark.parametrize("kind", ["trajectory", "two-cycle-timing"])
def test_actual_planner_lowering_native_writer_atomic_unique_and_before_timer(tmp_path, monkeypatch, kind):
    plan = _plan(tmp_path, "pipeline", kind, tmp_path / "CPU_ONLY_TRACE")
    before = copy.deepcopy(plan)
    replacements = []
    replace = worker.os.replace
    def observed(src, dst):
        replacements.append((Path(src), Path(dst)))
        return replace(src, dst)
    monkeypatch.setattr(worker.os, "replace", observed)
    for node in (0, 1):
        for local in range(8):
            _native_writer_statement(plan, tmp_path, kind=kind, node=node, local_rank=local)
    assert [dst.name for _, dst in replacements] == ["effective_plan.node0.json", "effective_plan.node1.json"]
    assert all(src != dst and not src.exists() and dst.is_file() for src, dst in replacements)
    for node in (0, 1):
        saved = json.loads((tmp_path / f"effective_plan.node{node}.json").read_text())
        assert saved.pop("effective_plan_sha256") == worker._effective_plan_sha256(plan)
        assert saved == plan
    assert plan == before
    function = ast.parse(inspect.getsource(worker.run_worker)).body[0]
    calls = [n for n in ast.walk(function) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    writers = [n for n in calls if n.func.id == "_publish_effective_plan"]
    assert len(writers) == 1
    assert writers[0].lineno < min(n.lineno for n in calls if n.func.id == "build_models")
    assert writers[0].lineno < next(n.lineno for n in ast.walk(function) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "timing_start_ns" for t in n.targets))


@pytest.mark.parametrize("kind", ["validate", "capacity", "correctness", "timing", "profile"])
def test_writer_does_not_add_files_to_unrelated_modes(tmp_path, kind):
    plan = _plan(tmp_path, "baseline", "trajectory", tmp_path / "CPU_ONLY_TRACE")
    _native_writer_statement(plan, tmp_path, kind=kind, node=0)
    assert not list(tmp_path.glob("effective_plan.node*.json"))


@pytest.mark.parametrize("node", [-1, 2])
def test_actual_metadata_producer_rejects_wrong_node_rank(tmp_path, node):
    plan = _plan(tmp_path, "pipeline", "trajectory", tmp_path / "CPU_ONLY_TRACE")
    with pytest.raises(RuntimeError, match="node rank"):
        _native_writer_statement(plan, tmp_path, kind="trajectory", node=node)


def _policy(rank, mode):
    if mode == "pipeline":
        fixture = json.loads((ROOT / "tests/fixtures/livewan_c51_policy_receipts_synthetic.json").read_text())
        assert fixture["evidence_scope"] == SCOPE
        row = copy.deepcopy(fixture["generator" if rank < 8 else "score"])
        row["planner_c51_bundle"] = dict(receipt_contract(), execution_status="PASS")
    else:
        row = {}
    row.update(evidence_scope=SCOPE, rank=rank, status="PASS", mode=mode,
        workload_request={"frames": 81, "schema_version": "distillschedule.livewan_request.v1"},
        actual_latent_shapes=[[1,16,21,46,80]], installed_plan_sha256="f"*64,
        logical_generator_updates=2, logical_critic_updates=10,
        physical_generator_commits=2 if mode == "baseline" or rank < 8 else 0,
        physical_critic_commits=10 if mode == "baseline" or rank >= 8 else 0,
        final_versions=dict(generator=2, critic=10, ema=0), profiled=False)
    return row


def _qualification(root, mode, trace):
    from distillschedule.evidence.trajectory_coarse import AppendOnlyTrajectoryWriter, TrajectoryCycleReceipt
    run = root / (SCOPE + "_" + mode)
    run.mkdir()
    arm = SCOPE + "_" + mode + "_arm"
    variant = "b1" if mode == "baseline" else VARIANT
    plan = _plan(run, mode, "trajectory", trace.path)
    anchors = _anchors(run)
    for node in (0, 1):
        _native_writer_statement(plan, run, kind="trajectory", node=node)
        launch = run / "launcher" / f"node_{node}"
        launch.mkdir(parents=True)
        manifest = dict(head=HEAD, node_rank=node, run_id=run.name, execution_mode=mode,
            pipeline_variant=variant, run_kind="trajectory", trajectory_cycles=2,
            trajectory_seed=SEED, trajectory_arm_id=arm, claim_boundary="DIAGNOSTIC_ONLY")
        (launch / "launch_manifest.txt").write_text("".join(f"{k}={v}\n" for k, v in manifest.items()))
    digest = worker._effective_plan_sha256(plan)
    for rank in range(16):
        raw = _policy(rank, mode)
        raw.update(run_kind="trajectory", trace_sha256=trace.sha256, **IDENTITY,
            effective_plan_sha256=digest, trajectory=dict(formal_timing_eligible=False, arm_id=arm))
        path = run / "receipts" / f"rank_{rank:03d}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(raw))
        writer = AppendOnlyTrajectoryWriter(run / "trajectory" / f"rank_{rank:03d}.jsonl", run_id=run.name, rank=rank)
        for cycle in (1, 2):
            samples = tuple(f"cycle={cycle}/slot={rank}/step={s}" for s in range(5))
            writer.append(TrajectoryCycleReceipt(run_id=run.name, workload="livewan", variant=arm,
                rank=rank, world_size=16, seed=SEED, cycle=cycle, expected_total_cycles=2, logical_step=(cycle-1)*5,
                matched_contract=dict(evidence_scope=SCOPE, width="640", height="368", frames="81", global_batch="16"),
                implementation=dict(execution_mode=mode, critic_physical_batch_size="1" if mode == "baseline" else "2",
                    critic_physical_backward_calls="5" if mode == "baseline" or rank >= 8 else "0",
                    critic_optimizer_commits="5" if mode == "baseline" or rank >= 8 else "0"),
                sample_identities=samples, rng_identities={k: "b"*64 for k in samples},
                versions=dict(generator=cycle, critic=cycle*5, ema=0),
                cycle_updates=dict(generator=1, critic=5, ema=0),
                cumulative_updates=dict(generator=cycle, critic=cycle*5, ema=0),
                call_census={"application/student": 2080}, losses=dict(generator_mean=1./cycle, critic_mean=2./cycle),
                gradient_norms=dict(generator_mean=3., critic_mean=4.),
                tensor_contracts=dict(latent=dict(shape=[1,16,21,46,80], dtype="torch.bfloat16")),
                state_fingerprints={**{k:"c"*64 for k in
                    ("generator_trainable", "generator_adam", "critic_trainable", "critic_adam")}, "anchor": anchors[cycle]},
                finite=True))
    summary = run / "summary.json"
    result = _invoke("summarize_livewan_trajectory_world16.py", "--run-root", run,
        "--cycles", 2, "--expected-head", HEAD, "--output", summary)
    assert result.returncode == 0, result.stdout + result.stderr
    return dict(run=run, summary=summary, variant=variant, arm=arm, plan=plan, mode=mode)


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    root = tmp_path_factory.mktemp(SCOPE + "_c51_full_bundle_chain")
    trace = _trace(root)
    baseline, pipeline = (_qualification(root, mode, trace) for mode in ("baseline", "pipeline"))
    comparison, gate = root / "comparison.json", root / "gate.json"
    args = ["--baseline-trajectory-summary", baseline["summary"], "--pipeline-trajectory-summary", pipeline["summary"],
        "--trajectory-comparison", comparison, "--expected-head", HEAD, "--pipeline-variant", VARIANT, "--output", gate]
    result = _invoke("compare_livewan_two_cycle_trajectories.py", "--baseline-run-root", baseline["run"],
        "--pipeline-run-root", pipeline["run"], "--baseline-summary", baseline["summary"],
        "--pipeline-summary", pipeline["summary"], "--expected-head", HEAD, "--pipeline-variant", VARIANT, "--output", comparison)
    assert result.returncode == 0, result.stdout + result.stderr
    result = _invoke("build_livewan_two_cycle_timing_gate.py", *args)
    assert result.returncode == 0, result.stdout + result.stderr
    return dict(root=root, trace=trace, baseline=baseline, pipeline=pipeline, comparison=comparison, gate=gate)


def _node_header(gate, *, head=HEAD, variant=VARIANT):
    source = (ROOT / "scripts/distillschedule/launch_livewan_crossscale_world16_node.sh").read_text()
    anchor = '"${PYTHON}" - "${CORRECTNESS_RECEIPT}" "${EXPECTED_HEAD}" "${RUN_KIND}"'
    script = source[source.index(anchor):].split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    return subprocess.run([sys.executable, "-", str(gate), head, "two-cycle-timing", "pipeline", variant, str(ROOT)],
        input=script + '\nassert "torch" not in sys.modules\n', env=dict(os.environ, PYTHONPATH=""),
        text=True, capture_output=True, timeout=15)


def _load_gate(chain, **overrides):
    args = dict(path=chain["gate"], expected_sha256=_sha(chain["gate"]), expected_head=HEAD,
        mode="pipeline", pipeline_variant=VARIANT, arm_id=chain["pipeline"]["arm"], seed=SEED,
        trace=chain["trace"], bootstrap=SimpleNamespace(sha256=IDENTITY["bootstrap_sha256"]),
        source_receipt={"source_receipt_sha256": IDENTITY["source_receipt_sha256"]})
    args.update(overrides)
    return worker._load_two_cycle_timing_gate(**args)


def test_actual_producer_summary_compare_v2_gate_node_header_worker_loader(chain):
    gate = json.loads(chain["gate"].read_text())
    assert gate["schema_version"] == "distillschedule.livewan_two_cycle_timing_gate.v2"
    assert gate["numerical_review"]["policy"] == "REJECT_EXTREME_ONLY_RECORD_ORDINARY_REVIEW"
    assert gate["arms"]["pipeline"]["artifact_binding"]["c51_effective_plans"]
    result = _node_header(chain["gate"])
    assert result.returncode == 0, result.stderr
    assert _load_gate(chain)["status"] == "PASS"


@pytest.mark.parametrize("mutation", ["missing", "hash", "shape_rehashed", "rank_sha", "runtime_count"])
def test_actual_summary_rejects_broken_native_producer_or_receipt(chain, tmp_path, mutation):
    run = chain["pipeline"]["run"]
    path = run / ("receipts/rank_000.json" if mutation in {"rank_sha", "runtime_count"} else "effective_plan.node0.json")
    before = path.read_bytes()
    try:
        if mutation == "missing":
            path.unlink()
        else:
            raw = json.loads(before)
            if mutation == "hash": raw["effective_plan_sha256"] = "f" * 64
            if mutation == "shape_rehashed":
                raw["workload"]["shape"]["resolution"] = [368, 704]
                raw.pop("effective_plan_sha256")
                raw["effective_plan_sha256"] = worker._effective_plan_sha256(raw)
            if mutation == "rank_sha": raw["effective_plan_sha256"] = "f" * 64
            if mutation == "runtime_count": raw["generator_rms_epilogue"]["calls"] = 0
            path.write_text(json.dumps(raw))
        result = _invoke("summarize_livewan_trajectory_world16.py", "--run-root", run,
            "--cycles", 2, "--expected-head", HEAD, "--output", tmp_path / "broken-summary.json")
        assert result.returncode != 0
    finally:
        path.write_bytes(before)


@pytest.mark.parametrize("field,value", [("expected_head", "f"*40), ("seed", 17),
    ("arm_id", "stale"), ("pipeline_variant", "c34-continuous-b16-residual-layout"), ("expected_sha256", "f"*64)])
def test_worker_loader_rejects_source_seed_arm_and_gate_hash_drift(chain, field, value):
    with pytest.raises(RuntimeError):
        _load_gate(chain, **{field: value})


def test_node_header_and_worker_loader_reject_post_gate_actual_plan_mutation(chain):
    path = chain["pipeline"]["run"] / "effective_plan.node1.json"
    before = path.read_bytes()
    try:
        path.write_text("{}")
        assert _node_header(chain["gate"]).returncode != 0
        with pytest.raises((ValueError, RuntimeError)):
            _load_gate(chain)
    finally:
        path.write_bytes(before)
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



def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()



def _invoke(name, *arguments):
    folder = "tools/assets" if name == "build_livewan_continuous_trace.py" else "scripts/distillschedule"
    return subprocess.run([sys.executable, str(ROOT / folder / name),
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
    _trajectory(seed, mode="pipeline", variant=VARIANT, run_id=SCOPE)
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

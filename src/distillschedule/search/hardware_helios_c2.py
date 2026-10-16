"""H100 C2 source/launcher/admission bridge into the existing model planner.

No GPU, SSH, install, timing estimator or implicit campaign resume. A calibrated
component is an actual coupled native G1/C5/EMA5 cycle over all eight ranks, not
independent role estimates. Memory is a conservative whole-device reservation,
not an invented peak. Old single-Full5 receipts are never accepted as C2 costs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

from distillschedule.core.enums import GradientMode, OperationKind
from distillschedule.core.graph import Dependency, ExecutionGraph, Operation
from distillschedule.runtime import h100_helios_c2_control as control
from distillschedule.runtime import h100_helios_two_cycle as runtime
from . import hardware_helios as full5
from .allocation import ExecutionAllocation
from .contracts import require
from .cost_model import ComponentContract, CostModel, OperationCost
from .model_search import validate_saved_prediction
from .plan_ir import BindingRule, PlanningProblem

digest = control.digest
DEPLOYMENT = "distillschedule.helios_h100_c2_deployment.v1"
FAMILY = "helios_h100_native.coupled_cycle"
WORKLOAD = "helios14b_nogan_h100_c2"
PRECISION = {"frozen_base": "torch.bfloat16", "trainable_lora": "torch.float32", "reduce_dtype": "torch.float32",
    "loss_scale_per_logical_b1": .5, "ema": "NATIVE_GPU_LOCAL_SHARD_REFERENCE_RECURRENCE"}
MEMORY = "WHOLE_GPU_CAPACITY_EXCLUSIVE_RESERVATION_NOT_MEASURED_PEAK"


def prepare_source(parent_root, destination, *, allow_local_provenance_commit=False):
    """Explicit local source-only derivation. Never alters the parent checkout."""
    require(allow_local_provenance_commit is True, "explicit permission for a local provenance commit required")
    parent_root, destination = Path(parent_root).resolve(), Path(destination).resolve()
    full5.inspect_native_source(parent_root)
    require(not destination.exists(), "immutable derived destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Local shared object store, source checkout only: no model/asset copies.
    subprocess.run(["git", "clone", "--shared", "--no-checkout", str(parent_root), str(destination)], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(destination), "-c", "core.hooksPath=/dev/null", "checkout", "--detach", runtime.PARENT_HEAD], check=True, capture_output=True, text=True)
    worker = destination / runtime.WORKER
    text = worker.read_text()
    seam = '        dist.barrier()\n        torch.cuda.synchronize(device)\n        started_perf = time.perf_counter()'
    require(text.count(seam) == 1 and control.sha(worker) == runtime.WORKER_SHA, "exact default-off hook seam changed")
    worker.write_text(text.replace(seam, runtime.HOOK + seam))
    files = {control.ADAPTER: Path(runtime.__file__).read_text(), control.CONTROL: Path(control.__file__).read_text(),
        control.ENTRY: control.ENTRY_TEXT, control.LAUNCHER: control.launcher_text((destination / control.OLD_LAUNCHER).read_text())}
    for relative, value in files.items():
        path = destination / relative
        require(not path.exists(), "new native C2 source already exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    subprocess.run(["git", "-C", str(destination), "add", "--", runtime.WORKER, *files], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(destination), "-c", "core.hooksPath=/dev/null", "-c", "user.name=Codex provenance",
        "-c", "user.email=codex-provenance@localhost", "-c", "commit.gpgsign=false", "commit", "-m",
        "Add default-off H100 Helios two-cycle native hook and CPU control"], check=True, capture_output=True, text=True)
    proof = control.inspect_source(destination)
    full5.inspect_native_source(parent_root)
    return proof


def _binding(d):
    return {"source_head": d["native_source"]["head"], "hardware_fingerprint": digest(d["allocation"]),
        "environment_fingerprint": digest({"site": d["site"], "versions": d["runtime_versions"]}),
        "asset_fingerprint": digest({"gate": d["asset_gate"], "retention": d["retention_api"]}),
        "semantic_fingerprint": digest({"contract": control.CONTRACT, "source_proof": d["source_proof"], "trace_replay": "fixed_source_cohort_replay"}),
        "precision_fingerprint": digest(PRECISION)}


def bind_deployment(legacy_deployment, source_proof, *, repo_root, evidence_scope="MEASURED"):
    legacy = full5.validate_deployment(legacy_deployment)
    source = control.read_ref(source_proof)
    d = dict(legacy, schema_version=DEPLOYMENT, native_source={"head": source["derived_head"], "repo_root": str(Path(repo_root).resolve())},
             source_proof=source_proof, evidence_scope=evidence_scope)
    d["binding"] = _binding(d)
    return validate_deployment(d)


def validate_deployment(value):
    require(isinstance(value, dict) and set(value) == {"schema_version", "native_source", "site", "allocation", "asset_gate",
        "retention_api", "runtime_versions", "master_port", "deadline_seconds", "source_proof", "binding", "evidence_scope"}, "unknown/missing C2 deployment field")
    d = json.loads(json.dumps(value, allow_nan=False))
    require(d["schema_version"] == DEPLOYMENT and d["evidence_scope"] in ("MEASURED", "CPU_FIXTURE_ONLY"), "C2 deployment schema/scope mismatch")
    legacy = {k: v for k, v in d.items() if k not in ("source_proof", "binding", "evidence_scope")}
    legacy.update(schema_version="distillschedule.hardware_helios_deployment.v1", native_source=dict(d["native_source"], head=runtime.PARENT_HEAD))
    full5.validate_deployment(legacy)
    full5._ref(d["source_proof"], "derived source proof")
    require(re.fullmatch(r"[0-9a-f]{40}", d["native_source"]["head"]) and d["native_source"]["head"] != runtime.PARENT_HEAD, "C2 must bind new derived source, never old Full5 HEAD")
    require(d["binding"] == _binding(d), "model/source/hardware/environment/assets/precision binding changed")
    require(len({g["capacity_bytes"] for g in d["allocation"]["devices"]}) == 1, "whole-device cost envelope requires equal observed capacities")
    return d


def inspect_deployment(deployment, *, source_root=None):
    d = validate_deployment(deployment)
    source = control.inspect_source(Path(source_root or d["native_source"]["repo_root"]))
    require(control.read_ref(d["source_proof"]) == source and source["derived_head"] == d["native_source"]["head"], "derived source proof mismatch")
    return {"source": source, "binding": d["binding"], "gpu_execution": "NOT_RUN", "live_assets": "NOT_CHECKED", "campaign": "STOPPED_NOT_RESUMED"}


def _native_cpu(root, action, payload):
    require(action in ("inventory", "audit", "verify-assets", "rebind-assets"), "unsupported native CPU action")
    control.inspect_source(root)
    code = '''
import json, sys
from pathlib import Path
root, action = Path(sys.argv[1]), sys.argv[2]
sys.path[:0] = [str(root/'src'), str(root), str(root/'scripts/distillschedule')]
from distillschedule.runtime import h100_helios_c2_control as c
payload = json.load(sys.stdin)
if action == 'audit':
    result = c.audit_run(root, Path(payload['run_root']))
else:
    native = c.native_module(root)
    if action == 'rebind-assets':
        frozen = native.freeze_policy(root, variant='no-gan', arm=payload['arm'])
        ref = payload['legacy_gate']
        native.verify_asset_gate(Path(ref['path']), ref['sha256'], expected_head=c.runtime.PARENT_HEAD, asset_root=Path(payload['asset_root']), frozen=frozen)
        result = c.read_ref(ref)
        result['exact_head'] = c.inspect_source(root)['derived_head']
        result['c2_source_rebind'] = dict(previous_gate=ref, full_hash_revalidated=True, scope='CPU_ASSET_BYTES_ONLY_NO_GPU')
    elif action == 'inventory':
        result = {arm: native.freeze_policy(root, variant='no-gan', arm=arm) for arm in payload['arms']}
    else:
        plan = payload['plan']
        c.check_plan(plan, root)
        gate, retention, controls = c._assets(plan, native)
        result = c._render(plan, native, gate, retention)
        result['correctness_gate'] = c.correctness_gate(plan, root, at_ns=c.time.time_ns())
        result['gpu_execution'] = 'NOT_RUN'
if 'torch' in sys.modules:
    raise RuntimeError('CPU control unexpectedly imported Torch')
print(json.dumps(result, allow_nan=False))
'''
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DISTILLSCHEDULE_", "NCCL_", "HELIOS_", "PYTHON"))}
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(Path(root).resolve()), action], input=json.dumps(payload),
        text=True, capture_output=True, cwd=root, env=env, timeout=7200, check=True)
    control.inspect_source(root)
    return json.loads(result.stdout)


def compile_plan(deployment, *, arm, mode, run_id, explicit_opt_in=False, correctness=None, source_root=None, model_plan=None):
    d = validate_deployment(deployment)
    require(explicit_opt_in is True, "C2 is default-off: explicit opt-in required")
    require(arm in runtime.POLICIES and mode in ("correctness", "timing"), "unsupported C2 arm/mode")
    require(isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id), "unsafe C2 run id")
    if mode == "timing": full5._ref(correctness, "fresh C2 correctness")
    else: require(correctness is None and model_plan is None, "correctness cannot inherit another timing/model selection")
    root = Path(source_root or d["native_source"]["repo_root"])
    inspected = inspect_deployment(d, source_root=root)
    frozen = _native_cpu(root, "inventory", {"arms": [arm]})[arm]
    require(digest(frozen["policy"]) == runtime.POLICIES[arm], "native frozen arm changed")
    if model_plan is not None:
        problem, costs = PlanningProblem.from_dict(model_plan["problem"]), CostModel.from_dict(model_plan["cost_model"])
        validate_model(problem, costs, d)
        predicted = validate_saved_prediction(model_plan["search_result"], problem, costs)
        require(predicted["status"] == "PREDICTED_FEASIBLE" and predicted["assignment"] == {"arm": arm}, "selected model does not lower this native arm")
    site = d["site"]
    run_root = Path(site["project_root"]) / "research_artifacts/five_workload_h100_20260930" / run_id
    argv = ["bash", str(Path(d["native_source"]["repo_root"]) / control.LAUNCHER), site["project_root"], run_id, "no-gan", mode,
            arm, d["native_source"]["head"], d["asset_gate"]["path"], d["asset_gate"]["sha256"]]
    environment = {"DISTILLSCHEDULE_PYTHON": site["python"], "HELIOS_RUNTIME_ROOT": site["helios_root"], "DISTILLSCHEDULE_ASSET_ROOT": site["asset_root"],
        "HF_HOME": site["hf_home"], "DISTILLSCHEDULE_MASTER_PORT": str(d["master_port"]), "DISTILLSCHEDULE_RUN_DEADLINE_SECONDS": str(d["deadline_seconds"]),
        control.PREFIX + "OPT_IN": "1"}
    if arm != "best": environment["DISTILLSCHEDULE_HELIOS_H100_RETENTION_API_RECEIPT"] = d["retention_api"]["path"]
    if correctness:
        environment.update(DISTILLSCHEDULE_HELIOS_H100_CORRECTNESS_SUMMARY=correctness["path"], DISTILLSCHEDULE_HELIOS_H100_CORRECTNESS_SHA256=correctness["sha256"])
    plan = {"schema_version": control.PLAN_SCHEMA, "deployment": d, "deployment_sha256": digest(d), "arm": arm, "mode": mode,
        "run_id": run_id, "run_root": str(run_root), "explicit_opt_in": True, "contract": control.CONTRACT, "frozen_policy": frozen,
        "source_proof_sha256": d["source_proof"]["sha256"], "argv": argv, "environment": environment, "correctness": correctness,
        "purpose": "PLANNER_SELECTED" if model_plan is not None else "EXPLICIT_CALIBRATION", "model_plan": model_plan,
        "evidence_scope": d["evidence_scope"], "execution": "NOT_RUN", "claimable_speedup": False, "numerical_status": "REVIEW",
        "strict_full_state_parity": "NOT_RUN", "campaign": "STOPPED_SEPARATE_GPU_AUTHORIZATION_REQUIRED"}
    plan["plan_sha256"] = digest(plan)
    return plan


def validate_plan(plan, *, source_root=None):
    require(plan.get("plan_sha256") == digest({k: v for k, v in plan.items() if k != "plan_sha256"}), "compiled plan hash changed")
    expected = compile_plan(plan["deployment"], arm=plan["arm"], mode=plan["mode"], run_id=plan["run_id"], explicit_opt_in=plan["explicit_opt_in"],
        correctness=plan["correctness"], source_root=source_root, model_plan=plan["model_plan"])
    require(plan == expected, "compiled plan differs from exact native/model lowering")
    return expected


def dry_run(plan_ref, *, verify_assets=False, source_root=None):
    plan = validate_plan(control.read_ref(plan_ref), source_root=source_root)
    require(plan["evidence_scope"] == "MEASURED", "fixture plan cannot render executable native launch")
    result = {"status": "PASS_CPU_SOURCE_AND_PLAN_ONLY", "gpu_execution": "NOT_RUN", "campaign": "STOPPED_NOT_RESUMED",
        "argv": plan["argv"], "environment": dict(plan["environment"], **{control.PREFIX + "PLAN": plan_ref["path"], control.PREFIX + "PLAN_SHA256": plan_ref["sha256"]}),
        "live_assets": "NOT_CHECKED_PREPARE_WILL_FAIL_CLOSED", "worker_mode": "TWO_CONTINUOUS_CYCLES_SAME_INITIALIZED_ENGINES"}
    if verify_assets:
        result["native_prepare"] = _native_cpu(Path(source_root or plan["deployment"]["native_source"]["repo_root"]), "verify-assets", {"plan": plan})
        result.update(status="PASS_CPU_NATIVE_PREPARE_READS", live_assets="PASS_FRESH_FULL_HASH_CHECK")
    return result


def audit_plan_run(plan_ref, summary_ref):
    plan = validate_plan(control.read_ref(plan_ref))
    d = plan["deployment"]
    require(d["evidence_scope"] == "MEASURED", "CPU fixtures are never native admission")
    saved = control.read_ref(summary_ref)
    require(Path(summary_ref["path"]) == Path(plan["run_root"]) / "summary.json", "summary path differs from native run")
    actual = _native_cpu(Path(d["native_source"]["repo_root"]), "audit", {"run_root": plan["run_root"]})
    require(saved == actual and actual["mode"] == plan["mode"] and actual["status"] == "PASS_C2_PROTOCOL_ONLY"
        and actual["plan_sha256"] == plan["plan_sha256"] and actual["evidence_scope"] == "MEASURED", "native raw C2 admission failed or summary was altered")
    return actual


def validate_observation(deployment, observation):
    d = validate_deployment(deployment)
    require(d["evidence_scope"] == "MEASURED", "CPU fixtures are never native cost admission")
    require(isinstance(observation, dict) and set(observation) == {"summary", "plan"}, "exact summary/plan references required")
    plan = validate_plan(control.read_ref(observation["plan"]))
    require(plan["deployment"] == d and plan["mode"] == "timing", "only same-deployment unprofiled C2 timing can calibrate")
    return audit_plan_run(observation["plan"], observation["summary"])


def component_contract(deployment, arm, cycle):
    d = validate_deployment(deployment)
    require(arm in runtime.POLICIES and type(cycle) is int and cycle in (1, 2), "unregistered C2 component")
    return ComponentContract(FAMILY, work={"workload": WORKLOAD, "contract": control.CONTRACT, "coupled_updates_per_cycle": "G1_C5_EMA5"}, precision=PRECISION,
        placement={"arm": arm, "allocation_sha256": digest(d["allocation"]), "world_size": 8},
        implementation={"adapter": FAMILY, "source": d["native_source"], "source_proof": d["source_proof"], "policy_sha256": runtime.POLICIES[arm]},
        context={"cycle": cycle, "entry": "INITIAL_CHECKPOINT" if cycle == 1 else "CONTINUOUS_AFTER_C1", "resources": "ALL_EIGHT_GPUS_EXCLUSIVE",
                 "memory_definition": MEMORY, "timer_scope": control.TIMER_SCOPE, "deployment_sha256": digest(d), "no_overlap_discount": True})


def planning_problem(deployment):
    d = validate_deployment(deployment)
    arms, ranks = tuple(runtime.POLICIES), tuple(range(8))
    ops = tuple(Operation(f"cycle{c}.native_cycle", (c-1)*5, "native_runtime", OperationKind.CRITIC_UPDATE, GradientMode.GRAD,
        required_versions={"completed_cycles": c-1}, produced_versions={"completed_cycles": c}, metadata={"cycle": c,
        "scope": "NATIVE_COUPLED_G1_C5_EMA5", "parameter_version_proof": "NATIVE_RAW_EIGHT_RANK_RECEIPTS"}) for c in (1, 2))
    graph = ExecutionGraph(ops, (Dependency(ops[0].operation_id, ops[1].operation_id, "same_initialized_native_engines"),),
        {"initial_versions": {"completed_cycles": 0}, "final_versions": {"completed_cycles": 2},
         "cycle_ends": {str(c): ops[c-1].operation_id for c in (1, 2)}, "memory_definition": MEMORY,
         "independent_overlap_discounts": False, "construction_peak": "NOT_ESTIMATED_NATIVE_COMPLETION_REQUIRED"})
    rules = tuple(BindingRule(ops[c-1].operation_id, {"arm": arm}, component_contract(d, arm, c).digest, ranks,
        tuple(f"gpu:{r}" for r in ranks), operation_contract_sha256=component_contract(d, arm, c).digest) for arm in arms for c in (1, 2))
    return PlanningProblem(WORKLOAD, d["binding"], graph, {"arm": arms}, rules, (), (), ExecutionAllocation.from_dict(d["allocation"]).memory_limits,
        tuple({"arm": arm} for arm in arms), evidence_scope=d["evidence_scope"], lowering={"adapter": FAMILY, "deployment_sha256": digest(d),
            "source_head": d["native_source"]["head"], "arbitrary_kernel_or_schedule_search": False})


def _derive(d, arm, cycle, observations):
    require(isinstance(observations, list) and observations, "no fresh native C2 calibration observations")
    values = [validate_observation(d, observation) for observation in observations]
    require(all(v["arm"] == arm for v in values) and len({v["run_id"] for v in values}) == len(values), "mixed/duplicate native calibration run")
    return [v["cycle_ns"][cycle-1] for v in values], d["allocation"]["devices"][0]["capacity_bytes"]


def validate_component_receipt(raw, cache=None):
    """Shared cost-model registration seam; always reopen raw evidence, no cache."""
    require(raw.get("schema_version") == "distillschedule.operation_calibration.v1", "unknown C2 calibration schema")
    d = validate_deployment(raw["deployment"])
    require(d["evidence_scope"] == raw["evidence_scope"] == "MEASURED" and raw["status"] == "PASS"
        and raw["profiled"] is False and raw["kind"] == "native_operation", "fixture/profiled/failed costs cannot be admitted")
    component = component_contract(d, raw["arm"], raw["cycle"])
    require(raw["component_contract"] == component.to_dict() and raw["key"] == raw["operation_contract_sha256"] == component.digest
        and raw["binding"] == d["binding"], "C2 component source/model contract changed")
    samples, reservation = _derive(d, raw["arm"], raw["cycle"], raw["observations"])
    require(raw["samples_ns"] == samples and raw["workspace_bytes_per_rank"] == reservation and raw["memory_definition"] == MEMORY,
        "C2 cost differs from actual timers or conservative reservation")
    require(raw["claimable_speedup"] is False and raw["strict_full_state_parity"] == "NOT_RUN" and raw["numerical_status"] == "REVIEW", "C2 diagnostic costs cannot claim full numerical parity")


def assemble_problem_and_costs(deployment, observations, *, output_dir):
    d = validate_deployment(deployment)
    require(d["evidence_scope"] == "MEASURED" and isinstance(observations, list), "native measured observations required")
    groups = {}
    for observation in observations:
        value = validate_observation(d, observation)
        groups.setdefault(value["arm"], []).append(observation)
    require(groups, "UNCALIBRATED: no fresh two-cycle H100 observations; old 94.3114s Full5 is inadmissible")
    output_dir = Path(output_dir)
    records = []
    for arm in sorted(groups):
        for cycle in (1, 2):
            component = component_contract(d, arm, cycle)
            samples, reservation = _derive(d, arm, cycle, groups[arm])
            raw = {"schema_version": "distillschedule.operation_calibration.v1", "binding": d["binding"], "key": component.digest,
                "samples_ns": samples, "workspace_bytes_per_rank": reservation, "memory_definition": MEMORY, "status": "PASS",
                "profiled": False, "kind": "native_operation", "operation_contract_sha256": component.digest, "component_contract": component.to_dict(),
                "deployment": d, "arm": arm, "cycle": cycle, "observations": groups[arm], "evidence_scope": "MEASURED", "claimable_speedup": False,
                "strict_full_state_parity": "NOT_RUN", "numerical_status": "REVIEW"}
            path = output_dir / f"{arm}.cycle{cycle}.json"
            control.write_new(path, raw)
            evidence = {k: raw[k] for k in ("status", "profiled", "kind", "operation_contract_sha256", "component_contract")}
            evidence.update(source_receipt=str(path.resolve()), source_receipt_sha256=control.sha(path))
            records.append(OperationCost(component.digest, tuple(samples), reservation, evidence))
    return planning_problem(d), CostModel(d["binding"], tuple(records), "MEASURED")


def validate_model(problem, costs, deployment):
    d = validate_deployment(deployment)
    require(problem.to_dict() == planning_problem(d).to_dict(), "selected model is not registered native C2 graph")
    costs.bind(d["binding"], problem_scope=d["evidence_scope"])
    allowed = {component_contract(d, arm, c).digest for arm in runtime.POLICIES for c in (1, 2)}
    for record in costs.records:
        require(record.key in allowed, "unregistered C2 cost record")
        if costs.scope == "MEASURED":
            raw = control.read_ref({"path": record.evidence["source_receipt"], "sha256": record.evidence["source_receipt_sha256"]})
            validate_component_receipt(raw)
            require(raw["samples_ns"] == list(record.samples_ns) and raw["workspace_bytes_per_rank"] == record.workspace_bytes_per_rank
                and raw["component_contract"] == record.evidence["component_contract"], "selected cost differs from native source")
    costs.verify_receipts()


def lower_selection(result, problem, costs, deployment, *, run_id, correctness, explicit_opt_in=False):
    d = validate_deployment(deployment)
    validate_model(problem, costs, d)
    prediction = validate_saved_prediction(result, problem, costs)
    require(prediction["status"] == "PREDICTED_FEASIBLE", "no calibrated feasible native C2 selection")
    for record in costs.records:
        if costs.scope == "MEASURED":
            raw = control.read_ref({"path": record.evidence["source_receipt"], "sha256": record.evidence["source_receipt_sha256"]})
            require(all(control.read_ref(o["summary"])["run_id"] != run_id for o in raw["observations"]), "selected confirmation must use fresh run id")
    return compile_plan(d, arm=prediction["assignment"]["arm"], mode="timing", run_id=run_id, explicit_opt_in=explicit_opt_in, correctness=correctness,
        model_plan={"problem": problem.to_dict(), "cost_model": costs.to_dict(), "search_result": result})

"""All timings/tensor sizes here are hand-checkable CPU fixtures, not GPU data."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from distillschedule.core.enums import GradientMode, OperationKind
from distillschedule.core.graph import Dependency, ExecutionGraph, Operation
from distillschedule.search.cost_model import CostModel, MissingCalibration, OperationCost
from distillschedule.search.implementation_library import implementation_inventory
from distillschedule.search.model_search import search_plans, validate_saved_prediction
from distillschedule.search.plan_ir import BindingRule, Lifetime, PlanningProblem, StateAllocation, validate_versions
from distillschedule.search.simulator import simulate
from distillschedule.search.precision import ExecutionPrecision

ROOT = Path(__file__).resolve().parents[1]
BINDING = {"source_head": "a" * 40, **{k: "b" * 64 for k in (
    "hardware_fingerprint", "environment_fingerprint", "asset_fingerprint",
    "semantic_fingerprint", "precision_fingerprint")}}


def fixture():
    ops = []
    edges = []
    for cycle in (1, 2):
        v = cycle - 1
        prefix = str(cycle)
        ops += [
            Operation(prefix + "f", cycle, "g", OperationKind.GENERATOR_ROLLOUT, GradientMode.GRAD,
                      adapter_mode="student", required_versions={"g": v}, metadata={"cycle": cycle}),
            Operation(prefix + "g", cycle, "g", OperationKind.GENERATOR_UPDATE, GradientMode.NO_GRAD,
                      adapter_mode="student", required_versions={"g": v, "c": v},
                      produced_versions={"g": cycle}, metadata={"cycle": cycle}),
            Operation(prefix + "c", cycle, "c", OperationKind.CRITIC_UPDATE, GradientMode.NO_GRAD,
                      adapter_mode="fake", required_versions={"g": cycle, "c": v},
                      produced_versions={"c": cycle}, metadata={"cycle": cycle}),
        ]
        edges += [Dependency(prefix + "f", prefix + "g"), Dependency(prefix + "g", prefix + "c")]
    edges += [Dependency("1g", "2f"), Dependency("1c", "2g")]
    graph = ExecutionGraph(tuple(ops), tuple(edges), {"initial_versions": {"g": 0, "c": 0},
                                                     "cycle_ends": {"1": "1c", "2": "2c"}})
    domains = {"kernel": ("slow", "fast"), "lookahead": ("off", "depth1"), "sharing": ("off", "on")}
    rules = []
    for op in ops:
        rank = 0 if op.role == "g" else 1
        for sharing in ("off", "on"):
            for kernel in ("slow", "fast"):
                rules.append(BindingRule(op.operation_id, {"sharing": sharing, "kernel": kernel},
                    kernel if op.kind == OperationKind.GENERATOR_ROLLOUT else ("commit" if op.role == "g" else "critic"),
                    (rank,), (f"gpu:{rank}",), ("shared" if sharing == "on" else op.role,)))
    states = (StateAllocation("g", (0,), 40, ("g",), True, "base-g"),
              StateAllocation("c", (1,), 40, ("c",), True, "base-c"),
              StateAllocation("shared", (0, 1), 20, ("g", "c"), True, "same-frozen-base", True))
    lives = tuple(Lifetime(f"saved{c}", f"{c}f", (f"{c}g",), (0,), 5) for c in (1, 2))
    baseline = {"kernel": "slow", "lookahead": "off", "sharing": "off"}
    problem = PlanningProblem("wan13b_nogan_resident_best", BINDING, graph, domains, tuple(rules),
        states, lives, {0: 100, 1: 100}, (baseline,),
        conditional_dependencies=({"when": {"lookahead": "off"}, "source": "1c", "target": "2f"},),
        evidence_scope="CPU_FIXTURE_ONLY")
    costs = CostModel(BINDING, tuple(OperationCost(k, (ns,), workspace, {"kind": "CPU_FIXTURE_ONLY"})
                    for k, ns, workspace in [("slow", 4, 10), ("fast", 2, 50), ("commit", 1, 1), ("critic", 4, 10)]),
                      scope="CPU_FIXTURE_ONLY")
    return problem, costs, baseline


def test_round_trip_reuses_core_graph_and_hash():
    p, c, a = fixture()
    other = PlanningProblem.from_dict(json.loads(json.dumps(p.to_dict())))
    assert other.digest == p.digest
    assert simulate(other, a, CostModel.from_dict(c.to_dict())) == simulate(p, a, c)


def test_producer_exit_lifetime_does_not_double_count_measured_output_workspace():
    p, costs, assignment = fixture()
    assignment = {**assignment,"kernel":"fast"}
    old = simulate(p, assignment, costs)
    p = replace(p, lifetimes=tuple(replace(x,allocation_point="producer_exit") for x in p.lifetimes))
    measured = simulate(p, assignment, costs)
    assert old["memory"][0]["peak_bytes"] == 95
    assert measured["memory"][0]["peak_bytes"] == 90
    assert measured["memory_reconciliation"] == "PASS"
    assert PlanningProblem.from_dict(json.loads(json.dumps(p.to_dict()))).digest == p.digest
    with pytest.raises(ValueError):
        replace(p,lifetimes=(replace(p.lifetimes[0],allocation_point="guessed"),))


def test_hand_computed_serial_and_depth1_cycle_accounting():
    p, c, a = fixture()
    serial = simulate(p, a, c)
    assert (serial["predicted_cycle1_ns"], serial["predicted_cycle2_ns"], serial["predicted_total_ns"]) == (9, 9, 18)
    overlap = simulate(p, {**a, "lookahead": "depth1"}, c)
    assert (overlap["predicted_cycle1_ns"], overlap["predicted_cycle2_ns"], overlap["predicted_total_ns"]) == (9, 5, 14)
    assert overlap["cycle2_work_started_before_cycle1_commit"] == ["2f"]
    assert overlap["cycle2_early_activity_sum_ns_not_wall"] == 4
    assert overlap["gpu_execution"] == "NOT_RUN"


def test_shared_adapter_excludes_concurrency_but_reduces_residency():
    p, c, a = fixture()
    off = simulate(p, {**a, "lookahead": "depth1"}, c)
    on = simulate(p, {**a, "lookahead": "depth1", "sharing": "on"}, c)
    assert on["predicted_cycle2_ns"] == 9 > off["predicted_cycle2_ns"]
    assert on["memory"][0]["static_bytes"] == 20 < off["memory"][0]["static_bytes"]
    assert on["memory_reconciliation"] == "PASS"


def test_fast_kernel_memory_tradeoff_changes_feasible_choice():
    p, c, a = fixture()
    p = replace(p, memory_limits={0: 80, 1: 80})
    assert simulate(p, {**a, "kernel": "fast"}, c)["status"] == "PREDICTED_CAPACITY_FAIL"
    shared = simulate(p, {**a, "kernel": "fast", "sharing": "on"}, c)
    assert shared["status"] == "PREDICTED_FEASIBLE"
    assert shared["memory"][0]["peak_bytes"] == 75


def test_memcpy_link_serialization_and_compute_can_overlap_only_when_declared():
    p, c, a = fixture()
    # Reserve one shared communication resource on all model calls. This
    # fixture tests the scheduler, not an assumption about real NCCL kernels.
    rules = tuple(replace(r, resources=r.resources + ("link:node0-node1",)) for r in p.rules)
    result = simulate(replace(p, rules=rules), {**a, "lookahead": "depth1"}, c)
    assert result["predicted_total_ns"] == 18


def test_unreserved_gpu_and_unresolved_branch_fail_closed():
    p, c, a = fixture()
    with pytest.raises(ValueError, match="reserve each executing GPU"):
        simulate(replace(p, rules=tuple(replace(r, resources=("host",)) for r in p.rules)), a, c)
    op = replace(p.graph.operations[0], condition="global_step % 5 == 0")
    with pytest.raises(ValueError, match="branch predicates"):
        replace(p, graph=replace(p.graph, operations=(op, *p.graph.operations[1:])))
    with pytest.raises(ValueError, match="two-cycle operation"):
        replace(p, graph=replace(p.graph, operations=(replace(op, condition=None, metadata={}), *p.graph.operations[1:])))


def test_stale_version_and_checkpoint_replay_cannot_be_hidden_by_reorder():
    p, _, _ = fixture()
    validate_versions(p.graph)
    # Removing the actual C-version producer dependency is illegal even if a
    # resource constraint would accidentally serialize it on today's hardware.
    graph = replace(p.graph, dependencies=tuple(e for e in p.graph.dependencies if (e.source, e.target) != ("1c", "2g")))
    with pytest.raises(ValueError, match="RAW|overtaken"):
        validate_versions(graph)
    replay = Operation("late_replay", 2, "g", OperationKind.GENERATOR_BACKWARD, GradientMode.GRAD,
                       required_versions={"g": 0}, metadata={"cycle": 2})
    with pytest.raises(ValueError, match="overtaken"):
        validate_versions(replace(p.graph, operations=p.graph.operations + (replay,)))


def test_trainable_sharing_and_unscoped_adapter_rejected():
    p, c, a = fixture()
    with pytest.raises(ValueError, match="trainable state"):
        replace(p, states=(*p.states[:2], replace(p.states[2], frozen=False)))
    graph = replace(p.graph, operations=(replace(p.graph.operations[0], adapter_mode=None), *p.graph.operations[1:]))
    with pytest.raises(ValueError, match="no role scope"):
        simulate(replace(p, graph=graph), {**a, "sharing": "on"}, c)


def test_uncalibrated_and_precision_mismatch_fail_closed():
    p, c, a = fixture()
    with pytest.raises(MissingCalibration):
        simulate(p, a, replace(c, records=c.records[1:]))
    with pytest.raises(ValueError, match="precision mismatch"):
        simulate(p, a, replace(c, binding={**BINDING, "precision_fingerprint": "c" * 64}))
    with pytest.raises(ValueError, match="fixture costs"):
        simulate(replace(p, evidence_scope="MEASURED"), a, c)


def test_profile_union_cannot_be_cost_calibration():
    _, c, _ = fixture()
    with pytest.raises(ValueError, match="passing and unprofiled"):
        replace(c, scope="MEASURED")


def test_search_generates_combination_not_in_seeds_and_matches_oracle():
    p, c, a = fixture()
    oracle = search_plans(p, c, exhaustive=True, max_proposals=8)
    result = search_plans(p, c, seed=42, max_proposals=300)
    assert result["predicted_best"]["predicted_cycle2_ns"] == oracle["predicted_best"]["predicted_cycle2_ns"] == 5
    assert result["predicted_best"]["assignment"] not in p.seeds
    assert result["unique_evaluated"] == 8
    assert result["strict_winner"] is None and result["claimable_speedup"] is False
    assert validate_saved_prediction(result, p, c) == result["predicted_best"]
    tampered = copy.deepcopy(result)
    tampered["predicted_best"]["predicted_cycle2_ns"] -= 1
    with pytest.raises(ValueError, match="altered"):
        validate_saved_prediction(tampered, p, c)


def test_search_seed_does_not_change_graph_or_workload_identity():
    p, c, _ = fixture()
    first = search_plans(p, c, seed=3, max_proposals=30)
    second = search_plans(p, c, seed=3, max_proposals=30)
    assert first["journal"] == second["journal"]
    assert first["problem_sha256"] == p.digest


def test_partial_calibration_never_extrapolates_or_classifies_as_oom():
    p, c, _ = fixture()
    result = search_plans(p, replace(c, records=()), max_proposals=15)
    assert result["status"] == "NO_CALIBRATED_FEASIBLE_PLAN"
    assert result["calibration_requests"]
    assert all(r["status"] == "UNCALIBRATED" for r in result["rejections"])


def test_implementation_inventory_sources_exist_without_gpu_imports():
    inventory = implementation_inventory(ROOT)
    assert len(inventory["entries"]) >= 10
    assert all(r["status"] == "SOURCE_INVENTORIED" for r in inventory["entries"])
    assert not inventory["new_kernel_generation"]


@pytest.mark.parametrize("bad", ["kernel", "source_head", "precision_fingerprint"])
def test_incomplete_problem_fields_rejected(bad):
    p, _, a = fixture()
    if bad == "kernel":
        del a[bad]
        with pytest.raises(ValueError):
            p.resolve(a)
    else:
        with pytest.raises(ValueError):
            replace(p, binding={k: v for k, v in p.binding.items() if k != bad})


@pytest.mark.parametrize("budget", [3, 6, 8])
def test_cli_model_based_search_is_explicitly_cpu_only(tmp_path, budget):
    p, c, _ = fixture()
    problem_path, cost_path = tmp_path / "problem.json", tmp_path / "costs.json"
    problem_path.write_text(json.dumps(p.to_dict()))
    cost_path.write_text(json.dumps(c.to_dict()))
    output = tmp_path / "result.json"
    command = [sys.executable, str(ROOT / "scripts/distillctl.py"), "workloads", "search",
               "--strategy", "model-based", "--workload", p.workload, "--problem", str(problem_path),
               "--cost-model", str(cost_path), "--max-proposals", "8", "--exhaustive",
               "--max-trials", str(budget), "--output", str(output)]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text())
    assert result["status"] == "PASS_CPU_SEARCH"
    assert result["native_trial_budget"]["actual_runs"] == 0
    assert result["native_trial_budget"]["total"] == budget
    assert result["native_trial_budget"]["new_candidates"] == budget - 3
    assert result["native_lowering"] == "NOT_IMPLEMENTED_FOR_GENERATED_PLANS"
    # A second invocation may not overwrite an existing evidence file.
    assert subprocess.run(command, cwd=ROOT, capture_output=True).returncode != 0


def test_tf32_is_explicit_not_inherited_and_does_not_change_globals():
    precision = ExecutionPrecision("bfloat16", "float32", "float32", True, True, "flash-attn2-pinned")
    torch = SimpleNamespace(backends=SimpleNamespace(
        cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
        cudnn=SimpleNamespace(allow_tf32=True)), get_float32_matmul_precision=lambda: "high")
    with pytest.raises(ValueError, match="environment"):
        precision.assert_runtime(torch, {})
    receipt = precision.assert_runtime(torch, precision.launch_environment())
    assert receipt["status"] == "PASS"
    assert receipt["parameter_dtype_gate"] == "REQUIRES_NATIVE_LAYER_RECEIPTS"
    assert replace(precision, matmul_allow_tf32=False).digest != precision.digest
    torch.backends.cuda.matmul.allow_tf32 = False
    with pytest.raises(ValueError, match="matmul"):
        precision.assert_runtime(torch, precision.launch_environment())
    assert not torch.backends.cuda.matmul.allow_tf32  # verifier must not repair it silently


def test_measured_costs_require_real_receipt_and_exact_reconciliation(tmp_path):
    from distillschedule.workloads.best_known.registry import file_sha256
    evidence = {"status": "PASS", "profiled": False, "kind": "isolated_microbenchmark",
                "operation_contract_sha256": "c" * 64}
    path = tmp_path / "calibration.json"
    raw = {"schema_version": "distillschedule.operation_calibration.v1", "binding": BINDING,
           "key": "test", "samples_ns": [1, 2, 3], "workspace_bytes_per_rank": 4, **evidence}
    path.write_text(json.dumps(raw))
    evidence.update(source_receipt=str(path), source_receipt_sha256=file_sha256(path))
    model = CostModel(BINDING, (OperationCost("test", (1, 2, 3), 4, evidence),))
    model.verify_receipts()
    changed = replace(model, records=(replace(model.records[0], samples_ns=(1, 2, 2)),))
    with pytest.raises(ValueError, match="samples"):
        changed.verify_receipts()
    path.unlink()
    with pytest.raises(ValueError, match="missing"):
        model.verify_receipts()

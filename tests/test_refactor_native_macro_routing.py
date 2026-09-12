"""Synthetic CPU routing receipts; no SSH, CUDA, model assets or timing evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from distillschedule.search.contracts import semantic_contract
from distillschedule.search.cost_model import CostModel, OperationCost
from distillschedule.search.execution import NativeTrialRunner, rebind_plan, resolve_candidate
from distillschedule.search.h3_implementation import H3, H3Implementation
from distillschedule.search.livewan_implementation import LIVEWAN, LiveWanImplementation
from distillschedule.search.model_adapters import adapter_for
from distillschedule.search.model_search import search_plans
from distillschedule.search.model_validation import ModelValidationSession
from distillschedule.search.native_macro_adapter import (
    ADAPTER, MEMORY, build_problem, operation_contract, precision_contract,
)
from distillschedule.search.registry import candidate_profile
from distillschedule.search.runtime_identity import observed_binding
from distillschedule.workloads.best_known.registry import (
    file_sha256, fingerprint, git_head, load_profile, repository_root, write_new_json,
)
from test_refactor_model_validation import fake_observations, measurement


def _assets_fixture(tmp_path, workload, implementation):
    path = repository_root() / "configs/distillschedule/best_known/assets_examples" / (workload + ".json")
    assets = json.loads(path.read_text())
    # Generated H3 additionally pins its Diffusers source. These are metadata
    # fixtures only: no remote asset payloads are copied onto the CPU controller.
    for name, spec in candidate_profile(implementation.candidate())["required_assets"].items():
        if name not in assets["assets"]:
            assets["assets"][name] = {"path": "/srv/CPU_TEST_ONLY/" + name, "sha256": "a" * 64,
                **({"identity_file": spec["identity_file"]} if spec["kind"] == "directory" else {})}
    if workload == LIVEWAN:
        source = tmp_path / "source.json"
        write_new_json(source, {"status": "PASS", "source_root": "/srv/CPU_TEST_ONLY/prepared-wan"})
        assets["assets"]["source_receipt"] = {"path": str(source), "sha256": file_sha256(source)}
    path = tmp_path / "assets.json"
    write_new_json(path, assets)
    return path


def _calibration_fixture(tmp_path, workload):
    # Deliberately not the direct-receipt default pair. Replay must preserve
    # these calibrated seeds, including their order, rather than pick a recipe.
    anchors = ((LiveWanImplementation("fsdp"), LiveWanImplementation("pipeline", True))
               if workload == LIVEWAN else (H3Implementation(1, 64), H3Implementation(2, 64)))
    assets_path = _assets_fixture(tmp_path, workload, anchors[0])
    binding = {"source_head": git_head(repository_root()),
        **observed_binding(fake_observations(), source_head=git_head(repository_root())),
        "asset_fingerprint": fingerprint(json.loads(assets_path.read_text())["assets"]),
        "precision_fingerprint": fingerprint(precision_contract(workload)),
        "semantic_fingerprint": fingerprint(semantic_contract(load_profile(workload)))}
    layout = {"schema_version": "distillschedule.native_macro_layout.v1", "adapter": ADAPTER,
        "workload": workload, "binding": binding, "status": "PASS", "profiled": False,
        "memory_definition": MEMORY, "seed_implementations": [i.id for i in anchors],
        "memory_limits": {str(r): 100 for r in range(16)},
        "implementations": {i.id: {"status": "PASS", "ranks": [{"rank": r, "status": "PASS",
            "source_head": binding["source_head"], "post_setup_static_bytes": 10}
            for r in range(16)]} for i in anchors}}
    path = tmp_path / "layout.json"
    write_new_json(path, layout)
    problem = build_problem(path)
    records = []
    for index, implementation in enumerate(anchors):
        for operation in problem.graph.operations:
            key = fingerprint(operation_contract(workload, operation, implementation))
            evidence = {"status": "PASS", "profiled": False, "kind": "native_operation",
                "operation_contract_sha256": key}
            payload = {"schema_version": "distillschedule.operation_calibration.v1",
                "binding": binding, "key": key, "samples_ns": [100 - index * 10],
                "workspace_bytes_per_rank": 20, **evidence}
            path = tmp_path / (key + ".json")
            write_new_json(path, payload)
            records.append(OperationCost(key, tuple(payload["samples_ns"]), 20,
                {**evidence, "source_receipt": str(path), "source_receipt_sha256": file_sha256(path)}))
    costs = CostModel(binding, tuple(records))
    costs.verify_receipts()
    prediction = search_plans(problem, costs, exhaustive=True, max_proposals=32)
    return problem, costs, prediction, assets_path


def _session(inputs, output, *, resume=False):
    problem, costs, prediction, _ = inputs
    return ModelValidationSession(problem=problem, costs=costs, prediction=prediction,
        output=output, max_trials=3, resume=resume)


@pytest.mark.parametrize("workload", [LIVEWAN, H3])
def test_native_macro_run_and_journal_resume_use_selected_adapter(tmp_path, workload):
    inputs = _calibration_fixture(tmp_path, workload)
    session = _session(inputs, tmp_path / "validation")
    calls = []

    def execute(candidate, phase, directory):
        calls.append((candidate.id, phase))
        lowered = json.loads((directory / "model-lowering.json").read_text())
        assert lowered["search_candidate"] == candidate.to_dict()
        assert fingerprint(lowered["model_plan"]["problem"]) == session.problem.digest
        return measurement(session, candidate, seconds=5 if candidate.id == session.seeds[1] else 10)

    result = session.run(execute)
    assert calls == [(session.seeds[0], "baseline"), (session.seeds[1], "incumbent"),
                     (session.seeds[1], "confirmation")]
    assert result["status"] == "COMPLETE_NATIVE_VALIDATION"
    assert result["timing_attempts"] == 3 and result["new_candidate_attempts"] == 0
    assert result["confirmed_candidate"]["candidate_id"] == session.seeds[1]
    assert result["strict_winner"] is None  # Synthetic REVIEW is not strict GPU evidence.
    assert len(session.events()) == 6
    resumed = _session(inputs, session.output, resume=True)
    assert resumed.run(lambda *args: pytest.fail("completed native journal must not relaunch")) == result


@pytest.mark.parametrize("workload", [LIVEWAN, H3])
def test_native_macro_resume_rechecks_completed_receipt_hash(tmp_path, workload):
    inputs = _calibration_fixture(tmp_path, workload)
    session = _session(inputs, tmp_path / "validation")
    session.run(lambda candidate, phase, directory: measurement(session, candidate))
    resumed = _session(inputs, session.output, resume=True)
    path = next(session.output.glob("trials/*/executor-receipt.json"))
    raw = json.loads(path.read_text())
    raw["cycle_wall_seconds"]["2"] = 0.01
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="completed receipt changed"):
        resumed.run(lambda *args: pytest.fail("changed journal must not launch"))


@pytest.mark.parametrize("workload", [LIVEWAN, H3])
@pytest.mark.parametrize("changed", ["layout", "calibration"])
def test_native_macro_run_revalidates_inputs_after_construction(tmp_path, workload, changed):
    inputs = _calibration_fixture(tmp_path, workload)
    session = _session(inputs, tmp_path / "validation")
    path = Path(session.problem.lowering["layout_receipt"] if changed == "layout" else
                session.costs.records[0].evidence["source_receipt"])
    path.write_text(path.read_text() + "\n")
    error = "native layout changed" if changed == "layout" else "calibration receipt hash mismatch"
    with pytest.raises(ValueError, match=error):
        session.run(lambda *args: pytest.fail("changed native inputs must not launch"))
    assert session.events() == []


def _assert_rebind_rpc(monkeypatch, receipt, assets_path, output, anchors):
    calls = []

    def rpc(runner, action, **payload):
        assert action == "resolve"
        assert runner.reference.id == anchors[0].id
        assert runner.bootstrap_peer.id == anchors[1].id
        assert runner.reference.id != runner.bootstrap_peer.id
        calls.append(payload)
        return resolve_candidate(payload["workload"], payload["candidate_id"],
            assets_path=assets_path, run_id=payload["run_id"], mode=payload["mode"])

    # Only transport is injected: real profile/lowering/hash validation and
    # the NativeTrialRunner constructor still run, with no SSH or GPU work.
    monkeypatch.setattr(NativeTrialRunner, "_rpc", rpc)
    rebound = rebind_plan(receipt, run_id="fresh-cpu-replay", coordinator_output=output)
    assert calls == [{"workload": receipt["profile_id"], "candidate_id": receipt["search_candidate"]["id"],
        "assets": receipt["asset_config"], "run_id": "fresh-cpu-replay", "mode": receipt["mode"]}]
    assert rebound["run_id"] == "fresh-cpu-replay"
    assert rebound["search_candidate"] == receipt["search_candidate"]
    assert rebound["asset_config"] == receipt["asset_config"]
    assert rebound["profile_sha256"] == receipt["profile_sha256"]
    assert receipt["run_id"] not in json.dumps(rebound["nodes"])
    assert "fresh-cpu-replay" in json.dumps(rebound["nodes"])
    if "model_plan" in receipt:
        assert rebound["model_plan"] == receipt["model_plan"]
    assert json.loads((output / "execution-plan.json").read_text()) == rebound


@pytest.mark.parametrize("workload", [LIVEWAN, H3])
def test_rebind_generated_macro_uses_validated_serialized_seed_anchors(tmp_path, monkeypatch, workload):
    problem, costs, prediction, assets_path = _calibration_fixture(tmp_path, workload)
    adapter = adapter_for(workload)
    receipt = adapter.lower_prediction(prediction, problem, costs, assets_path=assets_path,
        run_id="emitted-cpu-plan")
    path = tmp_path / "emitted-plan.json"
    write_new_json(path, receipt)
    anchors = tuple(adapter.implementation_type.from_assignment(seed) for seed in problem.seeds)
    _assert_rebind_rpc(monkeypatch, json.loads(path.read_text()), assets_path, tmp_path / "replay", anchors)


@pytest.mark.parametrize("workload", [LIVEWAN, H3])
def test_rebind_direct_macro_receipt_uses_distinct_typed_anchors_without_recipe_substitution(
        tmp_path, monkeypatch, workload):
    if workload == LIVEWAN:
        implementation = LiveWanImplementation("pipeline", True, True)
        anchors = (LiveWanImplementation("fsdp"), LiveWanImplementation("pipeline"))
    else:
        implementation = H3Implementation(1, 64)
        anchors = (H3Implementation(2, 128), H3Implementation(1, 128))
    assets_path = _assets_fixture(tmp_path, workload, implementation)
    receipt = resolve_candidate(workload, implementation.id, assets_path=assets_path, run_id="direct-cpu-plan")
    assert "model_plan" not in receipt
    _assert_rebind_rpc(monkeypatch, receipt, assets_path, tmp_path / "replay", anchors)


@pytest.mark.parametrize("model_based", [False, True])
def test_rebind_existing_generated_wan_routing_is_preserved(tmp_path, monkeypatch, model_based):
    from distillschedule.search.wan_implementation import WAN, WanImplementation
    from test_refactor_wan_model_adapter import calibration_fixture

    implementation = WanImplementation("resident", 1024, True)
    assets_path = _assets_fixture(tmp_path, WAN, implementation)
    anchors = (WanImplementation("fsdp", 512, False), WanImplementation("resident", 2048, True))
    if model_based:
        problem, costs = calibration_fixture(tmp_path,
            {"asset_fingerprint": fingerprint(json.loads(assets_path.read_text())["assets"])})
        prediction = search_plans(problem, costs, exhaustive=True, max_proposals=12)
        receipt = adapter_for(WAN).lower_prediction(prediction, problem, costs,
            assets_path=assets_path, run_id="emitted-cpu-plan")
    else:
        receipt = resolve_candidate(WAN, implementation.id, assets_path=assets_path, run_id="direct-cpu-plan")
    _assert_rebind_rpc(monkeypatch, json.loads(json.dumps(receipt)), assets_path, tmp_path / "replay", anchors)


@pytest.mark.parametrize("workload", [LIVEWAN, H3])
@pytest.mark.parametrize("changed", ["source", "assets", "seeds", "profile", "calibration"])
def test_rebind_macro_rejects_changed_source_assets_or_plan_before_rpc(tmp_path, monkeypatch, workload, changed):
    problem, costs, prediction, assets_path = _calibration_fixture(tmp_path, workload)
    receipt = adapter_for(workload).lower_prediction(prediction, problem, costs,
        assets_path=assets_path, run_id="emitted-cpu-plan")
    replacement = None
    if changed == "source":
        receipt["integration_head"] = "b" * 40
        error = "native lowering binding changed"
    elif changed == "assets":
        assets = json.loads(assets_path.read_text())
        next(iter(assets["assets"].values()))["sha256"] = "b" * 64
        replacement = tmp_path / "replacement-assets.json"
        write_new_json(replacement, assets)
        error = "replay assets/gates changed"
    elif changed == "seeds":
        receipt["model_plan"]["problem"]["seeds"] = tuple(reversed(problem.seeds))
        error = "unsupported altered native graph"
    elif changed == "profile":
        receipt["fully_resolved_profile"] = copy.deepcopy(receipt["fully_resolved_profile"])
        receipt["fully_resolved_profile"]["runtime"]["planner_implementation"] = {}
        error = "serialized profile was altered"
    else:
        path = Path(costs.records[0].evidence["source_receipt"])
        path.write_text(path.read_text() + "\n")
        error = "calibration receipt hash mismatch"
    monkeypatch.setattr(NativeTrialRunner, "_rpc", lambda *args, **kwargs: pytest.fail("must reject before RPC"))
    output = tmp_path / "replay"
    with pytest.raises(ValueError, match=error):
        rebind_plan(receipt, run_id="fresh-cpu-replay", coordinator_output=output, assets_path=replacement)
    assert not output.exists()


@pytest.mark.parametrize("workload", [LIVEWAN, H3])
@pytest.mark.parametrize("changed", ["source", "assets", "profile"])
def test_rebind_macro_revalidates_resolved_plan_before_persisting(tmp_path, monkeypatch, workload, changed):
    problem, costs, prediction, assets_path = _calibration_fixture(tmp_path, workload)
    receipt = adapter_for(workload).lower_prediction(prediction, problem, costs,
        assets_path=assets_path, run_id="emitted-cpu-plan")
    calls = []

    def rpc(runner, action, **payload):
        assert action == "resolve"
        calls.append(payload["run_id"])
        result = resolve_candidate(workload, payload["candidate_id"], assets_path=assets_path,
            run_id=payload["run_id"], mode=payload["mode"])
        if changed == "source":
            result["integration_head"] = "b" * 40
        elif changed == "assets":
            result["asset_fingerprint"] = "b" * 64
        else:
            result["fully_resolved_profile"]["runtime"]["planner_implementation"] = {}
        return result

    monkeypatch.setattr(NativeTrialRunner, "_rpc", rpc)
    error = {"source": "native lowering binding changed", "assets": "native assets changed",
             "profile": "serialized profile was altered"}[changed]
    with pytest.raises(ValueError, match=error):
        rebind_plan(receipt, run_id="fresh-cpu-replay", coordinator_output=tmp_path / "replay")
    assert calls == ["fresh-cpu-replay"]
    assert not (tmp_path / "replay/execution-plan.json").exists()


@pytest.mark.parametrize("workload", [LIVEWAN, H3])
@pytest.mark.parametrize("run_id", ["emitted-cpu-plan", "source-inspection"])
def test_rebind_macro_requires_fresh_explicit_run_id(tmp_path, monkeypatch, workload, run_id):
    problem, costs, prediction, assets_path = _calibration_fixture(tmp_path, workload)
    receipt = adapter_for(workload).lower_prediction(prediction, problem, costs,
        assets_path=assets_path, run_id="emitted-cpu-plan")
    monkeypatch.setattr(NativeTrialRunner, "_rpc", lambda *args, **kwargs: pytest.fail("must reject before RPC"))
    with pytest.raises(ValueError, match="fresh explicit run-id"):
        rebind_plan(receipt, run_id=run_id, coordinator_output=tmp_path / "replay")
    assert not (tmp_path / "replay").exists()

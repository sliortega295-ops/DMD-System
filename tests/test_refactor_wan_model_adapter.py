"""Synthetic receipt fixtures test native lowering, not GPU execution/results."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from distillschedule.search.contracts import semantic_contract
from distillschedule.search.cost_model import CostModel, OperationCost
from distillschedule.search.execution import resolve_candidate, validated_profile
from distillschedule.search.model_search import search_plans
from distillschedule.search.plan_ir import validate_versions
from distillschedule.search.registry import candidate_profile, get_candidate
from distillschedule.search.wan_contract import arm, registered_placements
from distillschedule.search.wan_implementation import PRECISION, WAN, WanImplementation, verify_execution_precision
from distillschedule.search.wan_model_adapter import (
    ADAPTER, build_problem, cost_key, implementations, lower_prediction, native_graph,
    operation_contract, validate_lowered_plan, validate_native_problem,
)
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint, git_head, load_profile, repository_root


def test_composed_candidates_preserve_semantics_and_use_existing_launcher():
    canonical = load_profile(WAN)
    for impl in implementations():
        candidate = get_candidate(WAN, impl.id)
        profile = candidate_profile(candidate)
        assert semantic_contract(profile) == semantic_contract(canonical)
        assert profile["runtime"]["launcher"] == canonical["runtime"]["launcher"]
        assert profile["runtime"]["arguments"][-1] == impl.native_placement
        assert profile["runtime"]["planner_implementation"] == impl.assignment
        assert profile["runtime"]["environment"]["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] == "1"
        assert profile["runtime"]["environment"]["DISTILLSCHEDULE_WAN13B_PLANNER_PRECISION"] == PRECISION.digest
        assert impl == WanImplementation.from_assignment(impl.assignment) == WanImplementation.parse(impl.id)
    assert "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE" not in candidate_profile(get_candidate(WAN, "fsdp-reference"))["runtime"]["environment"]


@pytest.mark.parametrize("key", ["joint-wan1-fsdp-c1024-e0", "joint-wan1-fsdp-c512-e1",
    "joint-wan1-resident-c8192-e1", "joint-wan1-resident-c1024-e1;evil", "joint-wan1-tp8-c512-e0"])
def test_unimplemented_combinations_never_fall_back(key):
    with pytest.raises(ValueError):
        get_candidate(WAN, key)


def test_runtime_chunk_1024_really_reaches_all_blocks_and_is_opt_in(monkeypatch):
    from distillschedule.runtime.wan13b_fair import configure_modulation_chunk_tokens
    impl = WanImplementation("resident", 1024, True)
    monkeypatch.setenv("DISTILLSCHEDULE_SEARCH_CANDIDATE_ID", impl.id)
    module = SimpleNamespace(blocks=[SimpleNamespace() for _ in range(30)])
    value = configure_modulation_chunk_tokens(module, placement=impl.native_placement)
    assert value["effective_tokens"] == [1024]
    assert all(b._distillschedule_modulation_chunk_tokens == 1024 for b in module.blocks)
    assert arm(impl.native_placement)["chunk"] == 1024
    assert arm(impl.native_placement)["fused"]
    assert registered_placements() == {impl.native_placement}
    with pytest.raises(ValueError):
        arm("world16-fsdp")
    monkeypatch.delenv("DISTILLSCHEDULE_SEARCH_CANDIDATE_ID")
    fresh = SimpleNamespace(blocks=[SimpleNamespace() for _ in range(30)])
    assert configure_modulation_chunk_tokens(fresh, placement=impl.native_placement)["effective_tokens"] == [512]


def test_precision_setup_requires_explicit_generated_plan_and_keeps_legacy_unchanged():
    torch = SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
        cudnn=SimpleNamespace(allow_tf32=False)), get_float32_matmul_precision=lambda: "high")
    assert verify_execution_precision(torch, {}) is None
    assert not torch.backends.cuda.matmul.allow_tf32
    impl = WanImplementation("resident", 1024, True)
    env = {"DISTILLSCHEDULE_SEARCH_CANDIDATE_ID": impl.id, **impl.environment()}
    with pytest.raises(ValueError):
        verify_execution_precision(torch, {**env, "NVIDIA_TF32_OVERRIDE": "0"})
    assert not torch.backends.cuda.matmul.allow_tf32
    receipt = verify_execution_precision(torch, env)
    assert receipt["status"] == "PASS"
    assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    with pytest.raises(ValueError):
        verify_execution_precision(torch, {**env, "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0"})


def test_graph_has_native_clustered_order_exact_versions_and_no_hidden_reorder():
    graph = native_graph()
    validate_versions(graph)
    assert len(graph.operations) == 14  # composite stages, not DiT call count!
    assert graph.metadata["final_versions"] == {"generator": 4, "critic": 20, "ema": 20}
    for cycle in (1, 2):
        ops = [op for op in graph.operations if op.metadata["cycle"] == cycle]
        assert [op.metadata["stage"] for op in ops] == ["generator_update", "critic_rollouts", *[
            f"critic{i}_update_ema" for i in range(5)]]
        inventory = [op.metadata["inventory"] for op in ops]
        assert sum(i.get("generator_updates", 0) for i in inventory) == 1
        assert sum(i.get("critic_updates", 0) for i in inventory) == 5
        assert sum(i.get("ema_updates", 0) for i in inventory) == 5


def calibration_fixture(tmp_path):
    binding = {"source_head": git_head(repository_root()), "precision_fingerprint": PRECISION.digest,
        "semantic_fingerprint": fingerprint(semantic_contract(load_profile(WAN))),
        **{k: "a" * 64 for k in ("asset_fingerprint", "environment_fingerprint", "hardware_fingerprint")}}
    layout = {"schema_version": "distillschedule.wan_stage_layout.v1", "status": "PASS", "profiled": False,
        "binding": binding, "adapter": ADAPTER, "memory_limits": {str(i): 1000 for i in range(16)},
        "memory_definition": "POST_SETUP_STATIC; STAGE_COST_INCLUDES_ALL_ADDITIONAL_LIVE_MEMORY",
        "implementations": {impl.id: {"status": "PASS", "ranks": [{"rank": i, "status": "PASS",
            "source_head": binding["source_head"], "precision_fingerprint": PRECISION.digest,
            "post_setup_static_bytes": 100} for i in range(16)]} for impl in implementations()}}
    path = tmp_path / "layout.json"
    path.write_text(json.dumps(layout))
    p = build_problem(path)
    records = []
    for impl in implementations():
        for op in p.graph.operations:
            key = cost_key(op, impl)
            evidence = {"status": "PASS", "profiled": False, "kind": "native_operation",
                        "operation_contract_sha256": fingerprint(operation_contract(op, impl))}
            duration = 1 if impl.chunk_tokens == 1024 and impl.lora_epilogue else 5
            raw = {"schema_version": "distillschedule.operation_calibration.v1", "binding": binding, "key": key,
                   "samples_ns": [duration], "workspace_bytes_per_rank": 10, **evidence}
            record_path = tmp_path / (key + ".json")
            record_path.write_text(json.dumps(raw))
            evidence.update(source_receipt=str(record_path), source_receipt_sha256=file_sha256(record_path))
            records.append(OperationCost(key, (duration,), 10, evidence))
    return p, CostModel(binding, tuple(records))


def test_adapter_search_and_lowering_discovers_non_seed_configuration(tmp_path):
    p, c = calibration_fixture(tmp_path)
    result = search_plans(p, c, exhaustive=True, max_proposals=12)
    assert result["predicted_best"]["assignment"] == WanImplementation("resident", 1024, True).assignment
    assert result["predicted_best"]["assignment"] not in p.seeds
    assert len(result["ranked"]) == 7 and len(result["rejections"]) == 5
    receipt = lower_prediction(result, p, c, run_id="unit-fixture-not-a-gpu-run")
    assert receipt["search_candidate"]["id"] == WanImplementation("resident", 1024, True).id
    validate_lowered_plan(receipt)
    validated_profile(receipt, root=repository_root())
    receipt["search_candidate"]["id"] = WanImplementation("resident", 2048, True).id
    with pytest.raises(ValueError, match="decision vector"):
        validate_lowered_plan(receipt)


def test_arbitrary_dag_and_wrong_operation_calibration_cannot_lower(tmp_path):
    p, c = calibration_fixture(tmp_path)
    graph = replace(p.graph, metadata={**p.graph.metadata, "schedule": "unimplemented-reorder"})
    with pytest.raises(ValueError, match="arbitrary graph"):
        validate_native_problem(replace(p, graph=graph))
    from distillschedule.search.simulator import simulate
    records = tuple(replace(r, evidence={**r.evidence, "operation_contract_sha256": "b" * 64}) for r in c.records)
    with pytest.raises(ValueError, match="calibration contract"):
        simulate(p, p.seeds[0], replace(c, records=records))
    with pytest.raises(ValueError, match="CPU fixture"):
        validate_native_problem(replace(p, evidence_scope="CPU_FIXTURE_ONLY"))

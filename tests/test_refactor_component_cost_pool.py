"""Synthetic receipt protocol tests; no values here are measured GPU costs."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from test_refactor_component_costs import BINDING, component
from distillschedule.search.cost_model import CostModel, OperationCost
from distillschedule.search.component_calibration import pool_component_models
from distillschedule.workloads.best_known.registry import file_sha256


def model(tmp_path, name, samples=(20, 30), workspace=50, contract=None, binding=None):
    contract = contract or component()
    binding = binding or BINDING
    raw = dict(schema_version="distillschedule.operation_calibration.v1", binding=binding,
        key=contract.digest, samples_ns=list(samples), workspace_bytes_per_rank=workspace,
        status="PASS", profiled=False, kind="native_operation",
        operation_contract_sha256=contract.digest, component_contract=contract.to_dict())
    path = tmp_path / f"{name}-source.json"
    path.write_text(json.dumps(raw))
    evidence = {k: raw[k] for k in ("status", "profiled", "kind",
                "operation_contract_sha256", "component_contract")}
    evidence.update(source_receipt=str(path), source_receipt_sha256=file_sha256(path))
    return CostModel(binding, (OperationCost(contract.digest, samples, workspace, evidence),))


def test_pools_same_component_without_replaying_or_double_counting_same_receipt(tmp_path):
    a = model(tmp_path, "a")
    b = model(tmp_path, "b", samples=(40,), workspace=60)
    merged = pool_component_models((a, b, a), output=tmp_path / "pooled")
    merged.verify_receipts()
    assert len(merged.records) == 1
    row = merged.get_component(component())
    assert row.samples_ns == (20, 30, 40)
    assert row.workspace_bytes_per_rank == 60
    raw = json.loads(open(row.evidence["source_receipt"]).read())
    assert len(raw["component_sources"]) == 2
    assert raw["aggregation"] == "POOLED_NATIVE_COMPONENT_SAMPLES"
    assert raw["formal_timing_eligible"] is False


def test_different_context_not_pooled_and_binding_mismatch_rejected(tmp_path):
    a = model(tmp_path, "a")
    changed = replace(component(), context={"optimizer_materialized": False})
    b = model(tmp_path, "b", contract=changed)
    merged = pool_component_models((a, b), output=tmp_path / "pooled")
    assert len(merged.records) == 2
    foreign = model(tmp_path, "foreign", binding={**BINDING, "source_head": "d"*40})
    with pytest.raises(ValueError, match="binding"):
        pool_component_models((a, foreign), output=tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_pooled_evidence_rechecks_leaf_hash_and_reconciles_samples(tmp_path):
    a, b = model(tmp_path, "a"), model(tmp_path, "b", samples=(40,))
    merged = pool_component_models((a, b), output=tmp_path / "pooled")
    row = merged.records[0]
    path = tmp_path / "a-source.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="source.*hash"):
        merged.verify_receipts()
    path.write_text(path.read_text()[:-1])
    target = Path(row.evidence["source_receipt"])
    raw = json.loads(open(target).read())
    raw["samples_ns"] = [1, 2, 3]
    open(target, "w").write(json.dumps(raw))
    forged = CostModel(BINDING, (replace(row, samples_ns=(1,2,3),
        evidence={**row.evidence, "source_receipt_sha256":file_sha256(target)}),))
    with pytest.raises(ValueError, match="pooled.*samples"):
        forged.verify_receipts()


def test_no_fixture_costs_or_nested_aggregation_and_no_output_overwrite(tmp_path):
    a = model(tmp_path, "a")
    with pytest.raises(ValueError, match="measured"):
        pool_component_models((replace(a, scope="CPU_FIXTURE_ONLY"),), output=tmp_path / "fixture")
    merged = pool_component_models((a,), output=tmp_path / "pooled")
    with pytest.raises(FileExistsError):
        pool_component_models((a,), output=tmp_path / "pooled")
    with pytest.raises(ValueError, match="flat.*source"):
        pool_component_models((merged,), output=tmp_path / "nested")

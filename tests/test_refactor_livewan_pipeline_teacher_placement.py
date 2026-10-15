"""CPU contract fixtures only; never substitute for H100 capacity/parity."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from distillschedule.runtime.livewan_hardware import livewan_hardware
from distillschedule.runtime.livewan_pipeline_placement import (
    FIELD, PipelineTeacherPlacement, apply_contract, audit_run, from_plan,
    validate_gate_binding, verify_teacher,
)


def plan():
    return {"metadata": {"execution_mode": "pipeline"},
            "workload": {"shape": {"batch_size": 16}},
            "training_backends": {role: {"kind": "fsdp", "parameter_offload": "none",
                "options": {"reshard_after_forward": True}}
                for role in ("generator", "score", "teacher")}}


@pytest.fixture
def h100(monkeypatch):
    monkeypatch.setenv("DISTILLSCHEDULE_HARDWARE_PROFILE", "h100-8-b16")
    return livewan_hardware()


def test_default_preserves_plan_and_gate():
    value = plan(); before = copy.deepcopy(value)
    assert not apply_contract(value, mode="pipeline", name="legacy-sharded").enabled
    assert value == before
    validate_gate_binding({"arms": {"pipeline": {}}}, name="legacy-sharded", mode="pipeline")


def test_only_teacher_placement_changes(h100):
    value = plan(); before = copy.deepcopy(value)
    spec = apply_contract(value, mode="pipeline", name="full-resident")
    assert spec == from_plan(value)
    assert value["workload"] == before["workload"]
    for role in ("generator", "score"):
        assert value["training_backends"][role] == before["training_backends"][role]
    assert value["training_backends"]["teacher"] == {
        "kind": "native", "parameter_offload": "none",
        "options": {"reshard_after_forward": False}}
    assert spec.receipt()["hardware_profile"]["physical_world_size"] == 8


def test_real_c34_plan_keeps_existing_worker_contract_and_routing(h100, monkeypatch):
    from distillschedule.core.config import load_plan
    from distillschedule.runtime import livewan_crossscale_worker as worker

    root = Path(__file__).resolve().parents[1]
    for key in ("LIVEWAN_STUDENT_CHECKPOINT", "LIVEWAN_CRITIC_CHECKPOINT",
                "LIVEWAN_TEACHER_CHECKPOINT", "LIVEWAN_OPERATION_TRACE",
                "LIVEWAN_SYSTEMS_BOOTSTRAP", "DISTILLSCHEDULE_OUTPUT_DIR"):
        monkeypatch.setenv(key, "/not-loaded/" + key)
    path = root / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_c34_continuous_b16_residual_layout.json"
    value = load_plan(path).plan.to_dict()
    before = copy.deepcopy(value)
    worker._validate_plan_identity(value, mode="pipeline")
    original_ranks = worker._plan_role_ranks(value, mode="pipeline")
    apply_contract(value, mode="pipeline", name="full-resident")
    worker._validate_plan_identity(value, mode="pipeline")
    assert worker._plan_role_ranks(value, mode="pipeline") == original_ranks
    assert original_ranks == ((0, 1, 2, 3), (4, 5, 6, 7))
    # Compare the entire real plan, not a hand-picked subset of math fields.
    value["metadata"].pop(FIELD)
    value["training_backends"]["teacher"] = before["training_backends"]["teacher"]
    assert value == before


@pytest.mark.parametrize("change", ["baseline", "batch8", "offload", "A100"])
def test_rejects_out_of_scope_without_partial_mutation(h100, monkeypatch, change):
    value = plan(); mode = "pipeline"
    if change == "baseline":
        value["metadata"]["execution_mode"] = mode = "baseline"
    if change == "batch8": value["workload"]["shape"]["batch_size"] = 8
    if change == "offload": value["training_backends"]["teacher"]["parameter_offload"] = "cpu"
    if change == "A100": monkeypatch.setenv("DISTILLSCHEDULE_HARDWARE_PROFILE", "a100-16-b16")
    before = copy.deepcopy(value)
    with pytest.raises(ValueError): apply_contract(value, mode=mode, name="full-resident")
    assert value == before


def test_tampered_plan_and_legacy_gate_fail_closed(h100):
    value = plan(); spec = apply_contract(value, mode="pipeline", name="full-resident")
    gate = {"arms": {"pipeline": {FIELD: spec.receipt()}}}
    with pytest.raises(ValueError, match="actual Teacher storage"):
        validate_gate_binding(gate, name="full-resident", mode="pipeline")
    # A paired baseline can use its own arm, not the candidate's placement.
    validate_gate_binding(gate, name="legacy-sharded", mode="baseline")
    with pytest.raises(ValueError): validate_gate_binding(gate, name="legacy-sharded", mode="pipeline")
    with pytest.raises(ValueError): validate_gate_binding({}, name="full-resident", mode="pipeline")
    value["training_backends"]["teacher"]["kind"] = "fsdp"
    with pytest.raises(ValueError): from_plan(value)


def test_actual_storage_guard_rejects_trainable_cpu_wrong_dtype_and_shards(h100):
    spec = PipelineTeacherPlacement("full-resident")
    placement = {"placement": "gpu_resident_replica", "fsdp_group_count": 0, "mesh_ranks": []}
    def parameter(**kwargs):
        return SimpleNamespace(device=SimpleNamespace(type="cuda"), dtype="torch.bfloat16",
                               requires_grad=False, numel=lambda: 8, element_size=lambda: 2, **kwargs)
    p = parameter(); model = SimpleNamespace(parameters=lambda: iter([p]))
    assert verify_teacher(spec, model, rank=4, placement=placement)["local_storage_bytes"] == 16
    assert verify_teacher(spec, None, rank=0, placement=None)["status"] == "NOT_APPLICABLE"
    for key, invalid in (("device", SimpleNamespace(type="cpu")), ("dtype", "torch.float32"),
                         ("requires_grad", True), ("to_local", lambda: None)):
        p = parameter(); setattr(p, key, invalid)
        with pytest.raises(ValueError): verify_teacher(spec, model, rank=4, placement=placement)
    with pytest.raises(ValueError): verify_teacher(spec, model, rank=0, placement=placement)


def test_all_rank_gate_binds_storage_and_rejects_missing_or_sharded_teacher(h100, tmp_path):
    value = plan(); spec = apply_contract(value, mode="pipeline", name="full-resident")
    (tmp_path / "effective_plan.node0.json").write_text(json.dumps(value))
    (tmp_path / "receipts").mkdir()
    for rank in range(8):
        score = rank >= 4
        runtime = {"status": "PASS" if score else "NOT_APPLICABLE", "rank": rank}
        if score: runtime.update(storage_device="cuda", storage_dtype="bfloat16", fsdp_groups=0,
                                 logical_numel=8, local_storage_bytes=16)
        row = {"status": "PASS", "rank": rank, "build": {FIELD: spec.receipt(),
            FIELD + "_runtime": runtime,
            "teacher": {"placement": "gpu_resident_replica", "fsdp_group_count": 0,
                        "mesh_ranks": []} if score else None}}
        (tmp_path / "receipts" / f"rank_{rank:03d}.json").write_text(json.dumps(row))
    result = audit_run(tmp_path, mode="pipeline", hardware=h100)
    assert len(result["rank_bindings"]) == 8
    gate = {"arms": {"pipeline": {FIELD: spec.receipt(), "teacher_storage_binding": result}}}
    validate_gate_binding(gate, name="full-resident", mode="pipeline", hardware=h100)
    path = tmp_path / "receipts/rank_007.json"
    original = path.read_text()
    # Even a metadata-only edit invalidates the previously admitted file hash.
    row = json.loads(original); row["unexpected_new_field"] = 1
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="storage evidence changed"):
        validate_gate_binding(gate, name="full-resident", mode="pipeline", hardware=h100)
    path.write_text(original)
    row = json.loads(path.read_text()); row["build"]["teacher"]["fsdp_group_count"] = 45
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="not a full GPU replica"):
        audit_run(tmp_path, mode="pipeline", hardware=h100)
    path.unlink()
    with pytest.raises(FileNotFoundError): audit_run(tmp_path, mode="pipeline", hardware=h100)

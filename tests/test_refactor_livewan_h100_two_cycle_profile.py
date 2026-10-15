"""CPU gates for the existing two-cycle profile on registered H100 C34."""
import copy
import importlib.util
import json
from pathlib import Path
import sqlite3
from unittest import mock

import pytest

from distillschedule.runtime import livewan_two_cycle_profile as profile
from distillschedule.runtime.livewan_hardware import livewan_hardware

ROOT = Path(__file__).resolve().parents[1]


def analyzer():
    path = ROOT / "scripts/distillschedule/analyze_livewan_crossscale_nsys.py"
    spec = importlib.util.spec_from_file_location("h100_profile_analyzer_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explicit_hardware_profile_not_inferred(monkeypatch):
    monkeypatch.delenv("DISTILLSCHEDULE_HARDWARE_PROFILE", raising=False)
    assert profile.profile_variant() == profile.VARIANT
    monkeypatch.setenv("DISTILLSCHEDULE_HARDWARE_PROFILE", "h100-8-b16")
    assert profile.profile_variant() == profile.H100_VARIANT
    monkeypatch.setenv("DISTILLSCHEDULE_HARDWARE_PROFILE", "unregistered")
    with pytest.raises(ValueError):
        profile.profile_variant()


def test_h100_profile_only_changes_native_timer_metadata(monkeypatch):
    from distillschedule.core.config import load_plan
    from distillschedule.runtime.livewan_crossscale_worker import (
        _apply_two_cycle_timing_contract, _validate_plan_identity)
    monkeypatch.setenv("DISTILLSCHEDULE_HARDWARE_PROFILE", "h100-8-b16")
    for key in ("LIVEWAN_STUDENT_CHECKPOINT", "LIVEWAN_CRITIC_CHECKPOINT",
                "LIVEWAN_TEACHER_CHECKPOINT", "LIVEWAN_OPERATION_TRACE",
                "LIVEWAN_SYSTEMS_BOOTSTRAP", "DISTILLSCHEDULE_OUTPUT_DIR"):
        monkeypatch.setenv(key, "/unread/" + key)
    path = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_c34_continuous_b16_residual_layout.json"
    plan = load_plan(path).plan.to_dict()
    native, observed = copy.deepcopy(plan), copy.deepcopy(plan)
    _apply_two_cycle_timing_contract(native, mode="pipeline")
    profile.apply_contract(observed, mode="pipeline")
    _validate_plan_identity(observed, mode="pipeline")
    a, b = native["metadata"]["continuous_pipeline"], observed["metadata"]["continuous_pipeline"]
    assert b["name"] == profile.NAME and b["profiled"] and not b["formal_timing_eligible"]
    for key in ("name", "profiled", "formal_timing_eligible"):
        b[key] = a[key]
    assert native == observed


def test_h100_gate_requires_full_hardware_and_c34(monkeypatch, tmp_path):
    monkeypatch.setenv("DISTILLSCHEDULE_HARDWARE_PROFILE", "h100-8-b16")
    value = dict(schema_version=profile.SCHEMA, status="PASS_DIAGNOSTIC_ONLY",
                 profile_authorized=True, formal_timing_authorized=False, cycles=2,
                 pipeline_variant=profile.H100_VARIANT,
                 hardware_profile=livewan_hardware().to_dict(),
                 profile_source=dict(profile_head="b"*40, workload_head="a"*40, instrumentation_only=True),
                 native_gate=dict(path="original_gate", sha256="c"*64))
    args = dict(expected_head="b"*40, mode="pipeline", pipeline_variant=profile.H100_VARIANT,
                arm_id="same", seed=1, trace=None, bootstrap=None, source_receipt={})
    with mock.patch("distillschedule.runtime.livewan_crossscale_worker._load_two_cycle_timing_gate",
                    return_value={"status": "NATIVE_GATE_UNCHANGED"}) as load:
        p = tmp_path / "valid.json"
        p.write_text(json.dumps(value))
        result = profile.load_gate(path=p, expected_sha256=profile.sha256(p), **args)
        assert result["status"] == "PASS_DIAGNOSTIC_ONLY" and not result["formal_timing_eligible"]
        assert load.call_args.kwargs["expected_head"] == "a"*40
        assert load.call_args.kwargs["pipeline_variant"] == profile.H100_VARIANT
        for key, change in (("hardware_profile", {}), ("pipeline_variant", profile.VARIANT),
                            ("formal_timing_authorized", True), ("cycles", 3)):
            bad = {**value, key: change}
            p.write_text(json.dumps(bad))
            with pytest.raises(RuntimeError, match="identity"):
                profile.load_gate(path=p, expected_sha256=profile.sha256(p), **args)


def test_h100_operator_gates_keep_role_and_multiplicity():
    hardware = livewan_hardware("h100-8-b16")
    targets = [
        "livewan_generator_rms_epilogue", "livewan_rollout_cast_cache", "livewan_score_rmsnorm",
        "livewan_generator_modulation", "livewan_generator_gate", "livewan_attention_checkpoint"]
    with __import__("contextlib").ExitStack() as stack:
        mocked = {name:stack.enter_context(mock.patch(
            "distillschedule.runtime."+name+".runtime_receipt_errors", return_value=[])) for name in targets}
        assert profile.profile_operator_errors({}, rank=4, hardware=hardware) == []
        for call in mocked.values():
            assert call.call_args.kwargs["rank"] == 8  # H100 Score rank4 must not look like G.
        assert mocked[targets[0]].call_args.kwargs["cycles"] == 4
        assert mocked[targets[0]].call_args.kwargs["fuse_cast_square"] is True
        assert mocked[targets[4]].call_args.kwargs["output_layout"] == "row_major"
        assert mocked[targets[5]].call_args.kwargs["expected_calls"] == 600


def make_db(path, *, ranks=range(8), enums=True, unresolved=False):
    db=sqlite3.connect(path)
    db.executescript("""
CREATE TABLE StringIds(id INTEGER,value TEXT);
CREATE TABLE NVTX_EVENTS(text TEXT,textId INTEGER,start INTEGER,end INTEGER,globalTid INTEGER);
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,globalPid INTEGER,demangledName INTEGER);
CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,deviceId INTEGER,globalPid INTEGER,bytes INTEGER,copyKind INTEGER);
""")
    db.executemany("INSERT INTO StringIds VALUES (?,?)", [(1,"gemm"), (2,"ncclAllGather"), (3,"UNRESOLVED_KERNEL")])
    if enums:
        db.execute("CREATE TABLE ENUM_CUDA_MEMCPY_OPER(id INTEGER,name TEXT,label TEXT)")
        db.executemany("INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES (?,?,?)",
                       [(1,"HtoD","Host-to-Device"),(2,"DtoH","Device-to-Host"),
                        (3,"HtoA","Host-to-Array"),(8,"DtoD","Device-to-Device")])
    for rank in ranks:
        pid=(1000+rank)<<24
        label=f"livewan/full5/pipeline/rank_{rank}"
        db.execute("INSERT INTO StringIds VALUES (?,?)", (100+rank,label))
        db.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?)", (None,100+rank,0,100,pid+1))
        phase="livewan/pipeline/g/test" if rank<4 else "livewan/pipeline/score/test"
        db.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?)",(phase,None,0,100,pid+1))
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?)",(10,30,rank,pid,3 if unresolved and rank==0 else 1))
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?)",(20,40,rank,pid,2))
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,?,?,?,?)",(50,60,rank,pid,64,1))
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,?,?,?,?)",(70,71,rank,pid,32,3))
    db.commit();db.close()


def test_h100_analyzer_keeps_four_four_roles_and_enum_directions(tmp_path):
    module=analyzer();db=tmp_path/"node.sqlite";make_db(db)
    node=module.analyze_node(db,0,physical_world_size=8)
    assert node['copy_reconciliation']['residual']==dict(events=0,bytes=0,sum_duration_ns=0)
    assert node['copy_reconciliation']['raw']['bytes']==8*96
    result=module.aggregate([node],physical_world_size=8)
    assert result["rank_count"]==8
    assert [r["role"] for r in result["per_rank"]]==["generator"]*4+["score"]*4
    assert len(result["phases"])==2 and all(r["rank_count"]==4 for r in result["phases"])
    assert result["copies"]["h2d"]["bytes"]==8*64 and result["copies"]["other"]["bytes"]==8*32
    assert result["copies"]["d2d"]["bytes"]==0  # Enum3 is not D2D.
    for row in result["per_rank"]:
        assert row["wall_ns"]==100 and row["compute_union_ns"]==20
        assert row["nccl_exposed_vs_compute_ns"]==10 and row["no_traced_gpu_activity_ns"]==59


@pytest.mark.parametrize("ranks,enums,error",[(range(7),True,"outer rank coverage"),(range(8),False,"direction enums")])
def test_h100_analyzer_rejects_partial_capture(tmp_path,ranks,enums,error):
    module=analyzer();db=tmp_path/"node.sqlite";make_db(db,ranks=ranks,enums=enums)
    with pytest.raises(RuntimeError,match=error):
        module.analyze_node(db,0,physical_world_size=8)


def test_h100_cannot_duplicate_node_as_world16(tmp_path):
    module=analyzer();db=tmp_path/"node.sqlite";make_db(db)
    node=module.analyze_node(db,0,physical_world_size=8)
    with pytest.raises(ValueError):module.aggregate([node],physical_world_size=16)
    with pytest.raises(ValueError):module.analyze_node(db,1,physical_world_size=8)


def test_literal_unresolved_kernel_is_not_silently_valid(tmp_path):
    module=analyzer();db=tmp_path/"node.sqlite";make_db(db,unresolved=True)
    node=module.analyze_node(db,0,physical_world_size=8)
    assert node["ranks"][0]["unresolved_kernel_duration_ratio"] >= .01

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from distillschedule.search.contracts import MEASUREMENT_SCHEMA, PROFILE_SCHEMA, semantic_contract, validate_measurement
from distillschedule.search.registry import WAN, LIVEWAN, catalog, candidates, candidate_profile, get_candidate, search_identity
from distillschedule.search.profile import parse_profile, validate_profile
from distillschedule.search.engine import SearchSession
from distillschedule.search.execution import resolve_candidate, validated_profile
from distillschedule.workloads.best_known.registry import fingerprint, load_profile, file_sha256


HARDWARE = {"nodes": 2, "world_size": 16, "gpu": "A100-SXM4-40GB"}
IDENTITY = search_identity(WAN, source_head="a" * 40, assets={"test": "UNIT_TEST_ONLY"}, hardware=HARDWARE)


def profile_fixture():
    rows = []
    for rank in range(16):
        categories = {k: {"union_ns": v, "bytes": 0, "events": 1} for k, v in
                      {"compute": 50, "nccl": 50, "h2d": 5, "d2h": 0, "active": 90,
                       "no_traced_gpu_activity": 10}.items()}
        rows.append({"rank": rank, "phase": "generator", "metrics": {"wall_ns": 100, "categories": categories},
                     "nccl_not_covered_by_compute_ns": 40, "quality": {"pointwise_name_family_union_ns": 10}})
    return {"schema_version": PROFILE_SCHEMA, "status": "PASS", "cycle": 2,
            "identity": IDENTITY, "hardware": HARDWARE, "sources": [{"path": "UNIT_TEST_ONLY"}], "rows": rows}


def measurement(c, seconds=50, numerical="REVIEW"):
    return {"schema_version": MEASUREMENT_SCHEMA, "status": "PASS", **IDENTITY,
            "candidate_id": c.id, "execution_fingerprint": fingerprint(c.to_dict()),
            "profiled": False, "same_process": True, "initializations": 1,
            "semantics_status": "PASS", "numerical_status": numerical,
            "profiler_status": "DISABLED", "resource_status": "EXCLUSIVE_LOCKED", "cleanup_status": "PASS",
            "timer_scope": "NATIVE_CYCLE_OUTSIDE_PROBES", "native_evidence": ["UNIT_TEST_NOT_GPU_EVIDENCE"],
            "cycle_wall_seconds": {"1": seconds + 10, "2": seconds},
            "ranks": [{"rank": r, "status": "PASS", **{k: "PASS" for k in
                        ("census", "versions", "sample_rng", "finite", "dtype", "memory")},
                       "cuda_oom_delta": 0, "cgroup_oom_delta": 0, "peak_allocated_bytes": 10,
                       "peak_reserved_bytes": 12, "min_driver_free_bytes": 20, "host_peak_bytes": 30}
                      for r in range(16)]}


def test_five_profiles_keep_locked_hashes_and_semantics():
    registered = catalog()
    assert len(registered) == 5
    locks = json.loads((ROOT / "configs/distillschedule/best_known/registry.lock.json").read_text())["profiles"]
    for item in registered:
        name = item["workload"]
        before = load_profile(name)
        for c in candidates(name):
            after = candidate_profile(c)
            assert semantic_contract(before) == semantic_contract(after)
        assert fingerprint(load_profile(name)) == locks[name]


def test_livewan_rejected_experiments_are_not_accidentally_cumulative():
    c28 = get_candidate(LIVEWAN, "c28")
    assert c28.parent == "c25"
    assert not any("pinned_prompt" in k or "critic_fa2" in k for k in c28.kernels)
    assert any("critic_fa2" in k for k in get_candidate(LIVEWAN, "c34").kernels)


def test_livewan_baseline_and_pipeline_are_real_native_plans():
    for key in ("fsdp-reference", "c24", "c34"):
        receipt = resolve_candidate(LIVEWAN, key)
        assert receipt["canonical_status"] != "CANONICAL"
        p = validated_profile(receipt, root=ROOT)
        assert receipt["native_plan"]["digest"] == p["runtime"]["typed_plan_template_sha256"]
        assert p["runtime"]["arguments"][3] == ("baseline" if key == "fsdp-reference" else "pipeline")


@pytest.mark.parametrize("path", ["search_candidate", "fully_resolved_profile", "profile_sha256", "workload_semantics_hash"])
def test_serialized_search_plan_tampering_rejected(path):
    value = resolve_candidate(WAN, "resident-lora-epilogue")
    value[path] = {} if isinstance(value[path], dict) else "tampered"
    with pytest.raises((ValueError, KeyError)):
        validated_profile(value, root=ROOT)


@pytest.mark.parametrize("field,new", [("source_head", "b" * 40), ("profiled", True),
    ("initializations", 2), ("semantics_status", "REVIEW"), ("numerical_status", "FAIL"),
    ("profiler_status", "UNKNOWN"), ("cleanup_status", "FAIL"), ("resource_status", "IDLE_ONLY")])
def test_native_measurement_fail_closed(field, new):
    c = get_candidate(WAN, "fsdp-reference")
    value = measurement(c)
    value[field] = new
    with pytest.raises(ValueError):
        validate_measurement(value, IDENTITY, c)


@pytest.mark.parametrize("change", ["missing_rank", "duplicate_rank", "oom", "dtype", "infinite_time"])
def test_runtime_gates_cannot_be_waived_by_numeric_review(change):
    c = get_candidate(WAN, "fsdp-reference")
    value = measurement(c)
    if change == "missing_rank": value["ranks"].pop()
    elif change == "duplicate_rank": value["ranks"][0]["rank"] = 1
    elif change == "oom": value["ranks"][0]["cgroup_oom_delta"] = 1
    elif change == "dtype": value["ranks"][0]["dtype"] = "FAIL"
    else: value["cycle_wall_seconds"]["2"] = float("nan")
    with pytest.raises(ValueError): validate_measurement(value, IDENTITY, c)


def test_review_and_strict_pass_are_separate():
    c = get_candidate(WAN, "fsdp-reference")
    assert validate_measurement(measurement(c), IDENTITY, c)["strict_eligible"] is False
    assert validate_measurement(measurement(c, numerical="PASS"), IDENTITY, c)["strict_eligible"] is True


def test_budget_confirmation_and_resume_are_bounded(tmp_path):
    calls = []
    def execute(c, phase, path):
        calls.append((c.id, phase))
        return measurement(c, 100 if phase == "baseline" else 50, "PASS" if phase == "baseline" else "REVIEW")
    s = SearchSession(workload=WAN, identity=IDENTITY, profile=profile_fixture(), output=tmp_path / "run", max_trials=2)
    result = s.run(execute)
    assert len(calls) == 4
    assert result["candidate_attempts"] == 2
    assert result["cycle2_speedup"] == 2
    assert result["strict_winner"]["candidate_id"] == "fsdp-reference"
    assert not result["diagnostic_winner"]["strict_eligible"]
    resumed = SearchSession(workload=WAN, identity=IDENTITY, profile=profile_fixture(), output=tmp_path / "run", max_trials=2, resume=True)
    assert resumed.run(execute) == result
    assert len(calls) == 4
    with pytest.raises(ValueError, match="different|historical-prior"):
        SearchSession(workload=WAN, identity={**IDENTITY, "source_head": "c" * 40}, profile=profile_fixture(), output=tmp_path / "run", max_trials=2, resume=True)


def test_crash_is_preserved_not_silently_retried(tmp_path):
    s = SearchSession(workload=WAN, identity=IDENTITY, profile=profile_fixture(), output=tmp_path / "run")
    def crash(*args): raise RuntimeError("simulated controller loss")
    with pytest.raises(RuntimeError): s.run(crash)
    with pytest.raises(ValueError, match="incomplete"):
        s.run(lambda c, *_: measurement(c))
    assert sum(e["type"] == "START" for e in s.events()) == 1


def test_nccl_failure_stops_but_capacity_failure_consumes_budget(tmp_path):
    def execute(c, phase, _):
        if phase == "candidate":
            return {"status": "FAIL", "failure_type": "CUDA_OOM", "native_evidence": ["UNIT_TEST_ONLY"]}
        return measurement(c)
    s = SearchSession(workload=WAN, identity=IDENTITY, profile=profile_fixture(), output=tmp_path / "capacity", max_trials=2)
    result = s.run(execute)
    assert result["candidate_attempts"] == 2 and result["diagnostic_winner"]["candidate_id"] == "fsdp-reference"
    def nccl(*_): return {"status": "FAIL", "failure_type": "NCCL_ERROR", "native_evidence": ["UNIT_TEST_ONLY"]}
    s = SearchSession(workload=WAN, identity=IDENTITY, profile=profile_fixture(), output=tmp_path / "nccl")
    with pytest.raises(RuntimeError, match="NCCL_ERROR"): s.run(nccl)


def sqlite_fixture(tmp_path):
    ranks = []
    for node in range(2):
        path = tmp_path / f"node{node}.sqlite"
        with sqlite3.connect(path) as db:
            db.executescript('''CREATE TABLE StringIds(id INTEGER PRIMARY KEY,value TEXT);
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INT,end INT,deviceId INT,globalPid INT,demangledName INT);
CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INT,end INT,deviceId INT,globalPid INT,copyKind INT,bytes INT);
CREATE TABLE NVTX_EVENTS(start INT,end INT,globalTid INT,text TEXT,textId INT);''')
            db.executemany("INSERT INTO StringIds VALUES(?,?)", [(1, "norm_kernel"), (2, "nccl_AllGather")])
            for device in range(8):
                rank = node * 8 + device
                pid = (100 + rank) << 24
                tid = pid + 123
                db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?)",
                               [(1, 41, device, pid, 1), (21, 71, device, pid, 2)])
                db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(?,?,?,?,?,?)", (61, 81, device, pid, 1, 1024))
                db.execute("INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,?)", (0, 100, tid, "cycle2", None))
                window = {"start_ns": 0, "end_ns": 100, "global_tid": tid, "nvtx_text": "cycle2", "coverage": "exact"}
                ranks.append({"node": node, "rank": rank, "device_id": device, "global_pid": pid,
                              "sqlite": str(path), "parser_status": "PASS", "parser_errors": [],
                              "capture": window, "phases": [{**window, "name": "generator", "role": "generator"}]})
        digest = file_sha256(path)
        doctor = tmp_path / f"doctor_node{node}.json"
        doctor.write_text(json.dumps({"producer": "nsys-ai", "producer_version": "UNIT_TEST_ONLY",
            "profile_path": str(path), "sections": [{"name": "Profile health", "checks": [
                {"name": n, "status": "ok"} for n in ("Schema compatibility", "GPUs", "NVTX events")]}]}))
        for r in ranks:
            if r["node"] == node:
                r["sqlite_sha256"] = digest
                r["nsys_ai_doctor"] = {"path": str(doctor), "sha256": file_sha256(doctor)}
    return {"schema_version": "distillschedule.search.profile_input.v1", "identity": IDENTITY,
            "cycle": 2, "hardware": HARDWARE, "ranks": ranks}


def test_sqlite_per_device_union_pid_and_reconciliation(tmp_path):
    manifest = sqlite_fixture(tmp_path)
    result = parse_profile(manifest)
    assert len(result["rows"]) == 16
    for row in result["rows"]:
        cats = row["metrics"]["categories"]
        assert cats["compute"]["union_ns"] == 40
        assert cats["nccl"]["union_ns"] == 50
        assert cats["active"]["union_ns"] == 80
        assert row["nccl_not_covered_by_compute_ns"] == 30
        assert cats["h2d"]["bytes"] == 1024
        assert row["quality"]["pointwise_name_family_union_ns"] == 40
    assert result["claim_boundary"].startswith("DIAGNOSTIC")


@pytest.mark.parametrize("bad", ["missing_rank", "duplicate_gpu", "parser", "bad_tid", "fake_nvtx", "nested_phase", "doctor", "hash"])
def test_profile_invalid_evidence_rejected(tmp_path, bad):
    v = sqlite_fixture(tmp_path)
    if bad == "missing_rank": v["ranks"].pop()
    elif bad == "duplicate_gpu": v["ranks"][0]["device_id"] = 1
    elif bad == "parser": v["ranks"][0]["parser_errors"] = ["ProcessEventsError"]
    elif bad == "bad_tid": v["ranks"][0]["capture"]["global_tid"] = 123
    elif bad == "fake_nvtx": v["ranks"][0]["capture"]["nvtx_text"] = "invented"
    elif bad == "nested_phase": v["ranks"][0]["phases"].append(v["ranks"][0]["phases"][0])
    elif bad == "doctor": v["ranks"][0]["nsys_ai_doctor"]["sha256"] = "0" * 64
    else: v["ranks"][0]["sqlite_sha256"] = "0" * 64
    with pytest.raises(ValueError): parse_profile(v)


def test_cli_cpu_only_no_torch_or_gpu_import():
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    command = [sys.executable, str(ROOT / "scripts/distillctl.py"), "workloads", "search",
               "--workload", "minimax_h3_grid1000_best", "--dry-run"]
    run = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    value = json.loads(run.stdout)
    assert value["status"] == "PASS_CPU_PLAN_ONLY" and value["gpu"] == "NOT_RUN"
    assert value["registration"]["pilot"] is False


def test_wan_legacy_arm_defaults_remain_unchanged():
    from distillschedule.search.wan_contract import arm, registered_placements
    with patch.dict(os.environ, {}, clear=True):
        assert len(registered_placements()) == 2
        assert arm("resident-async128-tree-chunk2048")["fused"] is True
        with pytest.raises(ValueError): arm("resident-async128-tree")
    with patch.dict(os.environ, {"DISTILLSCHEDULE_SEARCH_CANDIDATE_ID": "resident-chunk2048"}):
        assert arm("resident-async128-tree-chunk2048")["fused"] is False
        with pytest.raises(ValueError): arm("world16-fsdp")

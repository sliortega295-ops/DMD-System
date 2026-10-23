"""Controller tests use synthetic receipts; they never launch GPU workers."""
from __future__ import annotations

import json

import pytest

from distillschedule.search.contracts import MEASUREMENT_SCHEMA
from distillschedule.search.model_search import search_plans
from distillschedule.search.model_validation import ModelValidationSession
from distillschedule.search.wan_implementation import WanImplementation
from distillschedule.workloads.best_known.registry import fingerprint, git_head, repository_root
from distillschedule.search.runtime_identity import SCHEMA, observed_binding
from test_refactor_wan_model_adapter import calibration_fixture


def fake_observations():
    return [{"schema_version": SCHEMA, "status": "PASS", "source_head": git_head(repository_root()),
        "node_rank": n, "cuda_context_created": False,
        "environment": {k: "CPU_TEST_ONLY" for k in ("python", "executable", "executable_sha256", "packages",
            "torch", "torch_cuda", "nccl", "torch_extension_sha256")},
        "hardware": {"topology": "CPU_TEST_ONLY", "gpus": [{"device": i, "uuid": f"test-{n}-{i}",
            "name": "CPU_TEST", "driver": "test", "total_bytes": 1000} for i in range(8)]}}
        for n in range(2)]


def session(tmp_path, *, budget=8, resume=False):
    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    binding = observed_binding(fake_observations(), source_head=git_head(repository_root()))
    problem, costs = calibration_fixture(inputs, binding)
    result = search_plans(problem, costs, exhaustive=True, max_proposals=12)
    return ModelValidationSession(problem=problem, costs=costs, prediction=result,
        output=tmp_path / "validation", max_trials=budget, resume=resume)


def measurement(controller, candidate, *, seconds=10, numerical="REVIEW"):
    return {"schema_version": MEASUREMENT_SCHEMA, "status": "PASS", **controller.identity,
        "candidate_id": candidate.id, "execution_fingerprint": fingerprint(candidate.to_dict()),
        "profiled": False, "same_process": True, "initializations": 1,
        "semantics_status": "PASS", "numerical_status": numerical,
        "ranks": [{"rank": i, "status": "PASS", **{k: "PASS" for k in
            ("census", "versions", "sample_rng", "finite", "dtype", "memory")},
            "cuda_oom_delta": 0, "cgroup_oom_delta": 0,
            "peak_allocated_bytes": 100, "peak_reserved_bytes": 200,
            "min_driver_free_bytes": 100, "host_peak_bytes": 300} for i in range(16)],
        "profiler_status": "DISABLED", "resource_status": "EXCLUSIVE_LOCKED",
        "cleanup_status": "PASS", "cycle_wall_seconds": {"1": 100, "2": seconds},
        "timer_scope": "NATIVE_CYCLE_OUTSIDE_PROBES", "native_evidence": ["CPU_TEST_ONLY"],
        "runtime_identity_evidence": {"before": fake_observations(), "after": fake_observations()}}


def test_total_budget_includes_anchors_and_confirmation_and_does_not_average(tmp_path):
    s = session(tmp_path)
    calls = []

    def execute(c, phase, directory):
        calls.append((c.id, phase))
        return measurement(s, c, seconds=5 if "c1024-e1" in c.id else 10)

    r = s.run(execute)
    assert len(calls) == 8
    assert [p for _, p in calls] == ["baseline", "incumbent", *["candidate"] * 5, "confirmation"]
    assert calls[0][0] == WanImplementation("fsdp", 512, False).id
    assert calls[1][0] == WanImplementation("resident", 2048, True).id
    assert calls[-1][0] == WanImplementation("resident", 1024, True).id
    assert r["cycle2_speedup_diagnostic"] == 2
    assert r["strict_winner"] is None
    assert r["timing_attempts"] == 8
    assert len(r["prediction_errors"]) == 8
    assert s.run(lambda *args: pytest.fail("completed run restarted")) == r


def test_minimum_budget_still_measures_both_seeds_and_confirms(tmp_path):
    s = session(tmp_path, budget=3)
    result = s.run(lambda c, phase, out: measurement(s, c))
    assert result["timing_attempts"] == 3 and result["new_candidate_attempts"] == 0
    with pytest.raises(ValueError, match="budget"):
        session(tmp_path / "bad", budget=9)


def test_non_capacity_failure_and_interrupted_trial_cannot_be_retried(tmp_path):
    s = session(tmp_path)

    def error(*args):
        raise ConnectionError("lost SSH, not CUDA OOM")

    with pytest.raises(ConnectionError):
        s.run(error)
    assert len([e for e in s.events() if e["type"] == "START"]) == 1
    with pytest.raises(ValueError, match="incomplete"):
        s.run(lambda *args: pytest.fail("must not relaunch"))
    assert list(s.output.glob("trials/*/controller-failure.json"))


def test_tampered_measurement_is_preserved_and_does_not_release_budget(tmp_path):
    s = session(tmp_path)

    def execute(c, phase, out):
        return {**measurement(s, c), "precision_fingerprint": "b" * 64}

    with pytest.raises(ValueError, match="precision_fingerprint"):
        s.run(execute)
    assert list(s.output.glob("trials/*/executor-receipt.json"))
    with pytest.raises(ValueError, match="incomplete"):
        s.run(execute)


def test_echoing_expected_hardware_environment_without_observation_is_rejected(tmp_path):
    s = session(tmp_path)
    candidate = s.candidates[s.seeds[0]]
    raw = measurement(s, candidate)
    raw.pop("runtime_identity_evidence", None)
    with pytest.raises(ValueError, match="observed runtime"):
        s._validate_raw(raw, candidate)


def test_changed_python_or_gpu_rejected_before_native_preparation(tmp_path, monkeypatch):
    from distillschedule.search.execution import NativeTrialRunner
    s = session(tmp_path)
    assets = {"assets": {"test": "CPU_ONLY"}}
    identity = {**s.identity, "asset_fingerprint": fingerprint(assets["assets"])}
    runner = NativeTrialRunner(workload=s.problem.workload, identity=identity, assets=assets,
        output=tmp_path, run_prefix="CPU_TEST", bootstrap_peer=s.seeds[1], reference_id=s.seeds[0])
    calls = []

    def rpc(action, **kwargs):
        if action == "resolve":
            return {"nodes": []}  # CPU test does not execute a command
        assert action == "runtime_identity"
        calls.append(kwargs["_node_rank"])
        node = fake_observations()[kwargs["_node_rank"]]
        node["environment"]["executable"] = "/different/unvalidated/python"
        return node

    monkeypatch.setattr(runner, "_rpc", rpc)
    monkeypatch.setattr(runner, "_prepare", lambda *a: pytest.fail("must reject before GPU preparation"))
    with pytest.raises(ValueError, match="environment_fingerprint"):
        runner(s.candidates[s.seeds[0]], "baseline", tmp_path)
    assert calls == [0, 1]


def test_identity_probe_uses_launch_import_environment_not_inherited_ssh(monkeypatch):
    from distillschedule.search.runtime_identity import wan_probe_environment
    monkeypatch.setenv("PYTHONPATH", "/untrusted/shadow-torch")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/inherited/other-cuda")
    node = {"python": "/pinned/bin/python", "repo_root": "/source",
        "environment": {"DMD_SYSTEM_ROOT": "/source", "DISTILLSCHEDULE_PYENV": "/pinned",
            "HELIOS_RUNTIME_ROOT": "/source/helios", "PYTHONPATH": "/source/src:/source:/source/helios"}}
    env = wan_probe_environment(node)
    assert env["PYTHONPATH"] == "/source/src:/source:/source/helios"
    assert env["LD_LIBRARY_PATH"] == "/pinned/lib"
    assert "shadow-torch" not in str(env) and "other-cuda" not in str(env)


def test_gpu_identity_change_after_execution_is_not_an_accepted_measurement(tmp_path):
    s = session(tmp_path)
    c = s.candidates[s.seeds[0]]
    raw = measurement(s, c)
    raw["runtime_identity_evidence"]["after"][1]["hardware"]["gpus"][0]["uuid"] = "other-gpu"
    with pytest.raises(ValueError, match="hardware_fingerprint"):
        s._validate_raw(raw, c)


def test_livewan_identity_uses_actual_native_loader_environment(monkeypatch):
    import sys
    from distillschedule.search import execution, runtime_identity
    from distillschedule.workloads.best_known.dispatch import clean_environment
    root = repository_root()
    head = git_head(root)
    node = {"repo_root": str(root), "python": sys.executable,
            "environment": {"PYTHONPATH": "/explicit/native/import/path"}}
    receipt = {"profile_id": "livewan_crossscale_pipeline_best",
               "integration_head": head, "nodes": [node, node],
               "asset_config": {"nodes": [{}, {}]}}
    monkeypatch.setenv("LD_LIBRARY_PATH", "/inherited/wrong-libraries")
    monkeypatch.setattr(execution, "validated_profile", lambda *a, **kw: {})
    original = runtime_identity.subprocess.check_output
    calls = []

    def output(command, **kwargs):
        if command[0] == sys.executable:
            calls.append(kwargs["env"])
            return '{"status":"CPU_TEST_ONLY"}'
        return original(command, **kwargs)

    monkeypatch.setattr(runtime_identity.subprocess, "check_output", output)
    runtime_identity.collect_planned_node_identity(receipt, node_rank=0)
    assert calls == [clean_environment(node)]
    assert "LD_LIBRARY_PATH" not in calls[0]


def test_resume_revalidates_completed_receipts_not_only_cached_result(tmp_path):
    s = session(tmp_path, budget=3)
    s.run(lambda c, phase, out: measurement(s, c))
    path = next(s.output.glob("trials/*/executor-receipt.json"))
    raw = json.loads(path.read_text())
    raw["cycle_wall_seconds"]["2"] = 0.01
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="receipt changed"):
        s.run(lambda *args: pytest.fail("do not execute"))


def test_capacity_only_skips_new_candidate_with_source_bound_failure(tmp_path):
    s = session(tmp_path, budget=4)

    def execute(c, phase, out):
        if phase == "candidate":
            return {"status": "FAIL", "failure_type": "CUDA_OOM", **s.identity,
                "candidate_id": c.id, "native_evidence": ["CPU_TEST_NOT_ACTUAL_OOM"]}
        return measurement(s, c, numerical="PASS")

    result = s.run(execute)
    assert result["timing_attempts"] == 4
    assert result["capacity_failures"] == 1
    assert result["strict_winner"] is None  # no faster configuration was confirmed

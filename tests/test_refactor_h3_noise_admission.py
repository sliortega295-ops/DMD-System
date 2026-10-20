"""Explicit diagnostic admission never changes the original sampled comparator."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

import pytest

from distillschedule.runtime.p0_parity import DEFAULT_P0_PARITY
from distillschedule.search.h3_implementation import H3, H3Implementation
from distillschedule.search.h3_request import H3Request
from distillschedule.search.h3_native import compare_native_two_cycle
from test_refactor_h3_search_native import HEAD, _receipts, _with_rng
from distillschedule.search.h3_noise_admission import qualify, validate_policy
from distillschedule.workloads.best_known.registry import file_sha256


REF = H3Implementation(request=H3Request(480, 832), cache_policy="cpu_staged_trim")
CAND = H3Implementation(request=REF.request, cache_policy="cuda_no_trim")


def policy():
    return dict(schema_version="distillschedule.h3_noise_diagnostic_policy.v1",
                source_head=HEAD, reference_candidate_id=REF.id, candidate_id=CAND.id,
                max_cross_self_delta_l2_ratio=2.0, purpose="PLANNER_DIAGNOSTIC_ONLY",
                formal_timing_eligible=False, default_gate_unchanged=True)


def perturb(rows, amount):
    for rank in rows:
        for cycle in rank["cycles"]:
            for values in cycle["losses"].values():
                values[:] = [v + amount for v in values]
        for component in rank["state_fingerprints"]["components"].values():
            for row in component["rows"]:
                row["values_fp32"] = [v + amount for v in row["values_fp32"]]
            component["sample_sha256"] = hashlib.sha256(json.dumps(
                component["rows"], sort_keys=True, allow_nan=False).encode()).hexdigest()
    return rows


def inputs():
    return (_with_rng(_receipts(REF)), perturb(_with_rng(_receipts(REF)), .02),
            perturb(_with_rng(_receipts(CAND)), .022))


def test_noise_qualified_is_review_and_default_gate_still_fails():
    ref, repeat, cand = inputs()
    before = compare_native_two_cycle(ref, cand, expected_head=HEAD)
    assert before["status"] == "FAIL"
    result = qualify(ref, repeat, cand, policy=policy(), expected_head=HEAD)
    assert result["status"] == "REVIEW_NOISE_QUALIFIED_DIAGNOSTIC"
    assert result["diagnostic_search_allowed"] is True
    assert result["numerical_status"] == "REVIEW"
    assert result["formal_timing_eligible"] is False
    assert result["full_state_comparison"] == "NOT_RUN"
    assert result["strict_comparison"] == before
    assert result["strict_control_repeat_comparison"]["status"] == "FAIL"
    assert result["strict_comparison"]["thresholds"] == asdict(DEFAULT_P0_PARITY)
    assert all(m["cross_self_delta_l2_ratio"] == pytest.approx(1.1) for m in result["metrics"])
    assert compare_native_two_cycle(ref, cand, expected_head=HEAD) == before


@pytest.mark.parametrize("repeat_amount,candidate_amount,allowed", [
    (.02, .06, False), (0., .01, False), (0., 0., True), (.02, .04, True)])
def test_diagnostic_bound_and_zero_noise(repeat_amount, candidate_amount, allowed):
    ref = _with_rng(_receipts(REF))
    repeat = perturb(_with_rng(_receipts(REF)), repeat_amount)
    cand = perturb(_with_rng(_receipts(CAND)), candidate_amount)
    result = qualify(ref, repeat, cand, policy=policy(), expected_head=HEAD)
    assert result["diagnostic_search_allowed"] is allowed
    assert result["formal_timing_eligible"] is False


@pytest.mark.parametrize("mutation", ["source", "ratio", "purpose", "formal", "extra", "kernel", "reference"])
def test_policy_is_source_bound_and_cache_only(mutation):
    p = policy()
    if mutation == "source": p["source_head"] = "b" * 40
    if mutation == "ratio": p["max_cross_self_delta_l2_ratio"] = 10.
    if mutation == "purpose": p["purpose"] = "FORMAL_TIMING"
    if mutation == "formal": p["formal_timing_eligible"] = True
    if mutation == "extra": p["skip_rng"] = True
    if mutation == "kernel": p["candidate_id"] = H3Implementation(2, 64, cache_policy="cuda_no_trim", request=REF.request).id
    if mutation == "reference": p["reference_candidate_id"] = CAND.id
    with pytest.raises(ValueError): validate_policy(p, expected_head=HEAD)


@pytest.mark.parametrize("mutation", ["sample", "rng", "missing_state", "coordinate", "nonfinite", "repeat_config"])
def test_noise_policy_cannot_excuse_changed_semantics_or_missing_evidence(mutation):
    ref, repeat, cand = inputs()
    if mutation == "sample": cand[0]["cycles"][0]["census"]["samples"][0]["identity"] = "other"
    if mutation == "rng": cand[0]["cycles"][0]["rng_after"]["torch_cuda"] = "f" * 64
    if mutation == "missing_state": cand[0].pop("state_fingerprints")
    if mutation == "repeat_config": repeat = _with_rng(_receipts(CAND))
    if mutation in ("coordinate", "nonfinite"):
        component = cand[0]["state_fingerprints"]["components"]["student_parameters"]
        if mutation == "coordinate": component["rows"][0]["local_flat_indices"][-1] = 9
        else: component["rows"][0]["values_fp32"][0] = float("inf")
        component["sample_sha256"] = hashlib.sha256(json.dumps(
            component["rows"], sort_keys=True).encode()).hexdigest()
    with pytest.raises((ValueError, KeyError)):
        qualify(ref, repeat, cand, policy=policy(), expected_head=HEAD)


def witnessed(tmp_path, monkeypatch):
    from pathlib import Path
    from distillschedule.search import h3_native_bridge
    groups = inputs()
    value = qualify(*groups, policy=policy(), expected_head=HEAD)
    value["native_evidence"] = {}
    value["execution_plans"] = {}
    checked = {}
    for name, rows in zip(("reference", "repeat", "candidate"), groups):
        directory = tmp_path / name / "execution" / "ranks"
        directory.mkdir(parents=True)
        refs = []
        for row in rows:
            path = directory / f"rank_{row['rank']:03d}.json"
            path.write_text(json.dumps(row))
            refs.append(dict(path=str(path), sha256=file_sha256(path)))
        value["native_evidence"][name] = refs
        impl = CAND if name == "candidate" else REF
        plan = dict(integration_head=HEAD, mode="correctness", run_id=name, profile_id=H3,
                    asset_fingerprint="assets", workload_semantics_hash="semantics",
                    asset_config=dict(run_root=str(tmp_path)), search_candidate=impl.candidate().to_dict(),
                    profile_sha256=impl.id)
        path = tmp_path / (name + "-plan.json")
        path.write_text(json.dumps(plan))
        value["execution_plans"][name] = dict(path=str(path), sha256=file_sha256(path))
        checked[name] = (tmp_path / name, {}, rows, refs)
    # Only native transport/source verification is replaced. The admission
    # validator still reads and hashes all 48 files plus three execution plans.
    def checked_native(plan):
        assert plan["search_candidate"]["id"] == (CAND if plan["run_id"] == "candidate" else REF).id
        return checked[plan["run_id"]]
    monkeypatch.setattr(h3_native_bridge, "_checked_native", checked_native)
    return value


@pytest.mark.parametrize("mutation", [None, "witness", "metric", "alias", "missing", "formal", "hardlink", "plan"])
def test_bridge_recomputes_all_three_hash_bound_witness_sets(tmp_path, monkeypatch, mutation):
    from pathlib import Path
    from distillschedule.search.h3_native_bridge import validate_comparison
    value = witnessed(tmp_path, monkeypatch)
    if mutation is None:
        validate_comparison(value)
        return
    if mutation == "witness":
        Path(value["native_evidence"]["candidate"][0]["path"]).write_text('{}')
    if mutation == "metric": value["metrics"][0]["cross_delta_l2"] = 0.
    if mutation == "alias": value["native_evidence"]["repeat"] = value["native_evidence"]["reference"]
    if mutation == "missing": value["native_evidence"]["repeat"].pop()
    if mutation == "formal": value["formal_timing_eligible"] = True
    if mutation == "plan":
        Path(value["execution_plans"]["repeat"]["path"]).write_text('{}')
    if mutation == "hardlink":
        # Equal samples alone do not establish an independently executed repeat.
        import os
        ref = Path(value["native_evidence"]["reference"][0]["path"])
        repeat = Path(value["native_evidence"]["repeat"][0]["path"])
        repeat.unlink()
        os.link(ref, repeat)
        # This fails even before numeric admission; changing the digest cannot
        # turn a hardlinked control into a native independent execution.
        value["native_evidence"]["repeat"][0]["sha256"] = file_sha256(repeat)
    with pytest.raises(ValueError): validate_comparison(value)


@pytest.mark.parametrize("cached_policy,current_policy", [(None, True), (True, None), (True, "changed")])
def test_cached_preparation_cannot_cross_admission_modes(tmp_path, monkeypatch, cached_policy, current_policy):
    from distillschedule.search.execution import NativeTrialRunner
    runner = object.__new__(NativeTrialRunner)
    runner.output = tmp_path
    runner.identity = dict(source_head=HEAD)
    runner.h3_noise_policy = policy() if current_policy else None
    if current_policy == "changed": runner.h3_noise_policy["max_cross_self_delta_l2_ratio"] = 1.5
    runner.prepared = {CAND.id: dict(diagnostic_noise_policy=policy() if cached_policy else None)}
    monkeypatch.setattr(runner, "_rpc", lambda *a, **k: pytest.fail("must reject before transport or GPU"))
    with pytest.raises(ValueError, match="policy differs"):
        runner._prepare(CAND.candidate())


def test_public_source_only_calibration_records_explicit_policy(tmp_path, capsys):
    from distillschedule.search.workflow_cli import main
    from distillschedule.workloads.best_known.registry import git_head, repository_root
    head = git_head(repository_root())
    value = policy()
    value["source_head"] = head
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(value))
    output = tmp_path / "collection"
    assert main(["calibrate", "--workload", H3, "--candidate", REF.id, CAND.id,
                 "--expected-head", head, "--run-id", "cpu-noise-policy", "--h3-noise-policy", str(path),
                 "--output", str(output)]) == 0
    intent = json.loads((output / "intent.json").read_text())
    assert intent["diagnostic_noise_policy"] == value
    assert intent["correctness_runs"] == "TWO_CONTROL_PLUS_ONE_CANDIDATE"
    assert intent["gpu_execution"] == "NOT_RUN"

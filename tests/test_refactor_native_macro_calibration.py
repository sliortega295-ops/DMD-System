"""CPU receipt fixtures, not measured LiveWan/H3 performance or GPU proof.

Only ``native.checked_summary`` (profile/assets/remote-run admission) is stubbed.
LiveWan's raw rank/calibration files are explicit CPU envelopes; H3 also runs
the real raw reader, native summarizer/reconciliation, and sampled comparator.
The assembler, runtime identity, resource/teardown, hash, and CostModel gates
are real. Their MEASURED-shaped outputs remain ephemeral test data only.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from distillschedule.search import native
from distillschedule.search.contracts import semantic_contract
from distillschedule.search.cost_model import CostModel
from distillschedule.search.h3_implementation import H3, H3Implementation
from distillschedule.search.h3_native import (
    compare_native_two_cycle, read_native_receipts, summarize_native_two_cycle,
)
from distillschedule.search.livewan_implementation import LIVEWAN, LiveWanImplementation
from distillschedule.search.native_macro_adapter import (
    ADAPTER, MEMORY, native_graph, operation_contract, precision_contract,
)
from distillschedule.search.native_macro_calibration import assemble_native
from distillschedule.search.runtime_identity import SCHEMA, observed_binding
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint, load_profile
from test_refactor_h3_search_native import HEAD, _receipts, _with_rng


CASES = (
    pytest.param(LIVEWAN, LiveWanImplementation("fsdp"), id="livewan-fsdp"),
    pytest.param(LIVEWAN, LiveWanImplementation("pipeline", True, True, True), id="livewan-pipeline"),
    pytest.param(H3, H3Implementation(1, 64), id="h3"),
)
CAPACITY = 4096


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return native.binding(path)


def _duration(index, rank):
    # Alternate the slowest owner, so rank 15 cannot stand in for world max.
    return 100 * (index + 1) + (rank if index % 2 == 0 else 15 - rank)


def _calibration(workload, impl, rank):
    labels = (["livewan/critic_commit"] * 5 if workload == LIVEWAN else
              ["h3/student_step", "h3/critic_prepare_5xG"] + ["h3/critic_update_F"] * 5)
    rows, boundaries, stop = [], [], 0
    for cycle in (1, 2):
        for step, label in enumerate(labels):
            index = len(rows)
            start = stop
            stop += _duration(index, rank)
            rows.append(dict(cycle=cycle, label=label, status="PASS", start_ns=start,
                             stop_ns=stop, duration_ns=stop - start))
            if workload == LIVEWAN and (impl.placement == "fsdp" or rank >= 8):
                boundaries.append(dict(cycle=cycle, step=step, time_ns=stop))
    return dict(schema_version="distillschedule.async_range_calibration.v1",
        status="PASS", source_head=HEAD, rank=rank, implementation=impl.id,
        calibration_only=True, formal_timing_eligible=False, profiled=False,
        extra_synchronizations=0, extra_collectives=0, tensor_value_reads=0,
        post_setup_static_bytes=100 + rank, rows=rows, commit_boundaries=boundaries,
        fixture_scope="CPU_FIXTURE_ONLY_NOT_GPU_OBSERVATIONS")


def _observations():
    return [dict(schema_version=SCHEMA, status="PASS", source_head=HEAD,
        node_rank=node, cuda_context_created=False,
        environment={key: "CPU_TEST_ONLY" for key in (
            "python", "executable", "executable_sha256", "packages", "torch",
            "torch_cuda", "nccl", "torch_extension_sha256")},
        hardware=dict(topology="CPU_TEST_ONLY", gpus=[dict(device=i,
            uuid=f"CPU-TEST-{node}-{i}", name="CPU_TEST_ONLY", driver="CPU_TEST_ONLY",
            total_bytes=CAPACITY) for i in range(8)])) for node in (0, 1)]


def _fixture(tmp_path, monkeypatch, workload, impl):
    run = tmp_path / "cpu-calibration"
    run.mkdir()
    identity = dict(source_head=HEAD, asset_fingerprint="a" * 64,
        semantic_fingerprint=fingerprint(semantic_contract(load_profile(workload))),
        precision_fingerprint=fingerprint(precision_contract(workload)),
        **observed_binding(_observations(), source_head=HEAD))
    receipt = dict(profile_id=workload, mode="calibration", run_id=run.name,
        integration_head=HEAD, asset_fingerprint=identity["asset_fingerprint"],
        workload_semantics_hash=identity["semantic_fingerprint"], search_candidate={"id": impl.id})
    runtime = dict(before=_observations(), after=_observations())
    raw_paths, cal_paths = [], []
    raw_rows = _receipts(impl) if workload == H3 else [dict(rank=r) for r in range(16)]
    for raw in raw_rows:
        rank = raw["rank"]
        peak = 600 if rank == 3 else 200 + rank
        cal = _calibration(workload, impl, rank)
        if workload == LIVEWAN:
            path = run / "operation_calibration" / f"rank_{rank:03d}.json"
            _write(path, cal)
            cal_paths.append(path)
            # The real worker does NOT emit a top-level run_id here.
            raw.update(status="PASS", final_versions=dict(generator=2, critic=10, ema=0),
                peak_memory=dict(max_reserved_bytes=peak), operation_calibration=dict(
                    enabled=True, status="PASS", path=str(path), calibration_only=True,
                    formal_timing_eligible=False))
            path = run / "receipts" / f"rank_{rank:03d}.json"
        else:
            raw.update(calibration_only=True, stage_calibration=cal)
            raw["source_environment"]["H3_PLANNER_CALIBRATION"] = "1"
            for cycle in raw["cycles"]:
                for boundary in ("memory_before", "memory_after"):
                    cycle[boundary] = {"hbm": dict(peak_reserved_bytes=peak)}
            path = run / "execution/ranks" / f"rank_{rank:03d}.json"
            cal_paths.append(path)
        _write(path, raw)
        raw_paths.append(path)

    if workload == H3:
        summary = summarize_native_two_cycle(raw_rows, expected_head=HEAD, candidate_id=impl.id)
        summary["native_evidence"] = read_native_receipts(run)[1]
        comparison = compare_native_two_cycle(_with_rng(_receipts()),
            _with_rng(_receipts(impl)), expected_head=HEAD)
    else:
        summary = dict(schema_version="distillschedule.livewan_two_cycle_timing_summary.v1",
            status="PASS", errors=[], rank_count=16, run_id=run.name,
            calibration_only=True, formal_timing_eligible=False,
            fixture_scope="CPU_STUB_NATIVE_SUMMARY_NOT_WORKLOAD_VALIDATION")
        comparison = dict(schema_version="distillschedule.trajectory_coarse_comparison.v1",
            comparison_kind="semantic", expected_cycles=2, expected_world_size=16,
            complete=True, contract=dict(status="PASS", mismatches=[]),
            status="TRAJECTORY_REVIEW_50C_COMPLETED", fixture_scope="CPU_FIXTURE_ONLY")
    _write(run / "bound-summary.json", summary)
    preparation = dict(source_head=HEAD, candidate_id=impl.id,
        comparison=_write(tmp_path / "correctness/comparison.json", comparison),
        correctness=_write(tmp_path / "correctness/bound-summary.json", dict(
            status="PASS", formal_timing_eligible=False, fixture_scope="CPU_FIXTURE_ONLY")),
        strict_numerical_gate="NOT_ESTABLISHED_BY_CPU_FIXTURES")
    for node in (0, 1):
        samples = [dict(cgroup_events=dict(oom=0, oom_kill=0), gpus=[dict(
            device=i, total_bytes=CAPACITY, free_bytes=CAPACITY - (400 + node * 8 + i + sample * 10))
            for i in range(8)]) for sample in (0, 1)]
        _write(run / f"search-resources.node{node}.json", dict(status="PASS",
            node_rank=node, run_id=run.name, samples=samples, fixture_scope="CPU_FIXTURE_ONLY"))
        path = run / "launcher" / f"node_{node}" / "teardown_status.txt"
        path.parent.mkdir(parents=True)
        path.write_text(f"status=PASS\ngpu_process_count=0\nowner_run_id={run.name}\n")

    # Stub only external profile/assets admission. H3 still recomputes its
    # native summary and hashes from the files through real _checked_native.
    monkeypatch.setattr(native, "checked_summary", lambda actual: (
        run, native.read(run / "bound-summary.json")))
    return SimpleNamespace(run=run, receipt=receipt, preparation=preparation,
        identity=identity, runtime=runtime, workload=workload, impl=impl,
        raw_paths=raw_paths, cal_paths=cal_paths)


def _assemble(fixture):
    return assemble_native(fixture.receipt, fixture.preparation, fixture.identity, fixture.runtime)


@pytest.mark.parametrize("workload,impl", CASES)
def test_native_assembly_exports_hash_bound_costs_and_conservative_layout(tmp_path, monkeypatch, workload, impl):
    f = _fixture(tmp_path, monkeypatch, workload, impl)
    result = _assemble(f)
    owners = list(range(8, 16)) if workload == LIVEWAN and impl.placement == "pipeline" else list(range(16))
    graph = native_graph(workload)
    assert result["status"] == "PASS_NATIVE_MACRO_CALIBRATION"
    assert result["rank_count"] == 16 and result["operation_clock_owners"] == len(owners)
    assert result["operation_count"] == len(graph.operations) == (10 if workload == LIVEWAN else 14)
    assert result["formal_timing_eligible"] is False and result["speedup"] is None
    assert result["source_head"] == HEAD and result["implementation_id"] == impl.id
    for key in ("layout_fragment", "operation_costs", "provenance"):
        assert result[key] == native.binding(Path(result[key]["path"]))
    costs = CostModel.from_dict(native.read(Path(result["operation_costs"]["path"])))
    costs.verify_receipts()
    costs.bind(f.identity, problem_scope="MEASURED")
    assert costs.binding == f.identity
    for index, (operation, cost) in enumerate(zip(graph.operations, costs.records, strict=True)):
        payload = native.read(Path(cost.evidence["source_receipt"]))
        durations = [_duration(index, rank) for rank in owners]
        contract_hash = fingerprint(operation_contract(workload, operation, impl))
        assert cost.key == cost.evidence["operation_contract_sha256"] == contract_hash
        assert cost.samples_ns == (max(durations),)
        assert cost.workspace_bytes_per_rank == 600 - 115  # Global peak minus global static.
        assert payload["per_owner_duration_ns"] == durations and payload["owner_ranks"] == owners
        assert payload["binding"] == f.identity and payload["operation_id"] == operation.operation_id
        assert payload["cost_scope"] == graph.metadata["scope"]
        assert payload["calibration_only"] is True and payload["formal_timing_eligible"] is False
        assert payload["profiled"] is False and payload["provenance"] == result["provenance"]
        assert result["stage_world_max_seconds"][operation.operation_id] == max(durations) / 1e9

    layout = native.read(Path(result["layout_fragment"]["path"]))
    assert layout["schema_version"] == "distillschedule.native_macro_layout_fragment.v1"
    assert layout["adapter"] == ADAPTER and layout["memory_definition"] == MEMORY
    assert layout["binding"] == f.identity and layout["workload"] == workload
    assert layout["memory_limits"] == {str(rank): CAPACITY for rank in range(16)}
    assert set(layout["implementations"]) == {impl.id}  # Not a complete two-anchor layout.
    ranks = layout["implementations"][impl.id]["ranks"]
    assert [row["rank"] for row in ranks] == list(range(16))
    for rank, row in enumerate(ranks):
        peak = 600 if rank == 3 else 200 + rank
        assert row["source_head"] == HEAD and row["status"] == "PASS"
        assert row["post_setup_static_bytes"] == 100 + rank
        assert row["peak_reserved_bytes"] == peak
        assert row["sampled_driver_or_reserved_peak_bytes"] == max(peak, 410 + rank)
        assert row["sampled_min_driver_free_bytes"] == CAPACITY - (410 + rank)
    assert "NO_UNSAMPLED_PEAK_OR_ROBUST_HEADROOM_CLAIM" in layout["capacity_scope"]
    provenance = native.read(Path(result["provenance"]["path"]))
    expected_paths = set(f.raw_paths + f.cal_paths)
    expected_paths.update(f.run / f"search-resources.node{node}.json" for node in (0, 1))
    expected_paths.update(f.run / "launcher" / f"node_{node}" / "teardown_status.txt" for node in (0, 1))
    assert {row["path"] for row in provenance["native_sources"]} == {str(p) for p in expected_paths}
    assert all(row == native.binding(Path(row["path"])) for row in provenance["native_sources"])
    assert provenance["runtime_identity_evidence"] == f.runtime
    assert provenance["correctness"] == f.preparation["correctness"]
    assert provenance["comparison"] == f.preparation["comparison"]
    assert provenance["strict_numerical_gate"] == "NOT_ESTABLISHED_BY_CPU_FIXTURES"
    with pytest.raises(FileExistsError):
        _assemble(f)


def _mutate_calibration(f, key, value):
    path = f.cal_paths[-1]
    raw = native.read(path)
    calibration = raw["stage_calibration"] if f.workload == H3 else raw
    calibration[key] = value
    _write(path, raw)
    if f.workload == H3:
        # Refresh the summary's raw hash so this test reaches the calibration
        # gate itself, rather than failing earlier on stale native evidence.
        path = f.run / "bound-summary.json"
        summary = native.read(path)
        summary["native_evidence"] = read_native_receipts(f.run)[1]
        _write(path, summary)


@pytest.mark.parametrize("workload,impl", CASES)
@pytest.mark.parametrize("key,value", [
    ("status", "FAIL"), ("source_head", "b" * 40), ("rank", 14),
    ("implementation", "different-implementation"), ("calibration_only", False),
    ("formal_timing_eligible", True), ("profiled", True), ("extra_collectives", 1),
])
def test_changed_calibration_rejected_before_cost_directory(tmp_path, monkeypatch, workload, impl, key, value):
    f = _fixture(tmp_path, monkeypatch, workload, impl)
    _mutate_calibration(f, key, value)
    with pytest.raises(ValueError, match=f"rank15 calibration {key} mismatch"):
        _assemble(f)
    assert not (f.run / "macro-costs").exists()


@pytest.mark.parametrize("workload,impl", CASES)
@pytest.mark.parametrize("mutation,match", [
    ("resource_status", "resource evidence"), ("resource_owner", "resource evidence"),
    ("resource_samples", "resource evidence"), ("oom", "cgroup OOM increased"),
    ("capacity", "driver capacity changed"), ("teardown_status", "native teardown failed"),
    ("teardown_process", "native teardown failed"), ("teardown_owner", "native teardown failed"),
    ("correctness_hash", "native gate changed"), ("comparison_hash", "native gate changed"),
    ("runtime_identity", "observed runtime hardware_fingerprint differs"),
])
def test_bad_admission_evidence_never_emits_accepted_costs(tmp_path, monkeypatch, workload, impl, mutation, match):
    f = _fixture(tmp_path, monkeypatch, workload, impl)
    resource_path = f.run / "search-resources.node1.json"
    resource = native.read(resource_path)
    if mutation.startswith("resource_") or mutation in {"oom", "capacity"}:
        if mutation == "resource_status": resource["status"] = "FAIL"
        if mutation == "resource_owner": resource["run_id"] = "another-run"
        if mutation == "resource_samples": resource["samples"].pop()
        if mutation == "oom": resource["samples"][1]["cgroup_events"]["oom_kill"] = 1
        if mutation == "capacity": resource["samples"][1]["gpus"][0]["total_bytes"] += 1
        _write(resource_path, resource)
    elif mutation.startswith("teardown_"):
        path = f.run / "launcher/node_1/teardown_status.txt"
        before, after = {"teardown_status": ("status=PASS", "status=FAIL"),
            "teardown_process": ("gpu_process_count=0", "gpu_process_count=1"),
            "teardown_owner": (f"owner_run_id={f.run.name}", "owner_run_id=another-run")}[mutation]
        path.write_text(path.read_text().replace(before, after))
    elif mutation.endswith("_hash"):
        path = Path(f.preparation[mutation.removesuffix("_hash")]["path"])
        path.write_text(path.read_text() + "\n")
    else:
        f.runtime["after"][1]["hardware"]["gpus"][0]["driver"] = "different-runtime"
    with pytest.raises(ValueError, match=match):
        _assemble(f)
    assert not (f.run / "macro-costs").exists()


@pytest.mark.parametrize("placement", ["fsdp", "pipeline"])
@pytest.mark.parametrize("key,value", [
    ("path", "/another-run/operation_calibration/rank_015.json"),
    ("enabled", False), ("status", "FAIL"),
    ("calibration_only", False), ("formal_timing_eligible", True),
])
def test_livewan_rank_calibration_marker_binds_run_without_inventing_run_id(tmp_path, monkeypatch, placement, key, value):
    f = _fixture(tmp_path, monkeypatch, LIVEWAN, LiveWanImplementation(placement))
    raw = native.read(f.raw_paths[-1])
    assert "run_id" not in raw
    raw["operation_calibration"][key] = value
    _write(f.raw_paths[-1], raw)
    with pytest.raises(ValueError, match="native rank/calibration run binding failed"):
        _assemble(f)
    assert not (f.run / "macro-costs").exists()


def test_pipeline_generator_cannot_claim_score_commit_clock(tmp_path, monkeypatch):
    f = _fixture(tmp_path, monkeypatch, LIVEWAN, LiveWanImplementation("pipeline"))
    raw = native.read(f.cal_paths[0])
    raw["commit_boundaries"] = native.read(f.cal_paths[8])["commit_boundaries"]
    _write(f.cal_paths[0], raw)
    with pytest.raises(ValueError, match="G rank cannot claim Score commit clock"):
        _assemble(f)
    assert not (f.run / "macro-costs").exists()


def test_h3_raw_reconciliation_rejects_changed_receipt_hash(tmp_path, monkeypatch):
    f = _fixture(tmp_path, monkeypatch, H3, H3Implementation(1, 64))
    path = f.raw_paths[-1]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="H3 native evidence changed after summary"):
        _assemble(f)
    assert not (f.run / "macro-costs").exists()


@pytest.mark.parametrize("workload,impl", CASES)
def test_cost_model_rechecks_emitted_operation_hash_and_identity(tmp_path, monkeypatch, workload, impl):
    f = _fixture(tmp_path, monkeypatch, workload, impl)
    result = _assemble(f)
    costs = CostModel.from_dict(native.read(Path(result["operation_costs"]["path"])))
    path = Path(costs.records[0].evidence["source_receipt"])
    payload = native.read(path)
    payload["binding"]["source_head"] = "b" * 40
    _write(path, payload)
    with pytest.raises(ValueError, match="calibration receipt hash mismatch"):
        costs.verify_receipts()
    rebound = copy.deepcopy(costs.to_dict())
    rebound["records"][0]["evidence"]["source_receipt_sha256"] = file_sha256(path)
    with pytest.raises(ValueError, match="calibration receipt identity mismatch"):
        CostModel.from_dict(rebound).verify_receipts()

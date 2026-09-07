"""Derive a small, read-only Wan planner report from immutable native evidence.

Does not launch GPU work, modify a canonical profile, or promote numerical REVIEW.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root, out = args.evidence_root, args.output
    result = read(root / "validation/result.json")
    assert result["status"] == "COMPLETE_NATIVE_VALIDATION"
    source = result["identity"]["source_head"]
    events_path = root / "validation/events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    completed = [e for e in events if e["type"] == "COMPLETE"]
    assert len(completed) == sum(e["type"] == "START" for e in events) == 6
    search = read(root / "planning/search.json")
    predictions = {json.dumps(row["assignment"], sort_keys=True): row for row in search["ranked"]}
    spec = read(root / "validation/validation.json")
    rows, original_receipts = [], []
    for event in completed:
        path = Path(event["receipt"])
        assert sha(path) == event["receipt_sha256"]
        native = read(path)
        assert native["source_head"] == source and native["status"] == "PASS"
        assert native["profiled"] is False and native["initializations"] == 1
        assert native["same_process"] is True and len(native["ranks"]) == 16
        assert {r["rank"] for r in native["ranks"]} == set(range(16))
        for rank in native["ranks"]:
            assert all(rank[k] == "PASS" for k in ("status", "census", "versions", "sample_rng", "finite", "dtype", "memory"))
            assert rank["cgroup_oom_delta"] == rank["cuda_oom_delta"] == 0
        checked = event["result"]
        pred = spec["predictions"][event["candidate_id"]]["predicted_cycle2_ns"] / 1e9
        rows.append(dict(phase=event["phase"], candidate_id=event["candidate_id"],
            run_id=Path(native["native_evidence"][0]["path"]).parent.name,
            cycle1_seconds=checked["cycle1_seconds"], cycle2_seconds=checked["cycle2_seconds"],
            predicted_cycle2_seconds=pred, prediction_error_percent=100*(pred/checked["cycle2_seconds"]-1),
            ranks_pass=16, numerical_status=native["numerical_status"], strict_eligible=checked["strict_eligible"],
            source_head=source, profiled=False,
            native_reported_peak_allocated_gib=max(r["peak_allocated_bytes"] for r in native["ranks"])/2**30,
            min_sampled_driver_free_gib=min(r["min_driver_free_bytes"] for r in native["ranks"])/2**30,
            node_cgroup_sampled_peak_gib=max(r["host_peak_bytes"] for r in native["ranks"])/2**30,
            construction_world_max_seconds=native["construction"]["world_max_seconds"],
            cuda_oom_delta=0, cgroup_oom_delta=0,
            resource_status=native["resource_status"], cleanup_status=native["cleanup_status"]))
        original_receipts.append(dict(path=str(path), sha256=sha(path), native_evidence=native["native_evidence"]))
    calibration = []
    for path in sorted((root / "planning/receipts").glob("*/verified-summary.json")):
        value = read(path)
        assignment = spec["predictions"][value["implementation_id"]]["assignment"]
        pred = predictions[json.dumps(assignment, sort_keys=True)]
        calibration.append(dict(implementation_id=value["implementation_id"],
            predicted_cycle2_seconds=pred["predicted_cycle2_ns"] / 1e9,
            calibration_peak_allocated_gib=max(r["peak_allocated_bytes"] for r in value["ranks"])/2**30,
            calibration_peak_reserved_gib=max(r["peak_reserved_bytes"] for r in value["ranks"])/2**30,
            native_validation="MEASURED" if any(r["candidate_id"] == value["implementation_id"] for r in rows) else "NOT_RUN_BUDGET",
            source_head=value["source_head"], run_id=value["run_id"], ranks_pass=len(value["ranks"]),
            verified_summary_sha256=sha(path), raw_fragment=value["raw_fragment"]))
    assert len(calibration) == 7
    summary = dict(schema_version="distillschedule.wan_closed_loop_report.v1",
        status="COMPLETE_NATIVE_VALIDATION_DIAGNOSTIC", runtime_source_head=source,
        identity=result["identity"], measured_costs=98, rank_stage_observations=1568,
        calibration_implementations=calibration, native_trials=rows,
        native_result=result, budget=read(root / "validation/cumulative-budget-after.json"),
        search_seconds=search["search_seconds"], search_space=search["space_size"],
        bounded_exhaustive_agreement=True,
        prediction_absolute_error_percent_mean=statistics.mean(abs(r["prediction_error_percent"]) for r in rows),
        prediction_absolute_error_percent_max=max(abs(r["prediction_error_percent"]) for r in rows),
        measured_order_matches_predicted_order=(
            [r["candidate_id"] for r in sorted(rows[:-1], key=lambda r:r["cycle2_seconds"])] ==
            [r["candidate_id"] for r in sorted(rows[:-1], key=lambda r:r["predicted_cycle2_seconds"])]),
        native_receipts=original_receipts,
        failed_calibration_prerequisite=read(root / "manifests/transport_failure_reconciliation.json"),
        claims=dict(strict_equivalence=False, canonical_changed=False, global_optimum=False,
            new_better_than_incumbent_method=False,
            memory="Calibration captures nested FSDP allocator resets. Native allocator peak fields may be since-last-reset, not full-cycle peaks. Driver minima and cgroup peaks are sampled.",
            search="Seven admitted measured combinations; a complete cost table for this small space, not an unmeasured/general operator model.",
            next_order=["LiveWan", "H3 Grid1000", "Helios No-GAN", "Helios GAN"]))
    assert result["strict_winner"] is None and not result["canonical_winner_updated"]
    out.mkdir(parents=True, exist_ok=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    for name, values in [("native_trials.csv", rows), ("calibration.csv", calibration)]:
        with (out / name).open("x", newline="") as file:
            w=csv.DictWriter(file, fieldnames=list(values[0]), lineterminator="\n")
            w.writeheader()
            w.writerows({k:json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                         for k,v in row.items()} for row in values)
    lines = ["# Wan measured-cost planner: closed-loop native diagnostic", "",
        f"Runtime source: `{source}`. Later publication commits are not execution identities.", "",
        "Seven implementations / 98 operation costs / 1,568 rank-stage records. CPU search agrees with a bounded exhaustive oracle.",
        "The existing native dispatcher executed two cycles per trial with one initialization; only cycle 2 is compared. No Nsight.", "",
        "| Phase | Implementation | Cycle 1 (s) | Cycle 2 (s) | Prediction (s) | Error |",
        "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['phase']} | {r['candidate_id']} | {r['cycle1_seconds']:.6f} | {r['cycle2_seconds']:.6f} | {r['predicted_cycle2_seconds']:.6f} | {r['prediction_error_percent']:+.2f}% |")
    lines += ["", f"Confirmed diagnostic speedup vs matched FSDP: **{result['cycle2_speedup_diagnostic']:.6f}x**.",
        f"Mean absolute prediction error: {summary['prediction_absolute_error_percent_mean']:.2f}%; max: {summary['prediction_absolute_error_percent_max']:.2f}%.",
        "All six native trials: 16/16 runtime/semantic/resource gates PASS; final G4/C20/EMA20; zero observed new CUDA/cgroup OOM.",
        "Numerical status remains **REVIEW**, not strict P0 parity; strict_winner=null. The selected implementation rediscovered the incumbent; this is not a new optimization gain over it.",
        "", "## Scope and remaining work", "",
        "- Current Wan dimensions: placement, modulation chunk, existing no-grad LoRA epilogue. The native clustered schedule is fixed; this does not yet demonstrate searched sharing or G:C pool placement.",
        "- Costs are synchronized diagnostic stage envelopes, including internal communication. They are not profile unions or formal native timing; the table above separates prediction from measurement.",
        "- Native FSDP allocator peak can be reset in nested helpers. Use calibration peaks for the memory model, not the smaller native reported peak as a full-cycle assertion.",
        "- One prerequisite failed on both SSH clients Broken pipe during loading. It was preserved and classified LAUNCHER_ERROR, not OOM. Fresh recovery epoch followed explicit cleanup/transport/preflight checks; no failed run was overwritten.",
        "- Timing budget: 2 historical attempts + 6 current-source attempts = 8. Correctness/calibration costs are separate and not free; complete calibration of all seven combinations limits scalability claims.",
        "- Canonical profiles and main are unchanged. Next: LiveWan generated adapter and measured pool/DAG costs, then H3 Grid1000, then both Helios workloads. These new model-based pilots are NOT_RUN.", ""]
    (out / "report.md").write_text("\n".join(lines))
    hashes={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out / "artifact_sha256.json").write_text(json.dumps(hashes,indent=2)+"\n")
    print(json.dumps(dict(output=str(out), speedup=result['cycle2_speedup_diagnostic'], status=summary['status'])))


if __name__ == "__main__":
    main()

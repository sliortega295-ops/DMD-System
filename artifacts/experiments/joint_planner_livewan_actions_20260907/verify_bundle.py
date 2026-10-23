#!/usr/bin/env python3
"""Verify the small publication bundle, not replay its historical GPU commands."""
import csv
import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root)
    hashes = json.loads((root / "artifact_sha256.json").read_text())
    actual_files = {str(p.relative_to(root)) for p in root.rglob("*")
                    if p.is_file() and p.name != "artifact_sha256.json"}
    assert actual_files == set(hashes), "bundle file set differs"
    for name, expected in hashes.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "native-observation-summary.json").read_text())
    assert summary["status"] == "PASS_NATIVE_ACTION_OBSERVATIONS"
    assert summary["source_head"] == manifest["source_head"]
    assert summary["run_id"] == manifest["run_id"]
    assert summary["cycles"] == 2 and summary["rank_count"] == 16
    assert summary["final_versions"] == {"generator": 2, "critic": 10, "ema": 0}
    assert summary["matched_rank_sample_identities"] == 160
    assert not summary["formal_timing_eligible"] and not summary["planner_costs_ready"]
    assert not summary["independent_link_service_admitted"]
    assert summary["new_speedup"] is None
    assert {r["rank"] for r in summary["rank_summaries"]} == set(range(16))
    for row in summary["rank_summaries"]:
        assert all(row[key] == 0 for key in (
            "byte_residual", "event_residual", "shadow_event_residual", "host_event_residual"))
    for file, key in (("helper-observations.csv", "helpers"),
                      ("transport-observations.csv", "transport")):
        with (root / file).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == len(summary[key])
        for csv_row, json_row in zip(rows, summary[key]):
            for field in csv_row:
                assert csv_row[field] == str(json_row[field]), (file, field)
    assert sum(row["count"] for row in summary["helpers"]) == 336
    for kind in ("latent_d2h", "latent_h2d"):
        copies = [r for r in summary["transport"] if r["kind"] == kind]
        assert sum(r["count"] for r in copies) == 136
        assert sum(r["counted_bytes"] for r in copies) == 672645120
    for node in (0, 1):
        cleanup = json.loads((root / f"cleanup-readback.node{node}.json").read_text())
        assert cleanup["status"] == "PASS" and cleanup["node_lock_free"]
        assert not any(cleanup[k] for k in ("gpu_processes", "owned_run_processes", "port_listeners"))
        resource = next(r for r in summary["resources"] if r["node"] == node)
        assert resource["cgroup_oom_delta"] == resource["cgroup_oom_kill_delta"] == 0
    return {"status": "PASS_SMALL_BUNDLE_INTEGRITY_AND_RECONCILIATION",
            "files": len(hashes), "ranks": 16, "native_timing": "NOT_RUN"}


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))

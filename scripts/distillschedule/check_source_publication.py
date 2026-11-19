#!/usr/bin/env python3
"""Verify that the public tables and their release data agree, without CUDA."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re


IDS = tuple(f"W{i}" for i in range(1, 7))
TIMES = ("baseline_reference_s", "resident_offload_reference_s", "realhf_like_reference_s", "system_s")
FIGURES = {
    "artifacts/results/a100.csv": ["Figure 5"],
    "artifacts/results/h100.csv": ["Figure 5"],
    "artifacts/results/planner.json": ["Figure 7(a)", "Figure 9"],
}


def _workload_rows(guide: str, section: str) -> list[list[str]]:
    body = guide.split(f"\n## {section}\n", 1)[1].split("\n## ", 1)[0]
    return [list(map(str.strip, line.strip("|").split("|")))
            for line in body.splitlines() if re.match(r"^\| W[0-9]+ \|", line)]


def _cycle_table(root: Path, hardware: str, guide: str, names: dict[str, str]) -> None:
    with (root / f"artifacts/results/{hardware}.csv").open(newline="") as stream:
        table = csv.DictReader(stream)
        if table.fieldnames != ["id", "name", *TIMES, "reference_ratio"]:
            raise ValueError(f"unexpected {hardware} result columns")
        rows = list(table)
    if {row["id"]: row["name"] for row in rows} != names or tuple(row["id"] for row in rows) != IDS:
        raise ValueError(f"{hardware} rows must follow the paper's workload IDs")

    for row in rows:
        for key in TIMES:
            if key == "resident_offload_reference_s" and hardware == "a100" and row["id"] == "W1":
                if row[key] != "OOM":
                    raise ValueError("expected OOM for the A100 W1 resident baseline")
            elif int(row[key]) <= 0:
                raise ValueError(f"{hardware} cycle time must be positive")
        baseline, system, ratio = (float(row[k]) for k in ("baseline_reference_s", "system_s", "reference_ratio"))
        # Figure 5 rounds latency labels to seconds and speedups to hundredths.
        lo = (baseline - 0.5) / (system + 0.5)
        hi = (baseline + 0.5) / (system - 0.5)
        if not lo - 0.005 <= ratio <= hi + 0.005:
            raise ValueError(f"{hardware} {row['id']}: speedup outside the label rounding interval")

    expected = [[r["id"], *(r[key] for key in TIMES), r["reference_ratio"] + "×"] for r in rows]
    if _workload_rows(guide, hardware.upper()) != expected:
        raise ValueError(f"{hardware} Markdown and CSV results differ")


def verify(root: Path) -> dict:
    root = root.resolve()
    results = root / "artifacts/results"
    manifest = json.loads((results / "manifest.json").read_text())
    if manifest["schema"] != "dmd.public_results.v2" or manifest["workload_ids"] != "Table I":
        raise ValueError("unknown paper result schema")

    hashes: dict[str, str] = {}
    for row in manifest["files"]:
        relative = row["path"]
        if relative in hashes or relative not in FIGURES or row["paper_sources"] != FIGURES[relative]:
            raise ValueError("invalid or duplicate paper result file")
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("paper result resolves outside checkout")
        data = path.read_bytes()
        if len(data) >= 1_000_000:
            raise ValueError("large raw artifact is not a publication summary")
        hashes[relative] = hashlib.sha256(data).hexdigest()
        if len(data) != row["bytes"] or hashes[relative] != row["sha256"]:
            raise ValueError("publication bytes/hash changed: " + relative)
    if set(hashes) != set(FIGURES):
        raise ValueError("missing paper result file")
    catalog = json.loads((root / "src/distillschedule/workloads/catalog.json").read_text())
    if catalog["evidence_files"] != hashes:
        raise ValueError("catalog and paper-result checksums differ")
    catalog_by_paper_id = {row["paper_id"]: row for row in catalog["workloads"]}
    if len(catalog["workloads"]) != len(IDS) or set(catalog_by_paper_id) != set(IDS):
        raise ValueError("paper and CLI workload IDs must map one to one")

    guide = (root / "docs/guides/performance_validation.md").read_text()
    workload_rows = _workload_rows(guide, "Workloads")
    if tuple(row[0] for row in workload_rows) != IDS:
        raise ValueError("workload guide must follow the paper's workload order")
    names = {}
    for row in workload_rows:
        recipe = catalog_by_paper_id[row[0]]
        if f"](../../{recipe['recipe']})" not in row[-1]:
            raise ValueError(f"{row[0]}: wrong recipe link")
        names[row[0]] = row[1]
    for hardware in ("a100", "h100"):
        _cycle_table(root, hardware, guide, names)

    planner = json.loads((results / "planner.json").read_text())
    if (planner["schema_version"] != "dmd.paper_planner_results.v1"
            or planner["comparison_source"] != "Figure 9"
            or planner["approximate_sampling_source"] != "Figure 7(a)"
            or tuple(row["id"] for row in planner["rows"]) != IDS[1:]):
        raise ValueError("unknown planner figure scope")
    sampling = planner["approximate_sampling_seconds"]
    if tuple(sampling) != IDS or any(type(val) is not int or val <= 0 for val in sampling.values()):
        raise ValueError("approximate sampling must cover all six paper workloads")
    comparisons = _workload_rows(guide, "Planning and reproduction")
    if tuple(row[0] for row in comparisons) != IDS[1:]:
        raise ValueError("planner guide must follow Figure 9's workload order")
    for displayed, row in zip(comparisons, planner["rows"]):
        observed = tuple(map(float, (displayed[1], displayed[2], displayed[3].rstrip("%").replace("−", "-"))))
        manual, selected, gap = (row[key] for key in ("manual_cycle_s", "selected_cycle_s", "gap_percent"))
        if (observed != (manual, selected, gap) or min(manual, selected) <= 0
                or not math.isclose((selected / manual - 1) * 100, gap, abs_tol=0.005)):
            raise ValueError(f"{row['id']}: inconsistent Figure 9 plan comparison")
    return {"status": "PASS_PUBLICATION_INTEGRITY_ONLY", "files": len(hashes),
            "workloads": len(IDS), "planner_pairs": len(comparisons), "GPU_started": 0}


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parents[2]), indent=2))

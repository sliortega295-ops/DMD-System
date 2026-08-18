#!/usr/bin/env python3
"""Summarize one MiniMax FSDP-first memory-attribution run.

The report intentionally distinguishes observed allocator categories from
unattributed bytes.  It never treats ``reserved - allocated`` as a tensor class
and never claims NCCL memory from allocator counters alone.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

GIB = 1024**3
OOM_RE = re.compile(r"Tried to allocate ([0-9.]+) GiB.*?free.*?([0-9.]+) GiB.*?allocated.*?([0-9.]+) GiB.*?reserved.*?([0-9.]+) GiB", re.S)


def _gib(value: int | float) -> float:
    return float(value) / GIB


def _read(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _cat(row: dict[str, Any]) -> dict[str, int]:
    roles = sum(int(row.get(key, 0)) for key in (
        "student_parameter_bytes", "fake_parameter_bytes", "teacher_parameter_bytes"))
    grads = sum(int(row.get(key, 0)) for key in (
        "student_gradient_bytes", "fake_gradient_bytes", "teacher_gradient_bytes"))
    optim = sum(int(value) for key, value in row.items() if key.endswith("_optimizer_cuda_bytes"))
    saved = int(row.get("saved_tensors", {}).get("live_bytes_estimate", 0))
    allocated = int(row.get("torch_allocated_bytes", 0))
    reserved = int(row.get("torch_reserved_bytes", 0))
    inactive = max(0, reserved - allocated)
    observed = roles + grads + optim + saved
    return {
        "observed_parameter_bytes": roles,
        "observed_gradient_bytes": grads,
        "observed_optimizer_cuda_bytes": optim,
        "saved_tensor_live_estimate_bytes": saved,
        "allocator_reserved_unallocated_bytes": inactive,
        "allocator_active_residual_bytes": max(0, allocated - observed),
        "torch_allocated_bytes": allocated,
        "torch_reserved_bytes": reserved,
        "driver_free_bytes": int(row.get("driver_free_bytes", 0)),
    }


def _peak(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "NO_EVENTS"}
    peak = max(rows, key=lambda row: int(row.get("torch_allocated_bytes", 0)))
    categories = _cat(peak)
    return {
        "status": "OBSERVED",
        "event": peak.get("event"),
        "phase": peak.get("phase"),
        "role": peak.get("role"),
        "call_index": peak.get("call_index"),
        "timestamp": peak.get("timestamp"),
        **categories,
        "saved_pack_count": int(peak.get("saved_tensors", {}).get("pack_count", 0)),
        "saved_unpack_count": int(peak.get("saved_tensors", {}).get("unpack_count", 0)),
        "saved_pack_bytes": int(peak.get("saved_tensors", {}).get("pack_bytes", 0)),
        "saved_unpack_bytes": int(peak.get("saved_tensors", {}).get("unpack_bytes", 0)),
    }


def _exception(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        if row.get("event") != "exception":
            continue
        message = str(row.get("extra", {}).get("exception", ""))
        match = OOM_RE.search(message)
        result: dict[str, Any] = {
            "phase": row.get("phase"),
            "role": row.get("role"),
            "call_index": row.get("call_index"),
            "exception_type": row.get("extra", {}).get("exception_type"),
            "message": message,
        }
        if match:
            result["failed_allocation_gib"] = float(match.group(1))
            result["reported_driver_free_gib"] = float(match.group(2))
            result["reported_torch_allocated_gib"] = float(match.group(3))
            result["reported_torch_reserved_unallocated_gib"] = float(match.group(4))
        return result
    return None


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = sorted({field for record in records for field in record})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rank_files = sorted(args.input_dir.glob("rank_*.jsonl"))
    ranks: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for path in rank_files:
        rows = _read(path)
        rank = int(path.stem.split("_")[-1])
        peak = _peak(rows)
        item = {"rank": rank, "event_count": len(rows), **peak}
        failure = _exception(rows)
        if failure:
            item["exception"] = failure
        ranks.append(item)
        for row in rows:
            categories = _cat(row)
            events.append({"rank": rank, "event": row.get("event"), "phase": row.get("phase"),
                           "role": row.get("role"), "call_index": row.get("call_index"),
                           "timestamp": row.get("timestamp"), **categories,
                           "saved_pack_count": row.get("saved_tensors", {}).get("pack_count", 0),
                           "saved_unpack_count": row.get("saved_tensors", {}).get("unpack_count", 0)})
    oom_ranks = [item for item in ranks if "exception" in item]
    summary = {
        "schema_version": "distillschedule.minimax_memory_attribution_summary.v1",
        "status": "OOM_OBSERVED" if oom_ranks else ("PASS_RECEIPTS_NOT_ANALYZED" if ranks else "NO_EVENTS"),
        "input_dir": str(args.input_dir),
        "rank_files": [str(path) for path in rank_files],
        "rank_count": len(rank_files),
        "ranks": ranks,
        "oom_ranks": [item["rank"] for item in oom_ranks],
        "category_semantics": {
            "observed_parameter_bytes": "sum of local parameter metadata at peak; not a CUDA allocator attribution",
            "observed_gradient_bytes": "sum of non-null local gradients at peak",
            "observed_optimizer_cuda_bytes": "optimizer state tensors observed on CUDA at peak",
            "saved_tensor_live_estimate_bytes": "pack-minus-unpack logical estimate; no tensor references retained",
            "allocator_reserved_unallocated_bytes": "reserved minus allocated; fragmentation/free-cache, not a tensor class",
            "allocator_active_residual_bytes": "allocated bytes not covered by the observed metadata classes",
            "nccl_bytes": "NOT_INSTRUMENTED; allocator counters cannot identify NCCL workspace",
        },
        "reconciliation": {
            "allocator_partition_is_not_exact": True,
            "unattributed_allocator_residual_bytes": True,
            "nccl_reconciliation": "NOT_AVAILABLE_FROM_THIS_CANARY",
        },
    }
    (args.output_dir / "memory_attribution_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(args.output_dir / "memory_attribution_ranks.csv", ranks)
    _write_csv(args.output_dir / "memory_attribution_events.csv", events)
    lines = [
        "# MiniMax FSDP-first fine-grained memory attribution",
        "",
        f"- Status: **{summary['status']}**",
        f"- Rank files: {len(rank_files)}",
        f"- OOM ranks with diagnostic records: {summary['oom_ranks']}",
        "",
        "## Peak observed categories",
        "",
        "| rank | event | phase | role | allocated GiB | reserved GiB | parameter GiB | gradient GiB | optimizer CUDA GiB | saved-live estimate GiB | allocator residual GiB | driver free GiB |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in ranks:
        lines.append("| {rank} | {event} | {phase} | {role} | {alloc:.3f} | {reserved:.3f} | {params:.3f} | {grads:.3f} | {optim:.3f} | {saved:.3f} | {resid:.3f} | {free:.3f} |".format(
            rank=item["rank"], event=item.get("event"), phase=item.get("phase"), role=item.get("role"),
            alloc=_gib(item.get("torch_allocated_bytes", 0)), reserved=_gib(item.get("torch_reserved_bytes", 0)),
            params=_gib(item.get("observed_parameter_bytes", 0)), grads=_gib(item.get("observed_gradient_bytes", 0)),
            optim=_gib(item.get("observed_optimizer_cuda_bytes", 0)), saved=_gib(item.get("saved_tensor_live_estimate_bytes", 0)),
            resid=_gib(item.get("allocator_active_residual_bytes", 0)), free=_gib(item.get("driver_free_bytes", 0))))
    lines += [
        "",
        "## Boundaries",
        "",
        "The table is a diagnostic decomposition, not an exact allocator partition. Parameter and optimizer values are metadata observed from local tensors; saved-tensor values are logical pack/unpack counters. CUDA allocator residual includes temporary activations, attention/FlashAttention workspaces, FSDP materialization, NCCL/framework allocations, and any other live tensors. NCCL workspace is not separately instrumented and remains UNKNOWN.",
        "",
        "Raw JSONL and the machine-readable summary are the evidence; no timing or capacity claim is inferred from this diagnostic run.",
    ]
    (args.output_dir / "memory_attribution_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

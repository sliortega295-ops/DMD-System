#!/usr/bin/env python3
"""Overlap-aware world16 Nsight attribution for the Wan1.3B FSDP2 diagnostic."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

CATEGORIES = ("compute", "nccl", "copy")
COPY_KIND = {1: "h2d", 2: "d2h", 3: "d2d", 8: "d2d", 10: "d2d"}
OUTER = "WAN13B_FULL5/world16-fsdp"
STAGES = ["WAN13B/G_UPDATE", "WAN13B/G_ROLLOUTS"] + [
    f"WAN13B/C_UPDATE/step_{step}" for step in range(10, 15)
] + [f"WAN13B/EMA/step_{step}" for step in range(10, 15)]


def _tables(db: sqlite3.Connection) -> set[str]:
    return {str(r[0]) for r in db.execute("select name from sqlite_master where type='table'")}


def _cols(db: sqlite3.Connection, table: str) -> set[str]:
    return {str(r[1]) for r in db.execute(f'pragma table_info("{table}")')}


def _strings(db: sqlite3.Connection) -> dict[int, str]:
    if "StringIds" not in _tables(db):
        return {}
    return {int(i): str(v) for i, v in db.execute("select id,value from StringIds")}


def _name(value: object, strings: dict[int, str]) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return strings.get(int(value), f"UNRESOLVED_STRING_ID_{int(value)}")
    return str(value)


def _clip(interval: tuple[int, int], window: tuple[int, int]) -> tuple[int, int] | None:
    a, b = max(interval[0], window[0]), min(interval[1], window[1])
    return (a, b) if b > a else None


def _union(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _duration(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(b - a for a, b in _union(intervals))


def _intersection(a: Iterable[tuple[int, int]], b: Iterable[tuple[int, int]]) -> int:
    x, y = _union(a), _union(b)
    return _intersection_unioned(x, y)


def _intersection_unioned(
    x: list[tuple[int, int]], y: list[tuple[int, int]]
) -> int:
    i = j = total = 0
    while i < len(x) and j < len(y):
        total += max(0, min(x[i][1], y[j][1]) - max(x[i][0], y[j][0]))
        if x[i][1] <= y[j][1]:
            i += 1
        else:
            j += 1
    return total


def _exclusive(category_intervals: dict[str, list[tuple[int, int]]], window: tuple[int, int]) -> dict[str, int]:
    return _exclusive_unioned(
        {category: _union(intervals) for category, intervals in category_intervals.items()},
        window,
    )


def _exclusive_unioned(
    category_intervals: dict[str, list[tuple[int, int]]],
    window: tuple[int, int],
) -> dict[str, int]:
    points: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for category, intervals in category_intervals.items():
        for a, b in intervals:
            clipped = _clip((a, b), window)
            if clipped:
                points[clipped[0]][category] += 1
                points[clipped[1]][category] -= 1
    points[window[0]]
    points[window[1]]
    active = {c: 0 for c in CATEGORIES}
    result: Counter[str] = Counter()
    last = window[0]
    for now in sorted(points):
        if now > last:
            key = "+".join(c for c in CATEGORIES if active[c] > 0) or "no_traced_activity"
            result[key] += now - last
        for c, delta in points[now].items():
            active[c] += delta
        last = now
    return dict(result)


def _collective(name: str) -> str:
    value = name.lower()
    if "allgather" in value or "all_gather" in value:
        return "all_gather"
    if "reducescatter" in value or "reduce_scatter" in value:
        return "reduce_scatter"
    if "allreduce" in value or "all_reduce" in value:
        return "all_reduce"
    if any(x in value for x in ("sendrecv", "send", "recv")):
        return "p2p"
    if "broadcast" in value:
        return "broadcast"
    return "other_nccl"


def _phase_ranges(db: sqlite3.Connection) -> dict[str, list[tuple[int, int]]]:
    rows = db.execute(
        "select text,start,end from NVTX_EVENTS where text is not null and start is not null and end is not null and end>start"
    )
    out: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for text, start, end in rows:
        if str(text) == OUTER or str(text) in STAGES:
            out[str(text)].append((int(start), int(end)))
    return out


def analyze_node(path: Path, node: int) -> dict:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.execute("pragma query_only=on")
    try:
        tables = _tables(db)
        required = {"CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "NVTX_EVENTS", "StringIds"}
        missing = sorted(required - tables)
        if missing:
            raise RuntimeError(f"node{node} SQLite missing tables: {missing}")
        strings = _strings(db)
        phases = _phase_ranges(db)
        if len(phases.get(OUTER, [])) != 1:
            raise RuntimeError(f"node{node}: expected one {OUTER} range, found {len(phases.get(OUTER, []))}")
        outer = phases[OUTER][0]
        kcols = _cols(db, "CUPTI_ACTIVITY_KIND_KERNEL")
        name_col = "demangledName" if "demangledName" in kcols else "shortName"
        if name_col not in kcols or "deviceId" not in kcols:
            raise RuntimeError(f"node{node}: kernel schema lacks name/deviceId: {sorted(kcols)}")
        by_device = {i: {c: [] for c in CATEGORIES} for i in range(8)}
        kernel_rows = {i: [] for i in range(8)}
        nccl_rows = {i: [] for i in range(8)}
        unresolved_ns = Counter()
        kernel_ns = Counter()
        query = (
            f"select start,end,deviceId,{name_col} from CUPTI_ACTIVITY_KIND_KERNEL "
            "where end>? and start<?"
        )
        for start, end, device, raw_name in db.execute(query, outer):
            device = int(device)
            if device not in by_device:
                continue
            clipped = _clip((int(start), int(end)), outer)
            if not clipped:
                continue
            n = _name(raw_name, strings)
            category = "nccl" if any(t in n.lower() for t in ("nccl", "nvls", "collnet")) else "compute"
            by_device[device][category].append(clipped)
            kernel_rows[device].append((clipped[0], clipped[1], n))
            kernel_ns[device] += clipped[1] - clipped[0]
            if not n or n.startswith("UNRESOLVED_STRING_ID_"):
                unresolved_ns[device] += clipped[1] - clipped[0]
            if category == "nccl":
                nccl_rows[device].append((clipped[0], clipped[1], n, _collective(n)))

        copy_rows = {i: [] for i in range(8)}
        mcols = _cols(db, "CUPTI_ACTIVITY_KIND_MEMCPY")
        if "deviceId" not in mcols:
            raise RuntimeError(f"node{node}: memcpy schema lacks deviceId: {sorted(mcols)}")
        for start, end, device, num_bytes, kind in db.execute(
            "select start,end,deviceId,bytes,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY where end>? and start<?",
            outer,
        ):
            device = int(device)
            if device not in by_device:
                continue
            clipped = _clip((int(start), int(end)), outer)
            if not clipped:
                continue
            direction = COPY_KIND.get(int(kind), "other")
            by_device[device]["copy"].append(clipped)
            copy_rows[device].append((clipped[0], clipped[1], int(num_bytes or 0), direction))

        missing_devices = [d for d in range(8) if not kernel_rows[d]]
        if missing_devices:
            raise RuntimeError(f"node{node}: kernel device coverage missing {missing_devices}")
        devices = []
        for device in range(8):
            merged = {
                category: _union(by_device[device][category])
                for category in CATEGORIES
            }
            by_device[device] = merged
            wall = outer[1] - outer[0]
            compute = sum(b - a for a, b in merged["compute"])
            nccl = sum(b - a for a, b in merged["nccl"])
            copy = sum(b - a for a, b in merged["copy"])
            nccl_hidden = _intersection_unioned(merged["nccl"], merged["compute"])
            copy_hidden = _intersection_unioned(merged["copy"], merged["compute"])
            exclusive = _exclusive_unioned(merged, outer)
            residual = wall - sum(exclusive.values())
            if abs(residual) > 10:
                raise RuntimeError(f"node{node}/device{device}: exclusive reconciliation residual={residual}")
            directions = {}
            for direction in ("h2d", "d2h", "d2d", "other"):
                rows = [r for r in copy_rows[device] if r[3] == direction]
                directions[direction] = {
                    "events": len(rows),
                    "bytes": sum(r[2] for r in rows),
                    "union_ns": _duration((r[0], r[1]) for r in rows),
                    "sum_duration_ns": sum(r[1] - r[0] for r in rows),
                }
            coll = {}
            for kind in ("all_gather", "reduce_scatter", "all_reduce", "p2p", "broadcast", "other_nccl"):
                rows = [r for r in nccl_rows[device] if r[3] == kind]
                intervals = _union((r[0], r[1]) for r in rows)
                coll[kind] = {
                    "events": len(rows),
                    "union_ns": sum(b - a for a, b in intervals),
                    "sum_duration_ns": sum(r[1] - r[0] for r in rows),
                    "hidden_by_compute_ns": _intersection_unioned(
                        intervals, merged["compute"]
                    ),
                }
                coll[kind]["exposed_vs_compute_ns"] = coll[kind]["union_ns"] - coll[kind]["hidden_by_compute_ns"]
            devices.append({
                "node": node,
                "device": device,
                "rank": node * 8 + device,
                "wall_ns": wall,
                "compute_union_ns": compute,
                "nccl_union_ns": nccl,
                "nccl_hidden_by_compute_ns": nccl_hidden,
                "nccl_exposed_vs_compute_ns": nccl - nccl_hidden,
                "copy_union_ns": copy,
                "copy_hidden_by_compute_ns": copy_hidden,
                "copy_exposed_vs_compute_ns": copy - copy_hidden,
                "exclusive_ns": exclusive,
                "copy_directions": directions,
                "nccl_collectives": coll,
                "kernel_events": len(kernel_rows[device]),
                "unresolved_kernel_ratio": unresolved_ns[device] / max(kernel_ns[device], 1),
            })

        stage_reports = []
        for label in STAGES:
            ranges = phases.get(label, [])
            if len(ranges) != 1:
                continue
            window = ranges[0]
            per_device = []
            for device in range(8):
                clipped = {
                    c: [x for i in by_device[device][c] if (x := _clip(i, window))]
                    for c in CATEGORIES
                }
                compute = sum(b - a for a, b in clipped["compute"])
                nccl = sum(b - a for a, b in clipped["nccl"])
                copy = sum(b - a for a, b in clipped["copy"])
                hidden_n = _intersection_unioned(clipped["nccl"], clipped["compute"])
                hidden_c = _intersection_unioned(clipped["copy"], clipped["compute"])
                per_device.append({"device": device, "wall_ns": window[1]-window[0], "compute_ns": compute,
                    "nccl_ns": nccl, "nccl_exposed_ns": nccl-hidden_n, "copy_ns": copy,
                    "copy_exposed_ns": copy-hidden_c, "no_traced_ns": _exclusive_unioned(clipped, window).get("no_traced_activity", 0)})
            stage_reports.append({"label": label, "window": list(window), "per_device": per_device})
        return {"node": node, "sqlite": str(path), "outer_window": list(outer), "devices": devices,
                "stages": stage_reports, "kernel_device_coverage": 8,
                "max_unresolved_kernel_ratio": max(d["unresolved_kernel_ratio"] for d in devices),
                "intervals": by_device}
    finally:
        db.close()


def _seconds(value: int) -> float:
    return value / 1e9


def _aggregate(nodes: list[dict]) -> dict:
    devices = [d for n in nodes for d in n["devices"]]
    wall = sum(d["wall_ns"] for d in devices)
    sums = {k: sum(d[k] for d in devices) for k in (
        "compute_union_ns", "nccl_union_ns", "nccl_hidden_by_compute_ns", "nccl_exposed_vs_compute_ns",
        "copy_union_ns", "copy_hidden_by_compute_ns", "copy_exposed_vs_compute_ns")}
    exclusive_keys = sorted({k for d in devices for k in d["exclusive_ns"]})
    exclusive = {k: sum(d["exclusive_ns"].get(k, 0) for d in devices) for k in exclusive_keys}
    collectives = {}
    for kind in ("all_gather", "reduce_scatter", "all_reduce", "p2p", "broadcast", "other_nccl"):
        collectives[kind] = {key: sum(d["nccl_collectives"][kind][key] for d in devices) for key in (
            "events", "union_ns", "sum_duration_ns", "hidden_by_compute_ns", "exposed_vs_compute_ns")}
    directions = {}
    for direction in ("h2d", "d2h", "d2d", "other"):
        directions[direction] = {key: sum(d["copy_directions"][direction][key] for d in devices) for key in (
            "events", "bytes", "union_ns", "sum_duration_ns")}
    stages = []
    labels = [s["label"] for n in nodes for s in n["stages"]]
    for label in dict.fromkeys(labels):
        rows = [r for n in nodes for s in n["stages"] if s["label"] == label for r in s["per_device"]]
        stage_wall = sum(r["wall_ns"] for r in rows)
        stages.append({"label": label, "gpu_wall_ns": stage_wall, **{
            key: sum(r[key] for r in rows) for key in ("compute_ns", "nccl_ns", "nccl_exposed_ns", "copy_ns", "copy_exposed_ns", "no_traced_ns")}})
    return {"gpu_wall_ns": wall, "metrics": sums, "exclusive_ns": exclusive,
            "nccl_collectives": collectives, "copy_directions": directions, "stages": stages,
            "rank_count": len(devices), "per_rank": devices}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def _plot(path: Path, nodes: list[dict], aggregate: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [2.4, 1, 1.2]})
    colors = {"compute": "#2563eb", "nccl": "#f97316", "copy": "#8b5cf6"}
    # Exact activity intervals, one row per rank.
    global_start = min(n["outer_window"][0] for n in nodes)
    for n in nodes:
        base = n["outer_window"][0]
        for device in range(8):
            rank = n["node"] * 8 + device
            for category in CATEGORIES:
                for a, b in _union(n["intervals"][device][category]):
                    axes[0].broken_barh([((_seconds(a-base)), _seconds(b-a))], (rank-0.38, 0.76),
                                        facecolors=colors[category], alpha=0.85 if category != "compute" else 0.65)
    axes[0].set_yticks(range(16)); axes[0].set_ylabel("global rank"); axes[0].set_xlabel("seconds from node-local range start")
    axes[0].set_title("Wan1.3B world16 FSDP2: compute / NCCL / memcpy timeline")
    axes[0].legend(handles=[Patch(color=colors[c], label=c) for c in CATEGORIES], ncol=3, loc="upper right")
    wall = aggregate["gpu_wall_ns"]
    keys = ["compute", "nccl", "copy", "compute+nccl", "compute+copy", "nccl+copy", "compute+nccl+copy", "no_traced_activity"]
    vals = [100*aggregate["exclusive_ns"].get(k,0)/wall for k in keys]
    axes[1].barh(["16-rank GPU-seconds"], [vals[0]], label=keys[0], color="#2563eb")
    left=vals[0]
    palette=["#f97316","#8b5cf6","#0f766e","#7c3aed","#c2410c","#be123c","#d1d5db"]
    for k,v,c in zip(keys[1:],vals[1:],palette): axes[1].barh(["16-rank GPU-seconds"],[v],left=left,label=k,color=c); left+=v
    axes[1].set_xlim(0,100); axes[1].set_xlabel("exclusive share of summed per-rank phase wall (%)"); axes[1].legend(ncol=4,fontsize=8,loc="upper center",bbox_to_anchor=(0.5,-0.35))
    stages=aggregate["stages"]
    labels=[s["label"].replace("WAN13B/","") for s in stages]
    compute=[100*s["compute_ns"]/s["gpu_wall_ns"] for s in stages]
    nexp=[100*s["nccl_exposed_ns"]/s["gpu_wall_ns"] for s in stages]
    cexp=[100*s["copy_exposed_ns"]/s["gpu_wall_ns"] for s in stages]
    x=range(len(labels)); axes[2].bar(x,compute,label="compute union",color="#2563eb"); axes[2].bar(x,nexp,bottom=compute,label="NCCL exposed vs compute",color="#f97316"); axes[2].bar(x,cexp,bottom=[a+b for a,b in zip(compute,nexp)],label="copy exposed vs compute",color="#8b5cf6")
    axes[2].set_xticks(list(x),labels,rotation=35,ha="right",fontsize=8); axes[2].set_ylabel("% stage GPU-seconds"); axes[2].legend(ncol=3); axes[2].set_title("Stage activity (overlap-aware; bars are diagnostic, copy and NCCL may overlap each other)")
    fig.tight_layout(); fig.savefig(path,dpi=170); plt.close(fig)


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--node0-sqlite",type=Path,required=True);p.add_argument("--node1-sqlite",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args()
    nodes=[analyze_node(a.node0_sqlite,0),analyze_node(a.node1_sqlite,1)]
    aggregate=_aggregate(nodes)
    valid=(all(n["kernel_device_coverage"]==8 and n["max_unresolved_kernel_ratio"]<0.01 for n in nodes) and aggregate["rank_count"]==16)
    serializable_nodes=[]
    for n in nodes:
        serializable_nodes.append({k:v for k,v in n.items() if k!="intervals"})
    result={"schema_version":"distillschedule.wan13b_nsys.v1","status":"PASS" if valid else "FAIL_PROFILE_VALIDITY",
            "diagnostic_only":True,"formal_speedup_eligible":False,"phase_label":OUTER,"nodes":serializable_nodes,"aggregate":aggregate,
            "definitions":{"nccl_exposed_vs_compute":"NCCL interval union minus its intersection with non-NCCL kernel union on the same GPU",
            "no_traced_activity":"phase wall not covered by CUDA kernel or memcpy activity; not proof of CPU idle",
            "denominator":"sum of the 16 per-GPU outer-range walls; GPU-seconds, not global wall-clock"}}
    a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/"wan13b_nsys_attribution.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    rows=[]
    for d in aggregate["per_rank"]:
        rows.append({"rank":d["rank"],"wall_s":_seconds(d["wall_ns"]),"compute_union_s":_seconds(d["compute_union_ns"]),"nccl_union_s":_seconds(d["nccl_union_ns"]),"nccl_hidden_s":_seconds(d["nccl_hidden_by_compute_ns"]),"nccl_exposed_s":_seconds(d["nccl_exposed_vs_compute_ns"]),"copy_union_s":_seconds(d["copy_union_ns"]),"copy_exposed_s":_seconds(d["copy_exposed_vs_compute_ns"]),"no_traced_s":_seconds(d["exclusive_ns"].get("no_traced_activity",0)),"unresolved_kernel_ratio":d["unresolved_kernel_ratio"]})
    _write_csv(a.output_dir/"per_rank_attribution.csv",rows)
    stage_rows=[]
    for s in aggregate["stages"]:
        stage_rows.append({"stage":s["label"],"gpu_wall_s":_seconds(s["gpu_wall_ns"]),"compute_union_s":_seconds(s["compute_ns"]),"nccl_union_s":_seconds(s["nccl_ns"]),"nccl_exposed_s":_seconds(s["nccl_exposed_ns"]),"copy_union_s":_seconds(s["copy_ns"]),"copy_exposed_s":_seconds(s["copy_exposed_ns"]),"no_traced_s":_seconds(s["no_traced_ns"])})
    _write_csv(a.output_dir/"stage_attribution.csv",stage_rows)
    _plot(a.output_dir/"wan13b_nsys_timeline.png",nodes,aggregate)
    wall=aggregate["gpu_wall_ns"];m=aggregate["metrics"]
    md=f'''# Wan1.3B world16 FSDP2 Nsight attribution\n\nStatus: **{result['status']}** (diagnostic only).\n\nDenominator: summed 16-rank GPU phase walls = {_seconds(wall):.3f} GPU-s. Profile wall is not a formal latency result.\n\n| Category | Union GPU-s | Hidden by compute GPU-s | Exposed vs compute GPU-s | % GPU-wall |\n|---|---:|---:|---:|---:|\n| Compute | {_seconds(m['compute_union_ns']):.3f} | -- | -- | {100*m['compute_union_ns']/wall:.2f}% |\n| NCCL | {_seconds(m['nccl_union_ns']):.3f} | {_seconds(m['nccl_hidden_by_compute_ns']):.3f} | {_seconds(m['nccl_exposed_vs_compute_ns']):.3f} | {100*m['nccl_union_ns']/wall:.2f}% |\n| Memcpy | {_seconds(m['copy_union_ns']):.3f} | {_seconds(m['copy_hidden_by_compute_ns']):.3f} | {_seconds(m['copy_exposed_vs_compute_ns']):.3f} | {100*m['copy_union_ns']/wall:.2f}% |\n| No traced GPU activity | {_seconds(aggregate['exclusive_ns'].get('no_traced_activity',0)):.3f} | -- | -- | {100*aggregate['exclusive_ns'].get('no_traced_activity',0)/wall:.2f}% |\n\n`NCCL exposed vs compute` is an overlap-aware upper bound, not automatically removable wall time. `No traced activity` is not automatically CPU idle.\n'''
    (a.output_dir/"wan13b_nsys_attribution.md").write_text(md)
    print(json.dumps({"status":result["status"],"gpu_wall_s":_seconds(wall),"compute_pct":100*m["compute_union_ns"]/wall,"nccl_pct":100*m["nccl_union_ns"]/wall,"nccl_exposed_pct":100*m["nccl_exposed_vs_compute_ns"]/wall,"copy_pct":100*m["copy_union_ns"]/wall,"no_traced_pct":100*aggregate["exclusive_ns"].get("no_traced_activity",0)/wall},indent=2))
    return 0 if valid else 2

if __name__=="__main__": raise SystemExit(main())

#!/usr/bin/env python3
"""Measure host/device copy bandwidth without modifying model state."""

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import torch


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def measure_copy(dst, src, iterations, warmup, non_blocking):
    for _ in range(warmup):
        dst.copy_(src, non_blocking=non_blocking)
    torch.cuda.synchronize()
    samples_ms = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        dst.copy_(src, non_blocking=non_blocking)
        end.record()
        end.synchronize()
        samples_ms.append(float(start.elapsed_time(end)))
    return samples_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes-mib", type=int, nargs="+", default=[256, 1024, 4096])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    results = {
        "schema_version": 1,
        "timestamp": time.time(),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(args.device),
        "device_index": args.device,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "measurements": [],
    }

    for size_mib in args.sizes_mib:
        nbytes = size_mib * 1024 * 1024
        gpu = torch.empty(nbytes, dtype=torch.uint8, device=device)
        for pinned in (False, True):
            try:
                host_src = torch.empty(nbytes, dtype=torch.uint8, pin_memory=pinned)
                host_dst = torch.empty(nbytes, dtype=torch.uint8, pin_memory=pinned)
            except Exception as exc:
                results["measurements"].append(
                    {"size_mib": size_mib, "pinned": pinned, "status": "allocation_failed", "error": repr(exc)}
                )
                continue
            for direction, dst, src in (("h2d", gpu, host_src), ("d2h", host_dst, gpu)):
                for non_blocking in (False, True):
                    print(
                        f"starting {size_mib} MiB {direction} pinned={pinned} non_blocking={non_blocking}",
                        flush=True,
                    )
                    try:
                        samples = measure_copy(dst, src, args.iterations, args.warmup, non_blocking)
                        median_ms = statistics.median(samples)
                        results["measurements"].append(
                            {
                                "size_mib": size_mib,
                                "bytes": nbytes,
                                "direction": direction,
                                "pinned": pinned,
                                "non_blocking": non_blocking,
                                "status": "measured",
                                "median_ms": median_ms,
                                "p10_ms": percentile(samples, 0.10),
                                "p90_ms": percentile(samples, 0.90),
                                "bandwidth_gib_s": (nbytes / 1024**3) / (median_ms / 1000.0),
                                "samples_ms": samples,
                            }
                        )
                    except Exception as exc:
                        results["measurements"].append(
                            {
                                "size_mib": size_mib,
                                "direction": direction,
                                "pinned": pinned,
                                "non_blocking": non_blocking,
                                "status": "failed",
                                "error": repr(exc),
                            }
                        )
            del host_src, host_dst
        del gpu
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != "measurements"}, indent=2))
    for row in results["measurements"]:
        if row.get("status") == "measured":
            print(
                f"{row['size_mib']:5d} MiB {row['direction']} "
                f"pinned={row['pinned']} non_blocking={row['non_blocking']} "
                f"{row['median_ms']:.3f} ms {row['bandwidth_gib_s']:.2f} GiB/s"
            )
        else:
            print(row)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""C27 one-owned-GPU real-prompt canary; synthetic busy-stream latency only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import statistics
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--plan-only", action="store_true")
    a = p.parse_args()
    out = {"status": "NOT_RUN", "world16": "NOT_RUN", "production_installed": False,
           "kind": "REAL_TRACE_PROMPTS_SYNTHETIC_BUSY_STREAM", "cases": [],
           "claim_boundary": "INPUT_IDENTITY_AND_HOST_ENQUEUE_ONLY_NOT_WORKLOAD_SPEEDUP"}
    if a.plan_only:
        print(json.dumps(out))
        return 0
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        tmp = a.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, indent=2) + "\n")
        tmp.replace(a.output)

    save()
    try:
        import torch
        from distillschedule.runtime.livewan_crossscale_worker import FormalTrace, InputLoader
        from distillschedule.runtime.livewan_context_staging import PinnedScoreContextStager, BYTES_PER_REQUEST
        assert torch.cuda.device_count() == 1
        torch.cuda.set_device(0)
        trace = FormalTrace.load(a.trace, expected_cycles=2)
        loader = InputLoader(torch=torch, trace=trace)
        selected = list(dict.fromkeys(row.dataset_index for row in trace.rows))[:3]
        helper = PinnedScoreContextStager(torch, loader.prompts, allowed_indices=set(selected))
        out.update(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                   trace_sha256=trace.sha256, prompts_sha256=trace.prompts_sha256)
        device = torch.device("cuda:0")
        rng_cpu, rng_cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        for index in selected:
            row = SimpleNamespace(dataset_index=index)
            ref = loader.context_from_trace(row, device=device)
            got = helper.context_from_trace(row, device=device)
            assert torch.equal(ref, got) and ref.dtype == got.dtype == torch.float32
            assert ref.shape == got.shape == (1, 512, 4096)
            result = {"dataset_index": index, "output_bitwise_equal": True,
                      "source_pinned": helper.cache[index].is_pinned(), "arms": {}}
            for name, func in (("pageable_reference", loader.context_from_trace), ("pinned_candidate", helper.context_from_trace)):
                samples = []
                for _ in range(6):
                    torch.cuda.synchronize()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    # Explicit diagnostic queued-work emulator, NOT model timing.
                    torch.cuda._sleep(50_000_000)
                    t = time.perf_counter()
                    value = func(row, device=device)
                    host_ms = (time.perf_counter() - t) * 1000
                    end.record()
                    end.synchronize()
                    samples.append({"host_enqueue_ms": host_ms, "cuda_sleep_plus_copy_ms": start.elapsed_time(end)})
                    assert torch.equal(value, ref)
                result["arms"][name] = {"samples": samples,
                    "median_host_enqueue_ms": statistics.median(x["host_enqueue_ms"] for x in samples),
                    "median_cuda_sleep_plus_copy_ms": statistics.median(x["cuda_sleep_plus_copy_ms"] for x in samples)}
            out["cases"].append(result)
            save()
        assert torch.equal(rng_cpu, torch.get_rng_state()) and torch.equal(rng_cuda, torch.cuda.get_rng_state())
        r = helper.receipt()
        assert r["requests"] == 21 and r["h2d_bytes"] == 21 * BYTES_PER_REQUEST and r["cache_entries"] == 3 and r["contract_failures"] == 0
        out.update(status="PASS_CANARY_ONLY", rng_equal=True, staging_receipt=r,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   driver_free_bytes=torch.cuda.mem_get_info()[0])
        save()
        print(json.dumps(out))
        return 0
    except Exception:
        import traceback
        out.update(status="FAIL_CANARY", error=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())

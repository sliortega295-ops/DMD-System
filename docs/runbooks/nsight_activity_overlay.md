# Nsight semantic and CUDA activity overlay

The semantic PhaseTracer/NVTX bar is a host-side envelope. It may contain CUDA
compute, NCCL, copies, synchronization, optimizer work and waits. Use
`analyze_nsys_activity_overlay.py` to put the semantic envelope and real CUDA
activity strips on one aligned time axis.

```bash
python scripts/distillschedule/analyze_nsys_activity_overlay.py \
  --input 2=/path/to/rank2.sqlite \
  --input 9=/path/to/rank9.sqlite \
  --capture-range 2=<start_ns>:<end_ns> \
  --capture-range 9=<start_ns>:<end_ns> \
  --bin-ms 500 \
  --output-json /tmp/nsight_activity.json \
  --output-csv /tmp/nsight_activity.csv \
  --plot-dir /tmp/nsight_activity_plots
```

A provided closed profiler range is preferred. If omitted, the script uses the
union extent of discovered CUDA/NVTX records and labels the denominator as an
inferred fallback. Finalization or receipt tails should be excluded by passing
the exact closed range.

Each rank plot has two aligned layers:

1. semantic Phase A / A→B / Phase B / B→A envelopes;
2. Compute, NCCL, H2D, D2H and no-traced-GPU-activity strips.

The default display resolution is 0.5 seconds. A strip value is the **exact
interval union inside that display bin**, divided by the bin wall duration; the
underlying phase and capture percentages remain unbinned exact unions.
Exact, inferred and receipt-reconstructed semantic windows use different hatch
patterns.

The report contains independent interval unions for Compute, NCCL, H2D, D2H,
D2D and all-copy. They may overlap and must not be added. `Active` is the union
of all traced CUDA categories. `No traced GPU activity` is its complement; it
is not proof of CPU or scheduler idle.

Aggregates use GPU-seconds:

```text
sum(rank activity union) / sum(rank phase wall)
```

Profiled latency is diagnostic only. Formal speedup still comes from matched,
unprofiled exact wall-clock runs.

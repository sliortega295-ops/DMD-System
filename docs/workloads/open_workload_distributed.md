# Generic workload execution: replicated and sharded layouts

Active implementation/review: PR #34, `agent/general-workload-planning-20261012`.
PR #35's alternative ABI was inspected but is not copied into a second public
system. Its branch is preserved; this PR is the continuation line.

## User contract versus system decisions

A factory supplies ordinary model, loss, backward and update functions through
`Workload(Program(...), bindings, state, feeds)`. Model names and known-winner
recipes are not queried by the public capability matcher or shared DP service.
Parameters must change the actual factory computation/graph, not just metadata.
For data parallelism add `ReplicaSemantics(module_bindings, backward_calls,
local_batch_size)`: ordinary trainable-root forward interfaces, parameter
backward call IDs and an equal local-mean objective. These are training facts,
not optimization choices. Inputs may differ across ranks, but initial models,
optimizer states, logical graph, dtype and input geometry must agree.

The system currently matches and measures:

| Capability | Actual execution |
|---|---|
| Post-backward mean | SUM/world-size on parameter gradients after backward |
| DDP overlap, 1/128 MiB buckets | Native DDP plus the EXISTING `runtime.wan13b_fair.ddp_sum_then_divide_hook` |
| Immutable aliases | Same frozen parameter storage, independent adapters/optimizers |
| Nonreentrant checkpoint | User-declared replay-safe regions; backward still runs |
| FSDP2 reshard | Native `fully_shard`, declared trainable roots, parameter/gradient/optimizer sharding |
| FSDP2 keep-forward | Same sharding with `reshard_after_forward=False` |
| HSDP2 reshard/keep-forward | Native replicate-by-node, shard-within-node mesh, admitted only for complete equal multi-rank nodes |
| Per-module layouts | Bounded combinations of admitted replicated/DDP/FSDP2/HSDP2 layouts for separate declared model roots |

FSDP roots include their frozen base parameters, but other frozen-only model
roots remain resident. Aliases across separate roots are NOT combined with
FSDP2; unsupported mixed-dtype root groups and non-pointwise optimizer state are
rejected. These restrictions generate fallback candidates, not fake support.
FSDP2's new Parameter identities are installed into the EXISTING optimizer groups;
existing dense Adam moments are distributed rather than initialized to zero.

All choices have measured per-call costs from paired executions. The same
`search/service.py` ranks candidates under the declared per-device capacity.
The planner adds native forward/backward-completion dependencies before search;
an order that is only mathematically valid is insufficient for a stateful reducer.
Rank zero chooses and publishes one plan. The executor reconstructs the native
constraints and all ranks agree on identity/order before model collectives.

## Runnable CPU example (no models downloaded)

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test,native]'
OMP_NUM_THREADS=1 python -m torch.distributed.run --standalone --nproc-per-node=2 \
  scripts/distillctl.py workloads native program auto \
  --factory examples/open_workloads/replicated_input_gradient.py:build \
  --parameters examples/open_workloads/replicated_parameters.json \
  --distributed gloo --group-timeout 180 --output /tmp/new-program-run \
  --max-candidates 8 --samples 1 --max-program-runs 11 \
  --memory-limit-bytes 10000000 --execute
```

The example has random Conv3d models, a frozen input-differentiable critic and
actual Adam. `width` is feature channels; `height`/`image_width` are spatial
sizes. Change these, `frames` or `updates` in the JSON: the actual tensors and
logical calls change and costs must be recalibrated. No launcher, capability or
planner source edit is required. An unsupported arbitrary Python program is
not guessed into this ABI. Factories are trusted explicit user code.

`plan --calibration ... --memory-limit-bytes ... --output ...` is metadata-only
and can execute without torchrun or constructing a model. `describe`, `calibrate`
and `run` remain available individually. No `--execute` means no factory/group
execution. Each fresh `auto` uses an explicit total program-run budget including
the reference, candidate repetitions and selected-plan confirmation. Limited
candidate budgets are labeled incomplete enumeration, not global optimality.

For allocated CUDA ranks, `--distributed nccl` sets LOCAL_RANK before the factory;
the factory must return tensors on that rank's CUDA device. CUDA/NCCL source
paths exist but have NOT been GPU-validated in this work. Do not use CPU timing
or this example as video-model speedup or training-quality evidence.

## Evidence and limits

Tests use actual 2-process Gloo, native DDP/FSDP2, autograd and Adam, not mocked
collectives. They compare parameters and optimizer states to independently
concatenated single-process global-batch references. Two trainable roles retain
separate adapters/optimizers; frequency and input-shape changes are tested.
Tests also exercise public torchrun -> measured calibration -> common DP ->
selected execution, rank-specific stale input rejection and completion edges.
A separate measured FSDP candidate is replayed even when the CPU timing winner
is a replicated strategy; that replay is not called an automatic FSDP win.
Four-process CPU/Gloo coverage also exercises HSDP2 on two **logical** nodes on
one host, including warmed Adam moments, a global-batch reference and measured
common-plan replay (`test_open_workload_hsdp.py`). This does not establish
physical multi-node NCCL behavior or A100/H100 GPU validity (NOT_RUN).

CPU FSDP tests require Torch >=2.10. The distributed CI pins 2.10 so these tests
are executed rather than silently skipped. Older CPU environments retain the
replicated path and explicitly decline FSDP2. Original canonical paths are not
migrated to a newer Torch runtime by this test workflow.

This is still an admitted finite-program executor, NOT an unrestricted production
trainer: HSDP2 requires complete equal multi-rank node groups, and this callable
data-parallel path does not automatically lower arbitrary role pools or general
activation offload. Bounded role pools have a separate
[RoleTask/RoleFunctions contract](open_role_pools.md). Mid-run topology changes
and resumable streaming training remain unsupported. Factories initially
construct dense state and calibration clones it; sharding does not yet solve
large-model initialization capacity. CPU capacity uses storage accounting with
explicit DDP/FSDP staging allowances, not an RSS bound. GPU allocator behavior,
full pretrained workloads and end-to-end throughput remain unverified.

Calls stay serial across the data-parallel group. DDP overlaps gradient
communication WITHIN backward; it is not Generator/Score pipeline overlap.
Parameter `.backward()` and optimizer updates must be separate semantic calls;
input/latent gradients are not averaged. Unequal per-rank batches, hidden
stateful closures, sparse/higher-order parameter gradients and undeclared
side effects require additional contracts. Numerical mismatches remain REVIEW.

## A100/H100 hardware-specific calibration (opt-in public backend)

For an already allocated A100/H100 CUDA/NCCL group, the same `program auto`
command performs bounded native candidate calibration on the **current** GPUs.
It fingerprints GPU family/SM (A100 SM80, H100 SM90), total VRAM, device UUID,
Torch/CUDA/NCCL runtime and allocated rank topology. Cost receipts cannot be
replayed on another accelerator/environment without new measurements. GPU
family is a **capability identity**, not an instruction to pick a fixed winner.

Actual per-call costs remain the synchronized **host wall** times (including
completion fences and measured NCCL effects); separate CUDA events expose
on-stream GPU execution time. Automatically collected, scratch-only 4 MiB
pinned H2D and 1/8 MiB NCCL all-reduce samples are saved under
`calibration.json:hardware_probe`. Those microbenchmarks **never** replace
actual generator/backward/optimizer cost data or claim to represent an
end-to-end communication critical path. Use `--no-hardware-probe` to disable
the diagnostic probes while preserving actual candidate timing.

In CUDA `auto`, `--memory-limit-bytes` is optional. The default budget is the
minimum per-rank currently free + current process allocator-reserved reusable
VRAM, minus a 2 GiB reserve. Use `--gpu-reserve-gib N` (0..16) to adjust
reserve, or provide an explicit smaller `--memory-limit-bytes`. The budget and
per-rank snapshot are retained in the plan; execution checks current real
headroom again before attempting a selected CUDA plan. CPU still requires an
explicit memory budget (tracked tensors are not whole-process RSS).

These features do NOT supply an H100-only kernel, automatic role-pool lowering
of arbitrary Workload graphs, layerwise bandwidth roofline, full GPU Nsight trace or verified
A100/H100 numerical throughput. Hardware-specific strategy selection is
conditioned on **real** candidate measurements, not assumed winners. Mixed
GPU models within a rank group and inter-node transport require dedicated
admission and GPU verification before performance claims.

### Deep trace only after selecting a validated plan

`--deep-profile` on `program auto` or `program run` wraps only the selected
execution, not every calibration candidate. It emits a per-rank PyTorch Kineto
Chrome/Perfetto trace (`torch-profiler-rankXXXX.json`) with logical-call labels,
CPU/CUDA operator activity and native memory events. Paths and SHA256 digests
are embedded in `execution.json` and never reused as speedup evidence. This
flag adds instrumentation overhead; run a separate **unprofiled**, matched
train/optimizer/EMA validation when reporting speedups. CUDA traces require
supported local PyTorch/CUPTI installation; if unavailable, fail explicitly.

The `--deep-profile` flag is not a substitute for Nsight Systems on multi-GPU
critical paths or Nsight Compute on kernel roofline/MFU. Those require external
tools and valid access to the actual A100/H100 cluster. MFU is not inferred
from torch.profiler's partial FLOP coverage.

### Profile-guided bottleneck and upper-bound report

A metadata-only `program plan` now emits `opportunity_report` for its selected
measured candidate: per-call median wall time, share of the currently SERIAL
plan, maximum-to-median rank slowdown and per-call Amdahl *what-if* bounds for
halving or eliminating that call. Binding-level cost shares reveal whether the
large opportunity is in a forward, backward, communication-containing update,
or data movement; the report does not claim to isolate NCCL without profiling.
Amdahl numbers use the **observed**, single-allocation model and assume all
other work unchanged. They are not measured speedups, not the upper bound of a
future cross-role overlapping plan, and do not count incomplete FLOPs as MFU.
Selectively run `--deep-profile` for the most expensive unclear call, rather
than optimizing small kernels opportunistically or tracing every candidate.

#### Standalone A100/H100 probe without loading 14B model weights

On an existing single GPU: `python scripts/distillctl.py workloads native
program hardware --output /shared/gpu-profile.json --execute`. For multi-GPU
communication, launch the **same command** under an existing 2-node torchrun
allocation with `--distributed nccl` and one local GPU per rank. It performs
bounded scratch H2D and NCCL all-reduce operations, collecting a per-rank
hardware-fingerprint/probe receipt. It does not initialize Wan/Helios/H3
weights, attempt any optimization, or pretend that a microbenchmark is a
complete training-workload cost model. CPU testing is supported but labeled
`NOT_GPU_PROFILE` rather than claiming an A100/H100 measurement.

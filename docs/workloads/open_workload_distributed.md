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

CPU FSDP tests require Torch >=2.10. The distributed CI pins 2.10 so these tests
are executed rather than silently skipped. Older CPU environments retain the
replicated path and explicitly decline FSDP2. Original canonical paths are not
migrated to a newer Torch runtime by this test workflow.

This is still an admitted finite-program executor, NOT an unrestricted production
trainer: no automatic HSDP, arbitrary role pools, general activation offload,
mid-run topology changes or resumable streaming training. Factories initially
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

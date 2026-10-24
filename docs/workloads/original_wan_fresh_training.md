# Original Wan: reuse a selected strategy with fresh input cohorts

`auto-run` prepares and searches on a fixed, paired calibration cohort. The new
`prepare-stream` / `train-stream` path instead advances input cohorts and absolute
source steps while retaining the selected original engines and optimizer/EMA
state. Both use the original Wan/Helios-pyramid no-GAN mathematics, not the
separate Euler/Diffusers fixture protocol.

## Preparation

Provide a JSON object with one `cohorts` array. Each array item is a complete
`WanInputSources` object as documented in `original_wan_auto_run.md`: sixteen
feature paths, a seed and cohort index. There must be 2..64 entries with distinct
(seed, cohort) identities. Source clips can recur if intentionally declared;
the system does not claim that videos are globally unique. The task keeps
geometry, rank, horizons, update ratio and EMA interval fixed for this run.
Only absolute steps advance: cycle i starts at first_step + i * fake_updates.

```bash
python scripts/distillctl.py workloads native prepare-stream \
  --task task.json --sources cohort-sources.json \
  --training-config training.yaml --output /shared/wan-input-stream --execute
```

Without --execute this validates only. --resume reuses complete, matching input
captures; partial captures are not silently overwritten or resampled. This is
input-preparation resume, NOT a resumable training checkpoint.

## Training

After validating a selected plan and its matched-reference result, launch the
same original worker on each already allocated node:

```bash
python -m torch.distributed.run --nnodes=2 --nproc-per-node=8 \
  --node-rank=0 --master-addr=NODE0_HOST --master-port=29402 --max-restarts=0 \
  scripts/distillctl.py workloads native train-stream \
  --task /shared/new-wan-task/campaign/task.json \
  --assets /shared/new-wan-task/campaign/assets.json \
  --plan /shared/new-wan-task/campaign/selected-plan.json \
  --stream /shared/wan-input-stream/stream.json --output /shared/wan-training --execute
# Other node: node-rank=1, otherwise identical paths and rendezvous.
```

The complete stream is validated before model construction. Engines are built
once from the supplied original initialization. At completed-cycle boundaries,
only trace/task bindings and input-derived Python caches change. Parameters,
Adam moments, gradient buckets and EMA persist. Every rank agrees on cohort,
order and versions before running its original handlers. Selected G/R/C order is
retained; EMA operations are inserted using the current absolute-step cadence,
not blindly copied from the first cycle. Current graph validation checks every
operation and retained-window constraint.

The plan is reused as an execution POLICY, not as an accurate latency/capacity
prediction for new random inputs and exits. New data can still expose an OOM or
numerical issue; these fail rather than shrinking inputs or relabeling timings.
The output distinguishes fresh cohorts from fixed-trace diagnostics, reports
model-cycle time and input-plus-execution wall time separately, and saves final
trainable/Adam/EMA evidence. Comparisons reject different stream identities.

CPU tests execute real Adam/autograd on explicitly small scalar helper fixtures
through the original dispatcher and show parity across legal window orders.
They are NOT original-Wan model quality, CUDA/FSDP correctness, full GPU capacity
or throughput evidence. The full 16-GPU stream/restart path requires its own
execution receipt. This does not support mid-run geometry changes, automatic
checkpoint recovery, GAN, or other model protocols.

## Native zero-history branch (explicit Resident scope)

Set `stage_schedule_policy: source_native_amplified_v1` in a **new task** when
preparing fresh cohorts that can contain zero history. The existing default
`source_normal_only` and immutable benchmark traces are unchanged. Neither
history dropout nor seeds are altered to avoid this branch.

The new policy preserves the original three synchronized initial exit draws.
For a rank whose three history sums are all zero, the original scheduler uses
twice the stage horizon and replaces the initial exits with rank-local native
draws. Other ranks retain their initial exits. Capture records both decisions,
the actual rank, doubled budgets, and actual call counts; the worker checks the
predicate against the payload and validates each rank's own executed census.
There are no padded or discarded model calls. Capture uses reproducible isolated
CPU input streams, **not** the RNG continuation of an earlier upstream run.

This scope requires Resident forward execution without parameter collectives.
FSDP is explicitly rejected: native rank-local exit counts can differ and its
forward all-gathers do not yet have a collective-alignment implementation.
The stream-capable `wan_ar.incumbent21a_stream.v2` preset accepts this policy at
its original geometry. Graph limits, parameter/input version checks, optimizer,
EMA and capacity admission are not relaxed. Fixed-trace v1 is not relabeled.

Use the public `distillctl train` entry with `data.input_kind` set to
`original_wan_cohort_stream.v1`, `data.prepared` pointing to `stream.json`, and
an explicit checkpoint interval. Only complete-cycle, same-source, same-stream,
same-topology checkpoints can resume; transient graphs are rebuilt. CPU capture
and replay tests establish branch/census behavior, not model quality or GPU
restart equivalence.

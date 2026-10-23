# Wan incumbent21a-v1: task-bound opt-in bundle

This is a bounded port of an existing mechanism, **not a new optimization or a
promotion of historical timing**. Existing independent placement/chunk/no-grad
LoRA-epilogue/window choices, IDs, defaults and campaigns remain unchanged.

## Public route

The existing `workloads native` front door remains the execution route:

1. `validate --include-incumbent-bundle` advertises the typed candidate only when
   Triton is available and the task is exactly `OriginalWanTask()`.
2. `calibrate --implementation <strategy.json> --execute` runs the original
   native world16 handlers for two fixed-trace cycles and emits per-rank action,
   storage-lifetime, backend-effect and precision receipts.
3. `assemble --rank-receipts ...` requires sixteen completed current-source rank
   files, reconciles the bundle effects, and emits a measured collection/costs.
4. `plan --calibration <control-collection> <bundle-collection>` calls the SAME
   `distillschedule.search.service.select` and event-frontier scheduler used by
   the existing adapter. It saves a source/asset/cost/graph-bound plan. The bundle
   can lose to the control; no historical duration is a search cost.
5. `run --plan <saved-plan> --execute --cycles 2` independently validates the
   plan/costs, then uses `OriginalWanSession` and the original prepared action
   handlers. It does not dispatch to the historical runner. GPU work is still
   explicit and requires two nodes of eight GPUs; this document launches none.

CPU factory example, suitable for writing explicit strategy files:

```python
from distillschedule.strategies.wan_task import OriginalWanTask
from distillschedule.strategies.wan_strategy import implementations
choices = implementations(OriginalWanTask(), triton_available=True,
                          incumbent_bundle_available=True)
control = choices[0].to_dict()  # Original FSDP control, all bundle fields off.
bundle = choices[-1].to_dict() # resident/chunk2048/no-grad epilogue + typed v1.
```

Example command forms after source review/application, using explicit paths and
an already validated environment (do not execute without allocation authority):

```bash
"$PYTHON" "$CHECKOUT/scripts/distillctl.py" workloads native validate \
  --task "$RUN/task.json" --include-incumbent-bundle --output "$RUN/capabilities.json"
# On EACH existing node, under its already authorized 2-node torchrun allocation:
"$PYTHON" "$CHECKOUT/scripts/distillctl.py" workloads native calibrate \
  --task "$RUN/task.json" --assets "$RUN/assets.json" \
  --implementation "$RUN/bundle-strategy.json" --output "$RUN/bundle-calibration" --execute
# Repeat the same command with the independently declared FSDP control strategy.
"$PYTHON" "$CHECKOUT/scripts/distillctl.py" workloads native assemble \
  --task "$RUN/task.json" --rank-receipts "$RUN/bundle-calibration"/rank_*.json \
  --output "$RUN/bundle-costs"
# Assemble the control separately, then:
"$PYTHON" "$CHECKOUT/scripts/distillctl.py" workloads native plan \
  --task "$RUN/task.json" --calibration "$RUN/control-costs/collection.json" \
  "$RUN/bundle-costs/collection.json" --output "$RUN/selected-plan.json"
# On EACH existing node, under its authorized allocation:
"$PYTHON" "$CHECKOUT/scripts/distillctl.py" workloads native run \
  --task "$RUN/task.json" --assets "$RUN/assets.json" \
  --plan "$RUN/selected-plan.json" --cycles 2 --output "$RUN/confirmation" --execute
```

A CPU fixture collection cannot launch the worker. Current-source GPU numerical
acceptance remains required against matched reference trainables/Adam/EMA;
`compare_runs` retains REVIEW on failure. The existing worker does not promote a
completed run or its action-calibration timings to formal speedup eligibility.

## Capability and semantic boundary

`WanCandidateBundle` v1 has five explicit booleans, all default-off:
`grad_lora_fused_epilogue`, `modulated_norm_fusion`,
`generator_rollout_cuda_graph`, `generator_nograd_lora_cast_cache`, and
`critic_grad_cuda_graph`. Only the all-off or this coupled all-on bundle is
admitted; arbitrary five-axis cross-products are rejected. The old no-grad
LoRA epilogue is still its existing separate axis and must be enabled here.

Admission is restricted to the original 384x640/33-frame, B1/world16, rank256,
G1/C5/EMA5 task, steps10-14, original bootstrap G2/C10/EMA10, fixed legacy trace
bundle and binding, negative-prompt manifest, and actual base-model file hashes
reconciled with the immutable bootstrap inventory. New task geometries, new
traces, fresh-cohort streams, more than two cycles and another bootstrap fail
closed. Rollout signatures, pointer/layout guards, dropout restrictions, first-
cycle capture limits and literal G3/G4/27-calls/5-entries/three-shape census stay
unchanged. The common planner adds only a legal schedule restriction r0->r4;
window/memory choices remain measured decisions, not physical batching.

The exact original reducer, gradient convention, clipping, Adam, checkpointing,
EMA and FP32 trainable/BF16 frozen-storage policies remain outside graph replay.
Critic graphs return parameter gradients to original accumulators/hooks. Norm
counters are process-shared and are NOT added across duplicate role receipts.
Cast reuse retains the original Tensor.to result and ordinary LoRA-A GEMM.
Merged LoRA, rejected input GEMM, detached Score graphs, G-grad graphs, fused
optimizer and foreach EMA are not registered or activated by this bundle.

## Precision and evidence identity

The port uses the CURRENT generated Wan precision contract, equally for control
and bundle: `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`, declared matmul/cuDNN TF32 policy,
and no `NVIDIA_TF32_OVERRIDE`. Runtime flags, matmul precision, package/runtime
versions and observed parameter dtypes remain bound in actual receipts. No
candidate-only precision override is introduced.

The historical 21a observation was 20.31478561926633 s C2 world-max on A100 and
used a nonempty driver override. It is provenance only; this source/precision/
hardware combination has no measured time yet and no 20.3-second promise.
Historical comparison status contains `50C`, but that particular evidence had
only TWO cycles and one seed. It is not 50-cycle equivalence or quality proof.

Every bundle-bearing component cost includes all five capability fields and
actual SHA256s of the imported mechanisms plus existing cast/epilogue code.
Current source fingerprints, task, full assets, runtime, rank hardware and raw
rank receipts remain bound by the existing collection and saved-plan gates.
Neither old eager costs nor historical timing is borrowed.

## Explicitly not admitted

- Legacy five-profile/macro trajectory/timing lowering and old summarizers:
  `registry.candidate_profile` rejects this new ID rather than weakening eager
  counter assertions. Registry identity/parse works, but execution uses the
  task-bound public route above.
- Source-pinned world8 H100 hardware adapter and world8 logical-slot bridge:
  unchanged and NOT_ADMITTED for this bundle.
- General task-derived graph census: not implemented; literal guards retained.
- CUDA/numerical/current-source timing validation: Pending cluster validation.

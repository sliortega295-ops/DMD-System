# Wan1.3B explicit B1 weak-scaling experiment

This opt-in experiment admits **8 or 16 GPUs**, B1 per global rank. Logical
global batch is respectively **8 or 16**; it is not a fixed-global-batch
strong-scaling comparison. Every cycle remains one Generator optimizer update,
five Critic updates and five EMA updates. Report `global_batch / C2 wall`
samples/s, and compare FSDP2 versus the unchanged DMD-System winner at **the
same** admitted GPU count and source HEAD.

`WanDistributedExecution(experiment="weak_scaling", world_size=8|16)` explicitly
binds the live world, FSDP mesh, SUM/divisor, sample accounting and trace size.
The default remains world16. There is no assignment or monkeypatch to any
legacy `WORLD_SIZE` global. World8 is admitted only for the existing FSDP2
implementation and Resident async128/Tree/chunk2048 winner; unrelated arms do
not silently become world8-capable.

## Frozen workload and inputs

- Wan2.1-1.3B, native 30-block Helios compatibility loader; existing checkpoint,
  rank256 LoRA bootstrap, missing-pyramid patch seed and negative embedding.
- BF16 frozen backbone, FP32 trainable LoRA, FP32 gradient mean reduction.
- G1/C5/EMA5, unchanged optimizer, update order and B1 physical calls.
- 384x640; existing three-stage latent shapes 12x20/24x40/48x80; no Score packing.
- Frozen rank-qualified feature trace and seed20260903+global_rank.
- The rank8 bundle explicitly references ranks0..7 of the immutable rank16
  systems trace, with a new bundle identity and source-provenance receipt.
  Tensor/control payloads are unmodified. All six original sample slots and
  all exit/call inventories remain; **this is not truncation of logical steps**.
- These Gaussian feature inputs are systems-benchmark inputs, not real-video
  quality or convergence data. Deeper numerical review is deferred by the user,
  not replaced by a borrowed world16 numerical PASS.
- No BF16 merged-LoRA shadow, profiler, offload, model swap, or retuning.

## Timing and lifecycle

`probe_wan13b_weakscale_two_cycle.py` reuses the actual existing two-cycle
workload functions and mechanism validators. One initialization, two complete
cycles: C1 warm-up only, C2 claimable. Barrier/CUDA fences define each boundary;
host finite extraction, census and JSON occur after both windows. Setup is
separate. Validate-only imports must leave CUDA uninitialized.

The legacy string `world16-fsdp` remains an internal family selector for shared
B1 helpers, **not** the world8 mesh. Receipts separately state the actual mesh,
world size, gradient divisors, global sample count and `world8-fsdp` report name.
Canonical launchers/profiles and main defaults are unchanged.

Capacity and cluster timing remain **NOT_RUN at the implementation commit**.
Real CPU tests on both nodes cover all eight Gloo ranks' mean gradients,
sample/step counters, default world16 regression and validator API signatures.
CPU tests are not GPU capacity results.

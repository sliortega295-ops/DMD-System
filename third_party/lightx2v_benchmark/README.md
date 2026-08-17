# LightX2V MiniMax-H3 DMD benchmark source

This integration pins the upstream Apache-2.0 project
`ModelTC/LightX2V` at commit:

```text
d034a6b0ecaa31aa07c81aeb7bbe69b225f1d7be
```

The repository is attached as the `third_party/lightx2v` git submodule. The
submodule remains pristine. DMD-System's benchmark-only modifications are kept
as `benchmark.patch` and applied to an immutable local source copy by
`scripts/distillschedule/prepare_lightx2v_minimax.sh`.

## Why a pinned submodule instead of copied source

The MiniMax trainer depends on a substantial shared `lightx2v_train` runtime.
Pinning the upstream repository preserves source provenance and avoids silently
forking unrelated LightX2V inference/model code. A recursive checkout still
materializes the exact source needed by this repository.

## DMD semantics retained

The upstream MiniMax-H3 trainer performs one Student update followed by
`fake_update_ratio` Fake-Score updates. The official MiniMax config uses
`fake_update_ratio=5`, LoRA rank 128 for both Student and Fake, and FSDP2 over
`WORLD_SIZE` with sequence parallel disabled.

The benchmark patch does **not** change the DMD loss, optimizer math, model
weights, FSDP2 shard plan, LoRA targets, resolution, frame count, or update
order. It only adds:

- deterministic benchmark seeding when requested;
- an optional fixed MiniMax back-simulation end-step index;
- one exact outer-DMD-cycle NVTX/timing range and per-rank receipts.

Formal timing fixes `end_step_idx=3`, so each back simulation executes all four
MiniMax model evaluations. This removes run-to-run compute-count variation from
the upstream random end-step sampler. The ordinary random training behavior
remains unchanged when the benchmark environment variable is absent.

Benchmark patch SHA-256:

```text
e9fd51f8a1859392b334600a0c915fc04a609baf54eb152f85e2d0bdbf9587b9
```

This source integration does **not** itself claim a speedup over Helios. Helios
and MiniMax absolute walls may be reported as cross-workload context, while a
claimable speedup requires a DistillSchedule MiniMax candidate under the exact
same MiniMax contract.

## License / attribution

LightX2V is licensed under Apache License 2.0. The submodule carries the
upstream license and history. Files modified by `benchmark.patch` are clearly
marked here as DMD-System benchmark adaptations; no upstream authorship is
claimed for those changes.

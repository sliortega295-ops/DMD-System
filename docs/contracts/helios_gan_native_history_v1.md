# W3 GAN native-history v1 — explicit research opt-in

This ABI is independent of W2 native history. It retains the **same original
GAN training owner**, full FA3/down40/no-merge bundle, world16 FSDP2, C2048,
physical B1, five detached rollouts, joint GAN gradients and G1/C5/EMA1 clocks.
It is not a benchmark-request extension, stable recipe, No-GAN fallback,
numerical-equivalence result, CUDA-capacity result or performance result.

## Explicit input and execution binding

- Public preparation: `distillctl data prepare-helios-gan-stream` with
  `--history-policy source_native_amplified_v1`.
- Training `data.input_kind`:
  `original_helios_gan_native_history_cohort_stream.v1`.
- Execution remains explicit manual preset
  `helios_gan.fa3_down40_nomerge_stream.v1`, with the original eight local asset
  roles and pinned FA3 implementation. N70/merged execution remains OFF.
- The normal `original_helios_gan_cohort_stream.v1` default is unchanged. Its
  producer still rejects zero history. Its default loader rejects this new
  stream, and the native loader rejects normal/W2/old-request relabeling.

Use `helios_gan_inputs.portable_config()` to remove only the five unused legacy
system locators. Write/hash one portable configuration and bind that exact file
in both preparation and assets. Native math/configuration validation remains
unchanged. A finite record pool additionally requires the existing explicit
`distillschedule.helios_record_epochs.v1` input policy; it does not create a new
input sampler or change its seed/order/draw identities.

GAN capture remains separate from W2 capture. Original dataset/history,
corruption, scheduler, rollout noise, Score timestep, GAN crop and R1 epsilon
owners are unchanged. The actual original history predicate decides whether
each stage has its original budget2 or doubled budget4. Normal initial exits
remain common to all ranks. Amplified exits use the actual rank-local native
draws, with no seed search, resampling, padding, dummy calls, forced normal exit
or shortened rollout. The trace records and revalidates the exact grids.

The independent execution schema is
`distillschedule.helios_gan_native_history_execution.v1`, and semantics are
`original_helios_gan_rank_local_amplified.v1`. W2 success is not GAN admission.

## Physical owners and safe boundaries

1. G: validate the actual root plus40 complete `HeliosTransformerBlock` FSDP
   owners, one active original checkpoint function, and original sharded
   reshard-after-forward/backward entry state. Reuse the existing
   `fsdp2_variable_forward_phase`, materializing all41 units before the original
   `_run_gan_generator_update`. Retain them through its original GAN/Score
   latent-gradient boundary and G backward. Release and restore before the
   **original** engine clipping/Adam/scheduler/zero-grad step. Exceptions release
   the units without fabricating a commit.
2. Detached `r_all`: do not enter the G phase. Keep the original
   `generator_nograd_persistent_unshard_scope` exactly. Every rank performs at
   least one complete forward, which materializes the same41 units; all repeat
   forwards retain those units. Preserve five actual B1 rollouts, original
   allocator flushes and the original release before Critic updates. The native
   census requires all41 actual full owners before exit,41 releases, restored
   policy and zero retained unsharded units. Nested/materialized entry is still
   rejected.
3. C/GAN head, GAN/R1/crops, frozen-down40 and FA3 replay, joint-gradient/adapter
   boundaries, update order and EMA use the unchanged original owners. Dynamic
   G prefixes do not change fixed C replay work:400 original/replay frozen-down
   operations per cycle remain required. No counter is warmed or synthesized.

The G phase materializes the entire Generator while its activations and Score
state coexist. CPU sharding/component tests do **not** establish that this fits
the assigned GPUs. Fresh full-size CUDA/host/cgroup/HBM reserve admission is
required; source support never authorizes lowering memory or numeric gates.

## Continuation and source identity

Native G/C/GAN-head/Adam/scheduler/EMA/RNG state still uses the same original
`engine_state`, `save_checkpoint`, `restore_checkpoint` and atomic all-rank
publication owners at drained complete-cycle boundaries. The independent
checkpoint policy pins the GAN capability, execution source, source HEAD,
full transitive source fingerprint, immutable assets, exact stream SHA, actual
native owner layout and existing epoch/draw identity. Any mismatch rejects.

Bootstrap remains physical G2/C10/EMA800 and curriculum G200/C1000/EMA800;
after two new cycles it is G4/C20/EMA802, curriculum G202/C1010/EMA802. This is
not a recovered post50 checkpoint or an official checkpoint1000. New source
requires new source-bound W3 preparation from the existing feature pool, not
relabeling W2 traces or changing an old stream's fingerprint.

## Evidence boundary

The focused CPU test file characterizes metadata/schema, original helper
argument/phase dispatch, actual shared tiny-state checkpoint IO, source binding,
unchanged default behavior, and original persistence-control cleanup. Synthetic
FSDP/Score/metadata boundaries are labeled as such. Its target-dependent test
uses the actual original GAN capture, RNG, crop/R1 and official kwargs helpers
without a pretrained/full-size model. Missing dependencies mean **NOT_RUN**.
The capture RNG assertion covers execution with the original helper closure
already loaded, matching the existing W2 fixture. It is not a claim that first
transitive library import preserves Python RNG: an independent cold/warm target
probe observed Rich's import-time `getrandbits` call; warm helper loading and
the existing input-stream owner preserved both Python and CPU Torch streams.
Do not alter native sampling seeds or add a production RNG wrapper to conceal
that setup boundary.

Existing target tests for the unchanged common FSDP phase and whole40-block
checkpoint validator may be reused only at that physical component scope.
Actual16-rank GAN continuation/restart, optimized kernels, capacity, numerical
equivalence, decoded endpoint and quality remain **NOT_RUN** until separately
observed. Historical GAN numerical **REVIEW** is not promoted by this contract.

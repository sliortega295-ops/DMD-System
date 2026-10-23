# Parameterized ORIGINAL Wan optimization bridge

This opt-in layer joins event-frontier search to the **existing** Wan1.3B
Helios-pyramid no-GAN implementation. It does not substitute the separate
single-device Euler/Diffusers training protocol. The five canonical profiles
and the original model-based search interface remain unchanged.

## Implemented versus validated

The original task schema routes through `workloads native`. It generates
world16 FSDP or Resident/async128/Tree candidates, modulation chunks 512/1024/2048,
and the existing Generator no-grad LoRA epilogue. FSDP control remains
chunk512/epilogue-off. Choices reach original constructors, optimizer, loss,
score functions, pyramid sampler and EMA helpers.

Input capture and task preparation are now connected: see
[Original Wan auto-run](original_wan_auto_run.md). It starts from NEW pre-encoded
features, invokes original model-free input primitives, generates a task-bound
trace and prepares the existing budgeted calibration/search/confirmation.
No manual trace hash entry or old-trace resizing is required. VAE/text encoding
itself is not automated, and the all-zero rank-local amplified branch remains
unsupported without changing/resampling the source distribution.

Fresh-cohort training is a separate opt-in path described in
[Fresh training](original_wan_fresh_training.md). It advances absolute steps and
EMA cadence while retaining the selected engines and optimizer state. Existing
calibration/confirmation still repeat their fixed cohort for paired diagnosis;
those results must not be called fresh-data training.

CPU tests exercise capability generation, plan selection, capacity rejection,
original-helper binding, per-block chunk effects, actual native input/scheduler
primitives, and persistent dispatcher/Adam fixture state. They are **not**
full-Wan or CUDA tests. Multi-GPU execution, full-model numerical parity, speed
and generation quality remain NOT_RUN for this integration.

## Configurable contract

`OriginalWanTask` describes height/width multiples of 64, frames 4k+1, G1/Fn
(1..64), LoRA rank (1..256 matching bootstrap), original Generator-update
boundaries, three stage horizons and EMA cadence. Active pyramid sampling uses
`stage2_inference_steps`; the ordinary denoising list is inactive and cannot be
changed as if it controlled compute. Random exits are still sampled/captured,
so a horizon does not imply an identical call count on every rollout.

Still fixed: Wan1.3B 30-block model, world16/two nodes of eight, B1/global16,
original precision/history semantics, three stages and no-GAN. This does not
parameterize Helios14B, H3, LiveWan, arbitrary pool splits, automatic backbone
sharing or arbitrary intermediate-state offload.

### Explicit historical benchmark input scope

`stage_schedule_policy="benchmark_v4_strict_global"` is an opt-in extension for
parameter-generalization systems experiments using the previously declared v4
strict-global exit scope. It preserves sampled history dropout, zero-history
tensors and operation-qualified RNG; it does not resample passing inputs. For
all-zero history it retains the native doubled timestep grid, but the exits are
still the rank-zero synchronized exits declared by this benchmark. This is not
the upstream rank-local amplified-exit distribution, and must not be reported as
full-distribution training or quality equivalence. Every changed task needs new
trace identities and measured costs.

The default `source_normal_only` capture remains unchanged and rejects all-zero
history. Its serialized task identity is unchanged. No default profile, original
training helper, loss, gradient or optimizer is modified by this input scope.

## Search and execution

`validate` inventories candidates without model construction. `auto-run` is
one command per allocated node for new input preparation and the existing
campaign. Individual `capture-inputs`, `prepare-task`, `prepare-campaign`,
`calibrate`, `assemble`, `plan`, `run`, `prepare-stream` and `train-stream`
commands remain available for controlled use.

The default campaign performs seven component calibration arms and one native
confirmation, eight distributed launches total. `--without-fusion
--max-launches 5` requests four non-fused arms plus confirmation. Insufficient
budgets are rejected, failed arms remain failed, and unknown interrupted
launches are not automatically repeated. --resume preserves journals and
counts. CPU fixture costs cannot launch native GPU execution.

Plans only rank source-, hardware-, task- and asset-matched observed choices.
The selected layout is installed before NCCL/model construction, and worker
checks cover actual reduction, block chunks, fusion invocation and trace-derived
call counts. Final trainable, Adam and EMA states are compared with a matched
FSDP control; numerical REVIEW never becomes PASS merely because a process
completed. Calibration is diagnostic, not a formal speed baseline.

## Remaining scope gates

Real 16-rank original-Wan smoke/parity and performance, support for the upstream
amplified-history branch, raw-video/text encoding if needed, resumable long
training, and the other four native protocol/capability adapters remain to be
validated or implemented. Do not promote CPU integration to those claims.

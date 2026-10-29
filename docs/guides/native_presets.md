# Direct, versioned native optimization presets

The six `recipes/*/training.json` files now select an **explicit research
preset**, without a user-created saved plan or calibration campaign. They do
not search for a winner, import historical timing as a cost, download assets,
or claim new GPU validation. Use the editable source installation required by
the existing native APIs.

```json
{
  "adapter": "native",
  "mode": "manual",
  "preset": "helios_gan.fa3_down40_nomerge.v1",
  "task": "task.json",
  "assets": "qualified-native-assets.json"
}
```

This is the `execution` section of the complete
[W3 configuration](../../recipes/helios_gan/training.json), not a complete
training config. Replace recipe placeholders with qualified assets and an
output directory **outside the source checkout**. Then inspect without CUDA
or output creation:

```bash
distillctl doctor --config /project/recipe/training.json
distillctl train /project/recipe/training.json --dry-run
# Only on an authorized allocation, with a clean committed source checkout:
distillctl train /project/recipe/training.json
```

W1/W6 enter existing world16 torchrun ranks on two eight-GPU nodes. W2–W5
use the existing process-owned two-node dispatcher and operator-supplied node
manifest. This change introduces no launcher or node reservation service.

## Exact bindings, not a new training contract

The six original presets below require **exactly two native cycles, `checkpoint_every: 0`, no
resume, and their original fixed input contract**. They are not continuous
training, new-input generation or inference-export tutorials. Fresh-input
reference lifecycles remain separate in the [native guide](native_workloads.md).

| Workload / explicit preset | Existing typed implementation and mechanisms | Source origin | Numerical / lifecycle boundary |
| --- | --- | --- | --- |
| [W1](../../recipes/wan_ar/training.json) `wan_ar.incumbent21a.v1` | `WanStrategy`: resident/C2048/no-grad epilogue, original w5; coupled gradient-LoRA epilogue, exact norm, G rollout graph, G no-grad cast cache, C gradient graph | `21a0993d` | REVIEW; exact immutable model/bootstrap/trace/negative identities and G2/C10/EMA10; **no retain4** |
| [W2](../../recipes/helios_nogan/training.json) `helios_nogan.safe_ed8.v1` | `HeliosImplementation`: node-HSDP/C512/w5/p0; all-or-none `HeliosSafeBundle` norm, rollout sync elision, LoRA-B GEMM and checkpoint replay scope | `ed8e1ccb` | REVIEW; original 33-frame request, warm bootstrap and source-pinned kernels; no new shape admission |
| [W3](../../recipes/helios_gan/training.json) `helios_gan.fa3_down40_nomerge.v1` | `HeliosImplementation`: FSDP/C2048/w5/p1; `HeliosGANBundle` pinned SM80 FA3, down40 replay, separate GAN heads and LoRA-B scope | `ba52a244` | REVIEW; **merged-LoRA/N70 off**, actual merge count must remain zero; no H100 admission |
| [W4](../../recipes/self_forcing/training.json) `self_forcing.c51.v1` | `LiveWanImplementation`: complete `c51_full_v1` 8G+8Score pipeline, including C34 prerequisites and original C51-only source/template/kernel contracts | `aa30527d` | REVIEW; original 81-frame continuous-sigma task; **not C34**, not official Self-Forcing, no C51 drain/restart support |
| [W5](../../recipes/h3_fullseq/training.json) `h3_fullseq.cpu_staged_trim.v1` | `H3Implementation`: Grid1000, CPU-staged trimmed FA3 replay, depth2/N128 | `cfecbb41` | **FAIL unresolved**; research reproduction only, not a recommended/stable recipe; original sampled numerical gate retained, no timing eligibility granted |
| [W6](../../recipes/h3_ar/training.json) `h3_ar.zero3_ap.v1` | `H3ARImplementation("zero3-ap")`: existing ZeRO-3 selected-cache/unit prefetch, CPU-staged replay, pointwise/LoRA/SwiGLU/RMS/CUTLASS and communication overlap bundle | `e88d21c1` integration evidence | REVIEW; source/input/checkpoint audits and explicit live capacity policy; original previous/current AV, 8-step G1/F1/LoRA64 |

Source origins describe provenance, **not the currently authorized execution
HEAD**. `native_presets.py` (source checkout: `src/distillschedule/training/native_presets.py`)
contains the versioned metadata, mechanism owners and exact typed factories.
No frozen profile, source-import map, historical FAIL or publication receipt
is rewritten. Future changes to a preset's contract require a new preset ID.

### Separate W2 fresh-stream opt-in

`helios_nogan.safe_stream_ed8.v1` is a **new**, separately versioned manual
preset; the original `safe_ed8.v1` and reference route are unchanged. It binds
the actual SafeBundle and original producer/worker/lifecycle for 2..64 fresh
cohorts, with complete-cycle same-topology save/resume. It refuses saved-plan,
transport, replay and incompatible source/backend/control/cache state rather
than using an old benchmark payload. See the W2 recipe's
[fresh-stream guide](../../recipes/helios_nogan/SAFE_STREAM.md).
Original helper CPU updates and two-rank Gloo state/failure tests are source/
CPU evidence only: CUDA telemetry/kernel counters in the tiny fixtures are
explicit simulations, not actual FA2/Triton, node-HSDP, capacity or pretrained
validation. Native GPU execution remains **NOT_RUN**, status **REVIEW**.

## Shared lowering and runtime gates

For W2–W5, the direct path calls the existing `resolve_candidate` and
`resolve_configuration` against **current assets and current source**, then
uses the same candidate/profile/semantics/task/asset/node-command validators
as saved-plan mode. It dispatches via the unchanged `run_plan` →
`run_resolved` path. No search or saved-plan file is required. A deterministic
new binding ID includes config and output identity; reuse of a run directory
is still refused by the original dispatcher.

These bounded research bindings enter the original **correctness** mode.
That requests the original work and gates; it is not a numerical PASS, a
training qualification, a timing permission or a bypass of W5's failure.
Source/runtime, immutable asset, full tensor/cache/version, topology and live
capacity checks still belong to the original workers **before model use**.
CPU doctor makes the deferred physical scope explicit. A refusal from those
gates remains a refusal; nothing falls back to an older candidate or reference.

W1 passes the exact typed strategy to the same `wan_worker.run_worker` used
by its manual CLI; no strategy file is synthesized. W6 passes `zero3-ap` to
the original `h3_ar_cli` and worker with the same explicit capacity policy.
Their native task, asset, graph and live capacity checks are unchanged.

## Evidence boundaries and other modes

`describe_native` exposes source provenance, mechanism mapping, numerical
status, `continuous_training: false` and `gpu_execution: NOT_RUN` separately.
CPU tests prove public selection → real lowering → runtime binding, including
equivalence with an explicitly saved identical candidate. They do **not**
execute kernels or establish nonzero GPU mechanism counters. Actual execution
claims require the original worker's effective-plan/bundle/call receipts and
separate numerical/capacity outcomes; process exit zero is insufficient.

Presets require `mode: manual`. Combining a preset with `plan`,
`implementation` or `transport` is rejected, as are mismatched workloads,
unsupported shapes, altered source assertions or lifecycle requests. In
particular, W1's literal graph cannot consume a fresh stream; W2/W4/W5 fresh
streams cannot silently become optimized benchmark replay. The new W2
fresh-stream preset above is an explicit different contract, not an exception
to a frozen benchmark's gates.

Saved plans remain supported unchanged: select `mode: plan`, supply `plan`,
and omit `preset`. The original selected plan is authoritative, including its
source, calibrated hardware, gates and run ID. W1/W6 explicit manual primitive
strategies and the separate supported reference recipes also remain available.
Choose a reference **explicitly** if a preset is unsuitable; it is never an
automatic fallback. Automatic planning remains an optional workflow, not a
prerequisite for these direct bindings.

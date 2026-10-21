# Public framework integration: CPU acceptance

This change adds a public training lifecycle to the existing Program, graph,
solver and native adapters. It does not replace the six original algorithms
with the tiny example, merge the integration into `main`, or establish new
large-model performance or quality results.

## What was exercised

| Requirement | Evidence and boundary |
| --- | --- |
| Lightweight installation and discovery | A fresh wheel installs outside the checkout; discovery/doctor import no Torch or model code |
| Functional installed package | A separate package venv, reusing existing CPU dependencies without downloads, executes actual media preparation, updates, restart, generation/export and measured plan selection |
| Raw and preencoded data | Actual frame/video, PCM WAV and UTF-8 decoding; trusted model encoder; hash-bound native feature import; no implicit latent conversion or random fallback |
| Continuous training | Persistent model/optimizer ownership, explicit sample cursor, per-cycle versions/metrics, scheduler/EMA/history state |
| Fresh-process recovery | A 4-cycle checkpoint resumes to cycle 6; state, RNG, cursor, versions and policy equal an uninterrupted 6-cycle run |
| Optional planning | Reference/manual work without search; auto/saved selection uses the existing solver, admission and actual recorded operation order |
| Extensibility | An external model architecture and configurable tiny DMD workflow execute without a solver model-name branch |
| Six native integrations | W1-W6 public requests bind original APIs, task/assets/source/capacity and selected-plan parameters; CPU contract/lowering tests only |
| Rejection paths | Missing/drifting data, wrong shapes/dtypes, unsupported policies, invalid versions, incomplete/tampered checkpoints and unbound sources reject |

The example's Teacher is frozen random initialization. Its media statistics
encoder and diagnostic RGB/audio-feature output prove the training interfaces,
not pretrained audiovisual generation quality.

## Local verification

The available validated environment is Python 3.12 / Torch 2.14 CPU. No Python
environment was replaced; the original refactor suite's missing configuration
extra was supplied through a disposable test-only dependency overlay.

- Framework/data and existing open-workload suites: **354 passed, 2 skipped**.
  The two pre-existing skips require a real CUDA device. Nineteen non-failing warnings
  include existing selective-checkpoint notices and a cloned-scheduler notice;
  scheduler output/state/optimizer/EMA/history comparisons pass.
- Isolated functional wheel: **PASS_INSTALLED_FRAMEWORK_CPU**, including
  operation without a Git executable. The lightweight wheel check also passes.
- Existing required refactor suite: **2147 passed, 85 skipped, 2 failed**.
  Both failures import unavailable `triton` in the unchanged modulated-norm
  tests. Running those same two tests on the unmodified integration base
  reproduces both failures. They are not silently marked PASS or disabled.
- Source/CLI/import admission, Python compilation, shell syntax, focused
  linting and documentation links are checked separately.
- Local eight-lens review completed; nine confirmed issues and the subsequent
  repository-entrypoint compatibility regression were fixed and validated.
- Added CI runs CPU Torch 2.10 on Python 3.10/3.12. Its remote outcome is separate
  from these local results; the declared Torch 2.5 minimum was not tested here.
- New A100/H100 runs, SSH connections, model downloads and performance trials:
  **zero**.

## What this is not

The public Program lifecycle currently owns one process/device. Its checkpoint
is a complete-cycle, same-topology training checkpoint, not a mid-cycle or
cross-topology recovery mechanism. Exported inference weights are explicitly
not resumable training checkpoints.

The six original native integrations retain their actual bounded APIs. Most
currently expose two-cycle execution rather than arbitrary long training;
their universal checkpoint/resume and generation callbacks are **not yet
implemented** in the new front door. Existing upstream save/validation hooks
are preserved, not claimed absent or newly verified. See the exact
[capability matrix](../guides/support_matrix.md) and
[native guide](../guides/native_workloads.md). These remaining native lifecycle
adapters must not be described as six production-ready end-to-end trainers.

Historical A100/H100 evidence remains attached to its original source and
contract. This CPU integration does not promote it to validation of the new
branch. Large-model correctness, convergence, capacity and hardware-specific
restart/generation remain separate validation work.

# BAAI server adaptation gate (2026-08-08)

## Scope

This is a CPU-only source/configuration gate for GitHub main
`d2d9fdba053b18414bdec4484e427547f654847f`. It does not launch a model, reserve
a GPU, or establish a world16 performance/correctness result.

## Initial result

The source copied cleanly to
`/share/project/shuchenweng/yongyan_liu/DMD-System`, but the complete server
test suite initially reported `96 passed, 1 skipped, 3 failed`. All three
failures had the same cause: the example Helios plan referenced the removed
`configs/experiments/iteration45_m0_b0a_a100_substrate_384x640.yaml` file.
GitHub Actions at the source commit was also red; therefore this was an
upstream source-packaging defect rather than a cluster dependency failure.

## Adaptation

- retain a portable Stage-3 template in the runtime repository;
- derive Base/Real-Score and dataset paths from the typed plan;
- make the Helios runtime root and Python executable environment-selectable;
- force the rendered child invocation to import the validated external Helios
  checkout, avoiding accidental `runtime` package shadowing;
- add a fail-closed BAAI path/asset smoke script.

The validated node-local Python is
`/tmp/distillschedule-helios-official-torch210/bin/python` (Python 3.10,
PyTorch 2.10.0+cu128, DeepSpeed 0.18.4). Both nodes expose the shared DMD-System
tree and the pinned external Helios entrypoint.

## Measured result

- complete server tests: `99 passed, 1 skipped`;
- focused typed tests: `16 passed`;
- Helios typed-plan validation: PASS;
- dry-run plan digest:
  `2b31d1929c62172c2017bb4785f7e8332047bf04e8feb3e6b57bc5c772b63be4`;
- rendered shape: 384x640, latent window 9, 81 source frames;
- rendered branches: DMD/EMA on, GAN/Reward/VAE decode off;
- rendered checkpoint: canonical checkpoint 10;
- rendered child command points to the external validated `train_helios.py`.

## Boundary

The repository is **server-source compatible and dry-run compatible**. It is
not yet a native world16 executor: `distillctl run` currently owns one child
process and does not coordinate the two node launchers. Real 14B CUDA parity,
the Step10-A to Critic-pool transition, exact EMA continuation, and world16
timing remain Not Run through this cleaned entrypoint.


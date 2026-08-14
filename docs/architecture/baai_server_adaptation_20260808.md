# BAAI server adaptation snapshot (2026-08-08)

> **Historical source/configuration snapshot.** The live Route-A world16 status is maintained in [`../status/routea_world16.md`](../status/routea_world16.md), and the frozen workload is defined in [`../contracts/routea_full5_world16.md`](../contracts/routea_full5_world16.md).

## Scope

The original adaptation gate copied DMD-System to the BAAI environment, repaired repository packaging/path issues and established CPU/source/dry-run compatibility. It did not establish a successful world16 Full5 result, model parity or speedup.

## Initial finding

At GitHub main `d2d9fdba053b18414bdec4484e427547f654847f`, the server test suite initially reported `96 passed, 1 skipped, 3 failed`. All three failures came from an example Helios plan referencing a removed historical experiment config. GitHub Actions at that source commit was also red, so this was a repository packaging defect rather than a cluster dependency failure.

## Adaptation performed

The server-adaptation branch:

- retained a portable Stage-3 config template inside DMD-System;
- rendered model/data/checkpoint fields from the typed plan;
- made the Helios runtime root and Python/Accelerate executable paths environment-selectable;
- forced the child invocation to import the pinned external Helios checkout rather than an accidental package shadow; and
- added a BAAI source/asset smoke gate.

The validated environment at that point used Python 3.10, PyTorch 2.10.0+cu128 and DeepSpeed 0.18.4. Subsequent adaptation feedback reported the complete suite at `103 passed, 1 skipped` before the live world16 attempt cycle.

## Boundary exposed by real adaptation

Dry-run compatibility was not enough. Real attempts exposed that the repository still lacked:

- fail-closed Python/Accelerate/DeepSpeed identity resolution;
- one authoritative multi-node topology gate;
- a central two-host coordinator and peer cancellation;
- unambiguous EMA bootstrap/decay naming;
- normalized typed-versus-legacy config parity; and
- a summarizer proven on one successful typed world16 run.

Those gaps are tracked in [Issue #2](https://github.com/sliortega295-ops/DMD-System/issues/2). The cleanup following this snapshot adds reusable preflight and evidence tooling but does not claim that the central coordinator or successful Full5 closure already exists.

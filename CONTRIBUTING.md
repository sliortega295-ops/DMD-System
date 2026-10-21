# Contributing to DMD-System

DMD-System is a research runtime with strict equivalence and evidence boundaries. A change that makes a launch succeed but changes the DMD workload is not a valid fix.

## Choose the correct layer

- Put model-independent schemas, contracts, adapters and orchestration in `src/distillschedule/`.
- Keep already-measured compatibility mechanisms in root `runtime/` until their typed replacement passes operation/tensor/loss/version/optimizer/EMA parity.
- Put immutable workload semantics in `docs/contracts/`.
- Put mutable operating procedures in `docs/runbooks/`.
- Put measured/not-run outcomes in `docs/status/` and small curated manifests.
- Keep raw experiments, weights, traces, profiler databases and run directories out of Git.

## Frozen-plan changes

Do not edit a frozen plan to make an integration failure disappear. Changes to model/checkpoint/trace, shape, batch, logical steps, update ratio, optimizer, ZeRO stage, EMA behavior or branches require:

1. a new plan ID;
2. a new or versioned contract document;
3. new checkpoint/trace identities where applicable; and
4. a matched baseline before performance comparison.

Route-A specifically forbids a fallback to ZeRO-2.

## Fail closed before model construction

Multi-node launch code should validate, before loading a 14B model:

- executable availability and identity;
- source HEAD/clean state and entrypoint hash;
- resolved plan and backend contract;
- exact world size, node rank and local rank topology; and
- required asset paths/identities.

Never silently infer `world_size`, launcher type, ZeRO stage or EMA semantics from a historical config.

## Evidence and claims

Functional correctness, diagnostic timing and host-memory reserve are separate outcomes. A zero process return is insufficient. Preserve failed/partial outcomes honestly and do not calculate a speedup from incomplete rank evidence.

## Checks

Run at least:

```bash
python -m compileall -q src runtime scripts/distillctl.py scripts/distillschedule
find scripts -type f -name '*.sh' -print0 | xargs -0 -r -n1 bash -n
pytest -q tests/test_refactor_*.py
```

Cluster-facing changes additionally require the BAAI smoke gate and an immutable status update. A successful GPU run is not required to review source cleanup, but unexecuted cluster behavior must be labeled `Pending cluster validation`, not `PASS`.

## Six-workload organization

`distillctl workloads list` is the six-protocol catalog; the five immutable
legacy profiles are exposed with `--legacy`. Add current workflow guides under
`recipes/`, not duplicate model trainers. Developer tools belong under `tools/`
and standalone canaries under `benchmarks/kernels/`.

Do not rewrite `source_import_map.json` or publication receipts to hide a
changed source. A reviewed relocation/integration needs an explicit old/new
entry in `source_layout.lock.json`, an exact source origin and regression tests.
The compatibility `runtime/` namespace is packaged at its original audited path.

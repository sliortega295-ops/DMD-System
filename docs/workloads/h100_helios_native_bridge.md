# H100 Helios native plan binding — Full5 only, not C2

This bridge reuses the actual clean `8fc70e8ce9f6e0243604812120119c9f4c078e55`
launcher and CPU summarizer. It does not introduce a trainer, simulate durations,
copy old seconds into a cost model, silently change selective replay, or launch
GPUs. The H100 campaign was explicitly stopped; compiling a command is not
authorization to execute it. There is deliberately no `run` or `resume` command.

## Exact available surface

`distillschedule.search.hardware_helios` exports:

- `inspect_native_source(root)`: check clean registered Git HEAD and 11 native
  launcher/worker/policy/backend files; no Torch or GPU query.
- `compile_native_plan(deployment, arm, mode, run_id, source_root=None,
  correctness=None)`: call the **real native** `freeze_policy` and
  `policy_arguments` in a fresh CPU-only Python process, then bind the real
  eight-argument `launch_helios_hardware_p0_node.sh` command and launcher env.
- `validate_native_plan(plan, source_root=None)`: recompile, reject any changed
  arm, argv, policy, allocation, assets, timer, gate or claim.
- `audit_completed_run(plan, source_root=None)`: on the native receipt host,
  re-run the **real native** summarizer over raw rank/trace/health/source evidence;
  reconcile the saved summary, actual torchrun command, policy, capability and
  correctness bindings, GPU UUID/capacity/driver and package versions. Additional
  per-rank checks require real local-FA2 identity, two distinct B1 scopes,
  41-unit retain/reshard/restore, and actual interior/outer fence counters.

Supported arms: `best`, `h100-nograd-retain`, `h100-nograd-retain-nosync`.
Supported modes: `validate`, `correctness`, `timing`. Timing requires the exact
arm/source-bound fresh correctness path/hash; native preparation revalidates
age, all source evidence, assets and the retention API gate before model work.

The native source uses local-FA2, **stride-2 selective replay**, HSDP 2×4,
physical8/logical16, original 384×640 trace, independent B1 slots `(r, r+8)`,
0.5 per-B1 loss scale, G1/C5/EMA5, CUDA EMA/residuals. This is not the newer
planner's stride-0 recompute bundle. Full-state numerical equivalence stays
NOT_RUN/REVIEW; nothing here upgrades that evidence.

## Deployment input

Use `schema_version: distillschedule.hardware_helios_deployment.v1` and exactly:

- `native_source`: `head` above and absolute native `repo_root`.
- `site`: absolute `project_root`, `python`, `helios_root`, `asset_root`,
  `hf_home` (under the asset root). No free-form policy/env overrides.
- `allocation`: existing `ExecutionAllocation` schema, eight unique observed
  SM90 GPU UUIDs/capacities on node0; `physical_microbatch=1`, logical batch16,
  generator and critic rows `[r, r+8]`.
- `asset_gate`: absolute native full-hash receipt `path` plus `sha256`.
- `retention_api`: absolute registered API proof `path` and SHA
  `52f9b287011e681861731055b4b37ea7d65ab4cb39eb8ad0792025a7b0c23966`.
- `runtime_versions`: `python` version, `driver`, and exact `packages` mapping
  for torch/triton/deepspeed/accelerate/transformers/diffusers/peft/safetensors.
- `master_port`: explicit 1024–65535; `deadline_seconds`: explicit 1–7200.

`source_root` may point at a local clean identical source checkout while
compiling a future receipt-host command. This does **not** verify the remote
host, assets, environment or availability. Auditing requires original native
paths to exist; never rewrite remote raw references into invented local proof.

## CPU CLI (no GPU execution)

```
python -m distillschedule.search.hardware_helios_cli inspect-source --source-root /absolute/native/checkout
python -m distillschedule.search.hardware_helios_cli compile --deployment /absolute/deployment.json --source-root /absolute/local/native/checkout --arm h100-nograd-retain-nosync --mode correctness --run-id unique-id --output /absolute/new-plan.json
python -m distillschedule.search.hardware_helios_cli audit --plan /absolute/plan.json --output /absolute/new-audit.json
```

Output files use exclusive creation. The existing native launcher separately
enforces immutable run IDs, exact clean source, locks, idle/owned GPU inventory,
asset hashes, per-rank raw summaries and teardown. No locks/stop markers are
changed by this bridge.

## Explicit integration boundary for the main planner

This native worker executes **one Full5**, not two uninterrupted cycles. Its
`full5_seconds_max_rank` must NEVER populate `predicted_cycle2_ns`. No C2 search
adapter or calibrated cost record is registered by this patch. The exact native
program must first gain a semantics-preserving continuous-cycle adapter with
C1 state continuation, two-cycle version/RNG/optimizer/EMA evidence and C2 timer.
Only then can a root-owned registration connect real measured costs to
`search_plans` and an authorized `run_selected`, similar to `hardware_wan`.
The currently stopped H100 campaign cannot supply that new evidence.

Root-owned minimum public CLI registration: in
`src/distillschedule/workloads/best_known/cli.py`, immediately after `argv`
normalization in `main` (d569 line 11), add this branch beside existing
`hardware` dispatch (lines 12–14):

```python
if argv and argv[0] == "hardware-helios":
    from distillschedule.search.hardware_helios_cli import main as helios_hardware_main
    return helios_hardware_main(argv[1:])
```

The existing outer `src/distillschedule/cli.py:130–132` already forwards
`distillctl workloads ...` here, so no packaging, workflow, executor or cost
registration change is needed. This enables the same three CPU commands as
`distillctl workloads hardware-helios inspect-source|compile|audit`; its
parser rejects `run`, `resume`, `search`, and `--mode c2`. Add a dispatcher unit
test passing the remaining argv unchanged, plus CLI rejection tests. This
isolated patch does not edit the shared dispatcher; the root integrator owns it.

Do not register this Full5 binding as a C2 adapter in `model_adapters`, change
the existing Helios implementation, or reuse A100/world16 costs. In particular,
`search/cost_model.py:160–164` currently routes every `hardware_native.*`
component to Wan's provenance validator; that namespace cannot be reused as
a shortcut to admitting Helios Full5 evidence. The standalone module entrypoint
also remains available for inspection before integration.

## Tests

`tests/test_refactor_hardware_helios.py` uses stdlib `unittest`. Mocked unit
fixtures test adversarial admission/command/receipt cases, not GPU execution.
Set `DMD_HELIOS_H100_NATIVE_ROOT` to the registered clean source for the additional
real-source three-arm compiler test. Run with `PYTHONDONTWRITEBYTECODE=1` and
`CUDA_VISIBLE_DEVICES=''`; no package install or environment repair is needed.

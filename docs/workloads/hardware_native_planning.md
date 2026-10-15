# Hardware-native planning bridge: Wan on eight H100s

## Scope, and what is not claimed

`hardware_wan_coupled_v1` admits **W1 Wan1.3B no-GAN only**, on one node of
eight physical H100 devices, with **16 independent logical B1 samples**.
Each physical rank consumes slots `(rank, rank + 8)`. This is not physical B2,
and is not the world16 runner with its world size edited. Every logical loss
is scaled by 1/2 before the rank-mean reduction over eight workers.

The initial search domain is exactly:

| Placement | Native execution selected |
|---|---|
| `world16-fsdp` | Native world8 FSDP baseline; historical placement name retained |
| `resident` | Full replicated reference, synchronous gradient reduction, chunk512 |
| `resident-async128-tree-chunk2048` | Native resident bundle: async128, Tree, chunk2048, generator-only fused epilogue |

All three are choices in the same public `PlanningProblem`. An arm without
fresh calibration remains `UNCALIBRATED`; no historical time, default winner,
or optimistic estimate fills its costs. The CLI default calibrates FSDP and
the resident bundle; pass all three placements to calibrate the whole domain.

This is a **coupled native schedule/bundle search**, not independent M/L/Q/K
composition. It does not promise different resolutions, G:C ratios, global
batches, model assets, arbitrary pool splits, independently reordered G/C,
new offload policies, overlap discounts, or independently selectable kernels.
Those require additional native lowerers and context-qualified costs. Existing
worker files and canonical workload defaults are not overwritten.

W2 Helios no-GAN, W3 Helios GAN, W4 LiveWan, W5 H3 Grid1000, W6 native H3 AR,
and A100 execution are **NOT_ADMITTED by this adapter**. In particular W5 and
W6 are different workloads, not interchangeable H3 configurations.

## Public planner and execution path

```text
deployment + fresh native correctness/comparison/timing observations
  -> hardware_wan.assemble_problem_and_costs
  -> ComponentContract + CostModel + PlanningProblem
  -> existing model_search.search_plans (hierarchical or existing algorithms)
  -> lower_selection -> serialized hardware_wan_plan.v1
  -> hardware_wan_dispatch.run_selected
  -> source-pinned native launcher with the selected placement
  -> eight raw rank receipts + native summary + fresh launch sidecar
  -> measured C2 / prediction error / diagnostic closed-loop receipt
```

The bridge does not rank candidates or launch processes itself. The dispatcher
does not open SSH; an outer controller owns transport, authorization, allocation,
and inherited locks. A saved plan is revalidated against the exact adapter graph,
component identities, raw evidence, allocation, and recomputed prediction before
execution. A changed placement cannot be hidden by recomputing only the plan hash.

The model graph contains two sequential **whole native cycle operations**.
C1 loads the bootstrap state; C2 continues the same process after C1. Completion
counters are accounting edges, not fabricated parameter version timestamps.
Actual rank receipts must prove initial `G2/C10/EMA10`, C1 `G3/C15/EMA15`, and
C2 `G4/C20/EMA20`, with one G, five C and five EMA commits per cycle.

Timing samples are `max(rank.local_elapsed_seconds)` for each cycle. Search
minimizes the public planner's C2 objective; C1 is not averaged into C2. One
independent native run supplies one sample per cycle, not eight independent rank
samples. Shared `pool_component_models` collections are supported through their
flat content-hashed source chain; repeated run roots across pooled sources are
rejected rather than counted as independent trials.

Memory uses the maximum rank's measured **post-setup reserved peak** as a
conservative whole-cycle envelope for each GPU. No separate static allocation
is added on top, avoiding double-counting. Native construction peaks, allocator
fragmentation under another process, and changed device pools are not modeled.
Native startup/cleanup/OOM gates remain necessary even for a predicted-feasible
plan. There is no claim that this one scalar predicts arbitrary state lifetimes.

## Native dependency identity

The integration checkout and native backend are separate, explicit dependencies.
The only admitted native backend head is:

`7a668bcb12ad37d831dced3aca1be4ec7be0a2fe`

| Native file | Required SHA-256 |
|---|---|
| `scripts/distillschedule/launch_wan13b_hardware_two_cycle_node.sh` | `044788b6d478c4797656c65d2d85572648d1ceec9bbc551780cb8a199eb70fd0` |
| `scripts/distillschedule/probe_wan13b_two_cycle_timing_world16.py` | `4608568e7d305d0ca0c9bc1674e0d3a80e0d54c9c16cb0df6c79575d240668d9` |

These are an allowlist, not arbitrary user-supplied hashes. The dispatcher checks
both checkouts' current exact HEAD and clean status before and after execution,
and checks native launcher/worker bytes. Its native checkout contains all three
arms; no old worker is copied over the integration tree's current worker.

Native asset and precision contracts remain complete and fixed: bootstrap,
model, negative prompt, trace bundle/binding, trace rollout call census, seed,
fixed five-step replay, global16, B1, no score packing, G1/C5/EMA5, BF16 frozen
base, FP32 trainable LoRA, rank-mean gradient convention, and reference EMA.
`hardware_wan.SEMANTICS` and `PRECISION` are the exact admitted contracts.

The hardware fingerprint currently binds the **entire allocation**, including
physical UUID inventory, capacities, and role-slot mapping. The same mapping is
used for every arm. Costs are not transferable to a different pool, mapping,
environment, source revision, precision policy, or workload by changing labels.

## Deployment and raw evidence

Deployment schema: `distillschedule.hardware_wan_deployment.v1`.

Required fields:

- `binding`: the public six-field source/hardware/environment/asset/semantic/
  precision identity. `source_head` is the **integration** revision, not native7a.
- `native_source`: `head`, absolute `repo_root`, and the two allowlisted hashes.
- `allocation`: `ExecutionAllocation.to_dict()`, with eight distinct physical
  UUIDs and explicit generator/critic (optionally teacher) slot rows.
- `environment`: observed Python/package/binary identity, not per-arm NCCL knobs.
  The dispatcher requires explicit `ld_library_path`, including an empty value.
  MEASURED deployments must include `driver_version`. The metadata-only probe
  queries `nvidia-smi --query-gpu=driver_version`, requires one unique observed
  version, and compares it with this identity without importing Torch.
- `assets`, `semantics`, `precision`: complete matched native identities above.
- `evidence_scope`: `MEASURED` or `CPU_FIXTURE_ONLY`.
- `site`: absolute `project_root`, `python`, `helios_root`, `asset_root`, `run_root`.
  The unchanged launcher requires `run_root` to be
  `project_root/research_artifacts/five_workload_h100_20260930`.
- `transport`: controller-owned `host`, `ssh_port`, `address_family`. The node
  adapter does not send SSH or reinterpret the SSH alias as a hostname.

Every timing observation contains `placement`, absolute `run_root`, and
content-hashed `{path, sha256}` references to:

1. `summary.json`;
2. all eight `execution/ranks/rank_*.json` files;
3. `run_status.json` (exit status, actual cgroup OOM deltas, empty GPU cleanup);
4. `manifests/command.sh` (actual worker argv);
5. `manifests/environment.txt` (actual selected placement environment);
6. a fresh `distillschedule.hardware_wan_launch.v1` sidecar.

The bridge reopens the actual files and verifies their hashes. It reconciles
the summary's embedded ranks/health with the independent raw files; validates
16 logical slots and 96 distinct payload identities per cycle; checks finite
results, versions, call census, fixed cohort replay, candidate fused invocation
counts, source/semantic/precision fingerprints, and max-rank timer consistency.
Actual argv and environment must equal the registered placement lowering, not
merely contain the chosen placement string. Timing's correctness/comparison
files are also reread. Both comparator input summaries are located by their
single run IDs below the deployment's native run root and checked against
`summary_sha256.reference/candidate`, not just the selected arm. Each summary's
32 rollout artifacts (two cycles by 16 logical slots) must retain their recorded
SHA-256, and every rank must identify the same global state anchor. Rollout files
are streaming-hashed in place, without copying, unpickling, or importing Torch.
The anchor digest is **canonical JSON excluding its `anchor_sha256` field**,
not the hash of its formatted file bytes. The five state components and their
finite, unique global-coordinate sketches must remain complete and compatible
between the two sides. Runtime base/LoRA dtype observations are also retained.

The shared pure-data `validate_native_comparison` requires exactly 160 unique
rollout metrics `(cycle 1/2, logical slot 0..15, step 10..14)` plus the five
`generator_trainable`, `critic_trainable`, `generator_adam`, `critic_adam`, and
`generator_ema` sampled-state metrics. Every row must have finite statistics and
consistent same-shape/same-dtype evidence. Thresholds are exactly
`DEFAULT_P0_PARITY`; metric status is recomputed from the recorded statistics,
and top-level status/numerical status must agree with all rows. Finite threshold
deviations remain legal `HARDWARE_TWO_CYCLE_REVIEW`; non-finite values, missing
rows, duplicate keys, or shape/dtype disagreement are not legal REVIEW. Latent
dtype is the actual recorded dtype, not a new BF16-only constraint. This checks
the pinned comparator's retained evidence; it does not rerun tensor comparison
inside the planner or upgrade sampled state to full-state parity.

MEASURED admission recursively rejects explicit `CPU_FIXTURE_ONLY` markers in
timing, correctness, comparison, nested ranks/metrics, and state-anchor JSON.
Unmarked legacy native receipts still need the complete source/hash proof chain;
absence of a fixture marker alone proves nothing. The same prerequisite gate is
used before launching timing, before/after native comparison, during cost/plan
revalidation, and when validating the selected execution's actual receipt.

The fresh sidecar binds integration/native HEAD and clean state, deployment and
allocation hashes, run ID, actual argv/environment hashes, native file hashes,
observed devices/hostname/environment, and launch start/end timestamps. MEASURED
sidecars cannot omit actual host/environment/worker evidence. Each rank's two
timer windows must fall inside this launch interval. The dispatcher also refuses
an existing run directory. Attaching a fresh wrapper to old-source timing does
not qualify it as new-source calibration.

Each generated cost receipt is `distillschedule.operation_calibration.v1`, with
`ComponentContract.family = hardware_native.wan_cycle`. Its raw evidence is
reopened by the public CostModel provenance hook. Altering only a cost wrapper's
samples, memory, source, environment, or evidence scope cannot make it valid.

## API and CLI

CPU APIs:

```python
problem, costs = assemble_problem_and_costs(deployment, observations, output_dir=components)
result = search_plans(problem, costs, algorithm="hierarchical")
plan = lower_selection(result, problem, costs, deployment, run_id=fresh_run_id)
validate_lowered_plan(plan)
```

The normal receipt-host CLI is `distillctl workloads hardware` with `inspect`,
`calibrate`, `search`, and `run` subcommands. `inspect` checks source/manifests
without importing Torch or launching GPU work. `calibrate` and `run` are real
GPU actions and require the outer controller's authorization/ownership gates;
the existence of this CLI is not permission to occupy an already busy node.

## Evidence levels

| Level | What it establishes | What it does not establish |
|---|---|---|
| Source implementation | Public problem/cost/search/serialization and native lowering contracts | New hardware runtime success |
| CPU synthetic receipts/tests | Schema, identity, tamper rejection, public search and saved-plan round trip | GPU performance, kernel correctness, strict equivalence |
| Fresh MEASURED native diagnostic loop | Same-source/native/allocation execution, C2 timing and prediction error | Strict full-state parity or claimable speedup |

Fixtures are explicitly `CPU_FIXTURE_ONLY`, create non-executable saved plans,
and are rejected by native calibration/run paths. Their numbers are synthetic,
not reused historical measurements. Existing raw historical schemas may be
inspected with their explicitly named old head, but assembly admits only the
registered backend plus fresh integration-qualified launch evidence.

The native comparator's `REVIEW` is preserved. Sampled state and finite losses
are not strict full-state equality. Plans and closed-loop receipts keep
`strict_full_state_parity=NOT_RUN` and `claimable_speedup=false`; a predicted best
placement is not automatically a strict winner. This bridge's implementation and
CPU tests alone do not claim a completed GPU search/execution campaign.

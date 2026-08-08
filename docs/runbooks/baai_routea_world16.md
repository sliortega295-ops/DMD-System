# Runbook: BAAI Route-A Full5 world16 compatibility path

This runbook executes the frozen contract in [`../contracts/routea_full5_world16.md`](../contracts/routea_full5_world16.md). It does not authorize workload changes. The authoritative entrypoint is now the central coordinator; the per-host worker remains an implementation detail and manual recovery tool.

## 1. Prepare the same site profile on both nodes

```bash
cd /share/project/shuchenweng/yongyan_liu/DMD-System
source configs/distillschedule/server/baai_2x8_a100.env.example
```

Audit every path. The example is the canonical BAAI profile for this contract, not a portable public dataset configuration.

Pin the external compatibility source after verifying that it is clean:

```bash
export DISTILLSCHEDULE_EXPECTED_HELIOS_HEAD="$(git -C "${HELIOS_RUNTIME_ROOT}" rev-parse HEAD)"
export DISTILLSCHEDULE_EXPECTED_HELIOS_ENTRYPOINT_SHA256="$({ sha256sum "${HELIOS_RUNTIME_ROOT}/train_helios.py"; } | awk '{print $1}')"

git -C "${HELIOS_RUNTIME_ROOT}" status --short
```

The final command must print nothing. The worker rejects a dirty or mismatched Helios checkout.

## 2. Run the source/configuration smoke gate

Run this before reserving the real workload:

```bash
scripts/distillschedule/baai_server_smoke.sh
```

The smoke gate checks executables and assets, renders a node-qualified Accelerate config, runs the test suite, validates the typed plan and performs a dry run. It does **not** initialize a distributed group or construct a model.

## 3. Freeze launch identities

Use the same values on both hosts:

```bash
export RUN_ID=routea_world16_typed_revalidation_YYYYMMDD_NNN
export DMD_HEAD="$(git rev-parse HEAD)"
export PLAN=configs/distillschedule/plans/helios_routea_full5_world16_typed.json
export PLAN_SHA="$(sha256sum "${PLAN}" | awk '{print $1}')"
export MASTER_ADDR=<node-0-reachable-ip>
export MASTER_PORT=<unused-port>
```

Each worker validates its own node-qualified resolved invocation. Before model
construction, the coordinator finalizes a shared semantic plan/config/trace/checkpoint
contract while retaining node-local raw plan and invocation digests separately.

Confirm both nodes have no GPU process and that the selected master and probe ports are unused. By default the topology probe uses `MASTER_PORT + 1`; set `DISTILLSCHEDULE_PROBE_PORT` explicitly when that port is unavailable.

## 4. Start through the central coordinator

From a clean checkout that is visible at the same path on both hosts:

```bash
python scripts/distillschedule/run_routea_world16.py \
  --hosts <node0-ssh-target>,<node1-ssh-target> \
  --run-id "${RUN_ID}" \
  --repo-root "$PWD" \
  --site-env configs/distillschedule/server/baai_2x8_a100.env.example
```

The coordinator owns both SSH workers, requests peer cancellation after a failure,
runs strict finalization once both workers stop and exits nonzero unless the curated
Attempt 004 result is `PASS`. Use `--dry-run` to inspect commands and
`--summarize-only` only for already complete immutable evidence. A dry-run is never
reusable as real evidence.

Each worker first:

1. verifies repository/source/executable/asset identities;
2. renders and validates its own Accelerate config;
3. compiles the typed plan;
4. joins a cheap 16-rank Gloo control group and NCCL all-reduce; and
5. proceeds to `train_helios.py` only when `topology_probe.json` is `PASS`.

This prevents another expensive “two independent world8” checkpoint attempt. The
worker still supports direct manual invocation for diagnosis, but manual launches are
not the authoritative Attempt 004 path.

## 5. Preserve evidence and summarize

Raw run evidence stays under the ignored directory:

```text
artifacts/runs/<RUN_ID>/
├── preregistration/
│   ├── topology.json
│   ├── node_0_contract.json
│   ├── node_1_contract.json
│   └── evidence_contract.json
├── launcher/node_0/
├── launcher/node_1/
├── phase_trace/rank_*.jsonl
└── checkpoint-15/
```

Generate the curated result:

```bash
python scripts/distillschedule/summarize_routea_world16_typed.py \
  --run-id "${RUN_ID}" \
  --baseline-result "${DISTILLSCHEDULE_BASELINE_RESULT}"
```

The output is written to `artifacts/manifests/<RUN_ID>_result.json`. Review `status`,
`timing.status`, `memory_observation` and `evidence_errors` separately. There is no
64-GiB reserve PASS/FAIL gate: high cgroup or host use is retained as evidence but
rejects only when an actual CUDA/cgroup OOM or memory-attributed nonzero exit occurs.
Do not publish a performance number from `PARTIAL`, `CRASH`, `OOM`, `FAIL` or
invalid evidence.

## 6. Close the loop

A result may be linked from Issue #2 only after:

- both node manifests and all 16 rank traces are present;
- the topology manifest is `PASS`;
- checkpoint-15 inventory is non-empty;
- the summarizer completes without evidence errors; and
- source, normalized semantic-plan/config and runtime identities match across nodes.

Commit only the small curated result manifest and status-document update. Keep raw logs, checkpoints, traces and profiler outputs in experiment storage.

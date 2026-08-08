# Runbook: BAAI Route-A Full5 world16 compatibility path

This runbook executes the frozen contract in [`../contracts/routea_full5_world16.md`](../contracts/routea_full5_world16.md). It does not authorize workload changes. The current shell entrypoint is a **per-host worker**; an external scheduler or operator must start both nodes together because the repository does not yet provide the central coordinator tracked in Issue #2.

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

The worker records the resolved plan digest after includes and environment expansion. The result summarizer compares evidence from both nodes; a future central preregistration manifest will make this cross-node check pre-launch rather than post-launch.

Confirm both nodes have no GPU process and that the selected master and probe ports are unused. By default the topology probe uses `MASTER_PORT + 1`; set `DISTILLSCHEDULE_PROBE_PORT` explicitly when that port is unavailable.

## 4. Start both host workers

Start node 0 and node 1 close together through the cluster scheduler or two controlled shells.

Node 0:

```bash
scripts/distillschedule/launch_routea_world16_typed_node.sh \
  0 "${RUN_ID}" "${DMD_HEAD}" "${PLAN_SHA}" "${MASTER_ADDR}" "${MASTER_PORT}"
```

Node 1:

```bash
scripts/distillschedule/launch_routea_world16_typed_node.sh \
  1 "${RUN_ID}" "${DMD_HEAD}" "${PLAN_SHA}" "${MASTER_ADDR}" "${MASTER_PORT}"
```

Each worker first:

1. verifies repository/source/executable/asset identities;
2. renders and validates its own Accelerate config;
3. compiles the typed plan;
4. joins a cheap 16-rank Gloo control group and NCCL all-reduce; and
5. proceeds to `train_helios.py` only when `topology_probe.json` is `PASS`.

This prevents another expensive “two independent world8” checkpoint attempt. It does not provide cross-host peer cancellation: if one worker dies after the model launch, stop the peer through the scheduler and preserve both node logs.

## 5. Preserve evidence and summarize

Raw run evidence stays under the ignored directory:

```text
artifacts/runs/<RUN_ID>/
├── topology_probe.json
├── compiled_run.node0.json
├── compiled_run.node1.json
├── resolved_plan.node0.json
├── resolved_plan.node1.json
├── resolved_helios_config.node0.yaml
├── resolved_helios_config.node1.yaml
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

The output is written to `artifacts/manifests/<RUN_ID>_result.json`. Review `functional_status`, `timing.status`, `reserve_admission.status` and `evidence_errors` separately. Do not publish a performance number from `PARTIAL`, `CRASH`, `OOM`, `CONTRACT_FAIL` or `EVIDENCE_INVALID` output.

## 6. Close the loop

A result may be linked from Issue #2 only after:

- both node manifests and all 16 rank traces are present;
- the topology manifest is `PASS`;
- checkpoint-15 inventory is non-empty;
- the summarizer completes without evidence errors; and
- source, resolved-plan and runtime identities match across nodes.

Commit only the small curated result manifest and status-document update. Keep raw logs, checkpoints, traces and profiler outputs in experiment storage.

#!/usr/bin/env bash
# CPU/source/config-only gate for one BAAI host.  No model or GPU workload starts.
set -Eeuo pipefail

ROOT="${DMD_SYSTEM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
HELIOS_ROOT="${HELIOS_RUNTIME_ROOT:-/share/project/shuchenweng/yongyan_liu/DistillSchedule}"
PYENV="${DISTILLSCHEDULE_PYENV:-/tmp/distillschedule-helios-official-torch210}"
PYTHON="${DISTILLSCHEDULE_PYTHON:-${PYENV}/bin/python}"
ACCELERATE="${DISTILLSCHEDULE_ACCELERATE:-${PYENV}/bin/accelerate}"
DEEPSPEED="${PYENV}/bin/deepspeed"
PLAN="${DISTILLSCHEDULE_PLAN:-${ROOT}/configs/distillschedule/plans/helios_routea_full5_world16_typed.json}"
NODE_RANK="${DISTILLSCHEDULE_NODE_RANK:-0}"
TRACE="${DISTILLSCHEDULE_TRACE_BUNDLE:-/share/project/shuchenweng/DistillSchedule_assets/m0/traces/syn_feature_trace_rank16_v3_sync_steps20}"
CHECKPOINT="${DISTILLSCHEDULE_CHECKPOINT:-/share/project/shuchenweng/yongyan_liu/DistillSchedule/artifacts/runs/iteration124_m0_b4fast_zero1_shared_cw4_x16_checkpoint10/checkpoint-10}"
DATA_ROOT="${HELIOS_DATA_ROOT:-/share/project/shuchenweng/DistillSchedule_assets/m0/datasets/syn_feature_rank16}"
GENERATOR_ROOT="${HELIOS_DISTILLED_ROOT:-/share/project/shuchenweng/DistillSchedule_assets/Helios-Distilled}"
SCORE_ROOT="${HELIOS_BASE_ROOT:-/home/shuchenweng/xty/Helios-main/weights}"
OUT="${DISTILLSCHEDULE_OUTPUT_DIR:-/tmp/distillschedule-baai-smoke-$(id -u)-${NODE_RANK}-$$}"
[[ "${NODE_RANK}" == 0 || "${NODE_RANK}" == 1 ]] || { echo invalid_node_rank >&2; exit 2; }

require_file() { [[ -f "$1" ]] || { echo "missing required file: $1" >&2; exit 2; }; }
require_dir() { [[ -d "$1" ]] || { echo "missing required directory: $1" >&2; exit 2; }; }
for file in "${PYTHON}" "${ACCELERATE}" "${DEEPSPEED}" "${PLAN}" \
  "${ROOT}/scripts/accelerate_configs/zero1_cpu_optimizer_offload.json" \
  "${HELIOS_ROOT}/train_helios.py"; do require_file "${file}"; done
for directory in "${TRACE}" "${CHECKPOINT}" "${DATA_ROOT}" "${GENERATOR_ROOT}" "${SCORE_ROOT}"; do require_dir "${directory}"; done
[[ -z "$(git -C "${ROOT}" status --short)" ]] || { echo "DMD-System tree is dirty" >&2; exit 2; }
[[ -z "$(git -C "${HELIOS_ROOT}" status --short)" ]] || { echo "Helios compatibility tree is dirty" >&2; exit 2; }

mkdir -p "${OUT}"
cleanup() { rm -rf "${OUT}" 2>/dev/null || true; }
trap cleanup EXIT
ACCEL_CONFIG="${OUT}/accelerate.yaml"
cat >"${ACCEL_CONFIG}" <<EOF
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_config_file: ${ROOT}/scripts/accelerate_configs/zero1_cpu_optimizer_offload.json
  deepspeed_multinode_launcher: standard
fsdp_config: {}
gradient_accumulation_steps: 1
machine_rank: ${NODE_RANK}
main_training_function: main
mixed_precision: bf16
num_machines: 2
num_processes: 16
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

export PATH="${PYENV}/bin:${PATH}"
export DMD_SYSTEM_ROOT="${ROOT}"
export HELIOS_RUNTIME_ROOT="${HELIOS_ROOT}"
export DISTILLSCHEDULE_PYTHON="${PYTHON}"
export DISTILLSCHEDULE_ACCELERATE="${ACCELERATE}"
export DISTILLSCHEDULE_ACCELERATE_CONFIG="${ACCEL_CONFIG}"
export DISTILLSCHEDULE_NODE_RANK="${NODE_RANK}"
export DISTILLSCHEDULE_MASTER_ADDR="127.0.0.1"
export DISTILLSCHEDULE_MASTER_PORT="29677"
export DISTILLSCHEDULE_RUN_ID="baai-smoke-${NODE_RANK}"
export DISTILLSCHEDULE_OUTPUT_DIR="${OUT}/run"
export DISTILLSCHEDULE_TRACE_BUNDLE="${TRACE}"
export DISTILLSCHEDULE_CHECKPOINT="${CHECKPOINT}"
export DISTILLSCHEDULE_FROZEN_BLOCK_MMAP_CACHE="${OUT}/frozen"
export DISTILLSCHEDULE_IMMUTABLE_MODEL_MMAP_CACHE="${OUT}/immutable"
export HELIOS_DATA_ROOT="${DATA_ROOT}"
export HELIOS_DISTILLED_ROOT="${GENERATOR_ROOT}"
export HELIOS_BASE_ROOT="${SCORE_ROOT}"
export PYTHONPATH="${ROOT}/src:${ROOT}:${HELIOS_ROOT}"
mkdir -p "${DISTILLSCHEDULE_FROZEN_BLOCK_MMAP_CACHE}" "${DISTILLSCHEDULE_IMMUTABLE_MODEL_MMAP_CACHE}"

"${PYTHON}" -m compileall -q \
  "${ROOT}/src" "${ROOT}/runtime" "${ROOT}/scripts/distillctl.py" "${ROOT}/scripts/distillschedule"
"${PYTHON}" -m pytest -q "${ROOT}/tests/test_refactor_"*.py
"${PYTHON}" "${ROOT}/scripts/distillctl.py" validate "${PLAN}" --repo-root "${ROOT}" --json >"${OUT}/compiled_preflight.json"
"${PYTHON}" "${ROOT}/scripts/distillctl.py" run "${PLAN}" --repo-root "${ROOT}" --dry-run >"${OUT}/dry_run.log"
RENDERED="${DISTILLSCHEDULE_OUTPUT_DIR}/resolved_helios_config.node${NODE_RANK}.yaml"
require_file "${RENDERED}"

"${PYTHON}" "${ROOT}/scripts/distillschedule/build_routea_evidence_contract.py" node \
  --node-rank "${NODE_RANK}" \
  --plan "${PLAN}" \
  --repo-root "${ROOT}" \
  --compiled-preflight "${OUT}/compiled_preflight.json" \
  --rendered-config "${RENDERED}" \
  --accelerate-config "${ACCEL_CONFIG}" \
  --trace-bundle "${TRACE}" \
  --checkpoint "${CHECKPOINT}" \
  --asset "helios_runtime=${HELIOS_ROOT}" \
  --asset "generator=${GENERATOR_ROOT}" \
  --asset "score=${SCORE_ROOT}" \
  --asset "data=${DATA_ROOT}" \
  --output "${OUT}/node_contract.json" >"${OUT}/node_contract_summary.json"

"${PYTHON}" - "${OUT}/node_contract.json" <<'PY'
import json, pathlib, sys
value=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert value['schema_version']=='distillschedule.routea_world16.node_contract.v1'
assert value['shared']['batch_contract']=={
  'micro_batch_per_gpu':1,
  'gradient_accumulation_steps':1,
  'global_batch':16,
}
assert value['shared']['memory_acceptance']['host_reserve_gate'] is False
assert value['shared']['trace']['rank_count']==16
PY

echo "PASS BAAI Route-A source/config/evidence smoke; no model process or GPU workload launched"
echo "node_rank=${NODE_RANK}"
echo "plan_sha256=$(sha256sum "${PLAN}" | awk '{print $1}')"
echo "node_contract_sha256=$(sha256sum "${OUT}/node_contract.json" | awk '{print $1}')"

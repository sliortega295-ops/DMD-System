#!/usr/bin/env bash
# CPU/source/config gate only. It never starts the topology probe or a model process.
set -Eeuo pipefail

ROOT="${DMD_SYSTEM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
HELIOS_RUNTIME_ROOT="${HELIOS_RUNTIME_ROOT:-/share/project/shuchenweng/yongyan_liu/DistillSchedule}"
DISTILLSCHEDULE_PYENV="${DISTILLSCHEDULE_PYENV:-/tmp/distillschedule-helios-official-torch210}"
DISTILLSCHEDULE_PYTHON="${DISTILLSCHEDULE_PYTHON:-${DISTILLSCHEDULE_PYENV}/bin/python}"
DISTILLSCHEDULE_ACCELERATE="${DISTILLSCHEDULE_ACCELERATE:-${DISTILLSCHEDULE_PYENV}/bin/accelerate}"
DISTILLSCHEDULE_DEEPSPEED="${DISTILLSCHEDULE_DEEPSPEED:-${DISTILLSCHEDULE_PYENV}/bin/deepspeed}"
PLAN="${DISTILLSCHEDULE_PLAN:-${ROOT}/configs/distillschedule/plans/helios_routea_full5_world16_typed.json}"
OUT="${DISTILLSCHEDULE_OUTPUT_DIR:-/tmp/distillschedule-dmd-system-smoke-$(id -u)}"
ACCEL_TEMPLATE="${DISTILLSCHEDULE_ACCELERATE_TEMPLATE:-${ROOT}/scripts/accelerate_configs/multi_node_zero1_cpu_optimizer_offload.yaml}"
DS_CONFIG="${DISTILLSCHEDULE_DEEPSPEED_CONFIG:-${ROOT}/scripts/accelerate_configs/zero1_cpu_optimizer_offload.json}"
ACCEL_CONFIG="${OUT}/accelerate.node0.yaml"

require_file() { [[ -f "$1" ]] || { echo "missing_required_file=$1" >&2; exit 2; }; }
require_dir() { [[ -d "$1" ]] || { echo "missing_required_directory=$1" >&2; exit 2; }; }
require_executable() { [[ -x "$1" ]] || { echo "missing_required_executable=$1" >&2; exit 2; }; }

for executable in "${DISTILLSCHEDULE_PYTHON}" "${DISTILLSCHEDULE_ACCELERATE}" "${DISTILLSCHEDULE_DEEPSPEED}"; do
  require_executable "${executable}"
done
for file in "${PLAN}" "${ACCEL_TEMPLATE}" "${DS_CONFIG}" "${HELIOS_RUNTIME_ROOT}/train_helios.py"; do
  require_file "${file}"
done
for variable in \
  HELIOS_DISTILLED_ROOT HELIOS_BASE_ROOT HELIOS_DATA_ROOT \
  DISTILLSCHEDULE_TRACE_BUNDLE DISTILLSCHEDULE_CHECKPOINT \
  DISTILLSCHEDULE_HF_HOME DISTILLSCHEDULE_FLASH_ATTN_KERNEL \
  DISTILLSCHEDULE_PRECAST_GENERATOR_PATH DISTILLSCHEDULE_PRECAST_SCORE_PATH; do
  [[ -n "${!variable:-}" ]] || { echo "required_environment_missing=${variable}" >&2; exit 2; }
done
for directory in \
  "${HELIOS_DISTILLED_ROOT}" "${HELIOS_BASE_ROOT}" "${HELIOS_DATA_ROOT}" \
  "${DISTILLSCHEDULE_TRACE_BUNDLE}" "${DISTILLSCHEDULE_CHECKPOINT}" \
  "${DISTILLSCHEDULE_HF_HOME}" "${DISTILLSCHEDULE_FLASH_ATTN_KERNEL}" \
  "${DISTILLSCHEDULE_PRECAST_GENERATOR_PATH}" "${DISTILLSCHEDULE_PRECAST_SCORE_PATH}"; do
  require_dir "${directory}"
done

mkdir -p "${OUT}"
export PATH="${DISTILLSCHEDULE_PYENV}/bin:${PATH}"
export DMD_SYSTEM_ROOT="${ROOT}"
export HELIOS_RUNTIME_ROOT DISTILLSCHEDULE_PYTHON DISTILLSCHEDULE_ACCELERATE
export DISTILLSCHEDULE_RUN_ID="baai-server-smoke"
export DISTILLSCHEDULE_OUTPUT_DIR="${OUT}"
export DISTILLSCHEDULE_FROZEN_BLOCK_MMAP_CACHE="${OUT}/frozen"
export DISTILLSCHEDULE_IMMUTABLE_MODEL_MMAP_CACHE="${OUT}/immutable"
export DISTILLSCHEDULE_NODE_RANK=0
export DISTILLSCHEDULE_MASTER_ADDR=127.0.0.1
export DISTILLSCHEDULE_MASTER_PORT=29677
export PYTHONPATH="${ROOT}/src:${ROOT}"

"${DISTILLSCHEDULE_PYTHON}" "${ROOT}/scripts/distillschedule/render_accelerate_config.py" \
  --template "${ACCEL_TEMPLATE}" \
  --deepspeed-config "${DS_CONFIG}" \
  --output "${ACCEL_CONFIG}" \
  --num-machines 2 \
  --num-processes 16 \
  --machine-rank 0
export DISTILLSCHEDULE_ACCELERATE_CONFIG="${ACCEL_CONFIG}"

"${DISTILLSCHEDULE_PYTHON}" -m pytest -q "${ROOT}/tests"
"${DISTILLSCHEDULE_PYTHON}" "${ROOT}/scripts/distillctl.py" validate "${PLAN}" --repo-root "${ROOT}"
"${DISTILLSCHEDULE_PYTHON}" "${ROOT}/scripts/distillctl.py" run "${PLAN}" --repo-root "${ROOT}" --dry-run

echo "PASS server smoke; no distributed process group or model process was launched"
echo "resolved_artifacts=${OUT}"

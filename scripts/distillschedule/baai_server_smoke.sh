#!/usr/bin/env bash
set -euo pipefail

ROOT="${DMD_SYSTEM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
HELIOS_RUNTIME_ROOT="${HELIOS_RUNTIME_ROOT:-/share/project/shuchenweng/yongyan_liu/DistillSchedule}"
DISTILLSCHEDULE_PYTHON="${DISTILLSCHEDULE_PYTHON:-/tmp/distillschedule-helios-official-torch210/bin/python}"
PLAN="${DISTILLSCHEDULE_PLAN:-${ROOT}/configs/distillschedule/plans/helios_14b_384x640_full5.json}"
OUT="${DISTILLSCHEDULE_OUTPUT_DIR:-/tmp/distillschedule-dmd-system-smoke-$(id -u)}"

require_file() {
  [[ -f "$1" ]] || { echo "missing required file: $1" >&2; exit 2; }
}
require_dir() {
  [[ -d "$1" ]] || { echo "missing required directory: $1" >&2; exit 2; }
}

require_file "${DISTILLSCHEDULE_PYTHON}"
require_file "${HELIOS_RUNTIME_ROOT}/train_helios.py"
require_file "${HELIOS_RUNTIME_ROOT}/configs/experiments/iteration45_m0_b0a_a100_substrate_384x640.yaml"
require_dir "${HELIOS_DISTILLED_ROOT:-/share/project/shuchenweng/DistillSchedule_assets/Helios-Distilled}"
require_dir "${HELIOS_BASE_ROOT:-/home/shuchenweng/xty/Helios-main/weights}"
require_dir "${DISTILLSCHEDULE_TRACE_BUNDLE:-/share/project/shuchenweng/DistillSchedule_assets/m0/traces/syn_feature_trace_v2_steps20}"
require_dir "${DISTILLSCHEDULE_CHECKPOINT:-/share/project/shuchenweng/yongyan_liu/DistillSchedule/artifacts/runs/iteration65_m0_b4fast_canonical_checkpoint10_001/checkpoint-10}"

export DMD_SYSTEM_ROOT="$ROOT"
export HELIOS_RUNTIME_ROOT DISTILLSCHEDULE_PYTHON
export DISTILLSCHEDULE_OUTPUT_DIR="$OUT"
export PYTHONPATH="${ROOT}/src:${ROOT}"

"${DISTILLSCHEDULE_PYTHON}" -m pytest -q "${ROOT}/tests/test_refactor_"*.py
"${DISTILLSCHEDULE_PYTHON}" "${ROOT}/scripts/distillctl.py" validate "$PLAN" --repo-root "$ROOT"
"${DISTILLSCHEDULE_PYTHON}" "${ROOT}/scripts/distillctl.py" run "$PLAN" --repo-root "$ROOT" --dry-run

echo "PASS server smoke; no model process or GPU workload was launched"
echo "resolved artifacts: $OUT"


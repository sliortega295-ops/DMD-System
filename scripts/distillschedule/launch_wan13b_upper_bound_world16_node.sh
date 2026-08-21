#!/usr/bin/env bash
# Fail-closed per-node launcher for real Wan1.3B resident-vs-FSDP2 Full5.
set -Eeuo pipefail

NODE_RANK="${1:?node rank 0 or 1}"
RUN_ID="${2:?immutable run id}"
MODE="${3:?validate, correctness, timing, or profile}"
EXPECTED_HEAD="${4:?expected DMD-System HEAD}"
MASTER_ADDR="${5:?node0 routable address}"
MASTER_PORT="${6:?world16 rendezvous port}"
PLACEMENT="${7:?resident or world16-fsdp}"

[[ "${NODE_RANK}" == 0 || "${NODE_RANK}" == 1 ]] || exit 2
[[ "${MODE}" == validate || "${MODE}" == correctness || "${MODE}" == timing || "${MODE}" == profile ]] || exit 2
[[ "${PLACEMENT}" == resident || "${PLACEMENT}" == world16-fsdp ]] || exit 2
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || exit 2
[[ "${EXPECTED_HEAD}" =~ ^[0-9a-f]{40}$ ]] || exit 2

ROOT="${DMD_SYSTEM_ROOT:-/share/project/shuchenweng/yongyan_liu/DMD-System_wan13b_upper_bound_20260820}"
HELIOS="${HELIOS_RUNTIME_ROOT:-/share/project/shuchenweng/yongyan_liu/DistillSchedule_gan_rolephase_exec_20260812}"
PYENV="${DISTILLSCHEDULE_PYENV:-/tmp/distillschedule-helios-official-torch210}"
PYTHON="${PYENV}/bin/python"
TORCHRUN="${PYENV}/bin/torchrun"
ASSETS="${DISTILLSCHEDULE_ASSETS:-/share/project/shuchenweng/DistillSchedule_assets}"
MODEL_ROOT="${DISTILLSCHEDULE_WAN13B_MODEL_ROOT:?set DISTILLSCHEDULE_WAN13B_MODEL_ROOT}"
BOOTSTRAP="${DISTILLSCHEDULE_WAN13B_BOOTSTRAP:?set DISTILLSCHEDULE_WAN13B_BOOTSTRAP}"
BOOTSTRAP_HASH="${DISTILLSCHEDULE_WAN13B_BOOTSTRAP_HASH:?set DISTILLSCHEDULE_WAN13B_BOOTSTRAP_HASH}"
TRACE="${DISTILLSCHEDULE_TRACE_BUNDLE:-${ASSETS}/m0/traces/syn_feature_trace_rank16_v4_strict_sync_steps20}"
TRACE_HASH="${DISTILLSCHEDULE_TRACE_BUNDLE_HASH:-654d49015921b278ab82e2fc84ea1896692da3fd38ce82fbc500cb1ebfec46d3}"
NEGATIVE="${DISTILLSCHEDULE_NEGATIVE_PROMPT_ASSET:-${ASSETS}/m1/negative_prompt_official_v1}"
RUN_ROOT="${DISTILLSCHEDULE_RUN_ROOT:-${ROOT}/artifacts/runs/${RUN_ID}}"
EXECUTION="${RUN_ROOT}/execution"
NODE_LOG="${RUN_ROOT}/launcher/node_${NODE_RANK}"
LOCK_FILE="${DISTILLSCHEDULE_NODE_LOCK:-/tmp/distillschedule_gpu_node.lock}"
EXPECTED_HELIOS_TRANSFORMER_SHA256="${DISTILLSCHEDULE_EXPECTED_HELIOS_TRANSFORMER_SHA256:-776a26c3b2921d599821f0767910830471ff94a955f5013089f9f689ac20fee7}"
CORRECTNESS_RECEIPT="${DISTILLSCHEDULE_WAN13B_CORRECTNESS_RECEIPT:-}"
CORRECTNESS_SHA256="${DISTILLSCHEDULE_WAN13B_CORRECTNESS_SHA256:-}"

for executable in "${PYTHON}" "${TORCHRUN}"; do
  [[ -x "${executable}" ]] || { echo missing_executable="${executable}" >&2; exit 3; }
done
for path in "${ROOT}" "${HELIOS}" "${MODEL_ROOT}" "${BOOTSTRAP}" "${TRACE}" "${NEGATIVE}"; do
  [[ -e "${path}" ]] || { echo missing_path="${path}" >&2; exit 3; }
done
[[ "${BOOTSTRAP_HASH}" =~ ^[0-9a-f]{64}$ ]] || { echo invalid_bootstrap_hash >&2; exit 2; }
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_HEAD}" ]] || { echo source_head_mismatch >&2; exit 4; }
git -C "${ROOT}" diff --quiet && git -C "${ROOT}" diff --cached --quiet || { echo source_worktree_dirty >&2; exit 4; }
HELIOS_TRANSFORMER="${HELIOS}/helios/modules/transformer_helios.py"
[[ "$(sha256sum "${HELIOS_TRANSFORMER}" | awk '{print $1}')" == "${EXPECTED_HELIOS_TRANSFORMER_SHA256}" ]] || { echo helios_transformer_hash_mismatch >&2; exit 4; }

if [[ "${MODE}" == timing ]]; then
  [[ -f "${CORRECTNESS_RECEIPT}" ]] || { echo correctness_receipt_missing >&2; exit 3; }
  [[ "$(sha256sum "${CORRECTNESS_RECEIPT}" | awk '{print $1}')" == "${CORRECTNESS_SHA256}" ]] || { echo correctness_receipt_hash_changed >&2; exit 3; }
  "${PYTHON}" - "${CORRECTNESS_RECEIPT}" <<'PY'
import json,pathlib,sys
value=json.loads(pathlib.Path(sys.argv[1]).read_text())
if value.get("status") != "PASS" or not value.get("claimable_for_paired_timing"):
    raise SystemExit("cross-placement correctness gate is not PASS")
PY
fi

mkdir -p "${NODE_LOG}" "${EXECUTION}"
[[ ! -e "${NODE_LOG}/launch_manifest.txt" ]] || { echo run_id_already_used >&2; exit 82; }
exec 9>"${LOCK_FILE}"
flock -n 9 || { echo gpu_lock_busy >&2; exit 73; }
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')" ]] || { echo gpu_processes_present >&2; exit 81; }

export PATH="${PYENV}/bin:${PATH}"
export PYTHONPATH="${ROOT}/src:${ROOT}:${HELIOS}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="${ASSETS}/hf_home_official_torch210"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 NCCL_ASYNC_ERROR_HANDLING=1

CGROUP_EVENTS="/sys/fs/cgroup/memory.events"
CGROUP_CURRENT="/sys/fs/cgroup/memory.current"
CGROUP_PEAK="/sys/fs/cgroup/memory.peak"
cp "${CGROUP_EVENTS}" "${NODE_LOG}/cgroup_memory_events_before.txt" 2>/dev/null || true
cat "${CGROUP_CURRENT}" >"${NODE_LOG}/cgroup_memory_current_before.txt" 2>/dev/null || true
cat "${CGROUP_PEAK}" >"${NODE_LOG}/cgroup_memory_peak_before.txt" 2>/dev/null || true
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader \
  >"${NODE_LOG}/gpu_before.csv"

cat >"${NODE_LOG}/launch_manifest.txt" <<EOF
schema_version=distillschedule.wan13b_upper_bound_launch.v1
run_id=${RUN_ID}
mode=${MODE}
node_rank=${NODE_RANK}
dmd_system_head=${EXPECTED_HEAD}
helios_runtime_root=${HELIOS}
helios_transformer_sha256=${EXPECTED_HELIOS_TRANSFORMER_SHA256}
model_root=${MODEL_ROOT}
bootstrap=${BOOTSTRAP}
bootstrap_hash=${BOOTSTRAP_HASH}
trace=${TRACE}
trace_hash=${TRACE_HASH}
placement=${PLACEMENT}
world_size=16
global_batch=16
rollout_batch_size=1
score_packing=false
parameter_offload=false
activation_offload=false
optimizer_offload=false
parameter_streaming=false
model_swap=false
generator_residual_storage=cuda_out_of_place
critic_residual_storage=cuda_out_of_place
ema_device=cuda
EOF

WORKER=(
  "${ROOT}/scripts/distillschedule/probe_wan13b_upper_bound_world16.py"
  --run-id "${RUN_ID}" --output-dir "${EXECUTION}" --placement "${PLACEMENT}"
  --bootstrap "${BOOTSTRAP}" --expected-bootstrap-hash "${BOOTSTRAP_HASH}"
  --model-root "${MODEL_ROOT}"
  --trace-bundle "${TRACE}" --expected-trace-hash "${TRACE_HASH}"
  --negative-prompt "${NEGATIVE}"
)
[[ "${MODE}" == validate ]] && WORKER+=(--validate-only)
[[ "${MODE}" == timing ]] && WORKER+=(--timing-mode)
[[ "${MODE}" == profile ]] && WORKER+=(--profile-mode)

set +e
if [[ "${MODE}" == profile ]]; then
  NSYS_BIN="${DISTILLSCHEDULE_NSYS_BIN:-$(command -v nsys || true)}"
  [[ -x "${NSYS_BIN}" ]] || { echo nsys_missing >&2; exit 3; }
  NSYS_VERSION="$(${NSYS_BIN} --version | head -1)"
  [[ "${NSYS_VERSION}" =~ 2024\.6\.2|2024\.[789]|202[5-9]\. ]] || {
    echo "nsys_version_not_admissible=${NSYS_VERSION}" >&2
    exit 3
  }
  PROFILE_ROOT="${RUN_ROOT}/nsight"
  PROFILE_NODE_DIR="${PROFILE_ROOT}/node_${NODE_RANK}"
  RECEIPT_DIR="${PROFILE_ROOT}/range_receipts"
  REPORT_BASE="${PROFILE_NODE_DIR}/wan13b_fsdp2_node${NODE_RANK}"
  mkdir -p "${PROFILE_NODE_DIR}" "${RECEIPT_DIR}"
  {
    echo "nsys_bin=${NSYS_BIN}"
    echo "nsys_version=${NSYS_VERSION}"
    echo "capture_mode=process_wide_clipped_to_exact_nvtx"
  } >"${PROFILE_NODE_DIR}/profiler_identity.txt"
  [[ ! -e "${REPORT_BASE}.nsys-rep" && ! -e "${REPORT_BASE}.sqlite" ]] || {
    echo nsys_artifact_exists >&2
    exit 82
  }
  export DISTILLSCHEDULE_STATIC_FSDP2_NSYS_EXACT_RANGE=1
  export DISTILLSCHEDULE_STATIC_FSDP2_NSYS_RECEIPT_DIR="${RECEIPT_DIR}"
  # The CUDA profiler API stop path is known to deadlock controller ranks 0/8
  # under this two-node process tree.  Capture the complete process instead and
  # clip analysis to the exact balanced WAN13B_FULL5 NVTX/receipt window.
  export DISTILLSCHEDULE_STATIC_FSDP2_NSYS_CONTROL_API=0
  "${NSYS_BIN}" profile \
    --trace=cuda,nvtx \
    --trace-fork-before-exec=true \
    --wait=all \
    --sample=none \
    --cpuctxsw=none \
    --cuda-flush-interval=1000 \
    --force-overwrite=true \
    --output="${REPORT_BASE}" \
    "${TORCHRUN}" --no-python --nnodes=2 --nproc-per-node=8 --node-rank="${NODE_RANK}" \
      --master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}" \
      "${PYTHON}" "${WORKER[@]}" >"${NODE_LOG}/console.log" 2>&1
  status=$?
  if [[ ${status} -eq 0 ]]; then
    if [[ ! -s "${REPORT_BASE}.nsys-rep" ]]; then
      echo missing_nsys_report >&2
      status=87
    else
      "${NSYS_BIN}" export --type sqlite --force-overwrite true \
        --output "${REPORT_BASE}.sqlite" "${REPORT_BASE}.nsys-rep" \
        >"${PROFILE_NODE_DIR}/nsys_export.log" 2>&1
      status=$?
    fi
  fi
  if [[ ${status} -eq 0 ]]; then
    if [[ ! -s "${REPORT_BASE}.sqlite" ]]; then
      echo missing_nsys_sqlite >&2
      status=88
    else
      "${NSYS_BIN}" stats --report nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum --format csv \
        "${REPORT_BASE}.sqlite" >"${PROFILE_NODE_DIR}/nsys_stats_validation.log" 2>&1
      status=$?
    fi
  fi
  if [[ ${status} -eq 0 ]] && grep -Eq \
      "ProcessEventsError|Unknown runtime API|Cannot find bucket|Cannot find string|Errors occurred while processing raw events|does not contain (NV Tools Extension|CUDA kernel|CUDA API)" \
      "${PROFILE_NODE_DIR}/nsys_export.log" "${PROFILE_NODE_DIR}/nsys_stats_validation.log"; then
    echo nsys_profile_validation_failed >&2
    status=89
  fi
  if [[ ${status} -eq 0 ]]; then
    for _ in $(seq 1 300); do
      count="$(find "${RECEIPT_DIR}" -maxdepth 1 -type f -name 'rank_*.json' | wc -l)"
      [[ "${count}" == 16 ]] && break
      sleep 1
    done
    count="$(find "${RECEIPT_DIR}" -maxdepth 1 -type f -name 'rank_*.json' | wc -l)"
    if [[ "${count}" != 16 ]]; then
      echo "nsys_range_receipts=${count},expected=16" >&2
      status=90
    fi
  fi
  if [[ ${status} -eq 0 && "${NODE_RANK}" == 0 ]]; then
    "${PYTHON}" - "${RECEIPT_DIR}" "${PROFILE_ROOT}/range_validation.json" <<'PY'
import json
import pathlib
import sys
from distillschedule.runtime.nsys_range import validate_static_nsys_receipts

value = validate_static_nsys_receipts(
    pathlib.Path(sys.argv[1]),
    variant="no-gan",
    expected_stop_point="after_step14_ema_update",
    expected_controller_ranks=[],
)
pathlib.Path(sys.argv[2]).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY
    status=$?
  fi
else
  "${TORCHRUN}" --nnodes=2 --nproc-per-node=8 --node-rank="${NODE_RANK}" \
    --master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}" \
    "${WORKER[@]}" >"${NODE_LOG}/console.log" 2>&1
  status=$?
fi
set -e

cp "${CGROUP_EVENTS}" "${NODE_LOG}/cgroup_memory_events_after.txt" 2>/dev/null || true
cat "${CGROUP_CURRENT}" >"${NODE_LOG}/cgroup_memory_current_after.txt" 2>/dev/null || true
cat "${CGROUP_PEAK}" >"${NODE_LOG}/cgroup_memory_peak_after.txt" 2>/dev/null || true
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader \
  >"${NODE_LOG}/gpu_after.csv" || true
[[ ${status} -eq 0 ]] || exit "${status}"
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')" ]] || { echo gpu_processes_remain >&2; exit 88; }

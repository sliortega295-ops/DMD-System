#!/usr/bin/env bash
# Typed Route-A Full5 world16 node runner.  The typed DMD-System plan owns the
# schedule/config/lifecycle contract; the pinned Helios entrypoint owns model math.
set -euo pipefail
NODE_RANK="${1:?node rank 0 or 1}"
RUN_ID="${2:?run id}"
EXPECTED_HEAD="${3:?expected DMD-System HEAD}"
EXPECTED_PLAN_SHA="${4:?expected plan sha256}"
MASTER_ADDR="${5:?master addr}"
MASTER_PORT="${6:?master port}"
[[ "${NODE_RANK}" == 0 || "${NODE_RANK}" == 1 ]] || { echo invalid_node_rank >&2; exit 2; }

ROOT="${DMD_SYSTEM_ROOT:-/share/project/shuchenweng/yongyan_liu/DMD-System}"
HELIOS_ROOT="${HELIOS_RUNTIME_ROOT:-/share/project/shuchenweng/yongyan_liu/DistillSchedule}"
PYENV="${DISTILLSCHEDULE_PYENV:-/tmp/distillschedule-helios-official-torch210}"
PYTHON="${PYENV}/bin/python"
ACCELERATE="${PYENV}/bin/accelerate"
PLAN="${ROOT}/configs/distillschedule/plans/helios_routea_full5_world16_typed.json"
OUT="${ROOT}/artifacts/runs/${RUN_ID}"
NODE_LOG="${OUT}/launcher/node_${NODE_RANK}"
CACHE_ROOT="/tmp/distillschedule_typed_routea/${RUN_ID}"
LOCK_FILE="/tmp/distillschedule_gpu_node.lock"

[[ -x "${PYTHON}" && -x "${ACCELERATE}" ]] || { echo runtime_missing >&2; exit 3; }
[[ -f "${PLAN}" ]] || { echo plan_missing >&2; exit 3; }
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_HEAD}" ]] || { echo head_drift >&2; exit 4; }
[[ -z "$(git -C "${ROOT}" status --short)" ]] || { echo repository_not_clean >&2; exit 4; }
[[ "$(sha256sum "${PLAN}" | awk '{print $1}')" == "${EXPECTED_PLAN_SHA}" ]] || { echo plan_drift >&2; exit 4; }
for asset in \
  /share/project/shuchenweng/DistillSchedule_assets/m0/traces/syn_feature_trace_rank16_v3_sync_steps20 \
  /share/project/shuchenweng/DistillSchedule_assets/m0/datasets/syn_feature_rank16 \
  /share/project/shuchenweng/yongyan_liu/DistillSchedule/artifacts/runs/iteration124_m0_b4fast_zero1_shared_cw4_x16_checkpoint10/checkpoint-10 \
  /share/project/shuchenweng/DistillSchedule_assets/precast/helios-distilled-transformer-ode-mixed-bf16-v1 \
  /share/project/shuchenweng/DistillSchedule_assets/precast/helios-score-transformer-mixed-bf16-v1; do
  [[ -e "${asset}" ]] || { echo "asset_missing=${asset}" >&2; exit 5; }
done

exec 9>"${LOCK_FILE}"
flock -n 9 || { echo gpu_lock_busy >&2; exit 73; }
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')" ]]; then
  echo gpu_processes_present >&2; exit 81
fi
[[ ! -e "${CACHE_ROOT}" ]] || { echo node_cache_exists >&2; exit 82; }
mkdir -p "${NODE_LOG}" "${CACHE_ROOT}/frozen" "${CACHE_ROOT}/immutable"

export PATH="${PYENV}/bin:${PATH}"
export PYTHONPATH="${ROOT}/src:${ROOT}:${HELIOS_ROOT}"
export DISTILLSCHEDULE_RUN_ID="${RUN_ID}"
export DISTILLSCHEDULE_OUTPUT_DIR="${OUT}"
export DISTILLSCHEDULE_FROZEN_BLOCK_MMAP_CACHE="${CACHE_ROOT}/frozen"
export DISTILLSCHEDULE_IMMUTABLE_MODEL_MMAP_CACHE="${CACHE_ROOT}/immutable"
export DISTILLSCHEDULE_NODE_RANK="${NODE_RANK}"
export DISTILLSCHEDULE_MASTER_ADDR="${MASTER_ADDR}"
export DISTILLSCHEDULE_MASTER_PORT="${MASTER_PORT}"
export HELIOS_RUNTIME_ROOT="${HELIOS_ROOT}"
export DISTILLSCHEDULE_PYTHON="${PYTHON}"
export DISTILLSCHEDULE_ACCELERATE="${ACCELERATE}"
export DISTILLSCHEDULE_ACCELERATE_CONFIG="${ROOT}/scripts/accelerate_configs/multi_node_zero1_cpu_optimizer_offload.yaml"

MANIFEST="${NODE_LOG}/launch_manifest.txt"
MONITOR="${NODE_LOG}/resource.csv"
{
  echo schema_version=distillschedule.typed_routea_world16.node_launch.v1
  echo run_id="${RUN_ID}"
  echo node_rank="${NODE_RANK}"
  echo host="$(hostname)"
  echo started_at="$(date -Is)"
  echo dmd_system_head="${EXPECTED_HEAD}"
  echo dmd_system_plan_sha256="${EXPECTED_PLAN_SHA}"
  echo helios_head="$(git -C "${HELIOS_ROOT}" rev-parse HEAD)"
  echo helios_entrypoint_sha256="$(sha256sum "${HELIOS_ROOT}/train_helios.py" | awk '{print $1}')"
  echo master_addr="${MASTER_ADDR}"
  echo master_port="${MASTER_PORT}"
  echo cgroup_memory_current_before="$(cat /sys/fs/cgroup/memory.current)"
  echo cgroup_memory_max="$(cat /sys/fs/cgroup/memory.max)"
  echo memory_events_before="$(tr '\n' ';' </sys/fs/cgroup/memory.events)"
  nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv,noheader
} >"${MANIFEST}"

monitor() {
  echo 'unix_time,cgroup_bytes,host_used_bytes,gpu_used_mib,gpu_util_percent' >"${MONITOR}"
  while true; do
    now="$(date +%s.%N)"
    cgroup="$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo -1)"
    host_used="$(awk '/^MemTotal:/{t=$2}/^MemAvailable:/{a=$2}END{print (t-a)*1024}' /proc/meminfo)"
    gpu_mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd: - || echo unavailable)"
    gpu_util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | paste -sd: - || echo unavailable)"
    echo "${now},${cgroup},${host_used},${gpu_mem},${gpu_util}" >>"${MONITOR}"
    sleep 1
  done
}
MONITOR_PID= RUN_PID=
cleanup() {
  if [[ -n "${RUN_PID:-}" ]] && kill -0 "${RUN_PID}" 2>/dev/null; then
    kill -TERM -- "-${RUN_PID}" 2>/dev/null || true; sleep 2
    kill -KILL -- "-${RUN_PID}" 2>/dev/null || true; wait "${RUN_PID}" 2>/dev/null || true
  fi
  if [[ -n "${MONITOR_PID:-}" ]]; then kill "${MONITOR_PID}" 2>/dev/null || true; wait "${MONITOR_PID}" 2>/dev/null || true; fi
  {
    echo finished_at="$(date -Is)"
    echo cgroup_memory_current_after="$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo -1)"
    echo memory_events_after="$(tr '\n' ';' </sys/fs/cgroup/memory.events 2>/dev/null || true)"
  } >>"${MANIFEST}" 2>/dev/null || true
  rm -rf "${CACHE_ROOT}" 2>/dev/null || true
}
trap cleanup EXIT TERM INT HUP
monitor & MONITOR_PID=$!

cd "${ROOT}"
set +e
setsid timeout --signal=TERM --kill-after=180s 14400s \
  "${PYTHON}" scripts/distillctl.py run "${PLAN}" --repo-root "${ROOT}" \
  >"${NODE_LOG}/console.log" 2>&1 &
RUN_PID=$!
wait "${RUN_PID}"
RC=$?
RUN_PID=
set -e
echo command_return_code="${RC}" >>"${MANIFEST}"
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')" ]]; then
  echo gpu_processes_remain >&2
  (( RC == 0 )) && RC=88
fi
echo final_return_code="${RC}" >>"${MANIFEST}"
exit "${RC}"

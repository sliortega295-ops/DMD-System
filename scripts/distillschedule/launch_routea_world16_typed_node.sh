#!/usr/bin/env bash
# Route-A Full5 world16 worker for one host. It remains a worker, not a cluster coordinator.
set -Eeuo pipefail

NODE_RANK="${1:?node rank 0 or 1}"
RUN_ID="${2:?run id}"
EXPECTED_HEAD="${3:?expected DMD-System HEAD}"
EXPECTED_PLAN_SHA="${4:?expected top-level plan sha256}"
MASTER_ADDR="${5:?master address}"
MASTER_PORT="${6:?master port}"
[[ "${NODE_RANK}" == 0 || "${NODE_RANK}" == 1 ]] || { echo invalid_node_rank >&2; exit 2; }
[[ "${MASTER_PORT}" =~ ^[0-9]+$ ]] || { echo invalid_master_port >&2; exit 2; }
PROBE_PORT="${DISTILLSCHEDULE_PROBE_PORT:-$((MASTER_PORT + 1))}"
(( PROBE_PORT > 0 && PROBE_PORT < 65536 && PROBE_PORT != MASTER_PORT )) || {
  echo invalid_probe_port >&2; exit 2;
}

ROOT="${DMD_SYSTEM_ROOT:-/share/project/shuchenweng/yongyan_liu/DMD-System}"
HELIOS_ROOT="${HELIOS_RUNTIME_ROOT:-/share/project/shuchenweng/yongyan_liu/DistillSchedule}"
PYENV="${DISTILLSCHEDULE_PYENV:-/tmp/distillschedule-helios-official-torch210}"
PYTHON="${DISTILLSCHEDULE_PYTHON:-${PYENV}/bin/python}"
ACCELERATE="${DISTILLSCHEDULE_ACCELERATE:-${PYENV}/bin/accelerate}"
DEEPSPEED="${DISTILLSCHEDULE_DEEPSPEED:-${PYENV}/bin/deepspeed}"
PLAN="${DISTILLSCHEDULE_PLAN:-${ROOT}/configs/distillschedule/plans/helios_routea_full5_world16_typed.json}"
DS_CONFIG="${DISTILLSCHEDULE_DEEPSPEED_CONFIG:-${ROOT}/scripts/accelerate_configs/zero1_cpu_optimizer_offload.json}"
ACCEL_TEMPLATE="${DISTILLSCHEDULE_ACCELERATE_TEMPLATE:-${ROOT}/scripts/accelerate_configs/multi_node_zero1_cpu_optimizer_offload.yaml}"
OUT="${DISTILLSCHEDULE_OUTPUT_DIR:-${ROOT}/artifacts/runs/${RUN_ID}}"
NODE_LOG="${OUT}/launcher/node_${NODE_RANK}"
CACHE_ROOT="${DISTILLSCHEDULE_CACHE_ROOT:-/tmp/distillschedule_typed_routea/${RUN_ID}}"
LOCK_FILE="${DISTILLSCHEDULE_NODE_LOCK:-/tmp/distillschedule_gpu_node.lock}"
EXPECTED_HELIOS_HEAD="${DISTILLSCHEDULE_EXPECTED_HELIOS_HEAD:?pin the validated Helios git HEAD}"
EXPECTED_HELIOS_ENTRYPOINT_SHA="${DISTILLSCHEDULE_EXPECTED_HELIOS_ENTRYPOINT_SHA256:?pin train_helios.py sha256}"

for executable in "${PYTHON}" "${ACCELERATE}" "${DEEPSPEED}"; do
  [[ -x "${executable}" ]] || { echo "runtime_executable_missing=${executable}" >&2; exit 3; }
done
for file in "${PLAN}" "${DS_CONFIG}" "${ACCEL_TEMPLATE}" "${HELIOS_ROOT}/train_helios.py"; do
  [[ -f "${file}" ]] || { echo "required_file_missing=${file}" >&2; exit 3; }
done
for variable in \
  HELIOS_DISTILLED_ROOT HELIOS_BASE_ROOT HELIOS_DATA_ROOT \
  DISTILLSCHEDULE_TRACE_BUNDLE DISTILLSCHEDULE_CHECKPOINT \
  DISTILLSCHEDULE_HF_HOME DISTILLSCHEDULE_FLASH_ATTN_KERNEL \
  DISTILLSCHEDULE_PRECAST_GENERATOR_PATH DISTILLSCHEDULE_PRECAST_SCORE_PATH; do
  [[ -n "${!variable:-}" ]] || { echo "required_environment_missing=${variable}" >&2; exit 3; }
done
for asset in \
  "${HELIOS_DISTILLED_ROOT}" "${HELIOS_BASE_ROOT}" "${HELIOS_DATA_ROOT}" \
  "${DISTILLSCHEDULE_TRACE_BUNDLE}" "${DISTILLSCHEDULE_CHECKPOINT}" \
  "${DISTILLSCHEDULE_HF_HOME}" "${DISTILLSCHEDULE_FLASH_ATTN_KERNEL}" \
  "${DISTILLSCHEDULE_PRECAST_GENERATOR_PATH}" "${DISTILLSCHEDULE_PRECAST_SCORE_PATH}"; do
  [[ -e "${asset}" ]] || { echo "asset_missing=${asset}" >&2; exit 5; }
done

[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_HEAD}" ]] || { echo dmd_system_head_drift >&2; exit 4; }
[[ -z "$(git -C "${ROOT}" status --short)" ]] || { echo dmd_system_repository_not_clean >&2; exit 4; }
[[ "$(sha256sum "${PLAN}" | awk '{print $1}')" == "${EXPECTED_PLAN_SHA}" ]] || { echo plan_file_drift >&2; exit 4; }
[[ "$(git -C "${HELIOS_ROOT}" rev-parse HEAD)" == "${EXPECTED_HELIOS_HEAD}" ]] || { echo helios_head_drift >&2; exit 4; }
[[ -z "$(git -C "${HELIOS_ROOT}" status --short)" ]] || { echo helios_repository_not_clean >&2; exit 4; }
[[ "$(sha256sum "${HELIOS_ROOT}/train_helios.py" | awk '{print $1}')" == "${EXPECTED_HELIOS_ENTRYPOINT_SHA}" ]] || {
  echo helios_entrypoint_drift >&2; exit 4;
}

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

ACCEL_CONFIG="${CACHE_ROOT}/accelerate.node${NODE_RANK}.yaml"
"${PYTHON}" "${ROOT}/scripts/distillschedule/render_accelerate_config.py" \
  --template "${ACCEL_TEMPLATE}" \
  --deepspeed-config "${DS_CONFIG}" \
  --output "${ACCEL_CONFIG}" \
  --num-machines 2 \
  --num-processes 16 \
  --machine-rank "${NODE_RANK}" \
  >"${NODE_LOG}/launcher_contract.json"
export DISTILLSCHEDULE_ACCELERATE_CONFIG="${ACCEL_CONFIG}"

"${PYTHON}" "${ROOT}/scripts/distillctl.py" validate "${PLAN}" \
  --repo-root "${ROOT}" --json >"${NODE_LOG}/compiled_preflight.json"
RESOLVED_PLAN_DIGEST="$("${PYTHON}" - "${NODE_LOG}/compiled_preflight.json" <<'PY'
import json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(payload["plan_digest"])
PY
)"

MANIFEST="${NODE_LOG}/launch_manifest.txt"
MONITOR="${NODE_LOG}/resource.csv"
read_cgroup() { [[ -r "$1" ]] && cat "$1" || echo unavailable; }
{
  echo schema_version=distillschedule.typed_routea_world16.node_launch.v2
  echo run_id="${RUN_ID}"
  echo node_rank="${NODE_RANK}"
  echo host="$(hostname)"
  echo started_at="$(date -Is)"
  echo dmd_system_head="${EXPECTED_HEAD}"
  echo dmd_system_plan_sha256="${EXPECTED_PLAN_SHA}"
  echo resolved_plan_digest="${RESOLVED_PLAN_DIGEST}"
  echo helios_head="${EXPECTED_HELIOS_HEAD}"
  echo helios_entrypoint_sha256="${EXPECTED_HELIOS_ENTRYPOINT_SHA}"
  echo python_path="$(readlink -f "${PYTHON}")"
  echo python_sha256="$(sha256sum "${PYTHON}" | awk '{print $1}')"
  echo accelerate_path="$(readlink -f "${ACCELERATE}")"
  echo accelerate_sha256="$(sha256sum "${ACCELERATE}" | awk '{print $1}')"
  echo deepspeed_path="$(readlink -f "${DEEPSPEED}")"
  echo deepspeed_sha256="$(sha256sum "${DEEPSPEED}" | awk '{print $1}')"
  echo accelerate_config_sha256="$(sha256sum "${ACCEL_CONFIG}" | awk '{print $1}')"
  echo deepspeed_config_sha256="$(sha256sum "${DS_CONFIG}" | awk '{print $1}')"
  echo master_addr="${MASTER_ADDR}"
  echo master_port="${MASTER_PORT}"
  echo probe_port="${PROBE_PORT}"
  echo cgroup_memory_current_before="$(read_cgroup /sys/fs/cgroup/memory.current)"
  echo cgroup_memory_max="$(read_cgroup /sys/fs/cgroup/memory.max)"
  echo memory_events_before="$(read_cgroup /sys/fs/cgroup/memory.events | tr '\n' ';')"
  "${PYTHON}" --version 2>&1 | sed 's/^/python_version=/'
  { "${ACCELERATE}" --version 2>&1 || true; } | head -n 1 | sed 's/^/accelerate_version=/'
  { "${DEEPSPEED}" --version 2>&1 || true; } | head -n 1 | sed 's/^/deepspeed_version=/'
  nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv,noheader
} >"${MANIFEST}"

MONITOR_PID= RUN_PID=
cleanup() {
  if [[ -n "${RUN_PID:-}" ]] && kill -0 "${RUN_PID}" 2>/dev/null; then
    kill -TERM -- "-${RUN_PID}" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-${RUN_PID}" 2>/dev/null || true
    wait "${RUN_PID}" 2>/dev/null || true
  fi
  if [[ -n "${MONITOR_PID:-}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
  {
    echo finished_at="$(date -Is)"
    echo cgroup_memory_current_after="$(read_cgroup /sys/fs/cgroup/memory.current)"
    echo memory_events_after="$(read_cgroup /sys/fs/cgroup/memory.events | tr '\n' ';')"
  } >>"${MANIFEST}" 2>/dev/null || true
  rm -rf "${CACHE_ROOT}" 2>/dev/null || true
}
trap cleanup EXIT TERM INT HUP

# This catches two independent world8 groups before any 14B model is constructed.
set +e
timeout --signal=TERM --kill-after=30s 600s \
  "${ACCELERATE}" launch \
  --num_machines 2 \
  --num_processes 16 \
  --machine_rank "${NODE_RANK}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${PROBE_PORT}" \
  --mixed_precision bf16 \
  --config_file "${ACCEL_CONFIG}" \
  "${ROOT}/scripts/distillschedule/probe_world16_topology.py" \
  --output "${OUT}/topology_probe.json" \
  --expected-nodes 2 \
  --expected-local-world-size 8 \
  --expected-world-size 16 \
  >"${NODE_LOG}/topology_probe.log" 2>&1
PROBE_RC=$?
set -e
echo topology_probe_return_code="${PROBE_RC}" >>"${MANIFEST}"
(( PROBE_RC == 0 )) || exit "${PROBE_RC}"
"${PYTHON}" - "${OUT}/topology_probe.json" <<'PY'
import json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
if payload.get("status") != "PASS":
    raise SystemExit("topology probe manifest is not PASS")
summary = payload.get("summary", {})
if summary.get("world_size") != 16 or summary.get("nodes") != 2:
    raise SystemExit("topology probe dimensions do not match world16")
PY

monitor() {
  echo 'unix_time,cgroup_bytes,host_used_bytes,gpu_used_mib,gpu_util_percent' >"${MONITOR}"
  while true; do
    now="$(date +%s.%N)"
    cgroup="$(read_cgroup /sys/fs/cgroup/memory.current)"
    host_used="$(awk '/^MemTotal:/{t=$2}/^MemAvailable:/{a=$2}END{print (t-a)*1024}' /proc/meminfo)"
    gpu_mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd: - || echo unavailable)"
    gpu_util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | paste -sd: - || echo unavailable)"
    echo "${now},${cgroup},${host_used},${gpu_mem},${gpu_util}" >>"${MONITOR}"
    sleep 1
  done
}
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

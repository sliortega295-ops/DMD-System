#!/usr/bin/env bash
# Per-node fail-closed launcher for RB16-F5-PIPE-2G6C.
set -Eeuo pipefail

NODE_RANK="${1:?node rank 0 or 1}"
RUN_ID="${2:?immutable run id}"
SCHEDULE="${3:?sequential or pipeline}"
MODE="${4:?correctness or timing}"
EXPECTED_HEAD="${5:?expected DMD-System HEAD}"
MASTER_ADDR="${6:?master address}"
MASTER_PORT="${7:?master port}"
EXPECTED_RUNTIME_HEAD="${8:?expected Runtime HEAD}"
[[ "${NODE_RANK}" == 0 || "${NODE_RANK}" == 1 ]] || exit 2
[[ "${SCHEDULE}" == sequential || "${SCHEDULE}" == pipeline ]] || exit 2
[[ "${MODE}" == correctness || "${MODE}" == timing ]] || exit 2
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || exit 2
[[ "${MASTER_PORT}" =~ ^[0-9]+$ ]] || exit 2

ROOT="${DMD_SYSTEM_ROOT:-/share/project/shuchenweng/yongyan_liu/DMD-System_rb16_f5_pipeline_20260809}"
RUNTIME="${HELIOS_RUNTIME_ROOT:-/share/project/shuchenweng/yongyan_liu/DistillSchedule_rb16_f5_pipeline_20260809}"
PYENV="${DISTILLSCHEDULE_PYENV:-/tmp/distillschedule-helios-official-torch210}"
PYTHON="${PYENV}/bin/python"
TORCHRUN="${PYENV}/bin/torchrun"
ASSETS="${DISTILLSCHEDULE_ASSETS:-/share/project/shuchenweng/DistillSchedule_assets}"
TRACE="${DISTILLSCHEDULE_TRACE_BUNDLE:-${ASSETS}/m0/traces/syn_feature_trace_rank16_v3_sync_steps20}"
TRACE_HASH="${DISTILLSCHEDULE_TRACE_BUNDLE_HASH:-9e282f4bc24eb589d259e5e5603cae02fa5cd0c6cdd0f06f7c5416461e4564d0}"
LOGICAL="${DISTILLSCHEDULE_LOGICAL_STEP10:-${ASSETS}/m1/checkpoints/logical_world16_step10_v3}"
LOGICAL_HASH="${DISTILLSCHEDULE_LOGICAL_STEP10_HASH:-68eed293970be008764ee6acadb2aac6ce54de2ce1f55a17fca2246f0d04e4aa}"
TRANSITION="${DISTILLSCHEDULE_CANDIDATE_B_TRANSITION:?Candidate-B transition package required}"
TRANSITION_HASH="${DISTILLSCHEDULE_CANDIDATE_B_TRANSITION_HASH:?Candidate-B transition package hash required}"
SOURCE_TRANSITION_SHA="${DISTILLSCHEDULE_SOURCE_TRANSITION_SHA:-444a6740debbe37d7014ebaa45da23398f162652f6179d09537e60e3c304dee0}"
GENERATOR="${HELIOS_DISTILLED_ROOT:-${ASSETS}/Helios-Distilled}"
PRECAST_GENERATOR="${DISTILLSCHEDULE_PRECAST_GENERATOR_PATH:-${ASSETS}/precast/helios-distilled-transformer-ode-mixed-bf16-v1}"
SCORE="${HELIOS_BASE_ROOT:-/home/shuchenweng/xty/Helios-main/weights}"
PRECAST_SCORE="${DISTILLSCHEDULE_PRECAST_SCORE_PATH:-${ASSETS}/precast/helios-score-transformer-mixed-bf16-v1}"
CORRECTNESS_EVALUATION="${DISTILLSCHEDULE_CANDIDATE_B_CORRECTNESS_EVALUATION:-}"
CORRECTNESS_SHA="${DISTILLSCHEDULE_CANDIDATE_B_CORRECTNESS_SHA:-}"
RUN_ROOT="${DISTILLSCHEDULE_RUN_ROOT:-${ROOT}/artifacts/runs/${RUN_ID}}"
NODE_LOG="${RUN_ROOT}/launcher/node_${NODE_RANK}"
EXECUTION="${RUN_ROOT}/execution"
PREFLIGHT="${RUN_ROOT}/preflight"
LOCK_FILE="/tmp/distillschedule_gpu_node.lock"
ACTUAL_PORT="$((MASTER_PORT + 1))"

for executable in "${PYTHON}" "${TORCHRUN}"; do
  [[ -x "${executable}" ]] || { echo missing_executable="${executable}" >&2; exit 3; }
done
for path in "${ROOT}" "${RUNTIME}" "${TRACE}" "${LOGICAL}" "${TRANSITION}" \
  "${GENERATOR}" "${PRECAST_GENERATOR}" "${SCORE}" "${PRECAST_SCORE}"; do
  [[ -e "${path}" ]] || { echo missing_path="${path}" >&2; exit 3; }
done
if [[ "${MODE}" == timing ]]; then
  [[ -f "${CORRECTNESS_EVALUATION}" ]] || { echo correctness_evaluation_missing >&2; exit 3; }
  [[ "${CORRECTNESS_SHA}" =~ ^[0-9a-f]{64}$ ]] || { echo correctness_hash_missing >&2; exit 3; }
  [[ "$(sha256sum "${CORRECTNESS_EVALUATION}" | awk '{print $1}')" == "${CORRECTNESS_SHA}" ]] || { echo correctness_hash_changed >&2; exit 3; }
  "${PYTHON}" - "${CORRECTNESS_EVALUATION}" <<'PY'
import json, pathlib, sys
value=json.loads(pathlib.Path(sys.argv[1]).read_text())
if value.get("status") != "PASS" or value.get("errors") != []:
    raise SystemExit("Candidate-B correctness evaluation is not PASS")
PY
fi
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_HEAD}" ]] || { echo head_drift >&2; exit 4; }
[[ "$(git -C "${RUNTIME}" rev-parse HEAD)" == "${EXPECTED_RUNTIME_HEAD}" ]] || { echo runtime_head_drift >&2; exit 4; }
[[ -z "$(git -C "${ROOT}" status --short)" ]] || { echo repository_not_clean >&2; exit 4; }
[[ -z "$(git -C "${RUNTIME}" status --short)" ]] || { echo runtime_repository_not_clean >&2; exit 4; }

mkdir -p "${NODE_LOG}" "${PREFLIGHT}"
[[ ! -e "${NODE_LOG}/launch_manifest.txt" ]] || { echo run_id_already_used >&2; exit 82; }
exec 9>"${LOCK_FILE}"
flock -n 9 || { echo gpu_lock_busy >&2; exit 73; }
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')" ]]; then
  echo gpu_processes_present >&2
  exit 81
fi

MANIFEST="${NODE_LOG}/launch_manifest.txt"
MONITOR="${NODE_LOG}/resource.csv"
{
  echo schema_version=distillschedule.routeb_candidate_b_node_launch.v1
  echo experiment_id=RB16-F5-PIPE-2G6C
  echo claim_scope=batch12_resource_throughput_pareto_not_candidate_a_speedup
  echo run_id="${RUN_ID}"
  echo schedule="${SCHEDULE}"
  echo mode="${MODE}"
  echo node_rank="${NODE_RANK}"
  echo host="$(hostname)"
  echo started_at="$(date -Is)"
  echo dmd_system_head="${EXPECTED_HEAD}"
  echo runtime_head="${EXPECTED_RUNTIME_HEAD}"
  echo trace_bundle="${TRACE}"
  echo trace_bundle_hash="${TRACE_HASH}"
  echo logical_package="${LOGICAL}"
  echo logical_manifest_hash="${LOGICAL_HASH}"
  echo transition_package="${TRANSITION}"
  echo transition_package_hash="${TRANSITION_HASH}"
  echo source_transition_manifest_sha256="${SOURCE_TRANSITION_SHA}"
  echo topology=per_node_2G6C_global_4G12C
  echo critic_batch=DP12_MB1_GAS1_global12
  echo construction_wave=4
  echo reserve_gate=disabled_by_user_actual_oom_only
  echo correctness_evaluation="${CORRECTNESS_EVALUATION:-Not Run}"
  echo correctness_evaluation_sha256="${CORRECTNESS_SHA:-Not Run}"
  echo cgroup_memory_current_before="$(cat /sys/fs/cgroup/memory.current)"
  echo cgroup_memory_peak_before="$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo unavailable)"
  echo cgroup_memory_max="$(cat /sys/fs/cgroup/memory.max)"
  echo memory_events_before="$(tr '\n' ';' </sys/fs/cgroup/memory.events)"
} >"${MANIFEST}"

export PATH="${PYENV}/bin:${PATH}"
export PYTHONPATH="${ROOT}/src:${ROOT}:${RUNTIME}"
export HF_HOME="${ASSETS}/hf_home_official_torch210"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 NCCL_ASYNC_ERROR_HANDLING=1
export DISTILLSCHEDULE_NODE_RANK="${NODE_RANK}"

"${PYTHON}" "${ROOT}/scripts/distillschedule/validate_routeb_candidate_b_inputs.py" \
  --trace-bundle "${TRACE}" --expected-trace-bundle-hash "${TRACE_HASH}" \
  --logical-package "${LOGICAL}" --expected-logical-manifest-hash "${LOGICAL_HASH}" \
  --transition-package "${TRANSITION}" --expected-transition-package-hash "${TRANSITION_HASH}" \
  --expected-source-transition-manifest-sha256 "${SOURCE_TRANSITION_SHA}" \
  --generator-root "${GENERATOR}" --precast-generator-root "${PRECAST_GENERATOR}" \
  --score-root "${SCORE}" --precast-score-root "${PRECAST_SCORE}" \
  --output "${PREFLIGHT}/node_${NODE_RANK}_inputs.json" \
  >"${NODE_LOG}/input_validation.stdout.json"

for _ in $(seq 1 300); do
  [[ -f "${PREFLIGHT}/node_0_inputs.json" && -f "${PREFLIGHT}/node_1_inputs.json" ]] && break
  sleep 1
done
[[ -f "${PREFLIGHT}/node_0_inputs.json" && -f "${PREFLIGHT}/node_1_inputs.json" ]] || { echo peer_input_validation_timeout >&2; exit 83; }
"${PYTHON}" - "${PREFLIGHT}/node_0_inputs.json" "${PREFLIGHT}/node_1_inputs.json" <<'PY'
import json, pathlib, sys
a=json.loads(pathlib.Path(sys.argv[1]).read_text())
b=json.loads(pathlib.Path(sys.argv[2]).read_text())
if a != b or a.get("status") != "PASS":
    raise SystemExit("Candidate-B node input contracts differ or failed")
PY

"${TORCHRUN}" --nnodes=2 --nproc-per-node=8 --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}" \
  "${ROOT}/scripts/distillschedule/probe_world16_topology.py" \
  --output "${PREFLIGHT}/topology.json" >"${NODE_LOG}/topology.log" 2>&1
[[ -f "${PREFLIGHT}/topology.json" ]] || { echo topology_output_missing >&2; exit 84; }

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

MONITOR_PID=""
RUN_PID=""
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
    echo cgroup_memory_current_after="$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo -1)"
    echo cgroup_memory_peak_after="$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo unavailable)"
    echo memory_events_after="$(tr '\n' ';' </sys/fs/cgroup/memory.events 2>/dev/null || true)"
  } >>"${MANIFEST}" 2>/dev/null || true
}
trap cleanup EXIT TERM INT HUP
monitor &
MONITOR_PID=$!

RUN_NONCE="$("${PYTHON}" - "${RUN_ID}" <<'PY'
import hashlib,sys
print(int(hashlib.sha256(sys.argv[1].encode()).hexdigest()[:15],16))
PY
)"
WORKER=(
  "${TORCHRUN}" --nnodes=2 --nproc-per-node=8 --node-rank="${NODE_RANK}"
  --master-addr="${MASTER_ADDR}" --master-port="${ACTUAL_PORT}"
  "${ROOT}/scripts/distillschedule/probe_routeb_full5_candidate_b.py"
  --schedule "${SCHEDULE}" --run-id "${RUN_ID}" --run-nonce "${RUN_NONCE}"
  --output-dir "${EXECUTION}" --trace-bundle "${TRACE}"
  --logical-package "${LOGICAL}" --expected-logical-manifest-hash "${LOGICAL_HASH}"
  --transition-package "${TRANSITION}" --expected-transition-package-hash "${TRANSITION_HASH}"
  --expected-source-transition-manifest-sha256 "${SOURCE_TRANSITION_SHA}"
  --generator-root "${GENERATOR}" --precast-generator-root "${PRECAST_GENERATOR}"
  --score-root "${SCORE}" --precast-score-root "${PRECAST_SCORE}"
  --construction-wave 4 --activation-min-bytes $((32 * 1024 * 1024))
)
if [[ "${MODE}" == correctness ]]; then
  WORKER+=(--checksum sha256 --export-state)
else
  WORKER+=(
    --timing-mode --checksum gpu
    --correctness-evaluation "${CORRECTNESS_EVALUATION}"
    --expected-correctness-sha256 "${CORRECTNESS_SHA}"
  )
fi

set +e
setsid timeout --signal=TERM --kill-after=180s 28800s \
  "${WORKER[@]}" >"${NODE_LOG}/console.log" 2>&1 &
RUN_PID=$!
wait "${RUN_PID}"
RC=$?
RUN_PID=""
set -e
echo command_return_code="${RC}" >>"${MANIFEST}"
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')" ]]; then
  echo gpu_processes_remain >&2
  (( RC == 0 )) && RC=88
fi
echo final_return_code="${RC}" >>"${MANIFEST}"
exit "${RC}"

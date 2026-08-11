#!/usr/bin/env bash
# Per-node fail-closed worker for RB16-F5-PIPE-MATCH16.
set -Eeuo pipefail

NODE_RANK="${1:?node rank 0 or 1}"
RUN_ID="${2:?immutable run id}"
SCHEDULE="${3:?sequential or pipeline}"
MODE="${4:?transition, transition-save, correctness, critic-correctness, critic-timing, or timing}"
EXPECTED_HEAD="${5:?expected DMD-System HEAD}"
MASTER_ADDR="${6:?master address}"
MASTER_PORT="${7:?master port}"
EXPECTED_HELIOS_HEAD="${8:?expected Helios/Runtime HEAD}"
[[ "${NODE_RANK}" == 0 || "${NODE_RANK}" == 1 ]] || exit 2
[[ "${SCHEDULE}" == sequential || "${SCHEDULE}" == pipeline ]] || exit 2
[[ "${MODE}" == transition || "${MODE}" == transition-save || "${MODE}" == correctness || "${MODE}" == critic-correctness || "${MODE}" == critic-timing || "${MODE}" == timing ]] || exit 2
if [[ "${MODE}" == transition-save && "${SCHEDULE}" != sequential ]]; then
  echo transition_save_requires_sequential >&2
  exit 2
fi
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || exit 2
[[ "${MASTER_PORT}" =~ ^[0-9]+$ ]] || exit 2

ROOT="${DMD_SYSTEM_ROOT:-/share/project/shuchenweng/yongyan_liu/DMD-System_rb16_f5_pipeline_20260809}"
HELIOS_ROOT="${HELIOS_RUNTIME_ROOT:-/share/project/shuchenweng/yongyan_liu/DistillSchedule_rb16_f5_pipeline_20260809}"
PYENV="${DISTILLSCHEDULE_PYENV:-/tmp/distillschedule-helios-official-torch210}"
PYTHON="${PYENV}/bin/python"
TORCHRUN="${PYENV}/bin/torchrun"
ASSETS="${DISTILLSCHEDULE_ASSETS:-/share/project/shuchenweng/DistillSchedule_assets}"
TRACE="${DISTILLSCHEDULE_TRACE_BUNDLE:-${ASSETS}/m0/traces/syn_feature_trace_rank16_v3_sync_steps20}"
TRACE_HASH="${DISTILLSCHEDULE_TRACE_BUNDLE_HASH:-9e282f4bc24eb589d259e5e5603cae02fa5cd0c6cdd0f06f7c5416461e4564d0}"
LOGICAL="${DISTILLSCHEDULE_LOGICAL_STEP10:-${ASSETS}/m1/checkpoints/logical_world16_step10_v3}"
LOGICAL_HASH="${DISTILLSCHEDULE_LOGICAL_STEP10_HASH:-68eed293970be008764ee6acadb2aac6ce54de2ce1f55a17fca2246f0d04e4aa}"
NATIVE="${DISTILLSCHEDULE_NATIVE_WORLD8_STEP10:-${ASSETS}/m1/checkpoints/import_world8_step10_v1}"
GENERATOR="${HELIOS_DISTILLED_ROOT:-${ASSETS}/Helios-Distilled}"
PRECAST_GENERATOR="${DISTILLSCHEDULE_PRECAST_GENERATOR_PATH:-${ASSETS}/precast/helios-distilled-transformer-ode-mixed-bf16-v1}"
SCORE="${HELIOS_BASE_ROOT:-/home/shuchenweng/xty/Helios-main/weights}"
PRECAST_SCORE="${DISTILLSCHEDULE_PRECAST_SCORE_PATH:-${ASSETS}/precast/helios-score-transformer-mixed-bf16-v1}"
NEGATIVE="${DISTILLSCHEDULE_NEGATIVE_PROMPT_ASSET:-${ASSETS}/m1/negative_prompt_official_v1}"
TRANSITION_CHECKPOINT="${DISTILLSCHEDULE_TRANSITION_CHECKPOINT:-}"
TRANSITION_MANIFEST_HASH="${DISTILLSCHEDULE_TRANSITION_MANIFEST_HASH:-}"
TIMING_CORRECTNESS_EVALUATION="${DISTILLSCHEDULE_TIMING_CORRECTNESS_EVALUATION:-}"
TIMING_CORRECTNESS_HASH="${DISTILLSCHEDULE_TIMING_CORRECTNESS_HASH:-}"
RUN_ROOT="${DISTILLSCHEDULE_RUN_ROOT:-${ROOT}/artifacts/runs/${RUN_ID}}"
NODE_LOG="${RUN_ROOT}/launcher/node_${NODE_RANK}"
EXECUTION="${RUN_ROOT}/execution"
PREFLIGHT="${RUN_ROOT}/preflight"
LOCK_FILE="/tmp/distillschedule_gpu_node.lock"
NODE_CACHE="/tmp/distillschedule_routeb/${RUN_ID}/node_${NODE_RANK}"
ACTUAL_PORT="$((MASTER_PORT + 1))"

for executable in "${PYTHON}" "${TORCHRUN}"; do
  [[ -x "${executable}" ]] || { echo missing_executable="${executable}" >&2; exit 3; }
done
for path in "${ROOT}" "${HELIOS_ROOT}" "${TRACE}" "${LOGICAL}" "${NATIVE}" \
  "${GENERATOR}" "${PRECAST_GENERATOR}" "${SCORE}" "${PRECAST_SCORE}" "${NEGATIVE}"; do
  [[ -e "${path}" ]] || { echo missing_path="${path}" >&2; exit 3; }
done
if [[ "${MODE}" == critic-correctness || "${MODE}" == critic-timing ]]; then
  [[ -d "${TRANSITION_CHECKPOINT}" ]] || { echo transition_checkpoint_missing >&2; exit 3; }
  [[ "${TRANSITION_MANIFEST_HASH}" =~ ^[0-9a-f]{64}$ ]] || { echo transition_manifest_hash_missing >&2; exit 3; }
  [[ -f "${TRANSITION_CHECKPOINT}/transition_manifest.json" ]] || { echo transition_manifest_missing >&2; exit 3; }
  [[ "$(sha256sum "${TRANSITION_CHECKPOINT}/transition_manifest.json" | awk '{print $1}')" == "${TRANSITION_MANIFEST_HASH}" ]] || {
    echo transition_manifest_hash_changed >&2
    exit 3
  }
fi
if [[ "${MODE}" == critic-timing ]]; then
  [[ -f "${TIMING_CORRECTNESS_EVALUATION}" ]] || { echo timing_correctness_evaluation_missing >&2; exit 3; }
  [[ "${TIMING_CORRECTNESS_HASH}" =~ ^[0-9a-f]{64}$ ]] || { echo timing_correctness_hash_missing >&2; exit 3; }
  [[ "$(sha256sum "${TIMING_CORRECTNESS_EVALUATION}" | awk '{print $1}')" == "${TIMING_CORRECTNESS_HASH}" ]] || {
    echo timing_correctness_hash_changed >&2
    exit 3
  }
  "${PYTHON}" - "${TIMING_CORRECTNESS_EVALUATION}" <<'PY'
import json, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text())
if value.get("status") != "PASS" or value.get("errors") != []:
    raise SystemExit("timing correctness evaluation is not PASS")
PY
fi
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_HEAD}" ]] || { echo head_drift >&2; exit 4; }
[[ "$(git -C "${HELIOS_ROOT}" rev-parse HEAD)" == "${EXPECTED_HELIOS_HEAD}" ]] || { echo helios_head_drift >&2; exit 4; }
[[ -z "$(git -C "${ROOT}" status --short)" ]] || { echo repository_not_clean >&2; exit 4; }
[[ -z "$(git -C "${HELIOS_ROOT}" status --short)" ]] || { echo helios_repository_not_clean >&2; exit 4; }

mkdir -p "${NODE_LOG}" "${PREFLIGHT}"
[[ ! -e "${NODE_LOG}/launch_manifest.txt" ]] || { echo run_id_already_used >&2; exit 82; }
[[ ! -e "${NODE_CACHE}" ]] || { echo node_cache_exists >&2; exit 82; }
mkdir -p "${NODE_CACHE}/mmap"

exec 9>"${LOCK_FILE}"
flock -n 9 || { echo gpu_lock_busy >&2; exit 73; }
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')" ]]; then
  echo gpu_processes_present >&2
  exit 81
fi

MANIFEST="${NODE_LOG}/launch_manifest.txt"
MONITOR="${NODE_LOG}/resource.csv"
{
  echo schema_version=distillschedule.routeb_candidate_a_node_launch.v1
  echo experiment_id=RB16-F5-PIPE-MATCH16
  echo run_id="${RUN_ID}"
  echo schedule="${SCHEDULE}"
  echo mode="${MODE}"
  echo node_rank="${NODE_RANK}"
  echo host="$(hostname)"
  echo started_at="$(date -Is)"
  echo dmd_system_head="${EXPECTED_HEAD}"
  echo helios_head="${EXPECTED_HELIOS_HEAD}"
  echo state_provenance_package="${LOGICAL}"
  echo state_provenance_manifest_hash="${LOGICAL_HASH}"
  echo execution_trace_bundle="${TRACE}"
  echo execution_trace_bundle_hash="${TRACE_HASH}"
  echo cgroup_memory_current_before="$(cat /sys/fs/cgroup/memory.current)"
  echo cgroup_memory_peak_before="$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo unavailable)"
  echo cgroup_memory_max="$(cat /sys/fs/cgroup/memory.max)"
  echo memory_events_before="$(tr '\n' ';' </sys/fs/cgroup/memory.events)"
  echo reserve_gate=disabled_by_user_actual_oom_only
  echo transition_checkpoint="${TRANSITION_CHECKPOINT:-Not Run}"
  echo transition_manifest_sha256="${TRANSITION_MANIFEST_HASH:-Not Run}"
  echo timing_correctness_evaluation="${TIMING_CORRECTNESS_EVALUATION:-Not Run}"
  echo timing_correctness_sha256="${TIMING_CORRECTNESS_HASH:-Not Run}"
} >"${MANIFEST}"

export PATH="${PYENV}/bin:${PATH}"
export PYTHONPATH="${ROOT}/src:${ROOT}:${HELIOS_ROOT}"
export HF_HOME="${ASSETS}/hf_home_official_torch210"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 NCCL_ASYNC_ERROR_HANDLING=1
export DISTILLSCHEDULE_NODE_RANK="${NODE_RANK}"

"${PYTHON}" "${ROOT}/scripts/distillschedule/validate_routeb_full5_inputs.py" \
  --trace-bundle "${TRACE}" \
  --expected-trace-bundle-hash "${TRACE_HASH}" \
  --logical-package "${LOGICAL}" \
  --expected-logical-manifest-hash "${LOGICAL_HASH}" \
  --native-world8-root "${NATIVE}" \
  --generator-root "${GENERATOR}" \
  --precast-generator-root "${PRECAST_GENERATOR}" \
  --score-root "${SCORE}" \
  --precast-score-root "${PRECAST_SCORE}" \
  --negative-prompt-asset "${NEGATIVE}" \
  --output "${PREFLIGHT}/node_${NODE_RANK}_inputs.json" \
  >"${NODE_LOG}/input_validation.stdout.json"

for _ in $(seq 1 300); do
  [[ -f "${PREFLIGHT}/node_0_inputs.json" && -f "${PREFLIGHT}/node_1_inputs.json" ]] && break
  sleep 1
done
[[ -f "${PREFLIGHT}/node_0_inputs.json" && -f "${PREFLIGHT}/node_1_inputs.json" ]] || {
  echo peer_input_validation_timeout >&2; exit 83;
}
"${PYTHON}" - "${PREFLIGHT}/node_0_inputs.json" "${PREFLIGHT}/node_1_inputs.json" <<'PY'
import json, pathlib, sys
a=json.loads(pathlib.Path(sys.argv[1]).read_text())
b=json.loads(pathlib.Path(sys.argv[2]).read_text())
if a != b or a.get("status") != "PASS":
    raise SystemExit("node input contracts differ or failed")
PY

# Prove that this is one 16-rank process group, not two independent worlds.
"${TORCHRUN}" \
  --nnodes=2 --nproc-per-node=8 --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}" \
  "${ROOT}/scripts/distillschedule/probe_world16_topology.py" \
  --output "${PREFLIGHT}/topology.json" \
  >"${NODE_LOG}/topology.log" 2>&1
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
  rm -rf "${NODE_CACHE}" 2>/dev/null || true
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
  "${TORCHRUN}"
  --nnodes=2 --nproc-per-node=8 --node-rank="${NODE_RANK}"
  --master-addr="${MASTER_ADDR}" --master-port="${ACTUAL_PORT}"
  "${ROOT}/scripts/distillschedule/probe_routeb_full5_candidate_a.py"
  --schedule "${SCHEDULE}"
  --run-id "${RUN_ID}"
  --run-nonce "${RUN_NONCE}"
  --output-dir "${EXECUTION}"
  --trace-bundle "${TRACE}"
  --logical-package "${LOGICAL}"
  --native-world8-root "${NATIVE}"
  --generator-root "${GENERATOR}"
  --precast-generator-root "${PRECAST_GENERATOR}"
  --score-root "${SCORE}"
  --precast-score-root "${PRECAST_SCORE}"
  --negative-prompt-asset "${NEGATIVE}"
  --mmap-cache-root "${NODE_CACHE}/mmap"
  --construction-wave 4
)
if [[ "${MODE}" == transition ]]; then
  WORKER+=(--stop-after-transition --checksum sha256)
elif [[ "${MODE}" == transition-save ]]; then
  WORKER+=(--stop-after-transition --save-transition-checkpoint --checksum sha256)
elif [[ "${MODE}" == correctness ]]; then
  WORKER+=(--save-checkpoint --checksum sha256)
elif [[ "${MODE}" == critic-correctness ]]; then
  WORKER+=(
    --resume-transition-checkpoint "${TRANSITION_CHECKPOINT}"
    --expected-transition-manifest-sha256 "${TRANSITION_MANIFEST_HASH}"
    --save-checkpoint
    --checksum sha256
  )
elif [[ "${MODE}" == critic-timing ]]; then
  WORKER+=(
    --resume-transition-checkpoint "${TRANSITION_CHECKPOINT}"
    --expected-transition-manifest-sha256 "${TRANSITION_MANIFEST_HASH}"
    --timing-mode
    --checksum gpu
    --timing-correctness-evaluation "${TIMING_CORRECTNESS_EVALUATION}"
    --expected-timing-correctness-sha256 "${TIMING_CORRECTNESS_HASH}"
  )
else
  WORKER+=(--timing-mode --checksum gpu)
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

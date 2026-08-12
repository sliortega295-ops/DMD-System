#!/usr/bin/env bash
# Fail-closed per-node launcher for the real GAN 8G8Score->4G12C Full5 worker.
set -Eeuo pipefail

NODE_RANK="${1:?node rank 0 or 1}"
RUN_ID="${2:?immutable run id}"
MODE="${3:?validate, correctness, or timing}"
EXPECTED_HEAD="${4:?expected DMD-System source HEAD}"
MASTER_ADDR="${5:?node0 routable address}"
MASTER_PORT="${6:?topology rendezvous port}"
EXPECTED_SOURCE_IDENTITY="${7:?expected staged source identity hash}"

[[ "${NODE_RANK}" == 0 || "${NODE_RANK}" == 1 ]] || exit 2
[[ "${MODE}" == validate || "${MODE}" == correctness || "${MODE}" == timing ]] || exit 2
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || exit 2
[[ "${MASTER_PORT}" =~ ^[0-9]+$ ]] || exit 2
[[ "${EXPECTED_HEAD}" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "${EXPECTED_SOURCE_IDENTITY}" =~ ^[0-9a-f]{64}$ ]] || exit 2

ROOT="${DMD_SYSTEM_ROOT:-/share/project/shuchenweng/yongyan_liu/DMD-System_gan_rolephase_world16_exec_20260812}"
HELIOS="${HELIOS_RUNTIME_ROOT:-/share/project/shuchenweng/yongyan_liu/DistillSchedule_gan_rolephase_exec_20260812}"
PYENV="${DISTILLSCHEDULE_PYENV:-/tmp/distillschedule-helios-official-torch210}"
PYTHON="${PYENV}/bin/python"
TORCHRUN="${PYENV}/bin/torchrun"
ASSETS="${DISTILLSCHEDULE_ASSETS:-/share/project/shuchenweng/DistillSchedule_assets}"
CONFIG="${DISTILLSCHEDULE_GAN_CONFIG:-${ROOT}/configs/distillschedule/templates/helios_stage3_gan_384x640_systems.yaml}"
LOGICAL="${DISTILLSCHEDULE_GAN_LOGICAL_PACKAGE:-${ASSETS}/gan/systems_only_gan_pipeline_world8_v3/logical_package}"
LOGICAL_HASH="${DISTILLSCHEDULE_GAN_LOGICAL_HASH:-c66ff180fa4d00c78ba05ffd1826b7e358e0ab38d2f3f68f5f92dad5fe42dd44}"
TRACE="${DISTILLSCHEDULE_GAN_TRACE_BUNDLE:-${ASSETS}/gan/systems_only_gan_trace_rank16_v1}"
TRACE_HASH="${DISTILLSCHEDULE_GAN_TRACE_HASH:-d31d4d473a3574b5b39e50ad2b5e72e17aee7baf707130ea0513092553f7cd15}"
GENERATOR="${HELIOS_DISTILLED_ROOT:-${ASSETS}/Helios-Distilled}"
PRECAST_GENERATOR="${DISTILLSCHEDULE_PRECAST_GENERATOR_PATH:-${ASSETS}/precast/helios-distilled-transformer-ode-mixed-bf16-v1}"
SCORE="${HELIOS_BASE_ROOT:-/home/shuchenweng/xty/Helios-main/weights}"
PRECAST_SCORE="${DISTILLSCHEDULE_PRECAST_SCORE_PATH:-${ASSETS}/precast/helios-score-transformer-mixed-bf16-gan-systems-v2}"
NEGATIVE="${DISTILLSCHEDULE_NEGATIVE_PROMPT_ASSET:-${ASSETS}/m1/negative_prompt_official_v1}"
SOURCE_IDENTITY="${DISTILLSCHEDULE_SOURCE_IDENTITY:-${ROOT}/artifacts/manifests/gan_rolephase_source_identity.json}"
RUN_ROOT="${DISTILLSCHEDULE_RUN_ROOT:-${ROOT}/artifacts/runs/${RUN_ID}}"
PREFLIGHT="${RUN_ROOT}/preflight"
NODE_LOG="${RUN_ROOT}/launcher/node_${NODE_RANK}"
EXECUTION="${RUN_ROOT}/execution"
NODE_CACHE="/tmp/distillschedule_gan_rolephase/${RUN_ID}/node_${NODE_RANK}"
LOCK_FILE="${DISTILLSCHEDULE_NODE_LOCK:-/tmp/distillschedule_gpu_node.lock}"
PHASE_B_PORT="$((MASTER_PORT + 1))"
PHASE_A_REVERSE_PORT="$((MASTER_PORT + 10 + NODE_RANK))"
PHASE_B_STREAMED_BLOCKS="${DISTILLSCHEDULE_PHASE_B_STREAMED_BLOCKS:-0}"
CORRECTNESS_SUMMARY="${DISTILLSCHEDULE_CORRECTNESS_SUMMARY:-}"
CORRECTNESS_SHA256="${DISTILLSCHEDULE_CORRECTNESS_SHA256:-}"

[[ "${PHASE_B_STREAMED_BLOCKS}" == 0 || "${PHASE_B_STREAMED_BLOCKS}" == 4 ]] || exit 2
for executable in "${PYTHON}" "${TORCHRUN}"; do
  [[ -x "${executable}" ]] || { echo missing_executable="${executable}" >&2; exit 3; }
done
for path in "${ROOT}" "${HELIOS}" "${CONFIG}" "${LOGICAL}" "${TRACE}" \
  "${GENERATOR}" "${PRECAST_GENERATOR}" "${SCORE}" "${PRECAST_SCORE}" \
  "${NEGATIVE}" "${SOURCE_IDENTITY}"; do
  [[ -e "${path}" ]] || { echo missing_path="${path}" >&2; exit 3; }
done

if [[ "${MODE}" == timing ]]; then
  [[ -f "${CORRECTNESS_SUMMARY}" ]] || { echo correctness_summary_missing >&2; exit 3; }
  [[ "${CORRECTNESS_SHA256}" =~ ^[0-9a-f]{64}$ ]] || { echo correctness_summary_hash_missing >&2; exit 3; }
  [[ "$(sha256sum "${CORRECTNESS_SUMMARY}" | awk '{print $1}')" == "${CORRECTNESS_SHA256}" ]] || { echo correctness_summary_hash_changed >&2; exit 3; }
  "${PYTHON}" - "${CORRECTNESS_SUMMARY}" "${PHASE_B_STREAMED_BLOCKS}" <<'PY'
import json, pathlib, sys
value=json.loads(pathlib.Path(sys.argv[1]).read_text())
if value.get("status") != "PASS" or not value.get("claimable_correctness"):
    raise SystemExit("role-phase correctness summary is not PASS")
if int(value.get("phase_b_streamed_blocks", -1)) != int(sys.argv[2]):
    raise SystemExit("timing memory policy differs from correctness")
PY
fi

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

export PATH="${PYENV}/bin:${PATH}"
export PYTHONPATH="${ROOT}/src:${ROOT}:${HELIOS}"
export HF_HOME="${ASSETS}/hf_home_official_torch210"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 NCCL_ASYNC_ERROR_HANDLING=1
export DISTILLSCHEDULE_NODE_RANK="${NODE_RANK}"

"${PYTHON}" "${ROOT}/scripts/distillschedule/build_gan_rolephase_source_identity.py" \
  --dmd-root "${ROOT}" --helios-root "${HELIOS}" --git-head "${EXPECTED_HEAD}" \
  --verify "${SOURCE_IDENTITY}" >"${NODE_LOG}/source_identity.stdout.json"
"${PYTHON}" - "${SOURCE_IDENTITY}" "${EXPECTED_SOURCE_IDENTITY}" <<'PY'
import json,pathlib,sys
value=json.loads(pathlib.Path(sys.argv[1]).read_text())
if value.get("status") != "PASS" or value.get("identity_hash") != sys.argv[2]:
    raise SystemExit("staged source identity/hash mismatch")
PY

"${PYTHON}" "${ROOT}/scripts/distillschedule/validate_gan_rolephase_world16_inputs.py" \
  --config "${CONFIG}" \
  --logical-package "${LOGICAL}" --expected-logical-manifest-hash "${LOGICAL_HASH}" \
  --trace-bundle "${TRACE}" --expected-trace-bundle-hash "${TRACE_HASH}" \
  --generator-root "${GENERATOR}" --precast-generator-root "${PRECAST_GENERATOR}" \
  --score-root "${SCORE}" --precast-score-root "${PRECAST_SCORE}" \
  --negative-prompt "${NEGATIVE}" --source-identity "${SOURCE_IDENTITY}" \
  --output "${PREFLIGHT}/node_${NODE_RANK}_inputs.json" \
  >"${NODE_LOG}/input_validation.stdout.json"

for _ in $(seq 1 300); do
  [[ -f "${PREFLIGHT}/node_0_inputs.json" && -f "${PREFLIGHT}/node_1_inputs.json" ]] && break
  sleep 1
done
[[ -f "${PREFLIGHT}/node_0_inputs.json" && -f "${PREFLIGHT}/node_1_inputs.json" ]] || { echo peer_input_validation_timeout >&2; exit 83; }
"${PYTHON}" - "${PREFLIGHT}/node_0_inputs.json" "${PREFLIGHT}/node_1_inputs.json" <<'PY'
import json,pathlib,sys
a=json.loads(pathlib.Path(sys.argv[1]).read_text())
b=json.loads(pathlib.Path(sys.argv[2]).read_text())
if a != b or a.get("status") != "PASS":
    raise SystemExit("node input contracts differ or failed")
PY

MANIFEST="${NODE_LOG}/launch_manifest.txt"
MONITOR="${NODE_LOG}/resource.csv"
{
  echo schema_version=distillschedule.gan_rolephase_world16_node_launch.v1
  echo experiment_id=GAN-RP16-A8G8S-B16-TO-B4G12C-B12
  echo claim_scope=systems_only_phase_local_B16_B12_pareto_not_matched_batch_speedup
  echo run_id="${RUN_ID}"
  echo mode="${MODE}"
  echo node_rank="${NODE_RANK}"
  echo host="$(hostname)"
  echo started_at="$(date -Is)"
  echo dmd_system_head="${EXPECTED_HEAD}"
  echo source_identity_hash="${EXPECTED_SOURCE_IDENTITY}"
  echo logical_manifest_hash="${LOGICAL_HASH}"
  echo trace_bundle_hash="${TRACE_HASH}"
  echo phase_a=DP8_MB1_GAS2_B16_NODE_LOCAL_DEFAULT_WORLD8_ZERO3
  echo phase_b=DP12_MB1_GAS1_B12_GLOBAL_WORLD16_ZERO1_CPU_OPTIMIZER
  echo phase_b_streamed_blocks="${PHASE_B_STREAMED_BLOCKS}"
  echo construction_policy=concurrent_all_local_ranks_no_construction_wave
  echo reserve_gate=disabled_actual_cuda_or_cgroup_oom_only
  echo correctness_summary="${CORRECTNESS_SUMMARY:-Not Run}"
  echo correctness_summary_sha256="${CORRECTNESS_SHA256:-Not Run}"
  echo cgroup_memory_current_before="$(cat /sys/fs/cgroup/memory.current)"
  echo cgroup_memory_peak_before="$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo unavailable)"
  echo cgroup_memory_max="$(cat /sys/fs/cgroup/memory.max)"
  echo memory_events_before="$(tr '\n' ';' </sys/fs/cgroup/memory.events)"
} >"${MANIFEST}"

"${TORCHRUN}" --nnodes=2 --nproc-per-node=8 --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}" \
  "${ROOT}/scripts/distillschedule/probe_world16_topology.py" \
  --output "${PREFLIGHT}/topology.json" >"${NODE_LOG}/topology.log" 2>&1
[[ -f "${PREFLIGHT}/topology.json" ]] || { echo topology_output_missing >&2; exit 84; }

monitor() {
  echo 'unix_time,cgroup_bytes,host_used_bytes,gpu_used_mib,gpu_util_percent' >"${MONITOR}"
  while true; do
    host_used="$(awk '/^MemTotal:/{t=$2}/^MemAvailable:/{a=$2}END{print (t-a)*1024}' /proc/meminfo)"
    echo "$(date +%s.%N),$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo -1),${host_used},$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd: -),$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | paste -sd: -)" >>"${MONITOR}"
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
  "${PYTHON}" - "${NODE_CACHE}" <<'PY' 2>/dev/null || true
import pathlib,shutil,sys
path=pathlib.Path(sys.argv[1])
if path.is_dir() and str(path).startswith('/tmp/distillschedule_gan_rolephase/'):
    shutil.rmtree(path)
PY
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
  "${TORCHRUN}" --standalone --nproc-per-node=8
  "${ROOT}/scripts/distillschedule/probe_gan_rolephase_world16.py"
  --node-rank "${NODE_RANK}"
  --phase-b-master-addr "${MASTER_ADDR}"
  --phase-b-master-port "${PHASE_B_PORT}"
  --phase-a-reverse-master-port "${PHASE_A_REVERSE_PORT}"
  --run-id "${RUN_ID}" --run-nonce "${RUN_NONCE}"
  --output-dir "${EXECUTION}" --config "${CONFIG}"
  --logical-package "${LOGICAL}" --expected-logical-manifest-hash "${LOGICAL_HASH}"
  --trace-bundle "${TRACE}"
  --generator-root "${GENERATOR}" --precast-generator-root "${PRECAST_GENERATOR}"
  --score-root "${SCORE}" --precast-score-root "${PRECAST_SCORE}"
  --negative-prompt "${NEGATIVE}" --mmap-cache-root "${NODE_CACHE}/mmap"
  --phase-b-streamed-blocks "${PHASE_B_STREAMED_BLOCKS}"
)
[[ "${MODE}" == validate ]] && WORKER+=(--validate-only)
[[ "${MODE}" == timing ]] && WORKER+=(--timing-mode)

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

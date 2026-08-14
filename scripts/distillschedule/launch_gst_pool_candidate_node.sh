#!/usr/bin/env bash
# One-node worker for the isolated N2 G/S/T target-graph capacity candidate.
set -Eeuo pipefail

ROLE="${1:?generator or score}"
RUN_ID="${2:?run id}"
EXPECTED_HEAD="${3:?expected DMD-System HEAD}"
CONTRACT_DIGEST="${4:?evidence contract digest}"
CANDIDATE_MANIFEST="${5:?shared candidate manifest}"
RUN_ROOT="${6:?shared run root}"
[[ "${ROLE}" == generator || "${ROLE}" == score ]] || { echo invalid_role >&2; exit 2; }
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo invalid_run_id >&2; exit 2; }

ROOT="${DMD_SYSTEM_ROOT:-/share/project/shuchenweng/yongyan_liu/DMD-System_pr3_0cd40e4}"
HELIOS_ROOT="${HELIOS_RUNTIME_ROOT:-/share/project/shuchenweng/yongyan_liu/DistillSchedule_routea_compat_20260807}"
PYENV="${DISTILLSCHEDULE_PYENV:-/tmp/distillschedule-helios-official-torch210}"
PYTHON="${PYENV}/bin/python"
TORCHRUN="${PYENV}/bin/torchrun"
TRACE="${DISTILLSCHEDULE_TRACE_BUNDLE:-/share/project/shuchenweng/DistillSchedule_assets/m0/traces/syn_feature_trace_rank16_v3_sync_steps20}"
TRACE_HASH="${DISTILLSCHEDULE_TRACE_BUNDLE_HASH:-9e282f4bc24eb589d259e5e5603cae02fa5cd0c6cdd0f06f7c5416461e4564d0}"
GENERATOR_ROOT="${HELIOS_DISTILLED_ROOT:-/share/project/shuchenweng/DistillSchedule_assets/Helios-Distilled}"
SCORE_ROOT="${HELIOS_BASE_ROOT:-/home/shuchenweng/xty/Helios-main/weights}"
PRECAST_GENERATOR="${DISTILLSCHEDULE_PRECAST_GENERATOR_PATH:-/share/project/shuchenweng/DistillSchedule_assets/precast/helios-distilled-transformer-ode-mixed-bf16-v1}"
PRECAST_SCORE="${DISTILLSCHEDULE_PRECAST_SCORE_PATH:-/share/project/shuchenweng/DistillSchedule_assets/precast/helios-score-transformer-mixed-bf16-v1}"
TARGET_PACKAGE="${DISTILLSCHEDULE_GST_TARGET_PACKAGE:-/share/project/shuchenweng/DistillSchedule_assets/m1/checkpoints/target_world8_C11_package_v1}"
TARGET_NATIVE="${DISTILLSCHEDULE_GST_TARGET_NATIVE:-/share/project/shuchenweng/DistillSchedule_assets/m1/checkpoints/target_world8_C11_native_v1}"
NEGATIVE_ASSET="${DISTILLSCHEDULE_NEGATIVE_PROMPT_ASSET:-/share/project/shuchenweng/DistillSchedule_assets/m1/negative_prompt_official_v1}"
HF_HOME="${DISTILLSCHEDULE_HF_HOME:-/share/project/shuchenweng/DistillSchedule_assets/hf_home_official_torch210}"
LOCK_FILE="/tmp/distillschedule_gpu_node.lock"
PID_FILE="${RUN_ROOT}/launcher/${ROLE}.pid"
LOG="${RUN_ROOT}/launcher/${ROLE}.log"
MANIFEST="${RUN_ROOT}/launcher/${ROLE}_launch_manifest.txt"
EXCHANGE="${RUN_ROOT}/exchange"
RENDEZVOUS="${RUN_ROOT}/rendezvous"
CACHE="${DISTILLSCHEDULE_GST_MMAP_CACHE:-/share/project/shuchenweng/DistillSchedule_assets/m1/gst_generator_frozen_mmap_v1}"

for executable in "${PYTHON}" "${TORCHRUN}"; do [[ -x "${executable}" ]] || { echo missing_executable="${executable}" >&2; exit 3; }; done
for path in "${ROOT}" "${HELIOS_ROOT}" "${TRACE}" "${GENERATOR_ROOT}" "${SCORE_ROOT}" "${PRECAST_GENERATOR}" "${PRECAST_SCORE}" "${TARGET_PACKAGE}" "${TARGET_NATIVE}"; do
  [[ -e "${path}" ]] || { echo missing_path="${path}" >&2; exit 3; }
done
[[ "${ROLE}" != score || -e "${NEGATIVE_ASSET}" ]] || { echo missing_negative_prompt_asset >&2; exit 3; }
[[ -f "${CANDIDATE_MANIFEST}" ]] || { echo missing_candidate_manifest >&2; exit 3; }
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_HEAD}" ]] || { echo head_drift >&2; exit 4; }
[[ -z "$(git -C "${ROOT}" status --porcelain --untracked-files=no)" ]] || { echo tracked_repository_dirty >&2; exit 4; }

mkdir -p "${RUN_ROOT}/launcher" "${RUN_ROOT}/rank_results" "${EXCHANGE}" "${RENDEZVOUS}" "${CACHE}"
[[ ! -e "${PID_FILE}" ]] || { echo stale_pid_file >&2; exit 82; }
exec 9>"${LOCK_FILE}"
flock -n 9 || { echo gpu_lock_busy >&2; exit 73; }
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')" ]]; then
  echo gpu_processes_present >&2
  exit 81
fi

{
  echo schema_version=distillschedule.gst_pool_node_launch.v1
  echo run_id="${RUN_ID}"
  echo role="${ROLE}"
  echo host="$(hostname)"
  echo started_at="$(date -Is)"
  echo dmd_system_head="${EXPECTED_HEAD}"
  echo contract_digest="${CONTRACT_DIGEST}"
  echo candidate_sha256="$(sha256sum "${CANDIDATE_MANIFEST}" | awk '{print $1}')"
  echo cgroup_memory_current_before="$(cat /sys/fs/cgroup/memory.current)"
  echo cgroup_memory_peak_before="$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo unavailable)"
  echo cgroup_memory_max="$(cat /sys/fs/cgroup/memory.max)"
  echo memory_events_before="$(tr '\n' ';' </sys/fs/cgroup/memory.events)"
} >"${MANIFEST}"

export PATH="${PYENV}/bin:${PATH}"
export PYTHONPATH="${ROOT}/src:${ROOT}:${HELIOS_ROOT}"
export HF_HOME TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export NCCL_ASYNC_ERROR_HANDLING=1

VALIDATION=(
  "${PYTHON}" "${ROOT}/scripts/distillschedule/validate_gst_pool_inputs.py"
  --role "${ROLE}"
  --run-id "${RUN_ID}"
  --candidate-manifest "${CANDIDATE_MANIFEST}"
  --contract-digest "${CONTRACT_DIGEST}"
  --trace-bundle "${TRACE}"
  --expected-trace-bundle-hash "${TRACE_HASH}"
  --target-package "${TARGET_PACKAGE}"
  --output "${RUN_ROOT}/launcher/${ROLE}_input_validation.json"
)
if [[ "${ROLE}" == generator ]]; then
  VALIDATION+=(
    --model-root "${GENERATOR_ROOT}"
    --model-subfolder transformer_ode
    --precast-root "${PRECAST_GENERATOR}"
    --target-native-root "${TARGET_NATIVE}"
  )
else
  VALIDATION+=(
    --model-root "${SCORE_ROOT}"
    --model-subfolder transformer
    --precast-root "${PRECAST_SCORE}"
    --negative-prompt-asset "${NEGATIVE_ASSET}"
  )
fi
"${VALIDATION[@]}" >"${RUN_ROOT}/launcher/${ROLE}_input_validation.stdout.json"

if [[ "${ROLE}" == generator ]]; then
  WORKER=(
    "${TORCHRUN}" --standalone --nproc_per_node=8
    "${ROOT}/scripts/distillschedule/probe_gst_pool_generator.py"
    --run-id "${RUN_ID}"
    --candidate-manifest "${CANDIDATE_MANIFEST}"
    --contract-digest "${CONTRACT_DIGEST}"
    --output-dir "${RUN_ROOT}/rank_results"
    --exchange-dir "${EXCHANGE}"
    --rendezvous-dir "${RENDEZVOUS}"
    --trace-bundle "${TRACE}"
    --generator-root "${GENERATOR_ROOT}"
    --precast-generator-root "${PRECAST_GENERATOR}"
    --target-package "${TARGET_PACKAGE}"
    --target-native-root "${TARGET_NATIVE}"
    --mmap-cache-root "${CACHE}"
    --construction-wave 4
  )
else
  WORKER=(
    "${TORCHRUN}" --standalone --nproc_per_node=8
    "${ROOT}/scripts/distillschedule/probe_gst_pool_score.py"
    --run-id "${RUN_ID}"
    --candidate-manifest "${CANDIDATE_MANIFEST}"
    --contract-digest "${CONTRACT_DIGEST}"
    --output-dir "${RUN_ROOT}/rank_results"
    --exchange-dir "${EXCHANGE}"
    --rendezvous-dir "${RENDEZVOUS}"
    --trace-bundle "${TRACE}"
    --score-root "${SCORE_ROOT}"
    --precast-score-root "${PRECAST_SCORE}"
    --target-package "${TARGET_PACKAGE}"
    --negative-prompt-asset "${NEGATIVE_ASSET}"
    --construction-wave 4
  )
fi

set +e
"${ROOT}/scripts/distillschedule/remote_stage_wrapper.sh" "${PID_FILE}" "${WORKER[@]}" >"${LOG}" 2>&1
RC=$?
set -e
{
  echo finished_at="$(date -Is)"
  echo return_code="${RC}"
  echo cgroup_memory_current_after="$(cat /sys/fs/cgroup/memory.current)"
  echo cgroup_memory_peak_after="$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo unavailable)"
  echo memory_events_after="$(tr '\n' ';' </sys/fs/cgroup/memory.events)"
} >>"${MANIFEST}"
exit "${RC}"

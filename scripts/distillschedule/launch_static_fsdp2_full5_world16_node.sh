#!/usr/bin/env bash
# Fail-closed per-node launcher for static world16 FSDP2 Full5 baselines.
set -Eeuo pipefail

NODE_RANK="${1:?node rank 0 or 1}"
RUN_ID="${2:?immutable run id}"
VARIANT="${3:?no-gan or gan}"
MODE="${4:?validate, correctness, timing, or profile}"
EXPECTED_HEAD="${5:?expected DMD-System HEAD}"
MASTER_ADDR="${6:?node0 routable address}"
MASTER_PORT="${7:?world16 rendezvous port}"

[[ "${NODE_RANK}" == 0 || "${NODE_RANK}" == 1 ]] || exit 2
[[ "${VARIANT}" == no-gan || "${VARIANT}" == gan ]] || exit 2
[[ "${MODE}" == validate || "${MODE}" == correctness || "${MODE}" == timing || "${MODE}" == profile ]] || exit 2
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || exit 2
[[ "${EXPECTED_HEAD}" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "${MASTER_PORT}" =~ ^[0-9]+$ ]] || exit 2

ROOT="${DMD_SYSTEM_ROOT:-/share/project/shuchenweng/yongyan_liu/DMD-System_fsdp2_static_20260814}"
HELIOS="${HELIOS_RUNTIME_ROOT:-/share/project/shuchenweng/yongyan_liu/DistillSchedule_gan_rolephase_exec_20260812}"
PYENV="${DISTILLSCHEDULE_PYENV:-/tmp/distillschedule-helios-official-torch210}"
PYTHON="${PYENV}/bin/python"
TORCHRUN="${PYENV}/bin/torchrun"
ASSETS="${DISTILLSCHEDULE_ASSETS:-/share/project/shuchenweng/DistillSchedule_assets}"
GENERATOR="${HELIOS_DISTILLED_ROOT:-${ASSETS}/Helios-Distilled}"
PRECAST_GENERATOR="${DISTILLSCHEDULE_PRECAST_GENERATOR_PATH:-${ASSETS}/precast/helios-distilled-transformer-ode-mixed-bf16-v1}"
SCORE="${HELIOS_BASE_ROOT:-/home/shuchenweng/xty/Helios-main/weights}"
NEGATIVE="${DISTILLSCHEDULE_NEGATIVE_PROMPT_ASSET:-${ASSETS}/m1/negative_prompt_official_v1}"
GAN_CONFIG="${DISTILLSCHEDULE_GAN_CONFIG:-${ROOT}/configs/distillschedule/templates/helios_stage3_gan_384x640_systems.yaml}"

if [[ "${VARIANT}" == no-gan ]]; then
  TRACE="${DISTILLSCHEDULE_TRACE_BUNDLE:-${ASSETS}/m0/traces/syn_feature_trace_rank16_v4_strict_sync_steps20}"
  TRACE_HASH="${DISTILLSCHEDULE_TRACE_BUNDLE_HASH:-654d49015921b278ab82e2fc84ea1896692da3fd38ce82fbc500cb1ebfec46d3}"
  LOGICAL="${DISTILLSCHEDULE_LOGICAL_STEP10:-${ASSETS}/m1/checkpoints/logical_world16_step10_v3}"
  LOGICAL_HASH="${DISTILLSCHEDULE_LOGICAL_STEP10_HASH:-68eed293970be008764ee6acadb2aac6ce54de2ce1f55a17fca2246f0d04e4aa}"
  PRECAST_SCORE="${DISTILLSCHEDULE_PRECAST_SCORE_PATH:-${ASSETS}/precast/helios-score-transformer-mixed-bf16-v1}"
else
  TRACE="${DISTILLSCHEDULE_GAN_TRACE_BUNDLE:-${ASSETS}/gan/systems_only_gan_trace_rank16_v1}"
  TRACE_HASH="${DISTILLSCHEDULE_GAN_TRACE_HASH:-d31d4d473a3574b5b39e50ad2b5e72e17aee7baf707130ea0513092553f7cd15}"
  LOGICAL="${DISTILLSCHEDULE_GAN_LOGICAL_PACKAGE:-${ASSETS}/gan/systems_only_gan_pipeline_world8_v3/logical_package}"
  LOGICAL_HASH="${DISTILLSCHEDULE_GAN_LOGICAL_HASH:-c66ff180fa4d00c78ba05ffd1826b7e358e0ab38d2f3f68f5f92dad5fe42dd44}"
  PRECAST_SCORE="${DISTILLSCHEDULE_PRECAST_SCORE_PATH:-${ASSETS}/precast/helios-score-transformer-mixed-bf16-gan-systems-v2}"
fi

RUN_ROOT="${DISTILLSCHEDULE_RUN_ROOT:-${ROOT}/artifacts/runs/${RUN_ID}}"
EXECUTION="${RUN_ROOT}/execution"
NODE_LOG="${RUN_ROOT}/launcher/node_${NODE_RANK}"
PROFILE_DIR="${RUN_ROOT}/nsight"
LOCK_FILE="${DISTILLSCHEDULE_NODE_LOCK:-/tmp/distillschedule_gpu_node.lock}"
CORRECTNESS_SUMMARY="${DISTILLSCHEDULE_FSDP2_CORRECTNESS_SUMMARY:-}"
CORRECTNESS_SHA256="${DISTILLSCHEDULE_FSDP2_CORRECTNESS_SHA256:-}"

for executable in "${PYTHON}" "${TORCHRUN}"; do
  [[ -x "${executable}" ]] || { echo missing_executable="${executable}" >&2; exit 3; }
done
for path in "${ROOT}" "${HELIOS}" "${TRACE}" "${LOGICAL}" "${GENERATOR}" \
  "${PRECAST_GENERATOR}" "${SCORE}" "${PRECAST_SCORE}" "${NEGATIVE}"; do
  [[ -e "${path}" ]] || { echo missing_path="${path}" >&2; exit 3; }
done
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_HEAD}" ]] || { echo source_head_mismatch >&2; exit 4; }
git -C "${ROOT}" diff --quiet && git -C "${ROOT}" diff --cached --quiet || { echo source_worktree_dirty >&2; exit 4; }

if [[ "${MODE}" == timing ]]; then
  [[ -f "${CORRECTNESS_SUMMARY}" ]] || { echo correctness_summary_missing >&2; exit 3; }
  [[ "${CORRECTNESS_SHA256}" =~ ^[0-9a-f]{64}$ ]] || { echo correctness_hash_missing >&2; exit 3; }
  [[ "$(sha256sum "${CORRECTNESS_SUMMARY}" | awk '{print $1}')" == "${CORRECTNESS_SHA256}" ]] || { echo correctness_hash_changed >&2; exit 3; }
  "${PYTHON}" - "${CORRECTNESS_SUMMARY}" "${VARIANT}" <<'PY'
import json,pathlib,sys
value=json.loads(pathlib.Path(sys.argv[1]).read_text())
if value.get('status') != 'PASS' or not value.get('claimable_correctness'):
    raise SystemExit('static FSDP2 correctness gate is not PASS')
if value.get('variant') != sys.argv[2]:
    raise SystemExit('correctness/timing variant mismatch')
PY
fi

mkdir -p "${NODE_LOG}" "${EXECUTION}"
[[ ! -e "${NODE_LOG}/launch_manifest.txt" ]] || { echo run_id_already_used >&2; exit 82; }
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

{
  echo schema_version=distillschedule.static_fsdp2_world16_launch.v1
  echo run_id="${RUN_ID}"
  echo variant="${VARIANT}"
  echo mode="${MODE}"
  echo node_rank="${NODE_RANK}"
  echo dmd_system_head="${EXPECTED_HEAD}"
  echo trace_hash="${TRACE_HASH}"
  echo logical_hash="${LOGICAL_HASH}"
  echo topology=STATIC_WORLD16_TWO_BACKBONES_FSDP2
  echo global_batch=16
  echo transition_count=0
  echo parameter_offload=false
  echo frozen_parameter_streaming=false
  echo formal_timing_profiled=false
} >"${NODE_LOG}/launch_manifest.txt"

WORKER=(
  "${ROOT}/scripts/distillschedule/probe_static_fsdp2_full5_world16.py"
  --variant "${VARIANT}"
  --run-id "${RUN_ID}"
  --output-dir "${EXECUTION}"
  --logical-package "${LOGICAL}"
  --expected-logical-hash "${LOGICAL_HASH}"
  --trace-bundle "${TRACE}"
  --expected-trace-hash "${TRACE_HASH}"
  --generator-root "${GENERATOR}"
  --precast-generator-root "${PRECAST_GENERATOR}"
  --score-root "${SCORE}"
  --precast-score-root "${PRECAST_SCORE}"
  --negative-prompt "${NEGATIVE}"
)
[[ "${VARIANT}" == gan ]] && WORKER+=(--gan-config "${GAN_CONFIG}")
[[ "${MODE}" == validate ]] && WORKER+=(--validate-only)
[[ "${MODE}" == timing ]] && WORKER+=(--timing-mode)
[[ "${MODE}" == profile ]] && WORKER+=(--profile-mode)

if [[ "${MODE}" == profile ]]; then
  command -v nsys >/dev/null || { echo nsys_missing >&2; exit 3; }
  mkdir -p "${PROFILE_DIR}"
  "${TORCHRUN}" --nnodes=2 --nproc-per-node=8 --node-rank="${NODE_RANK}" \
    --master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}" \
    "${ROOT}/scripts/distillschedule/nsys_rank_exec.py" \
    --profile-dir "${PROFILE_DIR}" \
    "${PYTHON}" "${WORKER[@]}" >"${NODE_LOG}/console.log" 2>&1
  for local_rank in $(seq 0 7); do
    global_rank=$((NODE_RANK * 8 + local_rank))
    report="${PROFILE_DIR}/rank_$(printf '%02d' "${global_rank}").nsys-rep"
    sqlite="${PROFILE_DIR}/rank_$(printf '%02d' "${global_rank}").sqlite"
    [[ -f "${report}" ]] || { echo missing_nsys_report="${report}" >&2; exit 87; }
    nsys export --type sqlite --force-overwrite true --output "${sqlite}" "${report}" >/dev/null
  done
else
  "${TORCHRUN}" --nnodes=2 --nproc-per-node=8 --node-rank="${NODE_RANK}" \
    --master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}" \
    "${WORKER[@]}" >"${NODE_LOG}/console.log" 2>&1
fi

if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')" ]]; then
  echo gpu_processes_remain >&2
  exit 88
fi

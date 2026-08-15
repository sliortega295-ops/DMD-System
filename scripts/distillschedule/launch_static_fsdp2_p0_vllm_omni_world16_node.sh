#!/usr/bin/env bash
# Two-node training launcher with a dedicated vLLM-Omni Helios rollout pool.
set -Eeuo pipefail

NODE_RANK="${1:?node rank 0 or 1}"
RUN_ID="${2:?immutable run id}"
VARIANT="${3:?no-gan or gan}"
MODE="${4:?validate, correctness, or timing}"
EXPECTED_HEAD="${5:?expected DMD-System HEAD}"
MASTER_ADDR="${6:?node0 routable address}"
MASTER_PORT="${7:?world16 rendezvous port}"
MESH_POLICY="${8:?world16-fsdp or node-hsdp}"
ROLLOUT_BATCH_SIZE="${9:?1, 2, or 5}"
PACK_REAL_SCORE_PAIR="${10:?on or off}"

[[ "${NODE_RANK}" == 0 || "${NODE_RANK}" == 1 ]] || exit 2
[[ "${VARIANT}" == no-gan || "${VARIANT}" == gan ]] || exit 2
[[ "${MODE}" == validate || "${MODE}" == correctness || "${MODE}" == timing ]] || {
  echo "vLLM-Omni node launcher supports validate/correctness/timing; profile the rollout service separately" >&2
  exit 2
}
[[ "${MESH_POLICY}" == world16-fsdp || "${MESH_POLICY}" == node-hsdp ]] || exit 2
[[ "${ROLLOUT_BATCH_SIZE}" == 1 || "${ROLLOUT_BATCH_SIZE}" == 2 || "${ROLLOUT_BATCH_SIZE}" == 5 ]] || exit 2
[[ "${PACK_REAL_SCORE_PAIR}" == on || "${PACK_REAL_SCORE_PAIR}" == off ]] || exit 2
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || exit 2
[[ "${EXPECTED_HEAD}" =~ ^[0-9a-f]{40}$ ]] || exit 2

: "${DISTILLSCHEDULE_VLLM_OMNI_ADAPTER_FACTORY:?module:function adapter factory is required}"
: "${DISTILLSCHEDULE_VLLM_OMNI_ENDPOINT:?vLLM-Omni endpoint is required}"
: "${DISTILLSCHEDULE_VLLM_OMNI_POOL_ID:?dedicated rollout pool id is required}"
: "${DISTILLSCHEDULE_VLLM_OMNI_EXPECTED_REVISION:=191b9a8dbf22fc494b7db34ea5ae04153f44f94d}"
: "${DISTILLSCHEDULE_VLLM_OMNI_EXPECTED_PATCH_SHA256:=84c9a914184373a771a5839bd6198d05ba8f981154732fc6f7bfbf5feafb3d99}"
[[ "${DISTILLSCHEDULE_VLLM_OMNI_DEDICATED_POOL:-0}" == 1 ]] || {
  echo DISTILLSCHEDULE_VLLM_OMNI_DEDICATED_POOL_must_equal_1 >&2
  exit 3
}
export DISTILLSCHEDULE_VLLM_OMNI_EXPECTED_REVISION
export DISTILLSCHEDULE_VLLM_OMNI_EXPECTED_PATCH_SHA256

ROOT="${DMD_SYSTEM_ROOT:-/share/project/shuchenweng/yongyan_liu/DMD-System_fsdp2_p0_20260815}"
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
LOCK_FILE="${DISTILLSCHEDULE_NODE_LOCK:-/tmp/distillschedule_gpu_node.lock}"
CORRECTNESS_SUMMARY="${DISTILLSCHEDULE_FSDP2_CORRECTNESS_SUMMARY:-}"
CORRECTNESS_SHA256="${DISTILLSCHEDULE_FSDP2_CORRECTNESS_SHA256:-}"
PROFILE_DIAGNOSTICS="${DISTILLSCHEDULE_PROFILE_DIAGNOSTICS:-off}"
EMA_DEVICE="${DISTILLSCHEDULE_EMA_DEVICE:-cuda}"

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
  [[ "$(sha256sum "${CORRECTNESS_SUMMARY}" | awk '{print $1}')" == "${CORRECTNESS_SHA256}" ]] || { echo correctness_hash_changed >&2; exit 3; }
  "${PYTHON}" - "${CORRECTNESS_SUMMARY}" "${VARIANT}" "${MESH_POLICY}" "${ROLLOUT_BATCH_SIZE}" "${PACK_REAL_SCORE_PAIR}" <<'PY'
import json, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text())
expected = {
    "mesh_policy": sys.argv[3],
    "rollout_batch_size": int(sys.argv[4]),
    "pack_real_score_pair": sys.argv[5] == "on",
    "rollout_backend": "vllm-omni-latent",
}
if value.get("status") != "PASS" or not value.get("claimable_correctness"):
    raise SystemExit("vLLM-Omni correctness gate is not PASS")
if value.get("variant") != sys.argv[2] or value.get("policy") != expected:
    raise SystemExit("correctness/timing contract mismatch")
PY
fi

mkdir -p "${NODE_LOG}" "${EXECUTION}"
[[ ! -e "${NODE_LOG}/launch_manifest.txt" ]] || { echo run_id_already_used >&2; exit 82; }
exec 9>"${LOCK_FILE}"
flock -n 9 || { echo gpu_lock_busy >&2; exit 73; }
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')" ]] || { echo gpu_processes_present >&2; exit 81; }

export PATH="${PYENV}/bin:${PATH}"
export PYTHONPATH="${ROOT}/src:${ROOT}:${HELIOS}${DISTILLSCHEDULE_VLLM_OMNI_CLIENT_PYTHONPATH:+:${DISTILLSCHEDULE_VLLM_OMNI_CLIENT_PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="${ASSETS}/hf_home_official_torch210"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 NCCL_ASYNC_ERROR_HANDLING=1
export DISTILLSCHEDULE_NODE_RANK="${NODE_RANK}"

cat >"${NODE_LOG}/launch_manifest.txt" <<EOF
schema_version=distillschedule.static_fsdp2_p0_vllm_omni_launch.v2
run_id=${RUN_ID}
variant=${VARIANT}
mode=${MODE}
node_rank=${NODE_RANK}
dmd_system_head=${EXPECTED_HEAD}
mesh_policy=${MESH_POLICY}
rollout_batch_size=${ROLLOUT_BATCH_SIZE}
pack_real_score_pair=${PACK_REAL_SCORE_PAIR}
rollout_backend=vllm-omni-latent
vllm_omni_version_family=0.19
vllm_version_family=0.19
vllm_omni_base_tag=v0.19.0rc1
vllm_omni_base_revision=${DISTILLSCHEDULE_VLLM_OMNI_EXPECTED_REVISION}
vllm_omni_compatibility_patch_sha256=${DISTILLSCHEDULE_VLLM_OMNI_EXPECTED_PATCH_SHA256}
vllm_omni_endpoint=${DISTILLSCHEDULE_VLLM_OMNI_ENDPOINT}
vllm_omni_pool_id=${DISTILLSCHEDULE_VLLM_OMNI_POOL_ID}
dedicated_rollout_pool=true
output_type=latent
vae_decode_expected_calls=0
native_b1_correctness_reference=true
EOF

WORKER=(
  "${ROOT}/scripts/distillschedule/probe_static_fsdp2_p0_vllm_omni_world16.py"
  --variant "${VARIANT}" --run-id "${RUN_ID}" --output-dir "${EXECUTION}"
  --logical-package "${LOGICAL}" --expected-logical-hash "${LOGICAL_HASH}"
  --trace-bundle "${TRACE}" --expected-trace-hash "${TRACE_HASH}"
  --generator-root "${GENERATOR}" --precast-generator-root "${PRECAST_GENERATOR}"
  --score-root "${SCORE}" --precast-score-root "${PRECAST_SCORE}"
  --negative-prompt "${NEGATIVE}" --mesh-policy "${MESH_POLICY}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --rollout-backend vllm-omni-latent --ema-device "${EMA_DEVICE}"
)
[[ "${VARIANT}" == gan ]] && WORKER+=(--gan-config "${GAN_CONFIG}")
[[ "${PACK_REAL_SCORE_PAIR}" == off ]] && WORKER+=(--no-pack-real-score-pair)
[[ "${MODE}" == validate ]] && WORKER+=(--validate-only)
[[ "${MODE}" == timing ]] && WORKER+=(--timing-mode)
[[ "${PROFILE_DIAGNOSTICS}" == off ]] && WORKER+=(--no-profile-diagnostics)

"${TORCHRUN}" --nnodes=2 --nproc-per-node=8 --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}" \
  "${WORKER[@]}" >"${NODE_LOG}/console.log" 2>&1

[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')" ]] || { echo gpu_processes_remain >&2; exit 88; }

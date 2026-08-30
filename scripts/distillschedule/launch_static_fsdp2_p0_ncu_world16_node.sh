#!/usr/bin/env bash
# Diagnostic-only bounded Nsight Compute launcher for static FSDP2 P0.
set -Eeuo pipefail

NODE_RANK="${1:?node rank 0 or 1}"
RUN_ID="${2:?immutable run id}"
VARIANT="${3:?no-gan or gan}"
MODE="${4:?validate, correctness, timing, or profile}"
EXPECTED_HEAD="${5:?expected DMD-System HEAD}"
MASTER_ADDR="${6:?node0 routable address}"
MASTER_PORT="${7:?world16 rendezvous port}"
MESH_POLICY="${8:?world16-fsdp or node-hsdp}"
ROLLOUT_BATCH_SIZE="${9:?1, 2, or 5}"
PACK_REAL_SCORE_PAIR="${10:?on or off}"

[[ "${NODE_RANK}" == 0 || "${NODE_RANK}" == 1 ]] || exit 2
[[ "${VARIANT}" == no-gan || "${VARIANT}" == gan ]] || exit 2
[[ "${MODE}" == validate || "${MODE}" == correctness || "${MODE}" == timing || "${MODE}" == profile ]] || exit 2
[[ "${MESH_POLICY}" == world16-fsdp || "${MESH_POLICY}" == node-hsdp ]] || exit 2
[[ "${ROLLOUT_BATCH_SIZE}" == 1 || "${ROLLOUT_BATCH_SIZE}" == 2 || "${ROLLOUT_BATCH_SIZE}" == 5 ]] || exit 2
[[ "${PACK_REAL_SCORE_PAIR}" == on || "${PACK_REAL_SCORE_PAIR}" == off ]] || exit 2
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || exit 2
[[ "${EXPECTED_HEAD}" =~ ^[0-9a-f]{40}$ ]] || exit 2

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
PROFILE_NODE_DIR="${RUN_ROOT}/ncu/node_${NODE_RANK}"
RECEIPT_DIR="${RUN_ROOT}/ncu/range_receipts"
LOCK_FILE="${DISTILLSCHEDULE_NODE_LOCK:-/tmp/distillschedule_gpu_node.lock}"
CORRECTNESS_SUMMARY="${DISTILLSCHEDULE_FSDP2_CORRECTNESS_SUMMARY:-}"
CORRECTNESS_SHA256="${DISTILLSCHEDULE_FSDP2_CORRECTNESS_SHA256:-}"
PROFILE_DIAGNOSTICS="${DISTILLSCHEDULE_PROFILE_DIAGNOSTICS:-off}"
EMA_DEVICE="${DISTILLSCHEDULE_EMA_DEVICE:-cuda}"
DIAGNOSTIC_TIMING="${DISTILLSCHEDULE_DIAGNOSTIC_TIMING:-0}"
CRITIC_ONLY_STEP="${DISTILLSCHEDULE_CRITIC_ONLY_STEP:-}"
OPERATION_SOURCE="${DISTILLSCHEDULE_OPERATION_SOURCE:-off}"
OPERATION_SOURCE_RANK="${DISTILLSCHEDULE_OPERATION_SOURCE_RANK:-0}"
OPERATION_SOURCE_TARGET_BYTES="${DISTILLSCHEDULE_OPERATION_SOURCE_TARGET_BYTES:-113049600}"
OPERATION_SOURCE_MAX_EVENTS="${DISTILLSCHEDULE_OPERATION_SOURCE_MAX_EVENTS:-256}"
# The fixed P0 path keeps both model residuals GPU-resident.  The explicit
# environment variables remain available for historical CPU-staging controls
# and bounded A/B experiments; generic runtime defaults are unchanged.
GENERATOR_RESIDUAL_STORAGE="${DISTILLSCHEDULE_GENERATOR_RESIDUAL_STORAGE:-cuda_out_of_place}"
CRITIC_RESIDUAL_STORAGE="${DISTILLSCHEDULE_CRITIC_RESIDUAL_STORAGE:-cuda_out_of_place}"
GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE="${DISTILLSCHEDULE_HELIOS_GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE:-off}"
ALL_ROLE_LORA_SCALE1_ADD="${DISTILLSCHEDULE_HELIOS_ALL_ROLE_LORA_SCALE1_ADD:-off}"
OPTIMIZER_DISPATCH_WARMUP="${DISTILLSCHEDULE_OPTIMIZER_DISPATCH_WARMUP:-off}"
GENERATOR_NOGRAD_RESIDUAL_FUSION="${DISTILLSCHEDULE_HELIOS_GENERATOR_NOGRAD_RESIDUAL_FUSION:-off}"
EXACT_GRAD_RESIDUAL_FUSION="${DISTILLSCHEDULE_HELIOS_EXACT_GRAD_RESIDUAL_FUSION:-off}"
EXACT_MODULATED_NORM_EPILOGUE="${DISTILLSCHEDULE_HELIOS_EXACT_MODULATED_NORM_EPILOGUE:-off}"
EXACT_MODULATED_NORM_MATERIALIZATION="${DISTILLSCHEDULE_HELIOS_EXACT_MODULATED_NORM_MATERIALIZATION:-off}"
EXACT_MODULATED_NORM_BACKWARD="${DISTILLSCHEDULE_HELIOS_EXACT_MODULATED_NORM_BACKWARD:-off}"
ATTENTION_BACKEND="${DISTILLSCHEDULE_HELIOS_ATTENTION_BACKEND:-sdpa}"
MODULATION_CHUNK_TOKENS="${DISTILLSCHEDULE_HELIOS_MODULATION_CHUNK_TOKENS:-512}"
SELECTIVE_FLASH_CHECKPOINT_STRIDE="${DISTILLSCHEDULE_HELIOS_SELECTIVE_FLASH_CHECKPOINT_STRIDE:-0}"
GENERATOR_NOGRAD_CROSS_KV_CACHE="${DISTILLSCHEDULE_HELIOS_GENERATOR_NOGRAD_CROSS_KV_CACHE:-off}"
FLASH_ATTN_KERNEL="${DISTILLSCHEDULE_FLASH_ATTN_KERNEL:-${ASSETS}/hf_home_official_torch210/hub/models--kernels-community--flash-attn2/snapshots/6a810a7a010ba2aad6613f1f0f7b226fb8bdee3c/build/torch210-cxx11-cu128-x86_64-linux}"
FLASH_ATTN_EXTENSION="${FLASH_ATTN_KERNEL}/_flash_attn2_cuda_a511a4c.abi3.so"
FLASH_ATTN_EXTENSION_SHA256="c2acd487c5d1d77b7f6721eabc0fee4f8c5c8f9e682aadf43bfa99b394eecba7"
EXPECTED_HELIOS_TRANSFORMER_SHA256="${DISTILLSCHEDULE_EXPECTED_HELIOS_TRANSFORMER_SHA256:-776a26c3b2921d599821f0767910830471ff94a955f5013089f9f689ac20fee7}"

[[ -z "${CRITIC_ONLY_STEP}" || "${CRITIC_ONLY_STEP}" == 10 ]] || exit 2
[[ "${OPERATION_SOURCE}" == on || "${OPERATION_SOURCE}" == off ]] || exit 2
[[ "${OPERATION_SOURCE_RANK}" =~ ^([0-9]|1[0-5])$ ]] || exit 2
[[ "${OPERATION_SOURCE_TARGET_BYTES}" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "${OPERATION_SOURCE_MAX_EVENTS}" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "${GENERATOR_RESIDUAL_STORAGE}" == cpu_inplace || "${GENERATOR_RESIDUAL_STORAGE}" == cuda_out_of_place ]] || exit 2
[[ "${CRITIC_RESIDUAL_STORAGE}" == cpu_inplace || "${CRITIC_RESIDUAL_STORAGE}" == cuda_out_of_place ]] || exit 2
[[ "${GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE}" == on || "${GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE}" == off ]] || exit 2
[[ "${ALL_ROLE_LORA_SCALE1_ADD}" == on || "${ALL_ROLE_LORA_SCALE1_ADD}" == off ]] || exit 2
[[ "${OPTIMIZER_DISPATCH_WARMUP}" == on || "${OPTIMIZER_DISPATCH_WARMUP}" == off ]] || exit 2
[[ "${GENERATOR_NOGRAD_RESIDUAL_FUSION}" == on || "${GENERATOR_NOGRAD_RESIDUAL_FUSION}" == off ]] || exit 2
[[ "${EXACT_GRAD_RESIDUAL_FUSION}" == on || "${EXACT_GRAD_RESIDUAL_FUSION}" == off ]] || exit 2
[[ "${EXACT_MODULATED_NORM_EPILOGUE}" == on || "${EXACT_MODULATED_NORM_EPILOGUE}" == off ]] || exit 2
[[ "${EXACT_MODULATED_NORM_MATERIALIZATION}" == on || "${EXACT_MODULATED_NORM_MATERIALIZATION}" == off ]] || exit 2
[[ "${EXACT_MODULATED_NORM_BACKWARD}" == on || "${EXACT_MODULATED_NORM_BACKWARD}" == off ]] || exit 2
[[ "${ATTENTION_BACKEND}" == sdpa || "${ATTENTION_BACKEND}" == local-fa2 ]] || exit 2
[[ "${MODULATION_CHUNK_TOKENS}" == 512 || "${MODULATION_CHUNK_TOKENS}" == 1024 || "${MODULATION_CHUNK_TOKENS}" == 2048 ]] || exit 2
[[ "${SELECTIVE_FLASH_CHECKPOINT_STRIDE}" == 0 || "${SELECTIVE_FLASH_CHECKPOINT_STRIDE}" == 2 || "${SELECTIVE_FLASH_CHECKPOINT_STRIDE}" == 4 || "${SELECTIVE_FLASH_CHECKPOINT_STRIDE}" == 10 ]] || exit 2
[[ "${GENERATOR_NOGRAD_CROSS_KV_CACHE}" == on || "${GENERATOR_NOGRAD_CROSS_KV_CACHE}" == off ]] || exit 2
[[ "${GENERATOR_NOGRAD_RESIDUAL_FUSION}" != on || "${GENERATOR_RESIDUAL_STORAGE}" == cuda_out_of_place ]] || exit 2
[[ "${GENERATOR_NOGRAD_RESIDUAL_FUSION}" != on || "${ROLLOUT_BATCH_SIZE}" == 1 ]] || exit 2
[[ "${EXACT_GRAD_RESIDUAL_FUSION}" != on || "${GENERATOR_RESIDUAL_STORAGE}" == cuda_out_of_place ]] || exit 2
[[ "${EXACT_GRAD_RESIDUAL_FUSION}" != on || "${CRITIC_RESIDUAL_STORAGE}" == cuda_out_of_place ]] || exit 2
[[ "${EXACT_GRAD_RESIDUAL_FUSION}" != on || "${ROLLOUT_BATCH_SIZE}" == 1 ]] || exit 2
[[ "${EXACT_MODULATED_NORM_EPILOGUE}" != on || "${ROLLOUT_BATCH_SIZE}" == 1 ]] || exit 2
[[ "${EXACT_MODULATED_NORM_MATERIALIZATION}" != on || "${EXACT_MODULATED_NORM_EPILOGUE}" == on ]] || exit 2
[[ "${EXACT_MODULATED_NORM_BACKWARD}" != on || "${EXACT_MODULATED_NORM_EPILOGUE}" == on ]] || exit 2
[[ "${GENERATOR_NOGRAD_CROSS_KV_CACHE}" != on || "${ROLLOUT_BATCH_SIZE}" == 1 ]] || exit 2
[[ "${GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE}" != on || "${ALL_ROLE_LORA_SCALE1_ADD}" != on ]] || exit 2
[[ "${SELECTIVE_FLASH_CHECKPOINT_STRIDE}" == 0 || "${ATTENTION_BACKEND}" == local-fa2 ]] || exit 2
[[ "${OPERATION_SOURCE}" != on || "${SELECTIVE_FLASH_CHECKPOINT_STRIDE}" == 0 ]] || exit 2
# Both supported mesh policies use sharded DTensor optimizer state.  The
# runtime warmup validates the exact mesh names/placements and fails closed;
# do not reject the validated world16 FULL_SHARD topology in the launcher.
[[ "${EXPECTED_HELIOS_TRANSFORMER_SHA256}" =~ ^[0-9a-f]{64}$ ]] || exit 2
if [[ "${ATTENTION_BACKEND}" == local-fa2 ]]; then
  [[ -d "${FLASH_ATTN_KERNEL}" ]] || { echo flash_attn_kernel_missing="${FLASH_ATTN_KERNEL}" >&2; exit 3; }
  [[ -f "${FLASH_ATTN_EXTENSION}" ]] || { echo flash_attn_extension_missing="${FLASH_ATTN_EXTENSION}" >&2; exit 3; }
  [[ "$(sha256sum "${FLASH_ATTN_EXTENSION}" | awk '{print $1}')" == "${FLASH_ATTN_EXTENSION_SHA256}" ]] || {
    echo flash_attn_extension_hash_mismatch >&2
    exit 4
  }
fi
if [[ -n "${CRITIC_ONLY_STEP}" ]]; then
  [[ "${VARIANT}" == no-gan && "${MODE}" == profile ]] || exit 2
fi
if [[ "${OPERATION_SOURCE}" == on ]]; then
  [[ -n "${CRITIC_ONLY_STEP}" ]] || exit 2
fi

for executable in "${PYTHON}" "${TORCHRUN}"; do
  [[ -x "${executable}" ]] || { echo missing_executable="${executable}" >&2; exit 3; }
done
for path in "${ROOT}" "${HELIOS}" "${TRACE}" "${LOGICAL}" "${GENERATOR}" \
  "${PRECAST_GENERATOR}" "${SCORE}" "${PRECAST_SCORE}" "${NEGATIVE}"; do
  [[ -e "${path}" ]] || { echo missing_path="${path}" >&2; exit 3; }
done
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_HEAD}" ]] || { echo source_head_mismatch >&2; exit 4; }
git -C "${ROOT}" diff --quiet && git -C "${ROOT}" diff --cached --quiet || { echo source_worktree_dirty >&2; exit 4; }
HELIOS_TRANSFORMER="${HELIOS}/helios/modules/transformer_helios.py"
[[ "$(sha256sum "${HELIOS_TRANSFORMER}" | awk '{print $1}')" == "${EXPECTED_HELIOS_TRANSFORMER_SHA256}" ]] || { echo helios_transformer_hash_mismatch >&2; exit 4; }

if [[ "${MODE}" == timing && "${DIAGNOSTIC_TIMING}" != 1 ]]; then
  [[ -f "${CORRECTNESS_SUMMARY}" ]] || { echo correctness_summary_missing >&2; exit 3; }
  [[ "$(sha256sum "${CORRECTNESS_SUMMARY}" | awk '{print $1}')" == "${CORRECTNESS_SHA256}" ]] || { echo correctness_hash_changed >&2; exit 3; }
  "${PYTHON}" - "${CORRECTNESS_SUMMARY}" "${VARIANT}" "${MESH_POLICY}" "${ROLLOUT_BATCH_SIZE}" "${PACK_REAL_SCORE_PAIR}" "${GENERATOR_RESIDUAL_STORAGE}" "${CRITIC_RESIDUAL_STORAGE}" "${GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE}" "${ALL_ROLE_LORA_SCALE1_ADD}" "${OPTIMIZER_DISPATCH_WARMUP}" "${GENERATOR_NOGRAD_RESIDUAL_FUSION}" "${EXACT_GRAD_RESIDUAL_FUSION}" "${EXACT_MODULATED_NORM_EPILOGUE}" "${EXACT_MODULATED_NORM_MATERIALIZATION}" "${ATTENTION_BACKEND}" "${MODULATION_CHUNK_TOKENS}" "${SELECTIVE_FLASH_CHECKPOINT_STRIDE}" "${GENERATOR_NOGRAD_CROSS_KV_CACHE}" "${EXACT_MODULATED_NORM_BACKWARD}" <<'PY'
import json,pathlib,sys
value=json.loads(pathlib.Path(sys.argv[1]).read_text())
expected={"mesh_policy":sys.argv[3],"rollout_batch_size":int(sys.argv[4]),"pack_real_score_pair":sys.argv[5]=="on","generator_residual_storage":sys.argv[6],"critic_residual_storage":sys.argv[7],"generator_nograd_lora_fused_epilogue":sys.argv[8]=="on","all_role_lora_scale1_add":sys.argv[9]=="on","optimizer_dispatch_warmup":sys.argv[10]=="on","generator_nograd_residual_fusion":sys.argv[11]=="on","exact_grad_residual_fusion":sys.argv[12]=="on","exact_modulated_norm_epilogue":sys.argv[13]=="on","exact_modulated_norm_materialization":sys.argv[14]=="on","attention_backend":sys.argv[15],"modulation_chunk_tokens":int(sys.argv[16]),"selective_flash_checkpoint_stride":int(sys.argv[17]),"generator_nograd_cross_kv_cache":sys.argv[18]=="on","exact_modulated_norm_backward":sys.argv[19]=="on"}
if value.get("status") != "PASS" or not value.get("claimable_correctness"):
    raise SystemExit("P0 correctness gate is not PASS")
if value.get("variant") != sys.argv[2]:
    raise SystemExit("correctness/timing variant mismatch")
for key, expected_value in expected.items():
    if value.get("policy",{}).get(key) != expected_value:
        raise SystemExit(f"correctness/timing policy mismatch: {key}")
PY
fi

mkdir -p "${NODE_LOG}" "${EXECUTION}"
[[ ! -e "${NODE_LOG}/launch_manifest.txt" ]] || { echo run_id_already_used >&2; exit 82; }
exec 9>"${LOCK_FILE}"
flock -n 9 || { echo gpu_lock_busy >&2; exit 73; }
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')" ]] || { echo gpu_processes_present >&2; exit 81; }

export PATH="${PYENV}/bin:${PATH}"
# The validated Torch 2.10 environment is a relocatable conda archive.  New
# cluster images can provide an older system libstdc++ than the archive's ICU
# and sqlite extensions require, so prefer the archive's own runtime libraries.
# This only repairs dynamic-library resolution; it does not change the model or
# CUDA workload.
export LD_LIBRARY_PATH="${PYENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${ROOT}/src:${ROOT}:${HELIOS}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="${ASSETS}/hf_home_official_torch210"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 NCCL_ASYNC_ERROR_HANDLING=1
export DISTILLSCHEDULE_NODE_RANK="${NODE_RANK}"
if [[ "${ATTENTION_BACKEND}" == local-fa2 ]]; then
  export LOCAL_KERNELS="kernels-community/flash-attn2=${FLASH_ATTN_KERNEL}"
else
  unset LOCAL_KERNELS
fi

cat >"${NODE_LOG}/launch_manifest.txt" <<EOF
schema_version=distillschedule.static_fsdp2_p0_launch.v2
run_id=${RUN_ID}
variant=${VARIANT}
mode=${MODE}
node_rank=${NODE_RANK}
dmd_system_head=${EXPECTED_HEAD}
helios_runtime_root=${HELIOS}
helios_transformer_sha256=${EXPECTED_HELIOS_TRANSFORMER_SHA256}
mesh_policy=${MESH_POLICY}
rollout_batch_size=${ROLLOUT_BATCH_SIZE}
pack_real_score_pair=${PACK_REAL_SCORE_PAIR}
rollout_backend=native-fsdp2-batch
global_batch=16
transition_count=0
parameter_offload=false
frozen_parameter_streaming=false
profile_diagnostics=${PROFILE_DIAGNOSTICS}
ema_device=${EMA_DEVICE}
hardened_worker=true
critic_input_reuse=retain_until_one_consume
parity_policy=multi_metric
diagnostic_timing=${DIAGNOSTIC_TIMING}
critic_only_step=${CRITIC_ONLY_STEP:-none}
operation_source=${OPERATION_SOURCE}
operation_source_rank=${OPERATION_SOURCE_RANK}
operation_source_target_bytes=${OPERATION_SOURCE_TARGET_BYTES}
operation_source_max_events=${OPERATION_SOURCE_MAX_EVENTS}
generator_residual_storage=${GENERATOR_RESIDUAL_STORAGE}
critic_residual_storage=${CRITIC_RESIDUAL_STORAGE}
generator_nograd_lora_fused_epilogue=${GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE}
all_role_lora_scale1_add=${ALL_ROLE_LORA_SCALE1_ADD}
optimizer_dispatch_warmup=${OPTIMIZER_DISPATCH_WARMUP}
generator_nograd_residual_fusion=${GENERATOR_NOGRAD_RESIDUAL_FUSION}
exact_grad_residual_fusion=${EXACT_GRAD_RESIDUAL_FUSION}
exact_modulated_norm_epilogue=${EXACT_MODULATED_NORM_EPILOGUE}
exact_modulated_norm_materialization=${EXACT_MODULATED_NORM_MATERIALIZATION}
exact_modulated_norm_backward=${EXACT_MODULATED_NORM_BACKWARD}
attention_backend=${ATTENTION_BACKEND}
modulation_chunk_tokens=${MODULATION_CHUNK_TOKENS}
selective_flash_checkpoint_stride=${SELECTIVE_FLASH_CHECKPOINT_STRIDE}
generator_nograd_cross_kv_cache=${GENERATOR_NOGRAD_CROSS_KV_CACHE}
flash_attn_kernel=$([[ "${ATTENTION_BACKEND}" == local-fa2 ]] && printf '%s' "${FLASH_ATTN_KERNEL}" || printf 'none')
flash_attn_extension_sha256=$([[ "${ATTENTION_BACKEND}" == local-fa2 ]] && printf '%s' "${FLASH_ATTN_EXTENSION_SHA256}" || printf 'none')
EOF

WORKER=(
  "${ROOT}/scripts/distillschedule/probe_static_fsdp2_p0_world16_hardened.py"
  --variant "${VARIANT}" --run-id "${RUN_ID}" --output-dir "${EXECUTION}"
  --logical-package "${LOGICAL}" --expected-logical-hash "${LOGICAL_HASH}"
  --trace-bundle "${TRACE}" --expected-trace-hash "${TRACE_HASH}"
  --generator-root "${GENERATOR}" --precast-generator-root "${PRECAST_GENERATOR}"
  --score-root "${SCORE}" --precast-score-root "${PRECAST_SCORE}"
  --negative-prompt "${NEGATIVE}" --mesh-policy "${MESH_POLICY}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" --rollout-backend native-fsdp2-batch
  --ema-device "${EMA_DEVICE}"
  --attention-backend "${ATTENTION_BACKEND}"
  --modulation-chunk-tokens "${MODULATION_CHUNK_TOKENS}"
  --selective-flash-checkpoint-stride "${SELECTIVE_FLASH_CHECKPOINT_STRIDE}"
  --generator-residual-storage "${GENERATOR_RESIDUAL_STORAGE}"
  --critic-residual-storage "${CRITIC_RESIDUAL_STORAGE}"
)
[[ "${VARIANT}" == gan ]] && WORKER+=(--gan-config "${GAN_CONFIG}")
if [[ "${PACK_REAL_SCORE_PAIR}" == on ]]; then
  # Pass the positive flag explicitly.  The hardened worker's direct-invocation
  # fallback is intentionally conservative and appends --no-pack when neither
  # spelling is present; relying on its argparse default would therefore turn
  # an explicit launcher request for packing back off.
  WORKER+=(--pack-real-score-pair)
else
  WORKER+=(--no-pack-real-score-pair)
fi
if [[ "${GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE}" == on ]]; then
  WORKER+=(--generator-nograd-lora-fused-epilogue)
else
  WORKER+=(--no-generator-nograd-lora-fused-epilogue)
fi
if [[ "${ALL_ROLE_LORA_SCALE1_ADD}" == on ]]; then
  WORKER+=(--all-role-lora-scale1-add)
else
  WORKER+=(--no-all-role-lora-scale1-add)
fi
if [[ "${OPTIMIZER_DISPATCH_WARMUP}" == on ]]; then
  WORKER+=(--optimizer-dispatch-warmup)
else
  WORKER+=(--no-optimizer-dispatch-warmup)
fi
if [[ "${GENERATOR_NOGRAD_RESIDUAL_FUSION}" == on ]]; then
  WORKER+=(--generator-nograd-residual-fusion)
else
  WORKER+=(--no-generator-nograd-residual-fusion)
fi
if [[ "${EXACT_GRAD_RESIDUAL_FUSION}" == on ]]; then
  WORKER+=(--exact-grad-residual-fusion)
else
  WORKER+=(--no-exact-grad-residual-fusion)
fi
if [[ "${EXACT_MODULATED_NORM_EPILOGUE}" == on ]]; then
  WORKER+=(--exact-modulated-norm-epilogue)
else
  WORKER+=(--no-exact-modulated-norm-epilogue)
fi
if [[ "${EXACT_MODULATED_NORM_MATERIALIZATION}" == on ]]; then
  WORKER+=(--exact-modulated-norm-materialization)
else
  WORKER+=(--no-exact-modulated-norm-materialization)
fi
if [[ "${EXACT_MODULATED_NORM_BACKWARD}" == on ]]; then
  WORKER+=(--exact-modulated-norm-backward)
else
  WORKER+=(--no-exact-modulated-norm-backward)
fi
if [[ "${GENERATOR_NOGRAD_CROSS_KV_CACHE}" == on ]]; then
  WORKER+=(--generator-nograd-cross-kv-cache)
else
  WORKER+=(--no-generator-nograd-cross-kv-cache)
fi
[[ "${MODE}" == validate ]] && WORKER+=(--validate-only)
[[ "${MODE}" == timing ]] && WORKER+=(--timing-mode)
[[ "${MODE}" == profile ]] && WORKER+=(--profile-mode)
[[ "${PROFILE_DIAGNOSTICS}" == off ]] && WORKER+=(--no-profile-diagnostics)
if [[ -n "${CRITIC_ONLY_STEP}" ]]; then
  WORKER+=(--critic-only-step "${CRITIC_ONLY_STEP}")
fi
if [[ "${OPERATION_SOURCE}" == on ]]; then
  WORKER+=(
    --operation-source
    --operation-source-rank "${OPERATION_SOURCE_RANK}"
    --operation-source-target-bytes "${OPERATION_SOURCE_TARGET_BYTES}"
    --operation-source-max-events "${OPERATION_SOURCE_MAX_EVENTS}"
  )
fi

[[ "${MODE}" == profile ]] || { echo ncu_launcher_requires_profile_mode >&2; exit 2; }
NCU_BIN="${DISTILLSCHEDULE_NCU_BIN:-/opt/nvidia/nsight-compute/2023.2.1/ncu}"
NCU_TARGET_ID="${DISTILLSCHEDULE_NCU_TARGET_ID:?set NCU target id}"
NCU_NVTX_RANGE="${DISTILLSCHEDULE_NCU_NVTX_RANGE:?set exact NCU NVTX range}"
NCU_KERNEL_NAME="${DISTILLSCHEDULE_NCU_KERNEL_NAME:?set exact demangled kernel name}"
NCU_LAUNCH_SKIP="${DISTILLSCHEDULE_NCU_LAUNCH_SKIP:-0}"
NCU_SET="${DISTILLSCHEDULE_NCU_SET:-full}"
NCU_NVTX_FILTER="${NCU_NVTX_RANGE//\//\\/}/"
[[ -x "${NCU_BIN}" ]] || { echo ncu_missing="${NCU_BIN}" >&2; exit 3; }
[[ "${NCU_LAUNCH_SKIP}" =~ ^[0-9]+$ ]] || { echo invalid_ncu_launch_skip >&2; exit 2; }
[[ "${NCU_SET}" == basic || "${NCU_SET}" == full ]] || { echo invalid_ncu_set >&2; exit 2; }

mkdir -p "${PROFILE_NODE_DIR}" "${RECEIPT_DIR}"
REPORT_BASE="${PROFILE_NODE_DIR}/${NCU_TARGET_ID}"
[[ ! -e "${REPORT_BASE}.ncu-rep" ]] || { echo refusing_to_overwrite="${REPORT_BASE}.ncu-rep" >&2; exit 82; }
export DISTILLSCHEDULE_STATIC_FSDP2_NSYS_EXACT_RANGE=1
export DISTILLSCHEDULE_STATIC_FSDP2_NSYS_RECEIPT_DIR="${RECEIPT_DIR}"
{
  echo schema_version=distillschedule.static_fsdp2_p0_ncu_node.v2
  echo status=RUNNING
  echo diagnostic_only=true
  echo formal_speedup_eligible=false
  echo node_rank="${NODE_RANK}"
  echo variant="${VARIANT}"
  echo mesh_policy="${MESH_POLICY}"
  echo rollout_batch_size="${ROLLOUT_BATCH_SIZE}"
  echo pack_real_score_pair="${PACK_REAL_SCORE_PAIR}"
  echo target_id="${NCU_TARGET_ID}"
  echo nvtx_range="${NCU_NVTX_RANGE}"
  echo nvtx_filter="${NCU_NVTX_FILTER}"
  echo kernel_name="${NCU_KERNEL_NAME}"
  echo launch_skip="${NCU_LAUNCH_SKIP}"
  echo launch_count=1
  echo device=0
  echo ncu_set="${NCU_SET}"
  echo source_head="${EXPECTED_HEAD}"
} >"${PROFILE_NODE_DIR}/ncu_manifest.txt"

if [[ "${NODE_RANK}" == 0 ]]; then
  "${NCU_BIN}" \
    --target-processes all \
    --devices 0 \
    --nvtx \
    --nvtx-include "${NCU_NVTX_FILTER}" \
    --kernel-name-base demangled \
    --kernel-name "${NCU_KERNEL_NAME}" \
    --launch-skip "${NCU_LAUNCH_SKIP}" \
    --launch-count 1 \
    --set "${NCU_SET}" \
    --replay-mode kernel \
    --cache-control none \
    --clock-control none \
    --force-overwrite \
    --export "${REPORT_BASE}" \
    "${TORCHRUN}" --no-python --nnodes=2 --nproc-per-node=8 --node-rank="${NODE_RANK}" \
      --master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}" \
      "${PYTHON}" "${WORKER[@]}" >"${NODE_LOG}/console.log" 2>&1
  [[ -s "${REPORT_BASE}.ncu-rep" ]] || { echo missing_ncu_report="${REPORT_BASE}.ncu-rep" >&2; exit 87; }
  "${NCU_BIN}" --import "${REPORT_BASE}.ncu-rep" --page details --csv \
    --log-file "${PROFILE_NODE_DIR}/ncu_details.csv" >/dev/null 2>&1
  "${NCU_BIN}" --import "${REPORT_BASE}.ncu-rep" --page raw --csv \
    --log-file "${PROFILE_NODE_DIR}/ncu_raw.csv" >/dev/null 2>&1
  [[ -s "${PROFILE_NODE_DIR}/ncu_details.csv" ]] || { echo missing_ncu_details >&2; exit 88; }
else
  "${TORCHRUN}" --nnodes=2 --nproc-per-node=8 --node-rank="${NODE_RANK}" \
    --master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}" \
    "${WORKER[@]}" >"${NODE_LOG}/console.log" 2>&1
fi

for _ in $(seq 1 300); do
  count="$(find "${RECEIPT_DIR}" -maxdepth 1 -type f -name 'rank_*.json' | wc -l)"
  [[ "${count}" == 16 ]] && break
  sleep 1
done
count="$(find "${RECEIPT_DIR}" -maxdepth 1 -type f -name 'rank_*.json' | wc -l)"
[[ "${count}" == 16 ]] || { echo "ncu_range_receipts=${count},expected=16" >&2; exit 90; }

{
  echo status=PASS
  echo completed_at="$(date -Is)"
  if [[ "${NODE_RANK}" == 0 ]]; then
    echo report="${REPORT_BASE}.ncu-rep"
    echo report_sha256="$(sha256sum "${REPORT_BASE}.ncu-rep" | awk '{print $1}')"
    echo details_sha256="$(sha256sum "${PROFILE_NODE_DIR}/ncu_details.csv" | awk '{print $1}')"
  fi
} >>"${PROFILE_NODE_DIR}/ncu_manifest.txt"

[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')" ]] || { echo gpu_processes_remain >&2; exit 88; }

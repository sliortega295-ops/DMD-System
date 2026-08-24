#!/usr/bin/env bash
# Fail-closed per-node launcher for the static FSDP2 P0 Full5 candidates.
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
PROFILE_NODE_DIR="${RUN_ROOT}/nsight/node_${NODE_RANK}"
RECEIPT_DIR="${RUN_ROOT}/nsight/range_receipts"
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
GENERATOR_RESIDUAL_STORAGE="${DISTILLSCHEDULE_GENERATOR_RESIDUAL_STORAGE:-cpu_inplace}"
CRITIC_RESIDUAL_STORAGE="${DISTILLSCHEDULE_CRITIC_RESIDUAL_STORAGE:-cpu_inplace}"
GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE="${DISTILLSCHEDULE_HELIOS_GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE:-off}"
EXPECTED_HELIOS_TRANSFORMER_SHA256="${DISTILLSCHEDULE_EXPECTED_HELIOS_TRANSFORMER_SHA256:-776a26c3b2921d599821f0767910830471ff94a955f5013089f9f689ac20fee7}"

[[ -z "${CRITIC_ONLY_STEP}" || "${CRITIC_ONLY_STEP}" == 10 ]] || exit 2
[[ "${OPERATION_SOURCE}" == on || "${OPERATION_SOURCE}" == off ]] || exit 2
[[ "${OPERATION_SOURCE_RANK}" =~ ^([0-9]|1[0-5])$ ]] || exit 2
[[ "${OPERATION_SOURCE_TARGET_BYTES}" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "${OPERATION_SOURCE_MAX_EVENTS}" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "${GENERATOR_RESIDUAL_STORAGE}" == cpu_inplace || "${GENERATOR_RESIDUAL_STORAGE}" == cuda_out_of_place ]] || exit 2
[[ "${CRITIC_RESIDUAL_STORAGE}" == cpu_inplace || "${CRITIC_RESIDUAL_STORAGE}" == cuda_out_of_place ]] || exit 2
[[ "${GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE}" == on || "${GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE}" == off ]] || exit 2
[[ "${EXPECTED_HELIOS_TRANSFORMER_SHA256}" =~ ^[0-9a-f]{64}$ ]] || exit 2
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
  "${PYTHON}" - "${CORRECTNESS_SUMMARY}" "${VARIANT}" "${MESH_POLICY}" "${ROLLOUT_BATCH_SIZE}" "${PACK_REAL_SCORE_PAIR}" "${GENERATOR_RESIDUAL_STORAGE}" "${CRITIC_RESIDUAL_STORAGE}" "${GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE}" <<'PY'
import json,pathlib,sys
value=json.loads(pathlib.Path(sys.argv[1]).read_text())
expected={"mesh_policy":sys.argv[3],"rollout_batch_size":int(sys.argv[4]),"pack_real_score_pair":sys.argv[5]=="on","generator_residual_storage":sys.argv[6],"critic_residual_storage":sys.argv[7],"generator_nograd_lora_fused_epilogue":sys.argv[8]=="on"}
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
export PYTHONPATH="${ROOT}/src:${ROOT}:${HELIOS}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="${ASSETS}/hf_home_official_torch210"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 NCCL_ASYNC_ERROR_HANDLING=1
export DISTILLSCHEDULE_NODE_RANK="${NODE_RANK}"

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

if [[ "${MODE}" == profile ]]; then
  command -v nsys >/dev/null || { echo nsys_missing >&2; exit 3; }
  mkdir -p "${PROFILE_NODE_DIR}" "${RECEIPT_DIR}"
  REPORT_BASE="${PROFILE_NODE_DIR}/static_fsdp2_p0_node${NODE_RANK}"
  [[ ! -e "${REPORT_BASE}.nsys-rep" ]] || { echo refusing_to_overwrite="${REPORT_BASE}.nsys-rep" >&2; exit 82; }
  export DISTILLSCHEDULE_STATIC_FSDP2_NSYS_EXACT_RANGE=1
  export DISTILLSCHEDULE_STATIC_FSDP2_NSYS_RECEIPT_DIR="${RECEIPT_DIR}"
  "${NSYS_BIN:-$(command -v nsys)}" profile \
    --trace=cuda,nvtx \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --trace-fork-before-exec=true \
    --wait=all \
    --sample=none \
    --cpuctxsw=none \
    --cuda-memory-usage=false \
    --cuda-flush-interval=1000 \
    --force-overwrite=true \
    --output="${REPORT_BASE}" \
    "${TORCHRUN}" --no-python --nnodes=2 --nproc-per-node=8 --node-rank="${NODE_RANK}" \
      --master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}" \
      "${PYTHON}" "${WORKER[@]}" >"${NODE_LOG}/console.log" 2>&1
  [[ -s "${REPORT_BASE}.nsys-rep" ]] || { echo missing_nsys_report="${REPORT_BASE}.nsys-rep" >&2; exit 87; }
  SQLITE="${REPORT_BASE}.sqlite"
  nsys export --type sqlite --force-overwrite true --output "${SQLITE}" "${REPORT_BASE}.nsys-rep" >/dev/null
  [[ -s "${SQLITE}" ]] || { echo missing_nsys_sqlite="${SQLITE}" >&2; exit 88; }
  nsys stats --report nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum --format csv \
    "${REPORT_BASE}.nsys-rep" >"${PROFILE_NODE_DIR}/nsys_stats_validation.log" 2>&1
  if grep -Eq "does not contain (NV Tools Extension|CUDA kernel|CUDA API)" \
      "${PROFILE_NODE_DIR}/nsys_stats_validation.log"; then
    echo nsys_empty_capture >&2
    exit 89
  fi
  for _ in $(seq 1 300); do
    count="$(find "${RECEIPT_DIR}" -maxdepth 1 -type f -name 'rank_*.json' | wc -l)"
    [[ "${count}" == 16 ]] && break
    sleep 1
  done
  count="$(find "${RECEIPT_DIR}" -maxdepth 1 -type f -name 'rank_*.json' | wc -l)"
  [[ "${count}" == 16 ]] || { echo "nsys_range_receipts=${count},expected=16" >&2; exit 90; }
  "${PYTHON}" - "${RECEIPT_DIR}" "${VARIANT}" "${CRITIC_ONLY_STEP}" <<'PY'
import json
import os
import pathlib
import sys
from distillschedule.runtime.nsys_range import validate_static_nsys_receipts
root = pathlib.Path(sys.argv[1]); variant = sys.argv[2]; critic_step = sys.argv[3]
kwargs = {}
if critic_step:
    step = int(critic_step)
    kwargs = {
        "expected_window": (step, step),
        "expected_start_point": f"before_step{step}_critic_compute",
        "expected_stop_point": f"after_step{step}_critic_commit",
    }
result = validate_static_nsys_receipts(root, variant=variant, **kwargs)
if os.environ.get("DISTILLSCHEDULE_NODE_RANK") == "0":
    out = root.parent / "static_p0_receipts_validation.json"
    temporary = out.with_suffix(out.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, out)
print(json.dumps(result, sort_keys=True))
PY
  {
    echo schema_version=distillschedule.static_fsdp2_p0_nsys_node.v2
    echo status=PASS
    echo diagnostic_only=true
    echo formal_speedup_eligible=false
    echo node_rank="${NODE_RANK}"
    echo variant="${VARIANT}"
    echo mesh_policy="${MESH_POLICY}"
    echo rollout_batch_size="${ROLLOUT_BATCH_SIZE}"
    echo pack_real_score_pair="${PACK_REAL_SCORE_PAIR}"
    echo critic_only_step="${CRITIC_ONLY_STEP:-none}"
    echo operation_source="${OPERATION_SOURCE}"
    echo report="${REPORT_BASE}.nsys-rep"
    echo sqlite="${SQLITE}"
    echo report_sha256="$(sha256sum "${REPORT_BASE}.nsys-rep" | awk '{print $1}')"
  } >"${PROFILE_NODE_DIR}/profile_manifest.txt"
else
  "${TORCHRUN}" --nnodes=2 --nproc-per-node=8 --node-rank="${NODE_RANK}" \
    --master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}" \
    "${WORKER[@]}" >"${NODE_LOG}/console.log" 2>&1
fi

[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')" ]] || { echo gpu_processes_remain >&2; exit 88; }

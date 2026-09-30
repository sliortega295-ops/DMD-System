#!/usr/bin/env bash
# Fail-closed per-node launcher for the fair Wan1.3B placement ladder.
set -Eeuo pipefail

NODE_RANK="${1:?node rank 0 or 1}"
RUN_ID="${2:?immutable run id}"
MODE="${3:?validate, correctness, timing, diagnostic-timing, profile, trajectory, or two-cycle-timing}"
EXPECTED_HEAD="${4:?expected DMD-System HEAD}"
MASTER_ADDR="${5:?node0 routable address}"
MASTER_PORT="${6:?world16 rendezvous port}"
PLACEMENT="${7:?resident, resident-clean, resident-async128, resident-async128-tree, resident-async128-tree-chunk2048, resident-async128-tree-chunk2048-compile, resident-async256, resident-local-fast, resident-flat256, resident-flat512, resident-lora-zero2, resident-rollout-fused, resident-ddp, resident-fast, or world16-fsdp}"

[[ "${NODE_RANK}" == 0 || "${NODE_RANK}" == 1 ]] || exit 2
[[ "${MODE}" == validate || "${MODE}" == correctness || "${MODE}" == timing || \
   "${MODE}" == diagnostic-timing || "${MODE}" == profile || \
   "${MODE}" == trajectory || "${MODE}" == two-cycle-timing ]] || exit 2
[[ "${PLACEMENT}" == resident || "${PLACEMENT}" == resident-ddp || \
   "${PLACEMENT}" == resident-clean || "${PLACEMENT}" == resident-async128 || \
   "${PLACEMENT}" == resident-async128-tree || \
   "${PLACEMENT}" == resident-async128-tree-chunk2048 || \
   "${PLACEMENT}" == resident-async128-tree-chunk2048-compile || \
   "${PLACEMENT}" == resident-async256 || \
   "${PLACEMENT}" == resident-local-fast || \
   "${PLACEMENT}" == resident-flat256 || "${PLACEMENT}" == resident-flat512 || \
   "${PLACEMENT}" == resident-lora-zero2 || "${PLACEMENT}" == resident-rollout-fused || \
   "${PLACEMENT}" == resident-fast || \
   "${PLACEMENT}" == world16-fsdp ]] || exit 2
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || exit 2
[[ "${EXPECTED_HEAD}" =~ ^[0-9a-f]{40}$ ]] || exit 2

ROOT="${DMD_SYSTEM_ROOT:-/share/project/shuchenweng/yongyan_liu/DMD-System_wan13b_upper_bound_20260820}"
HELIOS="${HELIOS_RUNTIME_ROOT:-/share/project/shuchenweng/yongyan_liu/DistillSchedule_fsdp2_gpu_residual_20260816}"
PYENV="${DISTILLSCHEDULE_PYENV:-/tmp/distillschedule-helios-official-torch210}"
PYTHON="${PYENV}/bin/python"
TORCHRUN="${PYENV}/bin/torchrun"
# The packed Torch 2.10 environment ships a newer libstdc++ alongside its
# Python/ICU libraries.  Without putting that directory first, importing
# torch before sqlite3 leaves the process bound to the host libstdc++ and the
# later sqlite3 import fails on CXXABI_1.3.15.  Keep the packaged environment
# self-consistent rather than relying on import order.
export LD_LIBRARY_PATH="${PYENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
ASSETS="${DISTILLSCHEDULE_ASSETS:-/share/project/shuchenweng/DistillSchedule_assets}"
MODEL_ROOT="${DISTILLSCHEDULE_WAN13B_MODEL_ROOT:?set DISTILLSCHEDULE_WAN13B_MODEL_ROOT}"
BOOTSTRAP="${DISTILLSCHEDULE_WAN13B_BOOTSTRAP:?set DISTILLSCHEDULE_WAN13B_BOOTSTRAP}"
BOOTSTRAP_HASH="${DISTILLSCHEDULE_WAN13B_BOOTSTRAP_HASH:?set DISTILLSCHEDULE_WAN13B_BOOTSTRAP_HASH}"
TRACE="${DISTILLSCHEDULE_TRACE_BUNDLE:-${ASSETS}/m0/traces/syn_feature_trace_rank16_v4_strict_sync_steps20}"
TRACE_HASH="${DISTILLSCHEDULE_TRACE_BUNDLE_HASH:-654d49015921b278ab82e2fc84ea1896692da3fd38ce82fbc500cb1ebfec46d3}"
NEGATIVE="${DISTILLSCHEDULE_NEGATIVE_PROMPT_ASSET:-${ASSETS}/m1/negative_prompt_official_v1}"
RUN_ROOT="${DISTILLSCHEDULE_RUN_ROOT:-${ROOT}/artifacts/runs/${RUN_ID}}"
EXECUTION="${RUN_ROOT}/execution"
NODE_LOG="${RUN_ROOT}/launcher/node_${NODE_RANK}"
LOCK_FILE="${DISTILLSCHEDULE_NODE_LOCK:-/tmp/distillschedule_gpu_node.lock}"
WORKER="${ROOT}/scripts/distillschedule/probe_wan13b_fair_world16.py"
TRAJECTORY_CYCLES="${DISTILLSCHEDULE_TRAJECTORY_CYCLES:-0}"
TRAJECTORY_SEED="${DISTILLSCHEDULE_TRAJECTORY_SEED:-0}"
TRAJECTORY_ARM_ID="${DISTILLSCHEDULE_TRAJECTORY_ARM_ID:-}"
TWO_CYCLE_SEED="${DISTILLSCHEDULE_TWO_CYCLE_SEED:-0}"
TWO_CYCLE_ARM_ID="${DISTILLSCHEDULE_TWO_CYCLE_ARM_ID:-}"
if [[ "${MODE}" == trajectory ]]; then
  WORKER="${ROOT}/scripts/distillschedule/probe_wan13b_trajectory_world16.py"
  [[ "${TRAJECTORY_CYCLES}" == 1 || "${TRAJECTORY_CYCLES}" == 2 || \
     "${TRAJECTORY_CYCLES}" == 5 || \
     "${TRAJECTORY_CYCLES}" == 50 ]] || exit 2
  [[ "${TRAJECTORY_SEED}" =~ ^[0-9]+$ && -n "${TRAJECTORY_ARM_ID}" ]] || exit 2
fi
if [[ "${MODE}" == two-cycle-timing ]]; then
  WORKER="${ROOT}/scripts/distillschedule/probe_wan13b_two_cycle_timing_world16.py"
  [[ "${TWO_CYCLE_SEED}" =~ ^[0-9]+$ && -n "${TWO_CYCLE_ARM_ID}" ]] || exit 2
fi
EXPECTED_HELIOS_TRANSFORMER_SHA256="${DISTILLSCHEDULE_EXPECTED_HELIOS_TRANSFORMER_SHA256:-5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886}"
CORRECTNESS_RECEIPT="${DISTILLSCHEDULE_WAN13B_FAIR_CORRECTNESS_RECEIPT:-}"
CORRECTNESS_SHA256="${DISTILLSCHEDULE_WAN13B_FAIR_CORRECTNESS_SHA256:-}"
TRAJECTORY_COMPARISON_RECEIPT="${DISTILLSCHEDULE_WAN13B_TRAJECTORY_COMPARISON_RECEIPT:-}"
TRAJECTORY_COMPARISON_SHA256="${DISTILLSCHEDULE_WAN13B_TRAJECTORY_COMPARISON_SHA256:-}"
LORA_OPERATOR_ATTRIBUTION="${DISTILLSCHEDULE_WAN13B_LORA_OPERATOR_ATTRIBUTION:-0}"
LORA_ATTRIBUTION_TARGET_RANK="${DISTILLSCHEDULE_WAN13B_LORA_ATTRIBUTION_TARGET_RANK:-0}"
LORA_ATTRIBUTION_TARGET_NOGRAD_CALLS="${DISTILLSCHEDULE_WAN13B_LORA_ATTRIBUTION_TARGET_NOGRAD_CALLS:-3}"
GENERATOR_NOGRAD_LORA_CAST_CACHE="${DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_CAST_CACHE:-0}"
GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE="${DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE:-0}"
GRAD_LORA_FUSED_EPILOGUE="${DISTILLSCHEDULE_WAN13B_GRAD_LORA_FUSED_EPILOGUE:-0}"
[[ "${GRAD_LORA_FUSED_EPILOGUE}" == 0 || "${GRAD_LORA_FUSED_EPILOGUE}" == 1 ]] || exit 2
if [[ "${GRAD_LORA_FUSED_EPILOGUE}" == 1 && "${PLACEMENT}" != resident-async128-tree-chunk2048 ]]; then
  echo grad_epilogue_requires_registered_resident >&2; exit 2
fi

if [[ "${MODE}" == trajectory || "${MODE}" == two-cycle-timing ]]; then
  if [[ "${PLACEMENT}" == resident-async128-tree-chunk2048 ]]; then
    [[ "${GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE}" == 1 && \
       "${GENERATOR_NOGRAD_LORA_CAST_CACHE}" == 0 ]] || {
      echo registered_resident_winner_contract_mismatch >&2; exit 2;
    }
  elif [[ "${PLACEMENT}" == world16-fsdp ]]; then
    [[ "${GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE}" == 0 && \
       "${GENERATOR_NOGRAD_LORA_CAST_CACHE}" == 0 ]] || {
      echo registered_fsdp_baseline_contract_mismatch >&2; exit 2;
    }
  else
    echo registered_placement_not_supported >&2
    exit 2
  fi
fi

if [[ "${MODE}" == two-cycle-timing ]]; then
  [[ -f "${TRAJECTORY_COMPARISON_RECEIPT}" ]] || {
    echo trajectory_comparison_receipt_missing >&2; exit 3;
  }
  [[ "${TRAJECTORY_COMPARISON_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
    echo trajectory_comparison_hash_missing >&2; exit 3;
  }
  [[ "$(sha256sum "${TRAJECTORY_COMPARISON_RECEIPT}" | awk '{print $1}')" == \
      "${TRAJECTORY_COMPARISON_SHA256}" ]] || {
    echo trajectory_comparison_receipt_hash_changed >&2; exit 3;
  }
  "${PYTHON}" - "${TRAJECTORY_COMPARISON_RECEIPT}" "${TWO_CYCLE_ARM_ID}" \
    "${CORRECTNESS_RECEIPT}" <<'PY'
import json, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text())
arm_id = sys.argv[2]
correctness = json.loads(pathlib.Path(sys.argv[3]).read_text())
required = {
    "schema_version": "distillschedule.trajectory_coarse_comparison.v1",
    "comparison_kind": "semantic",
    "expected_cycles": 2,
    "expected_world_size": 16,
    "complete": True,
}
for key, expected in required.items():
    if value.get(key) != expected:
        raise SystemExit(f"trajectory comparison changed: {key}")
contract = value.get("contract", {})
if contract.get("status") != "PASS" or contract.get("mismatches"):
    raise SystemExit("trajectory comparison semantic contract is not PASS")
matches = []
for side_name in ("reference", "candidate"):
    side = value.get(side_name, {})
    if arm_id in side.get("variants", []):
        matches.append((side_name, side))
if len(matches) != 1:
    raise SystemExit("two-cycle arm must identify exactly one comparison side")
if matches[0][1].get("run_ids", []) != [correctness.get("run_id")]:
    raise SystemExit("trajectory comparison does not bind this correctness run")
PY
fi

if [[ "${LORA_OPERATOR_ATTRIBUTION}" == 1 && "${MODE}" != profile ]]; then
  echo lora_operator_attribution_requires_profile_mode >&2
  exit 2
fi

for executable in "${PYTHON}" "${TORCHRUN}"; do
  [[ -x "${executable}" ]] || { echo missing_executable="${executable}" >&2; exit 3; }
done
for path in "${ROOT}" "${HELIOS}" "${MODEL_ROOT}" "${BOOTSTRAP}" "${TRACE}" "${NEGATIVE}" "${WORKER}"; do
  [[ -e "${path}" ]] || { echo missing_path="${path}" >&2; exit 3; }
done
[[ "${BOOTSTRAP_HASH}" =~ ^[0-9a-f]{64}$ ]] || { echo invalid_bootstrap_hash >&2; exit 2; }
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_HEAD}" ]] || { echo source_head_mismatch >&2; exit 4; }
git -C "${ROOT}" diff --quiet && git -C "${ROOT}" diff --cached --quiet || { echo source_worktree_dirty >&2; exit 4; }
HELIOS_TRANSFORMER="${HELIOS}/helios/modules/transformer_helios.py"
[[ "$(sha256sum "${HELIOS_TRANSFORMER}" | awk '{print $1}')" == "${EXPECTED_HELIOS_TRANSFORMER_SHA256}" ]] || {
  echo helios_transformer_hash_mismatch >&2
  exit 4
}

if [[ "${MODE}" == timing || "${MODE}" == diagnostic-timing || \
      "${MODE}" == profile || "${MODE}" == two-cycle-timing ]]; then
  [[ -f "${CORRECTNESS_RECEIPT}" ]] || { echo correctness_receipt_missing >&2; exit 3; }
  [[ "${CORRECTNESS_SHA256}" =~ ^[0-9a-f]{64}$ ]] || { echo correctness_hash_missing >&2; exit 3; }
  [[ "$(sha256sum "${CORRECTNESS_RECEIPT}" | awk '{print $1}')" == "${CORRECTNESS_SHA256}" ]] || {
    echo correctness_receipt_hash_changed >&2
    exit 3
  }
  "${PYTHON}" - "${CORRECTNESS_RECEIPT}" "${PLACEMENT}" "${MODE}" \
    "${EXPECTED_HELIOS_TRANSFORMER_SHA256}" "${EXPECTED_HEAD}" \
    "${BOOTSTRAP_HASH}" "${TRACE_HASH}" "${TWO_CYCLE_SEED}" \
    "${TWO_CYCLE_ARM_ID}" <<'PY'
import json, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text())
placement = sys.argv[2]
mode = sys.argv[3]
expected_helios_sha256 = sys.argv[4]
expected_head, bootstrap_hash, trace_hash = sys.argv[5:8]
two_cycle_seed, two_cycle_arm_id = int(sys.argv[8]), sys.argv[9]
if mode == "two-cycle-timing":
    if value.get("schema_version") != "distillschedule.wan13b_trajectory_summary.v1":
        raise SystemExit("two-cycle timing requires a trajectory correctness summary")
    required = {
        "status": "PASS", "placement": placement, "cycles": 2,
        "rank_count": 16, "expected_head": expected_head,
        "diagnostic_only": True, "formal_timing_eligible": False,
        "final_versions": {"generator": 4, "critic": 20, "ema": 20},
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise SystemExit(f"two-cycle correctness binding changed: {key}")
    matched = value.get("matched_contract", {})
    expected_matched = {
        "bootstrap_hash": bootstrap_hash, "trace_bundle_hash": trace_hash,
        "trace_replay_policy": "FIXED_FIVE_STEP_COHORT_REPLAY",
        "seed": two_cycle_seed, "global_batch": 16,
        "updates": "G1_C5_EMA5",
        "rollout_batch_size": 1, "score_packing": False,
    }
    for key, expected in expected_matched.items():
        if matched.get(key) != expected:
            raise SystemExit(f"two-cycle matched workload changed: {key}")
    implementation = value.get("implementation_contract", {})
    if implementation.get("trajectory_arm_id") != two_cycle_arm_id:
        raise SystemExit("two-cycle arm differs from correctness arm")
    runtime = value.get("runtime_receipts", [])
    if len(runtime) != 16 or any(
        item.get("helios_transformer_sha256") != expected_helios_sha256
        for item in runtime
    ):
        raise SystemExit("two-cycle correctness uses a different Helios runtime")
elif value.get("identities", {}).get("helios_transformer_sha256") != expected_helios_sha256:
    raise SystemExit("correctness receipt uses a different Helios residual runtime")
if mode == "timing":
    if value.get("status") != "PASS" or not value.get("claimable_for_paired_timing"):
        raise SystemExit("fair numerical gate is not PASS")
    placements = {
        value.get("reference", {}).get("placement"),
        value.get("candidate", {}).get("placement"),
    }
    if placement not in placements:
        raise SystemExit(
            f"correctness receipt does not cover requested placement {placement}: {placements}"
        )
elif mode != "two-cycle-timing":
    if value.get("status") != "PASS" or value.get("mode") != "correctness":
        raise SystemExit("diagnostic timing requires a PASS semantic correctness summary")
    if value.get("placement") != placement or value.get("rank_count") != 16:
        raise SystemExit("diagnostic correctness receipt placement/rank mismatch")
    workload = value.get("workload", {})
    expected = {
        "world_size": 16,
        "global_batch": 16,
        "rollout_batch_size": 1,
        "score_packing": False,
        "updates": {"generator": 1, "critic": 5, "ema": 5},
        "final_versions": {"generator": 3, "critic": 15, "ema": 15},
        "unique_rank_step_rollout_identities": 80,
        "lora_compute_dtype": "torch.float32",
        "frozen_base_compute_dtype": "torch.bfloat16",
        "gradient_mean_convention": "SUM_THEN_DIVIDE_WORLD_SIZE",
    }
    for key, expected_value in expected.items():
        if workload.get(key) != expected_value:
            raise SystemExit(f"diagnostic semantic contract drift: {key}")
PY
fi

mkdir -p "${NODE_LOG}" "${EXECUTION}"
[[ ! -e "${NODE_LOG}/launch_manifest.txt" ]] || { echo run_id_already_used >&2; exit 82; }
exec 9>"${LOCK_FILE}"
flock -n 9 || { echo gpu_lock_busy >&2; exit 73; }
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')" ]] || {
  echo gpu_processes_present >&2
  exit 81
}

export PATH="${PYENV}/bin:${PATH}"
export PYTHONPATH="${ROOT}/src:${ROOT}:${HELIOS}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="${ASSETS}/hf_home_official_torch210"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 NCCL_ASYNC_ERROR_HANDLING=1
if [[ "${PLACEMENT}" == resident-async128-tree || \
      "${PLACEMENT}" == resident-async128-tree-chunk2048 || \
      "${PLACEMENT}" == resident-async128-tree-chunk2048-compile ]]; then
  export NCCL_ALGO=Tree
else
  unset NCCL_ALGO
fi
if [[ "${PLACEMENT}" == resident-async128-tree-chunk2048-compile ]]; then
  export DISTILLSCHEDULE_WAN13B_COMPILE_CACHE_ROOT="${DISTILLSCHEDULE_WAN13B_COMPILE_CACHE_ROOT:-/tmp/lyy-experiments/DMD-System/wan13b_compile_cache/${EXPECTED_HEAD}}"
else
  unset DISTILLSCHEDULE_WAN13B_COMPILE_CACHE_ROOT
fi
if [[ "${PLACEMENT}" == resident-fast || "${PLACEMENT}" == resident-local-fast || \
      "${PLACEMENT}" == resident-flat256 || "${PLACEMENT}" == resident-flat512 ]]; then
  export DISTILLSCHEDULE_WAN13B_EMA_IMPLEMENTATION=foreach
else
  export DISTILLSCHEDULE_WAN13B_EMA_IMPLEMENTATION=reference
fi

CGROUP_EVENTS="/sys/fs/cgroup/memory.events"
cp "${CGROUP_EVENTS}" "${NODE_LOG}/cgroup_memory_events_before.txt" 2>/dev/null || true
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader \
  >"${NODE_LOG}/gpu_before.csv"

optimizer_implementation=$( [[ "${PLACEMENT}" == resident-fast || "${PLACEMENT}" == resident-local-fast || "${PLACEMENT}" == resident-flat256 || "${PLACEMENT}" == resident-flat512 ]] && echo PYTORCH_FUSED || echo PYTORCH_DEFAULT )
gradient_clip_implementation=$( [[ "${PLACEMENT}" == resident-local-fast || "${PLACEMENT}" == resident-flat256 || "${PLACEMENT}" == resident-flat512 ]] && echo PYTORCH_FOREACH_TRUE || echo PYTORCH_DEFAULT )
ema_implementation=${DISTILLSCHEDULE_WAN13B_EMA_IMPLEMENTATION}
if [[ "${PLACEMENT}" == resident-lora-zero2 || "${PLACEMENT}" == resident-rollout-fused ]]; then
  optimizer_implementation=PYTORCH_FUSED_LOCAL_SHARD
  gradient_clip_implementation=GLOBAL_L2_FROM_LOCAL_SHARDS
  ema_implementation=WORLD16_LOCAL_FLAT_SHARD
fi
torch_compile_scope=$( [[ "${PLACEMENT}" == resident-async128-tree-chunk2048-compile ]] && echo GENERATOR_NOGRAD_TRANSFORMER_BLOCKS_ONLY || echo DISABLED )

cat >"${NODE_LOG}/launch_manifest.txt" <<EOF
schema_version=distillschedule.wan13b_fair_launch.v1
run_id=${RUN_ID}
mode=${MODE}
node_rank=${NODE_RANK}
dmd_system_head=${EXPECTED_HEAD}
helios_runtime_root=${HELIOS}
helios_transformer_sha256=${EXPECTED_HELIOS_TRANSFORMER_SHA256}
model_root=${MODEL_ROOT}
bootstrap=${BOOTSTRAP}
bootstrap_hash=${BOOTSTRAP_HASH}
trace=${TRACE}
trace_hash=${TRACE_HASH}
placement=${PLACEMENT}
world_size=16
global_batch=16
rollout_batch_size=1
score_packing=false
frozen_base_compute_dtype=torch.bfloat16
lora_compute_dtype=torch.float32
gradient_mean=SUM_THEN_DIVIDE_WORLD_SIZE
resident_ddp_bucket_mb=128
resident_async_bucket_mb=$( case "${PLACEMENT}" in resident-async128|resident-async128-tree|resident-async128-tree-chunk2048|resident-async128-tree-chunk2048-compile) echo 128 ;; resident-async256) echo 256 ;; *) echo NOT_APPLICABLE ;; esac )
resident_async_collective_order=$( [[ "${PLACEMENT}" == resident-async128 || "${PLACEMENT}" == resident-async128-tree || "${PLACEMENT}" == resident-async128-tree-chunk2048 || "${PLACEMENT}" == resident-async128-tree-chunk2048-compile || "${PLACEMENT}" == resident-async256 ]] && echo DETERMINISTIC_REVERSE_PARAMETER_BUCKET_ORDER || echo NOT_APPLICABLE )
modulation_chunk_tokens=$( case "${PLACEMENT}" in resident-async128-tree) echo 512 ;; resident-async128-tree-chunk2048|resident-async128-tree-chunk2048-compile) echo 2048 ;; *) echo RUNTIME_DEFAULT ;; esac )
nccl_algo=${NCCL_ALGO:-AUTO}
torch_compile_scope=${torch_compile_scope}
torch_compile_cache_root=${DISTILLSCHEDULE_WAN13B_COMPILE_CACHE_ROOT:-NOT_APPLICABLE}
lora_operator_attribution=${LORA_OPERATOR_ATTRIBUTION}
lora_attribution_target_rank=${LORA_ATTRIBUTION_TARGET_RANK}
lora_attribution_target_nograd_calls=${LORA_ATTRIBUTION_TARGET_NOGRAD_CALLS}
generator_nograd_lora_cast_cache=${GENERATOR_NOGRAD_LORA_CAST_CACHE}
generator_nograd_lora_fused_epilogue=${GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE}
grad_lora_fused_epilogue=${GRAD_LORA_FUSED_EPILOGUE}
tf32_cublas_override=${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-NOT_SET}
ema_implementation=${ema_implementation}
optimizer_implementation=${optimizer_implementation}
gradient_clip_implementation=${gradient_clip_implementation}
parameter_offload=false
activation_offload=false
optimizer_offload=false
parameter_streaming=false
model_swap=false
python_runtime_lib_path=${PYENV}/lib
claim_boundary=$( if [[ "${MODE}" == two-cycle-timing ]]; then echo STEADY_STATE_CYCLE2_UNPROFILED; elif [[ "${MODE}" == diagnostic-timing || "${MODE}" == profile || "${MODE}" == trajectory ]]; then echo DIAGNOSTIC_ONLY; else echo FORMAL_GATE_BOUND; fi )
EOF
if [[ "${MODE}" == trajectory ]]; then
  cat >>"${NODE_LOG}/launch_manifest.txt" <<EOF
trajectory_cycles=${TRAJECTORY_CYCLES}
trajectory_seed=${TRAJECTORY_SEED}
trajectory_arm_id=${TRAJECTORY_ARM_ID}
EOF
fi
if [[ "${MODE}" == two-cycle-timing ]]; then
  cat >>"${NODE_LOG}/launch_manifest.txt" <<EOF
two_cycle_count=2
two_cycle_claimable_cycle=2
two_cycle_trace_replay_policy=FIXED_FIVE_STEP_COHORT_REPLAY
two_cycle_seed=${TWO_CYCLE_SEED}
two_cycle_arm_id=${TWO_CYCLE_ARM_ID}
two_cycle_correctness_receipt=${CORRECTNESS_RECEIPT}
two_cycle_correctness_sha256=${CORRECTNESS_SHA256}
two_cycle_trajectory_comparison_receipt=${TRAJECTORY_COMPARISON_RECEIPT}
two_cycle_trajectory_comparison_sha256=${TRAJECTORY_COMPARISON_SHA256}
EOF
fi

WORKER_ARGS=(
  "${WORKER}"
  --run-id "${RUN_ID}"
  --output-dir "${EXECUTION}"
  --placement "${PLACEMENT}"
  --bootstrap "${BOOTSTRAP}"
  --expected-bootstrap-hash "${BOOTSTRAP_HASH}"
  --model-root "${MODEL_ROOT}"
  --trace-bundle "${TRACE}"
  --expected-trace-hash "${TRACE_HASH}"
  --negative-prompt "${NEGATIVE}"
)
[[ "${MODE}" == validate ]] && WORKER_ARGS+=(--validate-only)
[[ "${MODE}" == timing || "${MODE}" == diagnostic-timing ]] && WORKER_ARGS+=(--timing-mode)
[[ "${MODE}" == profile ]] && WORKER_ARGS+=(--profile-mode)
if [[ "${MODE}" == trajectory ]]; then
  WORKER_ARGS+=(
    --trajectory-cycles "${TRAJECTORY_CYCLES}"
    --trajectory-seed "${TRAJECTORY_SEED}"
    --trajectory-arm-id "${TRAJECTORY_ARM_ID}"
  )
fi
if [[ "${MODE}" == two-cycle-timing ]]; then
  WORKER_ARGS+=(
    --timing-seed "${TWO_CYCLE_SEED}"
    --timing-arm-id "${TWO_CYCLE_ARM_ID}"
    --expected-head "${EXPECTED_HEAD}"
    --correctness-receipt "${CORRECTNESS_RECEIPT}"
    --expected-correctness-hash "${CORRECTNESS_SHA256}"
    --trajectory-comparison-receipt "${TRAJECTORY_COMPARISON_RECEIPT}"
    --expected-trajectory-comparison-hash "${TRAJECTORY_COMPARISON_SHA256}"
  )
fi

set +e
if [[ "${MODE}" == profile ]]; then
  NSYS_BIN="${DISTILLSCHEDULE_NSYS_BIN:-$(command -v nsys || true)}"
  [[ -x "${NSYS_BIN}" ]] || { echo nsys_missing >&2; exit 3; }
  NSYS_VERSION="$(${NSYS_BIN} --version | head -1)"
  [[ "${NSYS_VERSION}" =~ 2024\.6\.2|2024\.[789]|202[5-9]\. ]] || {
    echo "nsys_version_not_admissible=${NSYS_VERSION}" >&2
    exit 3
  }
  PROFILE_ROOT="${RUN_ROOT}/nsight"
  PROFILE_NODE_DIR="${PROFILE_ROOT}/node_${NODE_RANK}"
  RECEIPT_DIR="${PROFILE_ROOT}/range_receipts"
  REPORT_BASE="${PROFILE_NODE_DIR}/wan13b_${PLACEMENT}_node${NODE_RANK}"
  mkdir -p "${PROFILE_NODE_DIR}" "${RECEIPT_DIR}"
  {
    echo "nsys_bin=${NSYS_BIN}"
    echo "nsys_version=${NSYS_VERSION}"
    echo "capture_mode=process_wide_clipped_to_exact_nvtx"
  } >"${PROFILE_NODE_DIR}/profiler_identity.txt"
  [[ ! -e "${REPORT_BASE}.nsys-rep" && ! -e "${REPORT_BASE}.sqlite" ]] || {
    echo nsys_artifact_exists >&2
    exit 82
  }
  export DISTILLSCHEDULE_STATIC_FSDP2_NSYS_EXACT_RANGE=1
  export DISTILLSCHEDULE_STATIC_FSDP2_NSYS_RECEIPT_DIR="${RECEIPT_DIR}"
  export DISTILLSCHEDULE_STATIC_FSDP2_NSYS_CONTROL_API=0
  "${NSYS_BIN}" profile \
    --trace=cuda,nvtx \
    --trace-fork-before-exec=true \
    --wait=primary \
    --sample=none \
    --cpuctxsw=none \
    --cuda-flush-interval=1000 \
    --force-overwrite=true \
    --output="${REPORT_BASE}" \
    "${TORCHRUN}" --no-python --nnodes=2 --nproc-per-node=8 --node-rank="${NODE_RANK}" \
      --master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}" \
      "${PYTHON}" "${WORKER_ARGS[@]}" >"${NODE_LOG}/console.log" 2>&1
  status=$?
  if [[ ${status} -eq 0 ]]; then
    [[ -s "${REPORT_BASE}.nsys-rep" ]] || { echo missing_nsys_report >&2; status=87; }
  fi
  if [[ ${status} -eq 0 ]]; then
    "${NSYS_BIN}" export --type sqlite --force-overwrite true \
      --output "${REPORT_BASE}.sqlite" "${REPORT_BASE}.nsys-rep" \
      >"${PROFILE_NODE_DIR}/nsys_export.log" 2>&1
    status=$?
  fi
  if [[ ${status} -eq 0 ]]; then
    [[ -s "${REPORT_BASE}.sqlite" ]] || { echo missing_nsys_sqlite >&2; status=88; }
  fi
  if [[ ${status} -eq 0 ]]; then
    "${NSYS_BIN}" stats --report nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum --format csv \
      "${REPORT_BASE}.sqlite" >"${PROFILE_NODE_DIR}/nsys_stats_validation.log" 2>&1
    status=$?
  fi
  if [[ ${status} -eq 0 ]] && grep -Eq \
      "ProcessEventsError|Unknown runtime API|Cannot find bucket|Cannot find string|Errors occurred while processing raw events|does not contain (NV Tools Extension|CUDA kernel|CUDA API)" \
      "${PROFILE_NODE_DIR}/nsys_export.log" "${PROFILE_NODE_DIR}/nsys_stats_validation.log"; then
    echo nsys_profile_validation_failed >&2
    status=89
  fi
  if [[ ${status} -eq 0 ]]; then
    for _ in $(seq 1 300); do
      count="$(find "${RECEIPT_DIR}" -maxdepth 1 -type f -name 'rank_*.json' | wc -l)"
      [[ "${count}" == 16 ]] && break
      sleep 1
    done
    count="$(find "${RECEIPT_DIR}" -maxdepth 1 -type f -name 'rank_*.json' | wc -l)"
    [[ "${count}" == 16 ]] || { echo "nsys_range_receipts=${count},expected=16" >&2; status=90; }
  fi
else
  "${TORCHRUN}" --nnodes=2 --nproc-per-node=8 --node-rank="${NODE_RANK}" \
    --master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}" \
    "${WORKER_ARGS[@]}" >"${NODE_LOG}/console.log" 2>&1
  status=$?
fi
set -e

cp "${CGROUP_EVENTS}" "${NODE_LOG}/cgroup_memory_events_after.txt" 2>/dev/null || true
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader \
  >"${NODE_LOG}/gpu_after.csv" || true
[[ ${status} -eq 0 ]] || exit "${status}"
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')" ]] || {
  echo gpu_processes_remain >&2
  exit 88
}
cat >"${NODE_LOG}/teardown_status.txt" <<EOF
status=PASS
gpu_process_count=0
owner_run_id=${RUN_ID}
EOF

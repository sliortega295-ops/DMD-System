#!/usr/bin/env bash
# Per-node fail-closed launcher for RB16-F5-CONT-8G4S4T-TO-2G6C.
set -Eeuo pipefail

NODE_RANK="${1:?node rank 0 or 1}"
RUN_ID="${2:?immutable run id}"
SCHEDULE="${3:?sequential or pipeline}"
MODE="${4:?correctness or timing}"
EXPECTED_HEAD="${5:?expected DMD-System HEAD}"
MASTER_ADDR="${6:?master address}"
MASTER_PORT="${7:?master port}"
EXPECTED_RUNTIME_HEAD="${8:?expected Runtime HEAD}"
STOP_AFTER_STEP10A="${DISTILLSCHEDULE_STOP_AFTER_STEP10A:-0}"
STEP10A_NSYS="${DISTILLSCHEDULE_STEP10A_NSYS:-0}"
STEP10A_PARAM_PERSISTENCE_THRESHOLD="${DISTILLSCHEDULE_STEP10A_PARAM_PERSISTENCE_THRESHOLD:-0}"
STEP10A_MODULE_GRANULARITY_THRESHOLD="${DISTILLSCHEDULE_STEP10A_MODULE_GRANULARITY_THRESHOLD:-0}"
STEP10A_BLOCK_LEAF_FETCH="${DISTILLSCHEDULE_STEP10A_BLOCK_LEAF_FETCH:-0}"
STEP10A_PREFETCH_BUCKET_ELEMENTS="${DISTILLSCHEDULE_STEP10A_PREFETCH_BUCKET_ELEMENTS:-16777216}"
STEP10A_MAX_LIVE_PARAMETERS="${DISTILLSCHEDULE_STEP10A_MAX_LIVE_PARAMETERS:-268435456}"
STEP10A_MAX_REUSE_DISTANCE="${DISTILLSCHEDULE_STEP10A_MAX_REUSE_DISTANCE:-268435456}"
STEP10A_OVERLAP_COMM="${DISTILLSCHEDULE_STEP10A_OVERLAP_COMM:-0}"
STEP10A_SAVED_TENSOR_CENSUS="${DISTILLSCHEDULE_STEP10A_SAVED_TENSOR_CENSUS:-0}"
STEP10A_BOUNDARY_PINNED_SYNC="${DISTILLSCHEDULE_STEP10A_BOUNDARY_PINNED_SYNC:-0}"
TRANSITION_H2D_POLICY="${DISTILLSCHEDULE_TRANSITION_H2D_POLICY:-module_to}"
TRANSITION_H2D_CHUNK_MIB="${DISTILLSCHEDULE_TRANSITION_H2D_CHUNK_MIB:-64}"
TRANSITION_H2D_SLOT_COUNT="${DISTILLSCHEDULE_TRANSITION_H2D_SLOT_COUNT:-2}"
TRANSITION_PINNED_CLEANUP_POLICY="${DISTILLSCHEDULE_TRANSITION_PINNED_CLEANUP_POLICY:-immediate}"
G3_BROADCAST_BUCKET_MIB="${DISTILLSCHEDULE_G3_BROADCAST_BUCKET_MIB:-0}"
[[ "${NODE_RANK}" == 0 || "${NODE_RANK}" == 1 ]] || exit 2
[[ "${SCHEDULE}" == sequential || "${SCHEDULE}" == pipeline ]] || exit 2
[[ "${MODE}" == correctness || "${MODE}" == timing ]] || exit 2
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || exit 2
[[ "${MASTER_PORT}" =~ ^[0-9]+$ ]] || exit 2
[[ "${STOP_AFTER_STEP10A}" == 0 || "${STOP_AFTER_STEP10A}" == 1 ]] || exit 2
[[ "${STEP10A_NSYS}" == 0 || "${STEP10A_NSYS}" == 1 ]] || exit 2
[[ "${STEP10A_PARAM_PERSISTENCE_THRESHOLD}" =~ ^[0-9]+$ ]] || exit 2
[[ "${STEP10A_MODULE_GRANULARITY_THRESHOLD}" =~ ^[0-9]+$ ]] || exit 2
[[ "${STEP10A_BLOCK_LEAF_FETCH}" == 0 || "${STEP10A_BLOCK_LEAF_FETCH}" == 1 ]] || exit 2
[[ "${STEP10A_PREFETCH_BUCKET_ELEMENTS}" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "${STEP10A_MAX_LIVE_PARAMETERS}" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "${STEP10A_MAX_REUSE_DISTANCE}" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "${STEP10A_OVERLAP_COMM}" == 0 || "${STEP10A_OVERLAP_COMM}" == 1 ]] || exit 2
[[ "${STEP10A_SAVED_TENSOR_CENSUS}" == 0 || "${STEP10A_SAVED_TENSOR_CENSUS}" == 1 ]] || exit 2
[[ "${STEP10A_BOUNDARY_PINNED_SYNC}" == 0 || "${STEP10A_BOUNDARY_PINNED_SYNC}" == 1 ]] || exit 2
[[ "${TRANSITION_H2D_POLICY}" == module_to || "${TRANSITION_H2D_POLICY}" == bounded_pinned || "${TRANSITION_H2D_POLICY}" == persistent_pinned ]] || exit 2
[[ "${TRANSITION_H2D_CHUNK_MIB}" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "${TRANSITION_H2D_SLOT_COUNT}" == 1 || "${TRANSITION_H2D_SLOT_COUNT}" == 2 ]] || exit 2
[[ "${TRANSITION_PINNED_CLEANUP_POLICY}" == immediate || "${TRANSITION_PINNED_CLEANUP_POLICY}" == deferred || "${TRANSITION_PINNED_CLEANUP_POLICY}" == late_deferred || "${TRANSITION_PINNED_CLEANUP_POLICY}" == post_full5 ]] || exit 2
[[ "${G3_BROADCAST_BUCKET_MIB}" =~ ^[0-9]+$ ]] || exit 2
if [[ "${STOP_AFTER_STEP10A}" == 1 && "${MODE}" != timing ]]; then
  echo stop_after_step10a_requires_timing_mode >&2
  exit 2
fi
if [[ "${STEP10A_NSYS}" == 1 && ( "${NODE_RANK}" != 0 || "${STOP_AFTER_STEP10A}" != 1 ) ]]; then
  echo step10a_nsys_requires_node0_bounded_diagnostic >&2
  exit 2
fi
if [[ "${STEP10A_SAVED_TENSOR_CENSUS}" == 1 && ( "${STOP_AFTER_STEP10A}" != 1 || "${MODE}" != timing ) ]]; then
  echo step10a_saved_tensor_census_requires_bounded_timing_mode >&2
  exit 2
fi
if [[ "${STEP10A_SAVED_TENSOR_CENSUS}" == 1 && "${STEP10A_BOUNDARY_PINNED_SYNC}" == 1 ]]; then
  echo step10a_census_and_boundary_pinning_are_mutually_exclusive >&2
  exit 2
fi
if [[ "${STEP10A_BLOCK_LEAF_FETCH}" == 1 && "${STEP10A_MODULE_GRANULARITY_THRESHOLD}" != 0 ]]; then
  echo step10a_block_leaf_and_threshold_are_mutually_exclusive >&2
  exit 2
fi

ROOT="${DMD_SYSTEM_ROOT:-/share/project/shuchenweng/yongyan_liu/DMD-System_rb16_f5_pipeline_20260809}"
RUNTIME="${HELIOS_RUNTIME_ROOT:-/share/project/shuchenweng/yongyan_liu/DistillSchedule_rb16_f5_pipeline_20260809}"
PYENV="${DISTILLSCHEDULE_PYENV:-/tmp/distillschedule-helios-official-torch210}"
PYTHON="${PYENV}/bin/python"
TORCHRUN="${PYENV}/bin/torchrun"
ASSETS="${DISTILLSCHEDULE_ASSETS:-/share/project/shuchenweng/DistillSchedule_assets}"
TRACE="${DISTILLSCHEDULE_TRACE_BUNDLE:-${ASSETS}/m0/traces/syn_feature_trace_rank16_v4_strict_sync_steps20}"
TRACE_HASH="${DISTILLSCHEDULE_TRACE_BUNDLE_HASH:-654d49015921b278ab82e2fc84ea1896692da3fd38ce82fbc500cb1ebfec46d3}"
LOGICAL="${DISTILLSCHEDULE_LOGICAL_STEP10:-${ASSETS}/m1/checkpoints/logical_world16_step10_v3}"
LOGICAL_HASH="${DISTILLSCHEDULE_LOGICAL_STEP10_HASH:-68eed293970be008764ee6acadb2aac6ce54de2ce1f55a17fca2246f0d04e4aa}"
NATIVE="${DISTILLSCHEDULE_NATIVE_WORLD8_STEP10:-${ASSETS}/m1/checkpoints/import_world8_step10_v1}"
GENERATOR="${HELIOS_DISTILLED_ROOT:-${ASSETS}/Helios-Distilled}"
PRECAST_GENERATOR="${DISTILLSCHEDULE_PRECAST_GENERATOR_PATH:-${ASSETS}/precast/helios-distilled-transformer-ode-mixed-bf16-v1}"
SCORE="${HELIOS_BASE_ROOT:-/home/shuchenweng/xty/Helios-main/weights}"
PRECAST_SCORE="${DISTILLSCHEDULE_PRECAST_SCORE_PATH:-${ASSETS}/precast/helios-score-transformer-mixed-bf16-v1}"
NEGATIVE="${DISTILLSCHEDULE_NEGATIVE_PROMPT_ASSET:-${ASSETS}/m1/negative_prompt_official_v1}"
CORRECTNESS_EVALUATION="${DISTILLSCHEDULE_CANDIDATE_B_CORRECTNESS_EVALUATION:-}"
CORRECTNESS_SHA="${DISTILLSCHEDULE_CANDIDATE_B_CORRECTNESS_SHA:-}"
RUN_ROOT="${DISTILLSCHEDULE_RUN_ROOT:-${ROOT}/artifacts/runs/${RUN_ID}}"
NODE_LOG="${RUN_ROOT}/launcher/node_${NODE_RANK}"
EXECUTION="${RUN_ROOT}/execution"
PREFLIGHT="${RUN_ROOT}/preflight"
LOCK_FILE="/tmp/distillschedule_gpu_node.lock"
NODE_CACHE="/tmp/distillschedule_full5_2g6c/${RUN_ID}/node_${NODE_RANK}"
ACTUAL_PORT="$((MASTER_PORT + 1))"

for executable in "${PYTHON}" "${TORCHRUN}"; do
  [[ -x "${executable}" ]] || { echo missing_executable="${executable}" >&2; exit 3; }
done
for path in "${ROOT}" "${RUNTIME}" "${TRACE}" "${LOGICAL}" "${NATIVE}" \
  "${GENERATOR}" "${PRECAST_GENERATOR}" "${SCORE}" "${PRECAST_SCORE}" "${NEGATIVE}"; do
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
[[ ! -e "${NODE_CACHE}" ]] || { echo node_cache_exists >&2; exit 82; }
mkdir -p "${NODE_CACHE}/mmap"
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
  echo schema_version=distillschedule.rb16_full5_2g6c_continuous_node_launch.v1
  echo experiment_id=RB16-F5-CONT-8G4S4T-TO-2G6C
  echo claim_scope=continuous_full5_batch12_pareto_not_matched_batch16_speedup
  echo run_id="${RUN_ID}"
  echo schedule="${SCHEDULE}"
  echo mode="${MODE}"
  echo stop_after_step10a="${STOP_AFTER_STEP10A}"
  echo step10a_nsys="${STEP10A_NSYS}"
  echo step10a_param_persistence_threshold="${STEP10A_PARAM_PERSISTENCE_THRESHOLD}"
  echo step10a_module_granularity_threshold="${STEP10A_MODULE_GRANULARITY_THRESHOLD}"
  echo step10a_block_leaf_fetch="${STEP10A_BLOCK_LEAF_FETCH}"
  echo step10a_prefetch_bucket_elements="${STEP10A_PREFETCH_BUCKET_ELEMENTS}"
  echo step10a_max_live_parameters="${STEP10A_MAX_LIVE_PARAMETERS}"
  echo step10a_max_reuse_distance="${STEP10A_MAX_REUSE_DISTANCE}"
  echo step10a_overlap_comm="${STEP10A_OVERLAP_COMM}"
  echo step10a_saved_tensor_census="${STEP10A_SAVED_TENSOR_CENSUS}"
  echo step10a_boundary_pinned_sync="${STEP10A_BOUNDARY_PINNED_SYNC}"
  echo transition_h2d_policy="${TRANSITION_H2D_POLICY}"
  echo transition_h2d_chunk_mib="${TRANSITION_H2D_CHUNK_MIB}"
  echo transition_h2d_slot_count="${TRANSITION_H2D_SLOT_COUNT}"
  echo transition_pinned_cleanup_policy="${TRANSITION_PINNED_CLEANUP_POLICY}"
  echo g3_broadcast_bucket_mib="${G3_BROADCAST_BUCKET_MIB}"
  echo node_rank="${NODE_RANK}"
  echo host="$(hostname)"
  echo started_at="$(date -Is)"
  echo dmd_system_head="${EXPECTED_HEAD}"
  echo runtime_head="${EXPECTED_RUNTIME_HEAD}"
  echo trace_bundle="${TRACE}"
  echo trace_bundle_hash="${TRACE_HASH}"
  echo logical_package="${LOGICAL}"
  echo logical_manifest_hash="${LOGICAL_HASH}"
  echo native_world8_step10_provenance_only="${NATIVE}"
  echo negative_prompt_asset="${NEGATIVE}"
  echo step10a_topology=8G_4S_4T
  echo step10a_generator_backend=ZeRO3_DP8_MB1_GAS2_global16
  echo step10a_process_world=node0_default_world8
  echo step10a_score_process_world=node1_default_world8
  echo step10a_generator_parameter_offload=none
  echo step10a_generator_optimizer_offload=cpu
  echo step10a_checkpoint_policy=zero3_metadata_tolerant_block_local_non_reentrant
  echo target_topology=per_node_2G6C_global_4G12C
  echo target_process_world=explicit_global_world16_after_phase_a_destroy
  echo critic_batch=DP12_MB1_GAS1_global12
  echo construction_policy=concurrent_all_local_ranks
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

"${PYTHON}" "${ROOT}/scripts/distillschedule/validate_routeb_full5_inputs.py" \
  --trace-bundle "${TRACE}" --expected-trace-bundle-hash "${TRACE_HASH}" \
  --logical-package "${LOGICAL}" --expected-logical-manifest-hash "${LOGICAL_HASH}" \
  --native-world8-root "${NATIVE}" \
  --generator-root "${GENERATOR}" --precast-generator-root "${PRECAST_GENERATOR}" \
  --score-root "${SCORE}" --precast-score-root "${PRECAST_SCORE}" \
  --negative-prompt-asset "${NEGATIVE}" \
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
    raise SystemExit("continuous Full5 node input contracts differ or failed")
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
  "${TORCHRUN}" --standalone --nproc-per-node=8
  "${ROOT}/scripts/distillschedule/probe_routeb_full5_2g6c_continuous.py"
  --node-rank "${NODE_RANK}"
  --phase-b-master-addr "${MASTER_ADDR}"
  --phase-b-master-port "${ACTUAL_PORT}"
  --schedule "${SCHEDULE}" --run-id "${RUN_ID}" --run-nonce "${RUN_NONCE}"
  --output-dir "${EXECUTION}" --trace-bundle "${TRACE}"
  --logical-package "${LOGICAL}" --expected-logical-manifest-hash "${LOGICAL_HASH}"
  --native-world8-root "${NATIVE}"
  --generator-root "${GENERATOR}" --precast-generator-root "${PRECAST_GENERATOR}"
  --score-root "${SCORE}" --precast-score-root "${PRECAST_SCORE}"
  --negative-prompt-asset "${NEGATIVE}" --mmap-cache-root "${NODE_CACHE}/mmap"
  --activation-min-bytes $((32 * 1024 * 1024))
  --step10a-param-persistence-threshold "${STEP10A_PARAM_PERSISTENCE_THRESHOLD}"
  --step10a-module-granularity-threshold "${STEP10A_MODULE_GRANULARITY_THRESHOLD}"
  --step10a-prefetch-bucket-elements "${STEP10A_PREFETCH_BUCKET_ELEMENTS}"
  --step10a-max-live-parameters "${STEP10A_MAX_LIVE_PARAMETERS}"
  --step10a-max-reuse-distance "${STEP10A_MAX_REUSE_DISTANCE}"
  --transition-h2d-policy "${TRANSITION_H2D_POLICY}"
  --transition-h2d-chunk-mib "${TRANSITION_H2D_CHUNK_MIB}"
  --transition-h2d-slot-count "${TRANSITION_H2D_SLOT_COUNT}"
  --transition-pinned-cleanup-policy "${TRANSITION_PINNED_CLEANUP_POLICY}"
  --g3-broadcast-bucket-mib "${G3_BROADCAST_BUCKET_MIB}"
)
if [[ "${STEP10A_BLOCK_LEAF_FETCH}" == 1 ]]; then
  WORKER+=(--step10a-block-leaf-fetch)
fi
if [[ "${STEP10A_OVERLAP_COMM}" == 1 ]]; then
  WORKER+=(--step10a-overlap-comm)
fi
if [[ "${STEP10A_SAVED_TENSOR_CENSUS}" == 1 ]]; then
  WORKER+=(--step10a-saved-tensor-census)
fi
if [[ "${STEP10A_BOUNDARY_PINNED_SYNC}" == 1 ]]; then
  WORKER+=(--step10a-boundary-pinned-sync)
fi
if [[ "${MODE}" == correctness ]]; then
  WORKER+=(--checksum sha256 --export-state)
else
  WORKER+=(
    --timing-mode --checksum gpu
    --correctness-evaluation "${CORRECTNESS_EVALUATION}"
    --expected-correctness-sha256 "${CORRECTNESS_SHA}"
  )
fi
if [[ "${STOP_AFTER_STEP10A}" == 1 ]]; then
  WORKER+=(--stop-after-step10a)
fi
if [[ "${STEP10A_NSYS}" == 1 ]]; then
  NSYS_REPORT_BASE="${NODE_LOG}/step10a_nsys"
  [[ ! -e "${NSYS_REPORT_BASE}.nsys-rep" ]] || { echo nsys_report_exists >&2; exit 82; }
  export DISTILLSCHEDULE_NSYS_ANNOTATE=1
  export DISTILLSCHEDULE_STEP10A_CUDA_PROFILER=1
  WORKER=(
    /usr/local/cuda/bin/nsys profile
    --trace=cuda,nvtx,osrt
    --capture-range=cudaProfilerApi
    --capture-range-end=stop
    --trace-fork-before-exec=true
    --sample=none
    --cpuctxsw=none
    --force-overwrite=false
    --output="${NSYS_REPORT_BASE}"
    "${WORKER[@]}"
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

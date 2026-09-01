#!/usr/bin/env bash
# One node of the immutable LiveWan cross-scale world16 experiment.
set -Eeuo pipefail

NODE_RANK="${1:?node rank 0 or 1}"
RUN_ID="${2:?immutable run id}"
RUN_KIND="${3:?validate, capacity, correctness, timing, or profile}"
EXECUTION_MODE="${4:?baseline or pipeline}"
EXPECTED_HEAD="${5:?exact DMD-System HEAD}"
MASTER_ADDR="${6:?node0 address}"
MASTER_PORT="${7:?rendezvous port}"

[[ "${NODE_RANK}" == 0 || "${NODE_RANK}" == 1 ]] || exit 2
[[ "${RUN_KIND}" == validate || "${RUN_KIND}" == capacity || \
   "${RUN_KIND}" == correctness || \
   "${RUN_KIND}" == timing || "${RUN_KIND}" == profile ]] || exit 2
[[ "${EXECUTION_MODE}" == baseline || "${EXECUTION_MODE}" == pipeline ]] || exit 2
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || exit 2
[[ "${EXPECTED_HEAD}" =~ ^[0-9a-f]{40}$ ]] || exit 2

ROOT="${DMD_SYSTEM_ROOT:?set DMD_SYSTEM_ROOT}"
# These job containers enter as root while the persistent shared worktree is
# owned by its original uid.  Tell Git about that exact owner for this launcher
# process only; do not mutate /root/.gitconfig or the shared checkout owner.
if [[ "$(id -u)" == 0 && -z "${SUDO_UID:-}" ]]; then
  export SUDO_UID="$(stat -c '%u' "${ROOT}")"
fi
PYENV="${DISTILLSCHEDULE_PYENV:-/tmp/distillschedule-helios-official-torch210}"
PYTHON_OVERLAY="${LIVEWAN_PYTHON_OVERLAY:?set isolated LIVEWAN_PYTHON_OVERLAY}"
PYTHON="${PYENV}/bin/python"
TORCHRUN="${PYENV}/bin/torchrun"
RUN_ROOT="${DISTILLSCHEDULE_OUTPUT_DIR:?set DISTILLSCHEDULE_OUTPUT_DIR}"
NODE_LOG="${RUN_ROOT}/launcher/node_${NODE_RANK}"
LOCK_FILE="${DISTILLSCHEDULE_NODE_LOCK:-/tmp/distillschedule_gpu_node.lock}"
SOURCE_ROOT="${LIVEWAN_WAN21_SOURCE:?set LIVEWAN_WAN21_SOURCE}"
SOURCE_RECEIPT="${LIVEWAN_SOURCE_RECEIPT:?set LIVEWAN_SOURCE_RECEIPT}"
PIPELINE_VARIANT="${LIVEWAN_PIPELINE_VARIANT:-b1}"
[[ "${PIPELINE_VARIANT}" == b1 || "${PIPELINE_VARIANT}" == b1-runahead || \
   "${PIPELINE_VARIANT}" == b1-runahead-critic-b2 || \
   "${PIPELINE_VARIANT}" == b1-runahead-critic-b2-local-context || \
   "${PIPELINE_VARIANT}" == b1-runahead-critic-b2-local-context-async32 || \
   "${PIPELINE_VARIANT}" == b1-runahead-critic-b2-local-context-async32-trace-balanced || \
   "${PIPELINE_VARIANT}" == b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context || \
   "${PIPELINE_VARIANT}" == b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-context-prefetch || \
   "${PIPELINE_VARIANT}" == b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-scale1 || \
   "${PIPELINE_VARIANT}" == b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-scale1-modulation || \
   "${PIPELINE_VARIANT}" == b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-fused-cast-add || \
   "${PIPELINE_VARIANT}" == b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-autocast-input-elision || \
   "${PIPELINE_VARIANT}" == b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-autocast-input-elision-split-dmd-shadow || \
   "${PIPELINE_VARIANT}" == b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-autocast-input-elision-split-dmd-shadow-fused-adam || \
   "${PIPELINE_VARIANT}" == c15-score-rope || \
   "${PIPELINE_VARIANT}" == c16-score-flash-attn2 || \
	   "${PIPELINE_VARIANT}" == c18-generator-cross-kv-cache || \
	   "${PIPELINE_VARIANT}" == c19-host-runahead || \
	   "${PIPELINE_VARIANT}" == c20-generator-flash-attn2 || \
	   "${PIPELINE_VARIANT}" == c21-continuous-b16 || \
	   "${PIPELINE_VARIANT}" == c21-c1-continuous-b16 || \
   "${PIPELINE_VARIANT}" == phase-expanded-critic16 || \
   "${PIPELINE_VARIANT}" == phase-expanded-critic16-fused-lora-elision || \
   "${PIPELINE_VARIANT}" == score-b2 || \
   "${PIPELINE_VARIANT}" == 6g10score || \
   "${PIPELINE_VARIANT}" == 6g10score-node-local || \
   "${PIPELINE_VARIANT}" == 6g10score-node-local-fused ]] || exit 2

if [[ "${EXECUTION_MODE}" == baseline ]]; then
  [[ "${PIPELINE_VARIANT}" == b1 ]] || { echo baseline_pipeline_variant_mismatch >&2; exit 2; }
  PLAN="${ROOT}/configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_fsdp2.json"
elif [[ "${PIPELINE_VARIANT}" == score-b2 ]]; then
  PLAN="${ROOT}/configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_scoreb2.json"
elif [[ "${PIPELINE_VARIANT}" == b1-runahead ]]; then
  PLAN="${ROOT}/configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead.json"
elif [[ "${PIPELINE_VARIANT}" == b1-runahead-critic-b2 ]]; then
  PLAN="${ROOT}/configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2.json"
elif [[ "${PIPELINE_VARIANT}" == b1-runahead-critic-b2-local-context ]]; then
  PLAN="${ROOT}/configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context.json"
elif [[ "${PIPELINE_VARIANT}" == b1-runahead-critic-b2-local-context-async32 ]]; then
  PLAN="${ROOT}/configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32.json"
elif [[ "${PIPELINE_VARIANT}" == b1-runahead-critic-b2-local-context-async32-trace-balanced ]]; then
  PLAN="${ROOT}/configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced.json"
elif [[ "${PIPELINE_VARIANT}" == b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context ]]; then
  PLAN="${ROOT}/configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context.json"
elif [[ "${PIPELINE_VARIANT}" == b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-context-prefetch ]]; then
  PLAN="${ROOT}/configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_context_prefetch.json"
elif [[ "${PIPELINE_VARIANT}" == b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-scale1 ]]; then
  PLAN="${ROOT}/configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_scale1.json"
elif [[ "${PIPELINE_VARIANT}" == b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-scale1-modulation ]]; then
  PLAN="${ROOT}/configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_scale1_modulation.json"
elif [[ "${PIPELINE_VARIANT}" == b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-fused-cast-add ]]; then
  PLAN="${ROOT}/configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_fused_cast_add.json"
elif [[ "${PIPELINE_VARIANT}" == b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-autocast-input-elision ]]; then
  PLAN="${ROOT}/configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_autocast_input_elision.json"
elif [[ "${PIPELINE_VARIANT}" == b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-autocast-input-elision-split-dmd-shadow ]]; then
  PLAN="${ROOT}/configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_autocast_input_elision_split_dmd_shadow.json"
elif [[ "${PIPELINE_VARIANT}" == b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-autocast-input-elision-split-dmd-shadow-fused-adam ]]; then
  PLAN="${ROOT}/configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_autocast_input_elision_split_dmd_shadow_fused_adam.json"
elif [[ "${PIPELINE_VARIANT}" == c15-score-rope ]]; then
  PLAN="${ROOT}/configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_autocast_input_elision_split_dmd_shadow_fused_adam_score_rope.json"
elif [[ "${PIPELINE_VARIANT}" == c16-score-flash-attn2 ]]; then
  PLAN="${ROOT}/configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_autocast_input_elision_split_dmd_shadow_fused_adam_score_rope_flash_attn2.json"
  export DISTILLSCHEDULE_FLASH_ATTN_KERNEL="${DISTILLSCHEDULE_FLASH_ATTN_KERNEL:-/share/project/shuchenweng/DistillSchedule_assets/hf_home_official_torch210/hub/models--kernels-community--flash-attn2/snapshots/6a810a7a010ba2aad6613f1f0f7b226fb8bdee3c/build/torch210-cxx11-cu128-x86_64-linux}"
  [[ -f "${DISTILLSCHEDULE_FLASH_ATTN_KERNEL}/_flash_attn2_cuda_a511a4c.abi3.so" ]] || {
    echo "missing_flash_attn2_artifact=${DISTILLSCHEDULE_FLASH_ATTN_KERNEL}" >&2; exit 3;
  }
elif [[ "${PIPELINE_VARIANT}" == c18-generator-cross-kv-cache ]]; then
  PLAN="${ROOT}/configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_autocast_input_elision_split_dmd_shadow_fused_adam_score_rope_flash_attn2_generator_cross_kv_cache.json"
  export DISTILLSCHEDULE_FLASH_ATTN_KERNEL="${DISTILLSCHEDULE_FLASH_ATTN_KERNEL:-/share/project/shuchenweng/DistillSchedule_assets/hf_home_official_torch210/hub/models--kernels-community--flash-attn2/snapshots/6a810a7a010ba2aad6613f1f0f7b226fb8bdee3c/build/torch210-cxx11-cu128-x86_64-linux}"
  [[ -f "${DISTILLSCHEDULE_FLASH_ATTN_KERNEL}/_flash_attn2_cuda_a511a4c.abi3.so" ]] || {
    echo "missing_flash_attn2_artifact=${DISTILLSCHEDULE_FLASH_ATTN_KERNEL}" >&2; exit 3;
  }
elif [[ "${PIPELINE_VARIANT}" == c19-host-runahead ]]; then
  PLAN="${ROOT}/configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_autocast_input_elision_split_dmd_shadow_fused_adam_score_rope_flash_attn2_generator_cross_kv_cache_host_runahead.json"
  export DISTILLSCHEDULE_FLASH_ATTN_KERNEL="${DISTILLSCHEDULE_FLASH_ATTN_KERNEL:-/share/project/shuchenweng/DistillSchedule_assets/hf_home_official_torch210/hub/models--kernels-community--flash-attn2/snapshots/6a810a7a010ba2aad6613f1f0f7b226fb8bdee3c/build/torch210-cxx11-cu128-x86_64-linux}"
  [[ -f "${DISTILLSCHEDULE_FLASH_ATTN_KERNEL}/_flash_attn2_cuda_a511a4c.abi3.so" ]] || {
    echo "missing_flash_attn2_artifact=${DISTILLSCHEDULE_FLASH_ATTN_KERNEL}" >&2; exit 3;
  }
elif [[ "${PIPELINE_VARIANT}" == c20-generator-flash-attn2 ]]; then
  PLAN="${ROOT}/configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_c20_generator_flash_attn2.json"
elif [[ "${PIPELINE_VARIANT}" == c21-continuous-b16 ]]; then
  PLAN="${ROOT}/configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_c21_continuous_b16.json"
  export DISTILLSCHEDULE_FLASH_ATTN_KERNEL="${DISTILLSCHEDULE_FLASH_ATTN_KERNEL:-/share/project/shuchenweng/DistillSchedule_assets/hf_home_official_torch210/hub/models--kernels-community--flash-attn2/snapshots/6a810a7a010ba2aad6613f1f0f7b226fb8bdee3c/build/torch210-cxx11-cu128-x86_64-linux}"
  [[ -f "${DISTILLSCHEDULE_FLASH_ATTN_KERNEL}/_flash_attn2_cuda_a511a4c.abi3.so" ]] || {
    echo "missing_flash_attn2_artifact=${DISTILLSCHEDULE_FLASH_ATTN_KERNEL}" >&2; exit 3;
  }
elif [[ "${PIPELINE_VARIANT}" == c21-c1-continuous-b16 ]]; then
  PLAN="${ROOT}/configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_c21_c1_continuous_b16.json"
  export DISTILLSCHEDULE_FLASH_ATTN_KERNEL="${DISTILLSCHEDULE_FLASH_ATTN_KERNEL:-/share/project/shuchenweng/DistillSchedule_assets/hf_home_official_torch210/hub/models--kernels-community--flash-attn2/snapshots/6a810a7a010ba2aad6613f1f0f7b226fb8bdee3c/build/torch210-cxx11-cu128-x86_64-linux}"
  [[ -f "${DISTILLSCHEDULE_FLASH_ATTN_KERNEL}/_flash_attn2_cuda_a511a4c.abi3.so" ]] || {
    echo "missing_flash_attn2_artifact=${DISTILLSCHEDULE_FLASH_ATTN_KERNEL}" >&2; exit 3;
  }
elif [[ "${PIPELINE_VARIANT}" == phase-expanded-critic16 ]]; then
  PLAN="${ROOT}/configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_phase_expanded_critic16.json"
elif [[ "${PIPELINE_VARIANT}" == phase-expanded-critic16-fused-lora-elision ]]; then
  PLAN="${ROOT}/configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_phase_expanded_critic16_fused_lora_elision.json"
elif [[ "${PIPELINE_VARIANT}" == 6g10score ]]; then
  PLAN="${ROOT}/configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline6g10score.json"
elif [[ "${PIPELINE_VARIANT}" == 6g10score-node-local ]]; then
  PLAN="${ROOT}/configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline6g10score_node_local.json"
elif [[ "${PIPELINE_VARIANT}" == 6g10score-node-local-fused ]]; then
  PLAN="${ROOT}/configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline6g10score_node_local_fused.json"
else
  PLAN="${ROOT}/configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8.json"
fi

for executable in "${PYTHON}" "${TORCHRUN}"; do
  [[ -x "${executable}" ]] || { echo "missing_executable=${executable}" >&2; exit 3; }
done
[[ -f "${PYTHON_OVERLAY}/.complete.json" ]] || {
  echo "missing_python_overlay_receipt=${PYTHON_OVERLAY}/.complete.json" >&2; exit 3;
}
for path in "${ROOT}" "${PLAN}" "${LIVEWAN_STUDENT_CHECKPOINT:?}" \
  "${LIVEWAN_CRITIC_CHECKPOINT:?}" "${LIVEWAN_TEACHER_CHECKPOINT:?}" \
  "${LIVEWAN_OPERATION_TRACE:?}" "${LIVEWAN_SYSTEMS_BOOTSTRAP:?}"; do
  [[ -e "${path}" ]] || { echo "missing_path=${path}" >&2; exit 3; }
done
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_HEAD}" ]] || {
  echo source_head_mismatch >&2; exit 4;
}
git -C "${ROOT}" diff --quiet && git -C "${ROOT}" diff --cached --quiet || {
  echo source_worktree_dirty >&2; exit 4;
}
[[ "$(git -C "${ROOT}/third_party/livewan" rev-parse HEAD)" == \
  1eabf43c777bc5b2816926ca998cd4e4c2dfa3d5 ]] || exit 4
[[ "$(git -C "${ROOT}/third_party/wan21" rev-parse HEAD)" == \
  9737cba9c1c3c4d04b33fcad41c111989865d315 ]] || exit 4

if [[ "${RUN_KIND}" == timing || "${RUN_KIND}" == profile ]]; then
  CORRECTNESS_RECEIPT="${LIVEWAN_PAIR_CORRECTNESS_RECEIPT:?run requires LIVEWAN_PAIR_CORRECTNESS_RECEIPT}"
  CORRECTNESS_SHA256="${LIVEWAN_PAIR_CORRECTNESS_SHA256:?run requires LIVEWAN_PAIR_CORRECTNESS_SHA256}"
  [[ -f "${CORRECTNESS_RECEIPT}" && "${CORRECTNESS_SHA256}" =~ ^[0-9a-f]{64}$ ]] || exit 3
  [[ "$(sha256sum "${CORRECTNESS_RECEIPT}" | awk '{print $1}')" == "${CORRECTNESS_SHA256}" ]] || exit 3
  "${PYTHON}" - "${CORRECTNESS_RECEIPT}" "${EXPECTED_HEAD}" "${RUN_KIND}" <<'PY'
import json, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text())
expected_head = sys.argv[2]
run_kind = sys.argv[3]
if run_kind == "timing":
    if value.get("status") != "PASS" or value.get("claimable_for_timing") is not True:
        raise SystemExit("paired correctness gate is not PASS")
    if set(value.get("heads", {}).values()) != {expected_head}:
        raise SystemExit("paired correctness gate uses a different source HEAD")
else:
    if value.get("profile_authorized") is not True or value.get("formal_timing_authorized") is not False:
        raise SystemExit("diagnostic profile gate is not PASS")
    source = value.get("profile_source", {})
    if source.get("profile_head") != expected_head or source.get("instrumentation_only") is not True:
        raise SystemExit("diagnostic gate does not authorize this exact profile HEAD")
    heads = value.get("heads", {})
    if not (
        heads.get("baseline") == heads.get("pipeline") == source.get("workload_head")
        and heads.get("profile") == expected_head
    ):
        raise SystemExit("diagnostic gate lost its correctness workload identity")
PY
fi

mkdir -p "${NODE_LOG}" "${RUN_ROOT}"
mkdir -p "$(dirname "${SOURCE_ROOT}")"
[[ ! -e "${NODE_LOG}/launch_manifest.txt" ]] || { echo run_id_already_used >&2; exit 82; }
exec 9>"${LOCK_FILE}"
flock -n 9 || { echo gpu_lock_busy >&2; exit 73; }
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')" ]] || {
  echo gpu_processes_present >&2; exit 81;
}

exec 8>"${SOURCE_ROOT}.prepare.lock"
flock 8
"${ROOT}/scripts/distillschedule/prepare_livewan_crossscale_source.sh" \
  --output "${SOURCE_ROOT}" --receipt "${SOURCE_RECEIPT}" \
  >"${NODE_LOG}/source_prepare.log" 2>&1
flock -u 8

export PATH="${PYENV}/bin:${PATH}"
export PYTHONPATH="${PYTHON_OVERLAY}:${ROOT}/src:${ROOT}:${ROOT}/third_party/livewan:${SOURCE_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
export DISTILLSCHEDULE_NODE_RANK="${NODE_RANK}"
export DISTILLSCHEDULE_RUN_ID="${RUN_ID}"
export DISTILLSCHEDULE_RUN_KIND="${RUN_KIND}"
export DISTILLSCHEDULE_PYTHON="${PYTHON}"
export LIVEWAN_SOURCE_RECEIPT="${SOURCE_RECEIPT}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_ASYNC_ERROR_HANDLING=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

{
  echo "run_id=${RUN_ID}"
  echo "node_rank=${NODE_RANK}"
  echo "host=$(hostname)"
  echo "head=${EXPECTED_HEAD}"
  echo "pipeline_variant=${PIPELINE_VARIANT}"
  echo "plan=${PLAN}"
  echo "livewan=$(git -C "${ROOT}/third_party/livewan" rev-parse HEAD)"
  echo "wan21=$(git -C "${ROOT}/third_party/wan21" rev-parse HEAD)"
  echo "python=$(${PYTHON} --version 2>&1)"
  echo "python_overlay=${PYTHON_OVERLAY}"
  echo "python_overlay_sha256=$(sha256sum "${PYTHON_OVERLAY}/.complete.json" | awk '{print $1}')"
  "${PYTHON}" - <<'PY'
import torch
print("torch=" + torch.__version__)
print("torch_cuda=" + str(torch.version.cuda))
print("nccl=" + str(torch.cuda.nccl.version()))
PY
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader
} >"${NODE_LOG}/launch_manifest.txt"
cp /sys/fs/cgroup/memory.events "${NODE_LOG}/cgroup_memory_events_before.txt" 2>/dev/null || true

monitor_pid=""
stop_monitor() {
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
  fi
}
trap stop_monitor EXIT
monitor() {
  echo 'unix_time,cgroup_bytes,host_available_bytes,gpu_used_mib,gpu_free_mib,gpu_util_percent' \
    >"${NODE_LOG}/resource_monitor.csv"
  while true; do
    now="$(date +%s.%N)"
    cgroup="$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo -1)"
    host_available="$(awk '/^MemAvailable:/{print $2*1024}' /proc/meminfo)"
    gpu_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd: - || echo unavailable)"
    gpu_free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | paste -sd: - || echo unavailable)"
    gpu_util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | paste -sd: - || echo unavailable)"
    echo "${now},${cgroup},${host_available},${gpu_used},${gpu_free},${gpu_util}" \
      >>"${NODE_LOG}/resource_monitor.csv"
    sleep 5
  done
}
monitor &
monitor_pid=$!

RESOLVED="${RUN_ROOT}/resolved_plan.node${NODE_RANK}.json"
"${PYTHON}" -m distillschedule.cli render "${PLAN}" --output "${RESOLVED}" \
  >"${NODE_LOG}/render.log" 2>&1

# Fail before 14B construction if the two hosts accidentally form split worlds
# or the default NCCL communicator is unusable.  The probe is intentionally
# model-free and the node lock remains held throughout it.
TOPOLOGY="${RUN_ROOT}/preflight/topology.json"
mkdir -p "${RUN_ROOT}/preflight"
set +e
"${TORCHRUN}" --nnodes=2 --nproc-per-node=8 --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}" \
  "${ROOT}/scripts/distillschedule/probe_world16_topology.py" \
  --output "${TOPOLOGY}" >"${NODE_LOG}/topology_probe.log" 2>&1
topology_status=$?
set -e
[[ ${topology_status} -eq 0 ]] || { echo topology_probe_failed >&2; exit "${topology_status}"; }

if [[ "${EXECUTION_MODE}" == pipeline ]]; then
  TRANSPORT_DIR="${RUN_ROOT}/preflight/transport"
  set +e
  "${TORCHRUN}" --nnodes=2 --nproc-per-node=8 --node-rank="${NODE_RANK}" \
    --master-addr="${MASTER_ADDR}" --master-port="$((MASTER_PORT + 1))" \
    "${ROOT}/scripts/distillschedule/probe_livewan_crossscale_transport.py" \
    --output-dir "${TRANSPORT_DIR}" --pipeline-variant "${PIPELINE_VARIANT}" \
    >"${NODE_LOG}/transport_probe.log" 2>&1
  transport_status=$?
  set -e
  [[ ${transport_status} -eq 0 ]] || { echo transport_probe_failed >&2; exit "${transport_status}"; }
fi

set +e
if [[ "${RUN_KIND}" == validate ]]; then
  "${PYTHON}" "${ROOT}/scripts/distillschedule/livewan_crossscale_entrypoint.py" \
    --plan "${RESOLVED}" --mode "${EXECUTION_MODE}" --run-kind validate \
    --repo-root "${ROOT}" >"${NODE_LOG}/console.log" 2>&1
  status=$?
else
  workload=(
    "${TORCHRUN}" --nnodes=2 --nproc-per-node=8 --node-rank="${NODE_RANK}"
    --master-addr="${MASTER_ADDR}" --master-port="$((MASTER_PORT + 2))"
    "${ROOT}/scripts/distillschedule/livewan_crossscale_entrypoint.py"
    --plan "${RESOLVED}" --mode "${EXECUTION_MODE}" --run-kind "${RUN_KIND}"
    --repo-root "${ROOT}"
  )
  if [[ "${RUN_KIND}" == profile ]]; then
    NSYS_BIN="${DISTILLSCHEDULE_NSYS_BIN:?profile requires DISTILLSCHEDULE_NSYS_BIN}"
    [[ -x "${NSYS_BIN}" ]] || { echo "missing_nsys=${NSYS_BIN}" >&2; exit 3; }
    mkdir -p "${RUN_ROOT}/nsight"
    PROFILE_REPORT_BASE="${RUN_ROOT}/nsight/node_${NODE_RANK}"
    PROFILE_REPORT="${PROFILE_REPORT_BASE}.nsys-rep"
    PROFILE_SQLITE="${PROFILE_REPORT_BASE}.sqlite"
    PROFILE_STATS="${PROFILE_REPORT_BASE}.stats.csv"
    PROFILE_EXPORT_LOG="${PROFILE_REPORT_BASE}.export.log"
    PROFILE_RECEIPT="${PROFILE_REPORT_BASE}.receipt.json"
    for path in "${PROFILE_REPORT}" "${PROFILE_SQLITE}" "${PROFILE_STATS}" \
      "${PROFILE_EXPORT_LOG}" "${PROFILE_RECEIPT}"; do
      [[ ! -e "${path}" ]] || { echo "profile_artifact_exists=${path}" >&2; exit 82; }
    done
    "${NSYS_BIN}" profile \
      --trace=cuda,nvtx --sample=none --cpuctxsw=none \
      --trace-fork-before-exec=true --wait=primary \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      --force-overwrite=false \
      --output="${PROFILE_REPORT_BASE}" \
      "${workload[@]}" >"${NODE_LOG}/console.log" 2>&1
    status=$?
    if [[ ${status} -eq 0 && ! -s "${PROFILE_REPORT}" ]]; then
      echo "missing_nsys_report=${PROFILE_REPORT}" >&2
      status=87
    fi
    if [[ ${status} -eq 0 ]]; then
      "${NSYS_BIN}" export --type=sqlite --force-overwrite=true \
        --output="${PROFILE_SQLITE}" "${PROFILE_REPORT}" \
        >"${PROFILE_EXPORT_LOG}" 2>&1
      status=$?
    fi
    if [[ ${status} -eq 0 && ! -s "${PROFILE_SQLITE}" ]]; then
      echo "missing_nsys_sqlite=${PROFILE_SQLITE}" >&2
      status=88
    fi
    if [[ ${status} -eq 0 ]]; then
      "${NSYS_BIN}" stats --report=nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum \
        --format=csv "${PROFILE_SQLITE}" >"${PROFILE_STATS}" 2>&1
      status=$?
    fi
    if [[ ${status} -eq 0 ]] && grep -Eq \
      'ProcessEventsError|Unknown runtime API|Cannot find bucket|Cannot find string|Errors occurred while processing raw events|does not contain (NV Tools Extension|CUDA kernel|CUDA API)' \
      "${NODE_LOG}/console.log" "${PROFILE_EXPORT_LOG}" "${PROFILE_STATS}"; then
      echo nsys_profile_validation_failed >&2
      status=89
    fi
    if [[ ${status} -eq 0 ]]; then
      "${PYTHON}" - "${PROFILE_RECEIPT}" "${NODE_RANK}" "${EXPECTED_HEAD}" \
        "${NSYS_BIN}" "${PROFILE_REPORT}" "${PROFILE_SQLITE}" "${PROFILE_STATS}" <<'PY'
import hashlib, json, os, pathlib, sqlite3, subprocess, sys, tempfile

destination = pathlib.Path(sys.argv[1])
paths = [pathlib.Path(value) for value in sys.argv[5:8]]
node_rank = int(sys.argv[2])

sqlite_path = paths[1]
database = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
try:
    outer_rows = database.execute(
        "SELECT text FROM NVTX_EVENTS "
        "WHERE text LIKE 'livewan/full5/%/rank_%' "
        "AND start IS NOT NULL AND end IS NOT NULL AND end > start"
    ).fetchall()
    observed_ranks = sorted(
        int(str(text).rsplit("rank_", 1)[1]) for (text,) in outer_rows
    )
    expected_ranks = list(range(node_rank * 8, node_rank * 8 + 8))
    device_rows = database.execute(
        "SELECT deviceId, COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "GROUP BY deviceId ORDER BY deviceId"
    ).fetchall()
    observed_devices = [int(device) for device, count in device_rows if int(count) > 0]
finally:
    database.close()

if observed_ranks != expected_ranks:
    raise SystemExit(
        f"incomplete_nvtx_rank_ranges: observed={observed_ranks} expected={expected_ranks}"
    )
if observed_devices != list(range(8)):
    raise SystemExit(
        f"incomplete_kernel_device_coverage: observed={observed_devices} expected={list(range(8))}"
    )

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

value = {
    "schema_version": 1,
    "status": "PASS_CAPTURE_CONTAINER",
    "diagnostic_only": True,
    "formal_timing_eligible": False,
    "node_rank": node_rank,
    "head": sys.argv[3],
    "nsys_bin": sys.argv[4],
    "nsys_version": subprocess.check_output([sys.argv[4], "--version"], text=True).strip(),
    "artifacts": {
        path.suffix.lstrip(".") or path.name: {
            "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)
        }
        for path in paths
    },
    "nvtx_rank_ranges": observed_ranks,
    "kernel_devices": observed_devices,
}
destination.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, destination)
PY
      status=$?
    fi
  else
    "${workload[@]}" >"${NODE_LOG}/console.log" 2>&1
    status=$?
  fi
fi
set -e

cp /sys/fs/cgroup/memory.events "${NODE_LOG}/cgroup_memory_events_after.txt" 2>/dev/null || true
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader \
  >"${NODE_LOG}/gpu_after.csv" || true
[[ ${status} -eq 0 ]] || exit "${status}"
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')" ]] || {
  echo gpu_processes_remain >&2; exit 88;
}

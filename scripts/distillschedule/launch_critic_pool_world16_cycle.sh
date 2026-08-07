#!/usr/bin/env bash
set -euo pipefail

SCHEDULE="${1:?usage: launch_critic_pool_world16_cycle.sh sequential|pipeline RUN_ID RUN_NONCE role_local|pair_local WAVES}"
RUN_ID="${2:?missing RUN_ID}"
RUN_NONCE="${3:?missing RUN_NONCE}"
PLACEMENT="${4:?missing placement}"
WAVES="${5:?missing waves}"
case "${SCHEDULE}" in sequential|pipeline) ;; *) echo "invalid schedule" >&2; exit 2 ;; esac
case "${PLACEMENT}" in role_local|pair_local) ;; *) echo "invalid placement" >&2; exit 2 ;; esac
[[ "${WAVES}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid waves" >&2; exit 2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_ROOT="${DISTILLSCHEDULE_POOL_ENV:-/tmp/distillschedule-helios-official-torch210}"
PYTHON="${ENV_ROOT}/bin/python"
TORCHRUN="${ENV_ROOT}/bin/torchrun"
NODE_RANK="${DISTILLSCHEDULE_NODE_RANK:?set DISTILLSCHEDULE_NODE_RANK=0 or 1}"
MASTER_ADDR="${DISTILLSCHEDULE_MASTER_ADDR:-172.24.129.137}"
MASTER_PORT="${DISTILLSCHEDULE_MASTER_PORT:-29616}"
TRACE_ROOT="${DISTILLSCHEDULE_POOL_TRACE:-/share/project/shuchenweng/DistillSchedule_assets/m1/traces/critic_pool_w16_step11_14_v1}"
TARGET_PACKAGE="${DISTILLSCHEDULE_TARGET_PACKAGE:-/share/project/shuchenweng/DistillSchedule_assets/m1/checkpoints/target_world8_C11_package_v1}"
TARGET_NATIVE="${DISTILLSCHEDULE_TARGET_NATIVE:-/share/project/shuchenweng/DistillSchedule_assets/m1/checkpoints/target_world8_C11_native_v1}"
ARTIFACT_ROOT="${ROOT}/artifacts/profile/m1/critic_pool_world16/${RUN_ID}_${SCHEDULE}_${PLACEMENT}_w${WAVES}"
LOG_ROOT="${ROOT}/artifacts/logs/iteration205_w16_critic_pool"
LOG="${LOG_ROOT}/${RUN_ID}_${SCHEDULE}_${PLACEMENT}_w${WAVES}_node${NODE_RANK}.log"
MONITOR="${ARTIFACT_ROOT}/node_monitor_${NODE_RANK}.csv"
MANIFEST="${ARTIFACT_ROOT}/launch_manifest_node${NODE_RANK}.txt"
LOCK="${DISTILLSCHEDULE_SUITE_OWNER_FILE:-${ROOT}/artifacts/locks/w16_three_rounds_20260730.lock/owner.txt}"
NSYS_REPORT_BASE="${DISTILLSCHEDULE_NSYS_REPORT_BASE:-}"
NSYS_RANGE_LABEL="${DISTILLSCHEDULE_NSYS_RANGE_LABEL:-distillschedule_profile}"
NSYS_TRACE="${DISTILLSCHEDULE_NSYS_TRACE:-cuda,nvtx,osrt}"
NSYS_CAPTURE_RANGE="${DISTILLSCHEDULE_NSYS_CAPTURE_RANGE:-cudaProfilerApi}"
NSYS_CAPTURE_SETTLE_SECONDS="${DISTILLSCHEDULE_NSYS_CAPTURE_SETTLE_SECONDS:-0}"
POOL_TRACE_REPLAY="${DISTILLSCHEDULE_POOL_TRACE_REPLAY:-exact}"

case "${POOL_TRACE_REPLAY}" in
  exact|cyclic) ;;
  *) echo "invalid pool trace replay policy" >&2; exit 2 ;;
esac
case "${NSYS_CAPTURE_RANGE}" in
  cudaProfilerApi|nvtx) ;;
  *) echo "invalid Nsight capture range" >&2; exit 2 ;;
esac

[[ -f "${LOCK}" ]] || { echo "suite ownership lock is absent" >&2; exit 3; }
[[ -x "${PYTHON}" && -x "${TORCHRUN}" ]] || { echo "pool environment unavailable" >&2; exit 3; }
[[ -d "${TRACE_ROOT}" && -d "${TARGET_PACKAGE}" && -d "${TARGET_NATIVE}" ]] || {
  echo "frozen trace/target inputs are unavailable" >&2
  exit 3
}
mkdir -p "${ARTIFACT_ROOT}" "${LOG_ROOT}"
if compgen -G "${ARTIFACT_ROOT}/rank_*.json" >/dev/null; then
  echo "refusing to reuse output directory: ${ARTIFACT_ROOT}" >&2
  exit 2
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
  echo "world16 launch requires idle local GPUs" >&2
  exit 2
fi

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_ENABLE_PARALLEL_LOADING=yes HF_PARALLEL_LOADING_WORKERS=4
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"

{
  echo "schema_version=distillschedule.critic_pool_world16_launch.v1"
  echo "run_id=${RUN_ID}"
  echo "schedule=${SCHEDULE}"
  echo "placement=${PLACEMENT}"
  echo "waves=${WAVES}"
  echo "run_nonce=${RUN_NONCE}"
  echo "node_rank=${NODE_RANK}"
  echo "master_addr=${MASTER_ADDR}"
  echo "master_port=${MASTER_PORT}"
  echo "git_commit=$(git -C "${ROOT}" rev-parse HEAD)"
  echo "git_status=$(git -C "${ROOT}" status --short | tr '\n' ';')"
  echo "host=$(hostname)"
  echo "host_ip=$(hostname -I | awk '{print $1}')"
  echo "torch=$("${PYTHON}" -c 'import torch; print(torch.__version__)')"
  echo "deepspeed=$("${PYTHON}" -c 'import deepspeed; print(deepspeed.__version__)' 2>/dev/null | tail -1)"
  echo "trace_root=${TRACE_ROOT}"
  echo "pool_trace_replay=${POOL_TRACE_REPLAY}"
  echo "target_package=${TARGET_PACKAGE}"
  echo "target_native=${TARGET_NATIVE}"
  echo "suite_owner_file=${LOCK}"
  echo "nsys_report_base=${NSYS_REPORT_BASE:-disabled}"
  echo "nsys_range_label=${NSYS_RANGE_LABEL}"
  echo "nsys_trace=${NSYS_TRACE}"
  echo "nsys_capture_range=${NSYS_CAPTURE_RANGE}"
  echo "nsys_capture_settle_seconds=${NSYS_CAPTURE_SETTLE_SECONDS}"
  echo "nsys_sample=${DISTILLSCHEDULE_NSYS_SAMPLE:-none}"
  echo "nsys_cpuctxsw=${DISTILLSCHEDULE_NSYS_CPUCTXSW:-none}"
  echo "resource_schema=unix_time,memory_current_bytes,rss_kib,pss_kib,proc_cpu_ticks,system_cpu_ticks,threads,ctxt,procs_running,procs_blocked,gpu_used_mib,gpu_util_percent"
  echo "checksum=gpu"
  echo "timing_mode=1"
  echo "execution_scope=isolated_critic_only_subgraph"
  echo "official_continuous_step11_to_step14=false"
  echo "memory_events_before=$(cat /sys/fs/cgroup/memory.events 2>/dev/null | tr '\n' ';' || true)"
} > "${MANIFEST}"

monitor() {
  echo "unix_time,memory_current_bytes,rss_kib,pss_kib,proc_cpu_ticks,system_cpu_ticks,threads,ctxt,procs_running,procs_blocked,gpu_used_mib,gpu_util_percent" > "${MONITOR}"
  while true; do
    now="$(date +%s.%N)"
    memory="$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo -1)"
    rss_kib=0; pss_kib=0; proc_cpu_ticks=0; threads=0
    pids=$(pgrep -f "[p]robe_critic_pool_world16_cycle.py" 2>/dev/null || true)
    for pid in ${pids}; do
      if [[ -r "/proc/${pid}/stat" ]]; then
        read -r utime stime nthr < <(awk '{print $14,$15,$20}' "/proc/${pid}/stat" 2>/dev/null)
        proc_cpu_ticks=$((proc_cpu_ticks + ${utime:-0} + ${stime:-0}))
        threads=$((threads + ${nthr:-0}))
      fi
      rollup="/proc/${pid}/smaps_rollup"; [[ -r "${rollup}" ]] || continue
      read -r rss pss < <(awk '/^Rss:/{rss=$2} /^Pss:/{pss=$2} END{print rss+0,pss+0}' "${rollup}" 2>/dev/null)
      rss_kib=$((rss_kib + rss)); pss_kib=$((pss_kib + pss))
    done
    system_cpu_ticks=$(awk '/^cpu / {x=0; for(i=2;i<=NF;i++)x+=$i; print x; exit}' /proc/stat 2>/dev/null)
    ctxt=$(awk '/^ctxt /{print $2; exit}' /proc/stat 2>/dev/null)
    procs_running=$(awk '/^procs_running /{print $2; exit}' /proc/stat 2>/dev/null)
    procs_blocked=$(awk '/^procs_blocked /{print $2; exit}' /proc/stat 2>/dev/null)
    gpu_memory="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd: - || echo unavailable)"
    gpu_util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | paste -sd: - || echo unavailable)"
    echo "${now},${memory},${rss_kib},${pss_kib},${proc_cpu_ticks},${system_cpu_ticks:-0},${threads},${ctxt:-0},${procs_running:-0},${procs_blocked:-0},${gpu_memory},${gpu_util}" >> "${MONITOR}"
    sleep "${DISTILLSCHEDULE_RESOURCE_MONITOR_INTERVAL:-1}"
  done
}
monitor &
MONITOR_PID=$!
cleanup() {
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
  echo "memory_events_after=$(cat /sys/fs/cgroup/memory.events 2>/dev/null | tr '\n' ';' || true)" >> "${MANIFEST}"
}
trap cleanup EXIT

RUNNER=()
if [[ -n "${NSYS_REPORT_BASE}" ]]; then
  [[ -n "${NSYS_RANGE_LABEL}" ]] || { echo "empty Nsight range label" >&2; exit 3; }
  export DISTILLSCHEDULE_NSYS_RANGE_LABEL="${NSYS_RANGE_LABEL}"
  export DISTILLSCHEDULE_NSYS_CAPTURE_RANGE="${NSYS_CAPTURE_RANGE}"
  export DISTILLSCHEDULE_NSYS_ANNOTATE="${DISTILLSCHEDULE_NSYS_ANNOTATE:-1}"
  echo "nsys_annotate=${DISTILLSCHEDULE_NSYS_ANNOTATE}" >> "${MANIFEST}"
  [[ ! -e "${NSYS_REPORT_BASE}.nsys-rep" && ! -e "${NSYS_REPORT_BASE}.sqlite" ]] || {
    echo "refusing to overwrite Nsight output: ${NSYS_REPORT_BASE}" >&2
    exit 3
  }
  mkdir -p "$(dirname "${NSYS_REPORT_BASE}")"
  CAPTURE_ARGS=(
    --capture-range="${NSYS_CAPTURE_RANGE}"
    --capture-range-end=stop
  )
  if [[ "${NSYS_CAPTURE_RANGE}" == "nvtx" ]]; then
    CAPTURE_ARGS+=(--nvtx-capture="${NSYS_RANGE_LABEL}")
  fi
  RUNNER=(
    /usr/local/cuda/bin/nsys profile
    --trace="${NSYS_TRACE}"
    "${CAPTURE_ARGS[@]}"
    --trace-fork-before-exec=true
    --sample="${DISTILLSCHEDULE_NSYS_SAMPLE:-none}"
    --cpuctxsw="${DISTILLSCHEDULE_NSYS_CPUCTXSW:-none}"
    --force-overwrite=true
    --output="${NSYS_REPORT_BASE}"
  )
fi

set +e
timeout --signal=TERM --kill-after=120s 14400s \
  "${RUNNER[@]}" "${TORCHRUN}" \
    --nnodes=2 \
    --nproc-per-node=8 \
    --node-rank="${NODE_RANK}" \
    --master-addr="${MASTER_ADDR}" \
    --master-port="${MASTER_PORT}" \
    "${ROOT}/scripts/distillschedule/probe_critic_pool_world16_cycle.py" \
      --schedule "${SCHEDULE}" \
      --placement "${PLACEMENT}" \
      --waves "${WAVES}" \
      --run-nonce "${RUN_NONCE}" \
      --checksum gpu \
      --timing-mode \
      --pool-trace-replay "${POOL_TRACE_REPLAY}" \
      --pool-trace "${TRACE_ROOT}" \
      --target-package "${TARGET_PACKAGE}" \
      --target-native-root "${TARGET_NATIVE}" \
      --output-dir "${ARTIFACT_ROOT}" \
  2>&1 | tee "${LOG}"
rc=${PIPESTATUS[0]}
set -e
echo "return_code=${rc}" >> "${MANIFEST}"
exit "${rc}"

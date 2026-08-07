#!/usr/bin/env bash
set -euo pipefail

CASE_ID="${1:?usage: run_nsys_locked.sh CASE_ID COMMAND [ARG ...]}"
shift
if (( $# == 0 )); then
  echo "a profiling command is required" >&2
  exit 2
fi
if [[ ! "${CASE_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid case id: ${CASE_ID}" >&2
  exit 2
fi

ROOT="${DISTILLSCHEDULE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
SUITE_ID="${DISTILLSCHEDULE_NSYS_SUITE_ID:-iteration203_nsight_baseline_suite}"
SUITE_ROOT="${DISTILLSCHEDULE_NSYS_SUITE_ROOT:-${ROOT}/artifacts/nsight/${SUITE_ID}}"
CASE_ROOT="${SUITE_ROOT}/${CASE_ID}"
REPORT_BASE="${CASE_ROOT}/${CASE_ID}"
MANIFEST="${CASE_ROOT}/run_manifest.txt"
VALIDATION_LOG="${CASE_ROOT}/capture_validation.log"
LOCK_PATH="${DISTILLSCHEDULE_NSYS_LOCK_PATH:-/tmp/distillschedule_gpu_node.lock}"
TIMEOUT_SECONDS="${DISTILLSCHEDULE_NSYS_TIMEOUT_SECONDS:-7200}"
RANGE_LABEL="${DISTILLSCHEDULE_NSYS_RANGE_LABEL:-distillschedule_profile}"
INNER_PROFILE="${DISTILLSCHEDULE_NSYS_INNER_PROFILE:-0}"

[[ "${TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "DISTILLSCHEDULE_NSYS_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 2
}
case "${INNER_PROFILE}" in 0|1) ;; *) echo "DISTILLSCHEDULE_NSYS_INNER_PROFILE must be 0 or 1" >&2; exit 2 ;; esac
[[ ! -e "${CASE_ROOT}" ]] || {
  echo "refusing to reuse immutable Nsight case: ${CASE_ROOT}" >&2
  exit 3
}
mkdir -p "${CASE_ROOT}"

exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
  {
    echo "status=BLOCKED_NODE_LOCK"
    echo "lock_path=${LOCK_PATH}"
    echo "time=$(date --iso-8601=seconds)"
  } >"${MANIFEST}"
  echo "node profiling lock is already held: ${LOCK_PATH}" >&2
  exit 75
fi

mapfile -t GPU_PROCESSES < <(
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'
)
if (( ${#GPU_PROCESSES[@]} != 0 )); then
  {
    echo "status=BLOCKED_GPU_BUSY"
    echo "lock_path=${LOCK_PATH}"
    echo "time=$(date --iso-8601=seconds)"
    printf 'gpu_process=%s\n' "${GPU_PROCESSES[@]}"
  } >"${MANIFEST}"
  printf 'GPU node is not idle:\n%s\n' "${GPU_PROCESSES[*]}" >&2
  exit 76
fi

export DISTILLSCHEDULE_NSYS_RANGE_LABEL="${RANGE_LABEL}"
export DISTILLSCHEDULE_NSYS_ANNOTATE=1
if [[ "${INNER_PROFILE}" == 1 ]]; then
  export DISTILLSCHEDULE_NSYS_INNER_REPORT_BASE="${REPORT_BASE}"
else
  unset DISTILLSCHEDULE_NSYS_INNER_REPORT_BASE
fi

{
  echo "schema_version=distillschedule.nsys_locked_run.v3"
  echo "status=RUNNING"
  echo "suite_id=${SUITE_ID}"
  echo "case_id=${CASE_ID}"
  echo "host=$(hostname)"
  echo "host_ips=$(hostname -I | xargs)"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "lock_path=${LOCK_PATH}"
  echo "range_label=${RANGE_LABEL}"
  echo "capture_range=cudaProfilerApi"
  echo "capture_range_end=stop"
  echo "trace_fork_before_exec=true"
  echo "trace=cuda,nvtx,osrt"
  echo "profile_placement=$([[ "${INNER_PROFILE}" == 1 ]] && echo inner || echo outer)"
  echo "timeout_seconds=${TIMEOUT_SECONDS}"
  echo "git_commit=$(git -C "${ROOT}" rev-parse HEAD)"
  echo "git_status_begin"
  git -C "${ROOT}" status --short
  echo "git_status_end"
  printf 'command='
  printf '%q ' "$@"
  printf '\n'
  /usr/local/cuda/bin/nsys --version
  nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free \
    --format=csv,noheader
} >"${MANIFEST}" 2>&1

set +e
if [[ "${INNER_PROFILE}" == 1 ]]; then
  timeout --signal=TERM --kill-after=120s "${TIMEOUT_SECONDS}s" \
    "$@" \
    >"${CASE_ROOT}/console.log" 2>&1
else
  timeout --signal=TERM --kill-after=120s "${TIMEOUT_SECONDS}s" \
    /usr/local/cuda/bin/nsys profile \
      --trace=cuda,nvtx,osrt \
      --capture-range=cudaProfilerApi \
      --capture-range-end=stop \
      --trace-fork-before-exec=true \
      --sample=none \
      --cpuctxsw=none \
      --force-overwrite=true \
      --output="${REPORT_BASE}" \
      "$@" \
      >"${CASE_ROOT}/console.log" 2>&1
fi
RC=$?
set -e

CAPTURE_VALID=0
STATS_RC=NOT_RUN
if [[ "${RC}" == 0 && -f "${REPORT_BASE}.nsys-rep" ]]; then
  set +e
  /usr/local/cuda/bin/nsys stats \
    --report nvtx_sum,cuda_gpu_kern_sum \
    --format csv \
    "${REPORT_BASE}.nsys-rep" \
    >"${VALIDATION_LOG}" 2>&1
  STATS_RC=$?
  set -e
  if [[ "${STATS_RC}" == 0 ]] \
      && ! grep -q "does not contain NV Tools Extension" "${VALIDATION_LOG}" \
      && ! grep -q "does not contain CUDA kernel data" "${VALIDATION_LOG}"; then
    CAPTURE_VALID=1
  fi
fi

{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "return_code=${RC}"
  echo "nsys_stats_return_code=${STATS_RC}"
  echo "capture_valid=${CAPTURE_VALID}"
  if [[ "${RC}" == 0 && -f "${REPORT_BASE}.nsys-rep" && "${CAPTURE_VALID}" == 1 ]]; then
    echo "status=SUCCESS"
  elif [[ "${RC}" == 124 ]]; then
    echo "status=TIMEOUT"
  elif [[ "${RC}" == 0 && ! -f "${REPORT_BASE}.nsys-rep" ]]; then
    echo "status=FAILED_NO_REPORT"
  elif [[ "${RC}" == 0 ]]; then
    echo "status=FAILED_EMPTY_CAPTURE"
  else
    echo "status=FAILED"
  fi
  if [[ -f "${REPORT_BASE}.nsys-rep" ]]; then
    echo "report=${REPORT_BASE}.nsys-rep"
    echo "report_bytes=$(stat -c %s "${REPORT_BASE}.nsys-rep")"
    echo "report_sha256=$(sha256sum "${REPORT_BASE}.nsys-rep" | awk '{print $1}')"
  fi
  if [[ -f "${REPORT_BASE}.sqlite" ]]; then
    echo "sqlite=${REPORT_BASE}.sqlite"
    echo "sqlite_bytes=$(stat -c %s "${REPORT_BASE}.sqlite")"
    echo "sqlite_sha256=$(sha256sum "${REPORT_BASE}.sqlite" | awk '{print $1}')"
  fi
  if [[ -f "${VALIDATION_LOG}" ]]; then
    echo "validation_log=${VALIDATION_LOG}"
    echo "validation_log_sha256=$(sha256sum "${VALIDATION_LOG}" | awk '{print $1}')"
  fi
  echo "memory_events_after=$(cat /sys/fs/cgroup/memory.events 2>/dev/null | tr '\n' ';')"
} >>"${MANIFEST}"

if [[ "${RC}" != 0 ]]; then
  exit "${RC}"
fi
if [[ ! -f "${REPORT_BASE}.nsys-rep" ]]; then
  exit 77
fi
if [[ "${CAPTURE_VALID}" != 1 ]]; then
  exit 78
fi
exit 0

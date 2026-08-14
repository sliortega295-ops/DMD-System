#!/usr/bin/env bash
# Cancel only the exact process tree recorded by remote_stage_wrapper.sh.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=process_tree.sh
source "${SCRIPT_DIR}/process_tree.sh"
PID_FILE="${1:?pid file is required}"
[[ -f "${PID_FILE}" ]] || exit 0

value() {
  awk -F= -v key="$1" '$1 == key {sub(/^[^=]*=/, ""); print; exit}' "${PID_FILE}"
}
PID="$(value child_pid)"
PGID="$(value child_pgid)"
START_TIME="$(value child_start_time)"
[[ "${PID}" =~ ^[0-9]+$ && "${PGID}" =~ ^[0-9]+$ ]] || {
  echo "cancel_remote_stage: invalid PID record ${PID_FILE}" >&2
  exit 2
}

if kill -0 "${PID}" 2>/dev/null; then
  current_pgid="$(ps -o pgid= -p "${PID}" 2>/dev/null | tr -d '[:space:]' || true)"
  [[ "${current_pgid}" == "${PGID}" ]] || {
    echo "cancel_remote_stage: PID/PGID identity changed; refusing broad kill" >&2
    exit 3
  }
  current_start="$(distillschedule_proc_start_time "${PID}" 2>/dev/null || true)"
  if [[ -n "${START_TIME}" && "${current_start}" != "${START_TIME}" ]]; then
    echo "cancel_remote_stage: PID start-time identity changed; refusing kill" >&2
    exit 4
  fi
  distillschedule_terminate_process_tree "${PID}" "${PGID}" 30
fi
{
  echo "cancel_requested_at=$(date -Is)"
  echo "cancel_requester_host=$(hostname)"
} >>"${PID_FILE}"

#!/usr/bin/env bash
# Cancel only the process group recorded by remote_stage_wrapper.sh.
set -Eeuo pipefail
PID_FILE="${1:?pid file is required}"
[[ -f "${PID_FILE}" ]] || exit 0

value() {
  awk -F= -v key="$1" '$1 == key {sub(/^[^=]*=/, ""); print; exit}' "${PID_FILE}"
}
PID="$(value child_pid)"
PGID="$(value child_pgid)"
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
  kill -TERM -- "-${PGID}" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 "${PID}" 2>/dev/null || break
    sleep 1
  done
  kill -KILL -- "-${PGID}" 2>/dev/null || true
fi
{
  echo "cancel_requested_at=$(date -Is)"
  echo "cancel_requester_host=$(hostname)"
} >>"${PID_FILE}"

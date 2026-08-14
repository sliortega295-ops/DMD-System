#!/usr/bin/env bash
# Own one remote stage process tree and publish a cancel-safe PID record.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=process_tree.sh
source "${SCRIPT_DIR}/process_tree.sh"

PID_FILE="${1:?pid file is required}"
shift
(( $# > 0 )) || { echo "remote_stage_wrapper: command is required" >&2; exit 2; }

mkdir -p "$(dirname "${PID_FILE}")"
[[ ! -e "${PID_FILE}" ]] || { echo "remote_stage_wrapper: pid file already exists: ${PID_FILE}" >&2; exit 82; }

HANDSHAKE_FILE="${PID_FILE}.handshake.$$.$RANDOM"
LAUNCHER_PID=""
CHILD_PID=""
CHILD_PGID=""
CHILD_START_TIME=""
finished=0

write_record() {
  local destination temporary
  destination="${PID_FILE}"
  temporary="${destination}.$$.$RANDOM.tmp"
  {
    echo "schema_version=distillschedule.remote_stage_worker.v2"
    echo "host=$(hostname)"
    echo "wrapper_pid=$$"
    echo "launcher_pid=${LAUNCHER_PID}"
    echo "child_pid=${CHILD_PID}"
    echo "child_pgid=${CHILD_PGID}"
    echo "child_start_time=${CHILD_START_TIME}"
    echo "started_at=$(date -Is)"
    printf 'command_sha256='
    printf '%q\0' "$@" | sha256sum | awk '{print $1}'
  } >"${temporary}"
  mv "${temporary}" "${destination}"
}

cleanup() {
  local signal="${1:-EXIT}"
  if [[ -n "${CHILD_PID}" && -n "${CHILD_PGID}" ]] && kill -0 "${CHILD_PID}" 2>/dev/null; then
    distillschedule_terminate_process_tree "${CHILD_PID}" "${CHILD_PGID}" 30 || true
  elif [[ -n "${LAUNCHER_PID}" ]] && kill -0 "${LAUNCHER_PID}" 2>/dev/null; then
    kill -TERM "${LAUNCHER_PID}" 2>/dev/null || true
  fi
  rm -f "${HANDSHAKE_FILE}" 2>/dev/null || true
  if [[ -e "${PID_FILE}" ]]; then
    {
      echo "wrapper_finished_at=$(date -Is)"
      echo "wrapper_cleanup_signal=${signal}"
      echo "wrapper_finished=${finished}"
    } >>"${PID_FILE}" 2>/dev/null || true
  fi
}
trap 'cleanup TERM; exit 143' TERM
trap 'cleanup INT; exit 130' INT
trap 'cleanup HUP; exit 129' HUP
trap 'cleanup EXIT' EXIT

# GNU setsid may fork when its caller is already a process-group leader. The
# PID returned by `$!` is therefore not a stable identity for the launched
# command. The new-session shell publishes its own PID after `setsid` has
# completed, then `exec`s the requested command without changing that PID.
setsid --wait bash -c '
  handshake=$1
  shift
  temporary="${handshake}.$$"
  printf "%s\n" "$$" >"${temporary}"
  mv "${temporary}" "${handshake}"
  exec "$@"
' distillschedule-stage "${HANDSHAKE_FILE}" "$@" &
LAUNCHER_PID=$!

deadline=$((SECONDS + 10))
while [[ ! -s "${HANDSHAKE_FILE}" && ${SECONDS} -lt ${deadline} ]]; do
  kill -0 "${LAUNCHER_PID}" 2>/dev/null || break
  sleep 0.01
done
[[ -s "${HANDSHAKE_FILE}" ]] || {
  echo "remote_stage_wrapper: child identity handshake failed" >&2
  wait "${LAUNCHER_PID}" 2>/dev/null || true
  exit 83
}
CHILD_PID="$(cat "${HANDSHAKE_FILE}")"
[[ "${CHILD_PID}" =~ ^[0-9]+$ ]] || {
  echo "remote_stage_wrapper: invalid child PID handshake" >&2
  exit 84
}
CHILD_PGID="$(ps -o pgid= -p "${CHILD_PID}" | tr -d '[:space:]')"
CHILD_START_TIME="$(distillschedule_proc_start_time "${CHILD_PID}" 2>/dev/null || true)"
[[ "${CHILD_PGID}" =~ ^[0-9]+$ ]] || {
  echo "remote_stage_wrapper: failed to resolve child PGID" >&2
  exit 85
}
[[ "${CHILD_START_TIME}" =~ ^[0-9]+$ ]] || {
  echo "remote_stage_wrapper: failed to resolve child start time" >&2
  exit 86
}
write_record "$@"
rm -f "${HANDSHAKE_FILE}"

set +e
wait "${LAUNCHER_PID}"
RC=$?
set -e
finished=1
{
  echo "child_return_code=${RC}"
  echo "child_finished_at=$(date -Is)"
} >>"${PID_FILE}"
exit "${RC}"

#!/usr/bin/env bash
# Own one remote stage process group and publish a cancel-safe PID record.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=process_tree.sh
source "${SCRIPT_DIR}/process_tree.sh"

PID_FILE="${1:?pid file is required}"
shift
(( $# > 0 )) || { echo "remote_stage_wrapper: command is required" >&2; exit 2; }

mkdir -p "$(dirname "${PID_FILE}")"
[[ ! -e "${PID_FILE}" ]] || { echo "remote_stage_wrapper: pid file already exists: ${PID_FILE}" >&2; exit 82; }

CHILD_PID=""
CHILD_PGID=""
CHILD_START_TIME=""
finished=0
write_record() {
  local destination="${PID_FILE}"
  local temporary="${destination}.$$.$RANDOM.tmp"
  {
    echo "schema_version=distillschedule.remote_stage_worker.v1"
    echo "host=$(hostname)"
    echo "wrapper_pid=$$"
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
  if [[ -n "${CHILD_PGID}" ]] && kill -0 "${CHILD_PID}" 2>/dev/null; then
    distillschedule_terminate_process_tree "${CHILD_PID}" "${CHILD_PGID}" 30 || true
  fi
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

setsid "$@" &
CHILD_PID=$!
CHILD_PGID="$(ps -o pgid= -p "${CHILD_PID}" | tr -d '[:space:]')"
CHILD_START_TIME="$(distillschedule_proc_start_time "${CHILD_PID}" 2>/dev/null || true)"
[[ -n "${CHILD_PGID}" ]] || { echo "remote_stage_wrapper: failed to resolve child PGID" >&2; exit 83; }
[[ "${CHILD_START_TIME}" =~ ^[0-9]+$ ]] || { echo "remote_stage_wrapper: failed to resolve child start time" >&2; exit 84; }
write_record "$@"

set +e
wait "${CHILD_PID}"
RC=$?
set -e
finished=1
{
  echo "child_return_code=${RC}"
  echo "child_finished_at=$(date -Is)"
} >>"${PID_FILE}"
exit "${RC}"

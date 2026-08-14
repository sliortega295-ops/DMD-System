#!/usr/bin/env bash
# Exact-PID process-tree termination helpers for remote experiment stages.
#
# The launch stack contains nested `setsid`/`timeout`/torchrun processes, so a
# single process-group signal is insufficient.  We snapshot descendants while
# the recorded root is alive, bind every PID to its /proc start time, and only
# signal those exact identities.  This avoids both orphaned GPU workers and
# unsafe name-pattern kills.

distillschedule_proc_start_time() {
  local pid="${1:?pid is required}"
  [[ -r "/proc/${pid}/stat" ]] || return 1
  awk '{print $22}' "/proc/${pid}/stat" 2>/dev/null
}

distillschedule_proc_is_live() {
  local pid="${1:?pid is required}"
  local expected_start="${2:?start time is required}"
  local state current_start
  [[ -r "/proc/${pid}/stat" ]] || return 1
  read -r state current_start < <(awk '{print $3, $22}' "/proc/${pid}/stat" 2>/dev/null) || return 1
  [[ "${current_start}" == "${expected_start}" && "${state}" != "Z" ]]
}

distillschedule_collect_process_tree() {
  local root_pid="${1:?root pid is required}"
  local root_start="${2:?root start time is required}"
  local parent child start existing
  local head=0
  local -a queue=("${root_pid}")

  DISTILLSCHEDULE_TREE_PIDS=("${root_pid}")
  DISTILLSCHEDULE_TREE_STARTS=("${root_start}")
  while (( head < ${#queue[@]} )); do
    parent="${queue[head]}"
    head=$((head + 1))
    while read -r child; do
      [[ "${child}" =~ ^[0-9]+$ ]] || continue
      existing=0
      for known in "${DISTILLSCHEDULE_TREE_PIDS[@]}"; do
        if [[ "${known}" == "${child}" ]]; then
          existing=1
          break
        fi
      done
      (( existing == 0 )) || continue
      start="$(distillschedule_proc_start_time "${child}" 2>/dev/null || true)"
      [[ "${start}" =~ ^[0-9]+$ ]] || continue
      DISTILLSCHEDULE_TREE_PIDS+=("${child}")
      DISTILLSCHEDULE_TREE_STARTS+=("${start}")
      queue+=("${child}")
    done < <(ps -eo pid=,ppid= | awk -v parent="${parent}" '$2 == parent {print $1}')
  done
}

distillschedule_terminate_process_tree() {
  local root_pid="${1:?root pid is required}"
  local root_pgid="${2:?root pgid is required}"
  local grace_seconds="${3:-30}"
  local root_start index pid start any_live deadline

  root_start="$(distillschedule_proc_start_time "${root_pid}" 2>/dev/null || true)"
  [[ "${root_start}" =~ ^[0-9]+$ ]] || return 0
  distillschedule_collect_process_tree "${root_pid}" "${root_start}"

  # Signal leaves before their parents.  Also signal the original group to
  # cover a same-group child created in the very small snapshot-to-signal gap.
  for ((index=${#DISTILLSCHEDULE_TREE_PIDS[@]} - 1; index >= 0; index--)); do
    pid="${DISTILLSCHEDULE_TREE_PIDS[index]}"
    start="${DISTILLSCHEDULE_TREE_STARTS[index]}"
    distillschedule_proc_is_live "${pid}" "${start}" && kill -TERM "${pid}" 2>/dev/null || true
  done
  kill -TERM -- "-${root_pgid}" 2>/dev/null || true

  deadline=$((SECONDS + grace_seconds))
  while (( SECONDS < deadline )); do
    any_live=0
    for index in "${!DISTILLSCHEDULE_TREE_PIDS[@]}"; do
      if distillschedule_proc_is_live \
        "${DISTILLSCHEDULE_TREE_PIDS[index]}" \
        "${DISTILLSCHEDULE_TREE_STARTS[index]}"; then
        any_live=1
        break
      fi
    done
    (( any_live == 1 )) || return 0
    sleep 0.1
  done

  for ((index=${#DISTILLSCHEDULE_TREE_PIDS[@]} - 1; index >= 0; index--)); do
    pid="${DISTILLSCHEDULE_TREE_PIDS[index]}"
    start="${DISTILLSCHEDULE_TREE_STARTS[index]}"
    distillschedule_proc_is_live "${pid}" "${start}" && kill -KILL "${pid}" 2>/dev/null || true
  done
  kill -KILL -- "-${root_pgid}" 2>/dev/null || true
}

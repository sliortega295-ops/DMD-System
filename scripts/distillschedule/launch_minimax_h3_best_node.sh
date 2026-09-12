#!/usr/bin/env bash
# Canonical H3 overlay lives in DMD-System; only pinned upstream LightX2V is external.
set -Eeuo pipefail
NODE_RANK="${1:?node rank}"; RUN_ID="${2:?run id}"; MODE="${3:?mode}"
EXPECTED_HEAD="${4:?HEAD}"; MASTER_ADDR="${5:?master}"; MASTER_PORT="${6:?port}"
[[ "$NODE_RANK" == 0 || "$NODE_RANK" == 1 ]] || exit 2
[[ "$MODE" == correctness || "$MODE" == diagnostic-timing || "$MODE" == calibration ]] || exit 2
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]{1,128}$ && "$RUN_ID" != . && "$RUN_ID" != .. && "$EXPECTED_HEAD" =~ ^[0-9a-f]{40}$ ]] || exit 2
ROOT="${DMD_SYSTEM_ROOT:?}"; PYTHON="${DISTILLSCHEDULE_PYTHON:?}"
RUN_ROOT="${DISTILLSCHEDULE_RUN_ROOT:?}"; CONFIG="${H3_CONFIG:?}"
if [[ "$(id -u)" == 0 && -z "${SUDO_UID:-}" ]]; then
  export SUDO_UID="$(stat -c '%u' "$ROOT")"
fi
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_HEAD" ]] || exit 4
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || exit 4
[[ "$(git -C "${LIGHTX2V_ROOT:?}" rev-parse HEAD)" == d034a6b0ecaa31aa07c81aeb7bbe69b225f1d7be ]] || exit 4
[[ -z "$(git -C "$LIGHTX2V_ROOT" status --porcelain --untracked-files=all)" ]] || exit 4
[[ "${H3_ACTIVATION_POLICY:?}" == none && "${H3_FA3_REPLAY_CACHE_STORAGE:?}" == cpu_staged ]] || exit 4
[[ -x "$PYTHON" && -f "$CONFIG" ]] || exit 3
[[ "${H3_PLANNER_TWO_CYCLE:-0}" == 0 || "${H3_PLANNER_TWO_CYCLE:-0}" == 1 ]] || exit 2
[[ "${H3_PLANNER_CALIBRATION:-0}" == 0 || "${H3_PLANNER_CALIBRATION:-0}" == 1 ]] || exit 2
if [[ -n "${DISTILLSCHEDULE_SEARCH_CANDIDATE_ID:-}" ]]; then
  if [[ "${DISTILLSCHEDULE_SEARCH_CANDIDATE_ID}" =~ ^joint-h3v2-d[12]-n(64|128)-w([1-5])$ ]]; then
    [[ "${H3_PLANNER_ROLLOUT_WINDOW:-}" == "${BASH_REMATCH[2]}" ]] || exit 2
  else
    [[ "${DISTILLSCHEDULE_SEARCH_CANDIDATE_ID}" =~ ^joint-h3v1-d[12]-n(64|128)$ && \
       -z "${H3_PLANNER_ROLLOUT_WINDOW:-}" ]] || exit 2
  fi
  [[ "${H3_PLANNER_TWO_CYCLE:-0}" == 1 && "${H3_PLANNER_CLOCK:-}" == host_fenced ]] || exit 2
  if [[ "$MODE" == calibration ]]; then
    [[ "${H3_PLANNER_CALIBRATION:-0}" == 1 ]] || exit 2
  else
    [[ "${H3_PLANNER_CALIBRATION:-0}" == 0 ]] || exit 2
  fi
else
  [[ -z "${H3_PLANNER_ROLLOUT_WINDOW:-}" ]] || exit 2
  [[ "$MODE" != calibration && "${H3_PLANNER_TWO_CYCLE:-0}" == 0 && \
     "${H3_PLANNER_CALIBRATION:-0}" == 0 && -z "${H3_PLANNER_CLOCK:-}" ]] || exit 2
fi
mkdir -p "$RUN_ROOT/launcher/node_$NODE_RANK" "${TRITON_CACHE_DIR:?}"
LOG="$RUN_ROOT/launcher/node_$NODE_RANK"
[[ ! -e "$LOG/launch_manifest.json" ]] || exit 82
exec 9>"${DISTILLSCHEDULE_NODE_LOCK:?}"
flock -n 9 || exit 73
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')" ]] || exit 81
cp /sys/fs/cgroup/memory.events "$LOG/cgroup_before.txt"
export EXPECTED_INTEGRATION_HEAD="$EXPECTED_HEAD"
export DISTILLSCHEDULE_RUN_KIND="$MODE"
NATIVE_PID=""
teardown() {
  local rc=$? count status=PASS
  trap - EXIT INT TERM
  set +e
  # Only the child this launcher started; peer cancellation remains owned by
  # the existing controller. The node lock stays held through this audit.
  if [[ -n "$NATIVE_PID" ]] && kill -0 "$NATIVE_PID" 2>/dev/null; then
    kill -TERM "$NATIVE_PID" 2>/dev/null
    wait "$NATIVE_PID"
  fi
  cp /sys/fs/cgroup/memory.events "$LOG/cgroup_after.txt" || status=FAIL
  "$PYTHON" - "$LOG" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
def read(name):
    return dict((k, int(v)) for k, v in (line.split() for line in (p / name).read_text().splitlines()))
a, b = read('cgroup_before.txt'), read('cgroup_after.txt')
if any(b.get(k, 0) > a.get(k, 0) for k in ('oom', 'oom_kill')):
    raise SystemExit('cgroup OOM increased')
PY
  [[ $? == 0 ]] || status=FAIL
  if count=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | wc -l); then
    [[ "$count" == 0 ]] || status=FAIL
  else
    count=UNKNOWN; status=FAIL
  fi
  printf 'status=%s\ngpu_process_count=%s\nowner_run_id=%s\n' "$status" "$count" "$RUN_ID" >"$LOG/teardown_status.txt"
  [[ "$status" == PASS ]] || { [[ "$rc" != 0 ]] || rc=90; }
  exit "$rc"
}
trap teardown EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
COMMAND=("$PYTHON" -m torch.distributed.run --nnodes=2 --nproc-per-node=8
  --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT"
  --module distillschedule.backends.training.minimax_h3.integrated_entrypoint --config "$CONFIG")
printf '%q ' "${COMMAND[@]}" >"$LOG/command.sh"
printf '\n' >>"$LOG/command.sh"
"$PYTHON" - "$ROOT" "$LIGHTX2V_ROOT" "$CONFIG" "$MODE" "${COMMAND[@]}" >"$LOG/launch_manifest.json" <<'PY'
import hashlib, json, os, pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1])
lightx = pathlib.Path(sys.argv[2])
required = [
    root / "src/distillschedule/backends/training/minimax_h3/trainer.py",
    root / "src/distillschedule/backends/training/minimax_h3/fa3_replay_cache.py",
    root / "src/distillschedule/backends/training/minimax_h3/grid_entrypoint.py",
    lightx / "lightx2v_train",
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"missing H3 integration source: {missing}")
from distillschedule.search.h3_implementation import current_implementation
impl = current_implementation()
config = json.loads(pathlib.Path(sys.argv[3]).read_text())
binding = impl.receipt_binding(config, os.environ, mode=sys.argv[4]) if impl else None
keys = [k for k in os.environ if k.startswith(('H3_', 'DISTILLSCHEDULE_', 'NCCL_', 'TORCH_', 'PYTORCH_'))]
keys += ['PYTHONPATH', 'LD_LIBRARY_PATH', 'LIGHTX2V_ROOT', 'HF_HOME', 'TRITON_CACHE_DIR']
print(json.dumps({"status": "PASS_SOURCE_PREFLIGHT", "h3_backend": "in_repo", "lightx2v": str(lightx),
    "integration_head": os.environ['EXPECTED_INTEGRATION_HEAD'],
    "lightx2v_head": subprocess.check_output(['git', '-C', str(lightx), 'rev-parse', 'HEAD'], text=True).strip(),
    "config_sha256": hashlib.sha256(pathlib.Path(sys.argv[3]).read_bytes()).hexdigest(),
    "source_clean": True, "python_executable": sys.executable, "mode": sys.argv[4],
    "planner_binding": binding, "command": sys.argv[5:],
    "environment": {k: os.environ[k] for k in sorted(set(keys)) if k in os.environ},
    "claim_boundary": "NATIVE_DIAGNOSTIC_NOT_SEARCH_READY_OR_FORMAL_SPEEDUP"}, sort_keys=True))
PY
set +e
"${COMMAND[@]}" >"$LOG/console.log" 2>&1 &
NATIVE_PID=$!
wait "$NATIVE_PID"
RC=$?
NATIVE_PID=""
set -e
exit "$RC"

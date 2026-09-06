#!/usr/bin/env bash
# Canonical H3 overlay lives in DMD-System; only pinned upstream LightX2V is external.
set -Eeuo pipefail
NODE_RANK="${1:?node rank}"; RUN_ID="${2:?run id}"; MODE="${3:?mode}"
EXPECTED_HEAD="${4:?HEAD}"; MASTER_ADDR="${5:?master}"; MASTER_PORT="${6:?port}"
[[ "$NODE_RANK" == 0 || "$NODE_RANK" == 1 ]] || exit 2
[[ "$MODE" == correctness || "$MODE" == diagnostic-timing ]] || exit 2
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ && "$EXPECTED_HEAD" =~ ^[0-9a-f]{40}$ ]] || exit 2
ROOT="${DMD_SYSTEM_ROOT:?}"; PYTHON="${DISTILLSCHEDULE_PYTHON:?}"
RUN_ROOT="${DISTILLSCHEDULE_RUN_ROOT:?}"; CONFIG="${H3_CONFIG:?}"
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_HEAD" ]] || exit 4
[[ "$(git -C "${LIGHTX2V_ROOT:?}" rev-parse HEAD)" == d034a6b0ecaa31aa07c81aeb7bbe69b225f1d7be ]] || exit 4
[[ "${H3_ACTIVATION_POLICY:?}" == none && "${H3_FA3_REPLAY_CACHE_STORAGE:?}" == cpu_staged ]] || exit 4
[[ -f "$CONFIG" ]] || exit 3
mkdir -p "$RUN_ROOT/launcher/node_$NODE_RANK" "${TRITON_CACHE_DIR:?}"
LOG="$RUN_ROOT/launcher/node_$NODE_RANK"
[[ ! -e "$LOG/launch_manifest.json" ]] || exit 82
exec 9>"${DISTILLSCHEDULE_NODE_LOCK:?}"
flock -n 9 || exit 73
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')" ]] || exit 81
cp /sys/fs/cgroup/memory.events "$LOG/cgroup_before.txt"
export EXPECTED_INTEGRATION_HEAD="$EXPECTED_HEAD"
"$PYTHON" - "$ROOT" "$LIGHTX2V_ROOT" >"$LOG/launch_manifest.json" <<'PY'
import json, pathlib, sys
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
print(json.dumps({"status":"PASS_SOURCE_PREFLIGHT","h3_backend":"in_repo","lightx2v":str(lightx)}, sort_keys=True))
PY
set +e
"$PYTHON" -m torch.distributed.run --nnodes=2 --nproc-per-node=8 \
  --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  --module distillschedule.backends.training.minimax_h3.integrated_entrypoint \
  --config "$CONFIG" >"$LOG/console.log" 2>&1
RC=$?
set -e
cp /sys/fs/cgroup/memory.events "$LOG/cgroup_after.txt"
"$PYTHON" - "$LOG" <<'PY'
import pathlib,sys
p=pathlib.Path(sys.argv[1])
def read(name):return dict((k,int(v)) for k,v in (line.split() for line in (p/name).read_text().splitlines()))
a,b=read('cgroup_before.txt'),read('cgroup_after.txt')
if any(b.get(k,0)>a.get(k,0) for k in ('oom','oom_kill')):
    raise SystemExit('cgroup OOM increased')
PY
exit "$RC"

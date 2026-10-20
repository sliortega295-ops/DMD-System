#!/usr/bin/env bash
# Explicitly continued by the user's expanded W1-W6 training/optimization request.
set -Eeuo pipefail
ROOT=/home/jiangzhikun/yongyan_liu/Helios-H100
O="$ROOT/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1"
PY="$ROOT/envs/five-workload-h100-20260930/bin/python"
mkdir -p "$O/queue_r5" "$O/logs"
exec 7>"$ROOT/locks/helios_asset_transfer.lock"; flock -w 120 -E 75 7
exec 9>"$ROOT/locks/helios_world8_gpu.lock"; flock -w 30 -E 75 9
exec 8>/tmp/distillschedule_gpu_node.lock; flock -w 30 -E 75 8
export DISTILLSCHEDULE_INHERITED_GPU_LOCKS=1
record() {
 "$PY" - "$O/campaign_r5_status.json" "$1" "$2" <<'PY'
import json,sys,os,datetime
from pathlib import Path
p=Path(sys.argv[1]);t=p.with_suffix('.tmp')
t.write_text(json.dumps({'run_id':'h100_all_workload_training_optimization_20261014_r5','phase':sys.argv[2],'status':sys.argv[3],'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authorization':'W1-W6 matrix then H100 optimization until user interrupts','prior_deadline_preserved':'2026-10-14T19:43:43Z','extension':'explicit subsequent user authorization; prior failures/budget ledger not reset','lock_scope':'FD7/9/8 held across audit/construction/run/teardown','foreign_process_signals':False,'artificial_GPU_occupancy':False},indent=2)+'\n');os.replace(t,p)
PY
}
CHILD=''
stop() {
 trap - INT TERM EXIT
 if [[ -n "$CHILD" ]] && kill -0 "$CHILD" 2>/dev/null; then
  kill -TERM -- "-$CHILD" 2>/dev/null || true
  for _ in {1..60}; do kill -0 "$CHILD" 2>/dev/null || break; sleep 1; done
 fi
 record user_or_controller_stop STOPPED_OWNED_CHILD_ONLY
 exit 143
}
trap stop INT TERM
trap 'rc=$?; record controller_exit "EXIT_$rc"' EXIT
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
"$PY" "$ROOT/repos/DMD-System-h100-resident-baselines-20261014/scripts/distillschedule/verify_inherited_gpu_locks.py" --project-root "$ROOT" --asset-fd 7
record queue LOCKS_HELD_CPU_PREPARATION
while :; do
 [[ ! -e "$O/queue_r5/STOP_AFTER_CURRENT" ]] || { record requested_stop STOP_AFTER_CURRENT; exit 0; }
 ITEM=''
 for ready in "$O/queue_r5/"*.ready.json; do
  [[ -e "$ready" ]] || continue
  prefix=${ready%.ready.json}
  [[ ! -e "$prefix.done.json" && ! -e "$prefix.failure.json" ]] || continue
  ITEM=$prefix; break
 done
 if [[ -z "$ITEM" ]]; then sleep 10; continue; fi
 "$PY" - "$ITEM.ready.json" "$ITEM.sh" <<'PY'
import json,hashlib,sys
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text());p=Path(sys.argv[2])
assert m['script_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
assert m['status']=='PASS_CPU_ADMISSION' and m['workload_math_changed'] is False
PY
 [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || { record "$ITEM" BLOCKED_FOREIGN_GPU_CONTEXT_NO_PREEMPTION; exit 81; }
 record "${ITEM##*/}" RUNNING
 set +e
 setsid bash "$ITEM.sh" > "$ITEM.launcher.log" 2>&1 &
 CHILD=$!
 wait "$CHILD";RC=$?
 CHILD=''
 set -e
 "$PY" - "$ITEM" "$RC" <<'PY'
import json,sys,datetime
from pathlib import Path
p=Path(sys.argv[1]+('.done.json' if sys.argv[2]=='0' else '.failure.json'))
with p.open('x') as f:json.dump({'exit_code':int(sys.argv[2]),'status':'PASS_COMMAND_PENDING_WORKLOAD_RECEIPT_REVIEW' if sys.argv[2]=='0' else 'FAIL_PRESERVED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},f,indent=2)
PY
 if [[ "$RC" != 0 ]]; then
  record "${ITEM##*/}" FAIL_PRESERVED_WAITING_SCOPED_DISPOSITION
  while [[ ! -e "$ITEM.disposition.json" ]]; do sleep 10; done
  "$PY" - "$ITEM.disposition.json" <<'PY'
import json,sys
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text());assert m['preserve_failed_artifacts'] is True and m['action']=='CONTINUE_OTHER_ADMITTED_STEPS'
PY
 else record "${ITEM##*/}" COMMAND_DONE_REVIEW_RECEIPTS; fi
 done

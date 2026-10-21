#!/usr/bin/env bash
# User-corrected W5/W6 continuation. Queue for the cooperative lease; no preemption.
set -Eeuo pipefail
ROOT=/home/jiangzhikun/yongyan_liu/Helios-H100
O="$ROOT/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1"
PY="$ROOT/envs/five-workload-h100-20260930/bin/python"
QUEUE="$O/queue_r6"
record() {
 "$PY" - "$O/campaign_r6_status.json" "$1" "$2" <<'PY'
from pathlib import Path
import datetime,json,os,sys
p=Path(sys.argv[1]);t=p.with_suffix('.partial')
t.write_text(json.dumps({'run_id':'h100_W5_W6_resume_20261015_r6','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':sys.argv[2],'status':sys.argv[3],'authorization':'User corrected W5/W6 are not complete; resume training matrix, then existing optimization scope','old_failure_ledger':'preserved; no search budget reset','lock_scope':'when acquired, asset/project/node through audit/construction/run/teardown','foreign_signals':False,'artificial_GPU_memory_allocation':False},indent=2)+'\n');os.replace(t,p)
PY
}
CHILD=''
stop() {
 trap - INT TERM
 if [[ -n "$CHILD" ]] && kill -0 "$CHILD" 2>/dev/null; then
  # The runner has a SIGTERM handler and start-tick-qualified descendant cleanup.
  kill -TERM -- "-$CHILD" 2>/dev/null || true
  for _ in {1..120}; do kill -0 "$CHILD" 2>/dev/null || break; sleep 1; done
 fi
 record user_stop OWNED_RUNNER_INTERRUPTED_REVIEW_TEARDOWN
 exit 143
}
trap stop INT TERM
record acquire WAITING_EXISTING_PROJECT_GPU_LEASE_NO_GPU_LAUNCH
exec 9>"$ROOT/locks/helios_world8_gpu.lock"
flock 9
exec 8>/tmp/distillschedule_gpu_node.lock
flock -w 120 -E 75 8
exec 7>"$ROOT/locks/helios_asset_transfer.lock"
flock -w 120 -E 75 7
export DISTILLSCHEDULE_INHERITED_GPU_LOCKS=1
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || {
 record audit BLOCKED_FOREIGN_GPU_CONTEXTS_NO_SIGNALS
 exit 81
}
"$PY" "$ROOT/repos/DMD-System-h100-w6-lora-unit-inplace-20261015-r1/scripts/distillschedule/verify_inherited_gpu_locks.py" --project-root "$ROOT" --asset-fd 7
record acquired LEASE_HELD_ACTIVE_W5_W6_CAMPAIGN
while :; do
 [[ ! -e "$QUEUE/STOP_AFTER_CURRENT" ]] || { record user_stop STOP_AFTER_CURRENT; exit 0; }
 ITEM=''
 for READY in "$QUEUE/"*.ready.json; do
  [[ -e "$READY" ]] || continue
  PREFIX=${READY%.ready.json}
  [[ ! -e "$PREFIX.done.json" && ! -e "$PREFIX.failure.json" ]] || continue
  ITEM=$PREFIX;break
 done
 if [[ -z "$ITEM" ]]; then
  record queue WAITING_NEXT_BOUNDED_W5_W6_ADMISSION_NOT_COMPLETE
  sleep 60;continue
 fi
 "$PY" - "$ITEM.ready.json" "$ITEM.sh" <<'PY'
import hashlib,json,sys
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text());p=Path(sys.argv[2])
assert m['status']=='PASS_CPU_ADMISSION' and m['workload_math_changed'] is False
assert m['script_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
PY
 [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || {
  record audit BLOCKED_FOREIGN_GPU_CONTEXTS_NO_SIGNALS
  exit 81
 }
 record "${ITEM##*/}" RUNNING_BOUNDED_STAGE
 set +e
 setsid bash "$ITEM.sh" > "$ITEM.launcher.log" 2>&1 & CHILD=$!
 wait "$CHILD";RC=$?;CHILD=''
 set -e
 "$PY" - "$ITEM" "$RC" <<'PY'
import datetime,json,sys
from pathlib import Path
p=Path(sys.argv[1]+('.done.json' if sys.argv[2]=='0' else '.failure.json'))
with p.open('x') as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':int(sys.argv[2]),'status':'PASS_COMMAND_REVIEW_ACTUAL_RECEIPTS' if sys.argv[2]=='0' else 'FAIL_PRESERVED'},f,indent=2)
PY
 if [[ "$RC" != 0 ]]; then
  record "${ITEM##*/}" FAIL_PRESERVED_WAITING_SCOPED_REVIEW_NO_AUTOMATIC_RETRY
  while [[ ! -e "$ITEM.disposition.json" ]]; do
   [[ ! -e "$QUEUE/STOP_AFTER_CURRENT" ]] || { record user_stop STOP_AFTER_CURRENT; exit 0; }
   sleep 60
  done
  "$PY" - "$ITEM.disposition.json" <<'PY'
import json,sys
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text())
assert m['preserve_failed_artifacts'] is True and m['action']=='CONTINUE_OTHER_ADMITTED_STEPS'
PY
 else record "${ITEM##*/}" COMMAND_DONE_WORKLOAD_REVIEW_PENDING; fi
done

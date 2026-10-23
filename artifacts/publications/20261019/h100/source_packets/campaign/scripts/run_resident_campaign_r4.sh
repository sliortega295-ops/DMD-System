#!/usr/bin/env bash
# Resume after a telemetry-only bootstrap failure. Never repeat completed GAN.
set -Eeuo pipefail
ROOT=/home/jiangzhikun/yongyan_liu/Helios-H100
O="$ROOT/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1"
PY="$ROOT/envs/five-workload-h100-20260930/bin/python"
REPO="$ROOT/repos/DMD-System-h100-resident-baselines-20261014"
DEADLINE_UTC=2026-10-14T19:43:43Z
exec 7>"$ROOT/locks/helios_asset_transfer.lock"; flock -w 120 -E 75 7
exec 9>"$ROOT/locks/helios_world8_gpu.lock"; flock -w 30 -E 75 9
exec 8>/tmp/distillschedule_gpu_node.lock; flock -w 30 -E 75 8
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
[[ "$(git -C "$REPO" rev-parse HEAD)" == 7a668bcb12ad37d831dced3aca1be4ec7be0a2fe && -z "$(git -C "$REPO" status --porcelain)" ]]
export DISTILLSCHEDULE_INHERITED_GPU_LOCKS=1
"$PY" "$REPO/scripts/distillschedule/verify_inherited_gpu_locks.py" --project-root "$ROOT" --asset-fd 7
record() {
 "$PY" - "$O/campaign_r4_status.json" "$1" "$2" "$DEADLINE_UTC" <<'INNER'
import json,sys,os,datetime
from pathlib import Path
p=Path(sys.argv[1]);t=p.with_suffix('.tmp')
x={'run_id':'h100_resident_baselines_recovery_20261014_r4','phase':sys.argv[2],'status':sys.argv[3],
   'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fixed_original_deadline_utc':sys.argv[4],
   'resource_scope':'three cooperative locks FD7/9/8 across all stages','W1_W4_resident_baselines':True, 'H3_training': 'NOT_RUN',
   'new_budget_reset':False,'foreign_process_signals':False,'artificial_GPU_occupancy':False}
t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
INNER
}
trap 'rc=$?; trap - EXIT; record campaign_exit "EXIT_${rc}"; exit "$rc"' EXIT
for item in 005_livewan_resident_r2; do
 WAIT_START=$SECONDS
 while [[ ! -e "$O/commands/$item.ready.json" ]]; do
  [[ "$(date -u +%s)" -lt "$(date -u -d "$DEADLINE_UTC" +%s)"  ]] || { record "$item" FAIL_READINESS_DEADLINE; exit 84; }
  record "$item" LOCKS_HELD_WAITING_CPU_ADMISSION
  sleep 10
 done
 "$PY" - "$O/commands/$item.ready.json" "$O/commands/$item.sh" <<'INNER'
import json,hashlib,sys
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text());p=Path(sys.argv[2])
assert m['script_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
assert m['training_math_changed'] is False
INNER
 REMAIN=$(( $(date -u -d "$DEADLINE_UTC" +%s) - $(date -u +%s) ))
 [[ "$REMAIN" -gt 0 ]] || exit 84
 record "$item" RUNNING
 set +e
 timeout --signal=TERM --kill-after=60s "${REMAIN}s" bash "$O/commands/$item.sh"
 RC=$?
 set -e
 if [[ "$RC" != 0 ]]; then
  record "$item" FAIL_RETAINED_WAITING_SCOPED_DISPOSITION
  while [[ ! -e "$O/commands/$item.failure-disposition.json" ]]; do
   [[ "$(date -u +%s)" -lt "$(date -u -d "$DEADLINE_UTC" +%s)" ]] || exit 84
   sleep 15
  done
  "$PY" - "$O/commands/$item.failure-disposition.json" <<'INNER'
import json,sys
from pathlib import Path
x=json.loads(Path(sys.argv[1]).read_text())
assert x['action']=='SKIP_FAILED_WORKLOAD_PRESERVE_EVIDENCE' and x['change_training_contract'] is False
INNER
  record "$item" FAIL_PRESERVED_NOT_RUN_TIMING
  continue
 fi
 record "$item" PASS
 "$PY" - "$O/commands/$item.done.json" <<'INNER'
import json,sys,datetime
from pathlib import Path
p=Path(sys.argv[1]);assert not p.exists();p.write_text(json.dumps({'status':'PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})+'\n')
INNER
done
record all_registered_steps_processed COMPLETE_CHECK_PER_WORKLOAD_STATUS

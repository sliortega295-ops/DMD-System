#!/usr/bin/env bash
# One owned H100 node, eight physical workers, sixteen independent logical rows.
set -euo pipefail
if [[ $# -lt 6 ]]; then
  echo "usage: $0 ROOT RUN_ID MODE PLACEMENT EXPECTED_HEAD ASSET_ROOT [worker binding arguments...]" >&2
  exit 2
fi
ROOT=$1 RUN_ID=$2 MODE=$3 PLACEMENT=$4 EXPECTED_HEAD=$5 ASSETS=$6
shift 6
case "$MODE" in capacity|correctness|timing) ;; *) exit 2 ;; esac
case "$PLACEMENT" in resident|world16-fsdp|resident-async128-tree-chunk2048) ;; *) exit 2 ;; esac
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_.-]+$ ]] || exit 2
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${DISTILLSCHEDULE_PYTHON:?set isolated Python executable}
HELIOS=${DISTILLSCHEDULE_HELIOS_ROOT:?set frozen Helios source snapshot}
RUN=$ROOT/research_artifacts/five_workload_h100_20260930/$RUN_ID
mkdir -p "$ROOT/locks"
# Both locks stay open through construction, evidence extraction and teardown.
# An owned campaign may already hold these exact open-file descriptions.
case "${DISTILLSCHEDULE_INHERITED_GPU_LOCKS:-0}" in
  0)
    exec 9>"$ROOT/locks/helios_world8_gpu.lock"
    flock -n -E 75 9
    exec 8>/tmp/distillschedule_gpu_node.lock
    flock -n -E 75 8
    ;;
  1)
    "$PYTHON" "$REPO/scripts/distillschedule/verify_inherited_gpu_locks.py" \
      --project-root "$ROOT" --project-fd 9 --node-fd 8 --asset-fd 7
    ;;
  *) echo invalid_inherited_gpu_locks_flag >&2; exit 2 ;;
esac
unset DISTILLSCHEDULE_INHERITED_GPU_LOCKS
test ! -e "$RUN"
mkdir -p "$RUN/logs" "$RUN/manifests"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$REPO" status --porcelain)"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 8
nvidia-smi --query-gpu=name --format=csv,noheader | awk '!/H100/ {bad=1} END {exit bad}'
nvidia-smi > "$RUN/logs/nvidia_before.txt"
cat /sys/fs/cgroup/memory.events > "$RUN/logs/cgroup_before.txt"
cat /sys/fs/cgroup/memory.current > "$RUN/logs/cgroup_current_before.txt"
if [[ -r /sys/fs/cgroup/memory.peak ]]; then
  cat /sys/fs/cgroup/memory.peak > "$RUN/logs/cgroup_peak_before.txt"
else
  printf 'NOT_AVAILABLE_ON_THIS_KERNEL\n' > "$RUN/logs/cgroup_peak_before.txt"
fi
export DISTILLSCHEDULE_HARDWARE_PROFILE=h100-8-b16
export PYTHONPATH="$REPO/src:$REPO:$HELIOS"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1
export HF_HOME="$ASSETS/hf_home_official_torch210"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DISTILLSCHEDULE_WAN13B_EMA_IMPLEMENTATION=reference
export DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_CAST_CACHE=0
export DISTILLSCHEDULE_WAN13B_LORA_OPERATOR_ATTRIBUTION=0
unset DISTILLSCHEDULE_STATIC_FSDP2_NSYS_EXACT_RANGE
if [[ "$PLACEMENT" == resident-async128-tree-chunk2048 ]]; then
  export NCCL_ALGO=Tree
  export DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE=1
else
  unset NCCL_ALGO
  export DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE=0
fi
COMMAND=("$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=8
  "$REPO/scripts/distillschedule/probe_wan13b_two_cycle_timing_world16.py"
  --run-id "$RUN_ID" --output-dir "$RUN/execution" --mode "$MODE"
  --placement "$PLACEMENT" --expected-head "$EXPECTED_HEAD"
  --timing-seed 20260903 --timing-arm-id "h100-$PLACEMENT"
  --model-root "$ASSETS/wan13b/Wan2.1-T2V-1.3B-Diffusers-0fad780a"
  --bootstrap "$ASSETS/wan13b/systems_only_bootstrap_v3_head04e6e44"
  --expected-bootstrap-hash 6a713476d9839cf1ea37689d63cb1cc2b405b9f9915ca5314ccdadb67b3cd60d
  --trace-bundle "$ASSETS/m0/traces/syn_feature_trace_rank16_v4_strict_sync_steps20"
  --expected-trace-hash 654d49015921b278ab82e2fc84ea1896692da3fd38ce82fbc500cb1ebfec46d3
  --negative-prompt "$ASSETS/m1/negative_prompt_official_v1" "$@")
printf '%q ' "${COMMAND[@]}" > "$RUN/manifests/command.sh"
printf '\n' >> "$RUN/manifests/command.sh"
env | sort | awk '/^(DISTILLSCHEDULE_|NCCL_|CUDA_VISIBLE_DEVICES=|HF_HOME=|HF_HUB_OFFLINE=|TRANSFORMERS_OFFLINE=|OMP_NUM_THREADS=|PYTHONPATH=)/' > "$RUN/manifests/environment.txt"
set +e
timeout --signal=TERM --kill-after=60s "${DISTILLSCHEDULE_RUN_DEADLINE_SECONDS:-1500}s" \
  "${COMMAND[@]}" > "$RUN/logs/torchrun.log" 2>&1
RC=$?
set -e
# Torchrun owns its children; never pkill or touch an unrelated process.
for _ in $(seq 1 30); do
  test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" && break
  sleep 1
done
cat /sys/fs/cgroup/memory.events > "$RUN/logs/cgroup_after.txt"
cat /sys/fs/cgroup/memory.current > "$RUN/logs/cgroup_current_after.txt"
if [[ -r /sys/fs/cgroup/memory.peak ]]; then
  cat /sys/fs/cgroup/memory.peak > "$RUN/logs/cgroup_peak_after.txt"
else
  printf 'NOT_AVAILABLE_ON_THIS_KERNEL\n' > "$RUN/logs/cgroup_peak_after.txt"
fi
nvidia-smi > "$RUN/logs/nvidia_after.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$RUN/logs/gpu_processes_after.txt"
"$PYTHON" - "$RUN" "$RC" "$EXPECTED_HEAD" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1]); rc=int(sys.argv[2])
def events(name):
    return {k:int(v) for k,v in (line.split() for line in (p/f'logs/cgroup_{name}.txt').read_text().splitlines())}
a,b=events('before'),events('after')
delta={k:b[k]-a[k] for k in ('oom','oom_kill')}
gpu=(p/'logs/gpu_processes_after.txt').read_text().splitlines()
status={'status':'PASS' if rc==0 and delta=={'oom':0,'oom_kill':0} and not gpu else 'FAIL',
        'exit_code':rc,'expected_head':sys.argv[3],'cgroup_oom_delta':delta,
        'cgroup_before':a,'cgroup_after':b,'gpu_processes_after':gpu,
        'host_memory_scope':'shared node/cgroup; peak is not exclusively attributable to this run'}
(p/'run_status.json').write_text(json.dumps(status,indent=2,sort_keys=True)+'\n')
print(json.dumps(status,indent=2))
sys.exit(status['status']!='PASS')
PY
PYTHONPATH="$REPO/src:$REPO" "$PYTHON" "$REPO/scripts/distillschedule/summarize_wan13b_hardware_two_cycle.py" \
  --run-root "$RUN" --hardware-profile h100-8-b16 --output "$RUN/summary.json"
exit "$RC"

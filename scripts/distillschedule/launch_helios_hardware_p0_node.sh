#!/usr/bin/env bash
# Explicit one-host H100 migration; eight workers, sixteen original B1 slots.
# CPU --prepare-launch validates assets and renders the locked policy. The
# validate worker initializes PG/control evidence only, never full models.
set -Eeuo pipefail
if [[ $# != 8 ]]; then
  echo "usage: $0 PROJECT_ROOT RUN_ID VARIANT MODE ARM EXPECTED_HEAD ASSET_GATE ASSET_GATE_SHA256" >&2
  exit 2
fi
PROJECT_ROOT=$1 RUN_ID=$2 VARIANT=$3 MODE=$4 ARM=$5 EXPECTED_HEAD=$6 ASSET_GATE=$7 ASSET_GATE_SHA256=$8
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || exit 2
[[ "$EXPECTED_HEAD" =~ ^[0-9a-f]{40}$ && "$ASSET_GATE_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
case "$VARIANT" in no-gan|gan) ;; *) exit 2 ;; esac
case "$MODE" in validate|correctness|timing) ;; *) exit 2 ;; esac
case "$ARM" in baseline|best|resident|official-swap) ;; *) exit 2 ;; esac
REPO=$(cd "$(dirname "$0")/../.." && pwd -P)
PYTHON=${DISTILLSCHEDULE_PYTHON:?set validated isolated Python executable}
HELIOS=${HELIOS_RUNTIME_ROOT:?set frozen relocated Helios source root}
ASSET_ROOT=${DISTILLSCHEDULE_ASSET_ROOT:?set owned relocated artifact root}
HF_CACHE=${HF_HOME:?set explicit relocated offline cache root}
PORT=${DISTILLSCHEDULE_MASTER_PORT:-28791}
DEADLINE=${DISTILLSCHEDULE_RUN_DEADLINE_SECONDS:-1800}
[[ "$PORT" =~ ^[0-9]+$ && "$PORT" -ge 1024 && "$PORT" -le 65535 ]] || exit 2
[[ "$DEADLINE" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ -x "$PYTHON" && -d "$PROJECT_ROOT" && -d "$ASSET_ROOT" && -d "$HELIOS" ]] || exit 3
SUMMARY="$REPO/scripts/distillschedule/summarize_helios_hardware_p0.py"
RUN="$PROJECT_ROOT/research_artifacts/five_workload_h100_20260930/$RUN_ID"
mkdir -p "$PROJECT_ROOT/locks"
# Project and node locks stay open across audit, construction, run, summary,
# timeout cleanup and teardown. Cooperating launchers use the same node lock.
case "${DISTILLSCHEDULE_INHERITED_GPU_LOCKS:-0}" in
  0)
    exec 9>"$PROJECT_ROOT/locks/helios_world8_gpu.lock"
    flock -n -E 75 9
    exec 8>/tmp/distillschedule_gpu_node.lock
    flock -n -E 75 8
    ;;
  1)
    # Keep one campaign lease across fresh-process arms and construction gaps.
    # Reject missing/wrong descriptors rather than bypassing the lock audit.
    "$PYTHON" "$REPO/scripts/distillschedule/verify_inherited_gpu_locks.py" \
      --project-root "$PROJECT_ROOT"
    ;;
  *) echo "invalid inherited GPU lock mode" >&2; exit 2 ;;
esac
[[ ! -e "$RUN" ]] || { echo "immutable run ID already exists: $RUN" >&2; exit 82; }
mkdir -p "$(dirname "$RUN")"
mkdir "$RUN"
mkdir "$RUN/logs" "$RUN/manifests" "$RUN/execution"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$REPO" status --porcelain --untracked-files=all)"
test -r /sys/fs/cgroup/memory.events
nvidia-smi --query-compute-apps=pid --format=csv,noheader > "$RUN/logs/gpu_processes_before.txt"
test ! -s "$RUN/logs/gpu_processes_before.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader,nounits > "$RUN/logs/gpu_inventory.csv"
"$PYTHON" - "$RUN/logs/gpu_inventory.csv" "$RUN/logs/gpu_inventory.json" <<'PY'
import csv, json, sys
from pathlib import Path
rows = list(csv.reader(Path(sys.argv[1]).read_text().splitlines(), skipinitialspace=True))
if len(rows) != 8 or [int(row[0]) for row in rows] != list(range(8)) or any('H100' not in row[1] for row in rows) or len({row[2] for row in rows}) != 8:
    raise SystemExit('exact eight H100 GPUs required; hardware is never inferred from a logical world16 config')
gpus = [dict(index=int(i), name=name, uuid=uuid, driver=driver, memory_mib=int(memory)) for i, name, uuid, driver, memory in rows]
Path(sys.argv[2]).write_text(json.dumps({'gpus': gpus}, indent=2) + '\n')
PY

CHILD="" TEARDOWN=0
snapshot() {
  local label=$1
  cat /proc/meminfo > "$RUN/logs/cpu_memory_${label}.txt"
  cat /sys/fs/cgroup/memory.events > "$RUN/logs/cgroup_${label}.txt"
  for name in current peak max; do
    if [[ -r "/sys/fs/cgroup/memory.$name" ]]; then
      cat "/sys/fs/cgroup/memory.$name" > "$RUN/logs/cgroup_${name}_${label}.txt"
    else
      printf 'NOT_AVAILABLE_ON_THIS_KERNEL\n' > "$RUN/logs/cgroup_${name}_${label}.txt"
    fi
  done
  nvidia-smi > "$RUN/logs/nvidia_${label}.txt"
}
cleanup_owned_workers() {
  # The dedicated session/process group was created by this launcher. Never
  # signal a GPU PID discovered in an audit or any foreign process by name.
  if [[ -n "$CHILD" ]] && kill -0 -- "-$CHILD" 2>/dev/null; then
    kill -TERM -- "-$CHILD" 2>/dev/null || true
    for _ in {1..30}; do
      kill -0 -- "-$CHILD" 2>/dev/null || break
      sleep 1
    done
    if kill -0 -- "-$CHILD" 2>/dev/null; then
      kill -KILL -- "-$CHILD" 2>/dev/null || true
    fi
    wait "$CHILD" 2>/dev/null || true
  fi
}
teardown() {
  local rc=$1
  [[ "$TEARDOWN" == 0 ]] || return 0
  TEARDOWN=1
  cleanup_owned_workers
  for _ in {1..30}; do
    [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] && break
    sleep 1
  done
  snapshot after
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$RUN/logs/gpu_processes_after.txt"
  git -C "$REPO" status --porcelain --untracked-files=all > "$RUN/logs/source_status_after.txt"
  PYTHONPATH="$REPO/src:$REPO" "$PYTHON" "$SUMMARY" --seal-run --run-root "$RUN" --expected-head "$EXPECTED_HEAD" --exit-code "$rc"
}
on_exit() {
  local rc=$?
  trap - EXIT INT TERM
  set +e
  teardown "$rc"
  local health_rc=$?
  if [[ "$rc" == 0 && "$health_rc" != 0 ]]; then rc=$health_rc; fi
  exit "$rc"
}
snapshot before
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Reject free-form overrides. All performance flags come from the registry
# policy, all model reads from the relocated precasts, and --*-root arguments
# retain the unchanged source.path identities even if /share is absent.
PREPARE=(--prepare-launch --run-root "$RUN" --run-id "$RUN_ID" --variant "$VARIANT"
  --mode "$MODE" --arm "$ARM" --expected-head "$EXPECTED_HEAD" --asset-root "$ASSET_ROOT"
  --asset-gate "$ASSET_GATE" --asset-gate-sha256 "$ASSET_GATE_SHA256" --helios-root "$HELIOS" --hf-home "$HF_CACHE")
if [[ "$MODE" == timing ]]; then
  PREPARE+=(--correctness-summary "${DISTILLSCHEDULE_HELIOS_H100_CORRECTNESS_SUMMARY:?set fresh H100 correctness summary}"
    --correctness-sha256 "${DISTILLSCHEDULE_HELIOS_H100_CORRECTNESS_SHA256:?set exact fresh correctness summary SHA}")
fi
PYTHONPATH="$REPO/src:$REPO" PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "$SUMMARY" "${PREPARE[@]}" > "$RUN/logs/prepare.log" 2>&1
mapfile -d '' -t WORKER < "$RUN/manifests/worker_argv.nul"
mapfile -d '' -t UNSET < "$RUN/manifests/unset_environment.nul"
# No unrelated prior workload variables may silently tune this run. Only the
# complete frozen environment rendered above is reintroduced below.
while IFS= read -r key; do
  case "$key" in DISTILLSCHEDULE_*|NCCL_*|HELIOS_*|LOCAL_KERNELS|PYTORCH_CUDA_ALLOC_CONF|PYTORCH_ALLOC_CONF) unset "$key" ;; esac
done < <(compgen -e)
for key in "${UNSET[@]}"; do unset "$key"; done
while IFS= read -r -d '' binding; do export "$binding"; done < "$RUN/manifests/environment.nul"
# Prefer only the already selected environment's native libraries. No install,
# mutation, download or environment repair is done by this launcher.
PYENV=$(cd "$(dirname "$PYTHON")/.." && pwd -P)
export PATH="$PYENV/bin:$PATH"
export LD_LIBRARY_PATH="$PYENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
env | sort | awk '/^(DISTILLSCHEDULE_|NCCL_|CUDA_VISIBLE_DEVICES=|HF_HOME=|HF_HUB_OFFLINE=|TRANSFORMERS_OFFLINE=|OMP_NUM_THREADS=|PYTHONPATH=|LD_LIBRARY_PATH=|LOCAL_KERNELS=|PYTORCH_)/' > "$RUN/manifests/environment.txt"
"$PYTHON" - "$RUN/manifests/environment_versions.json" <<'PY'
import importlib.metadata, json, platform, sys
from pathlib import Path
versions = {}
for package in ('torch', 'triton', 'deepspeed', 'accelerate', 'transformers', 'diffusers', 'peft', 'safetensors'):
    versions[package] = importlib.metadata.version(package)
Path(sys.argv[1]).write_text(json.dumps({'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'packages': versions}, indent=2) + '\n')
PY
COMMAND=("$PYTHON" -m torch.distributed.run --nnodes=1 --nproc-per-node=8 --node-rank=0
  --master-addr=127.0.0.1 --master-port="$PORT" "${WORKER[@]}")
printf '%q ' "${COMMAND[@]}" > "$RUN/manifests/command.sh"
printf '\n' >> "$RUN/manifests/command.sh"
set +e
setsid timeout --signal=TERM --kill-after=60s "${DEADLINE}s" "${COMMAND[@]}" > "$RUN/logs/torchrun.log" 2>&1 &
CHILD=$!
wait "$CHILD"
RC=$?
set -e
teardown "$RC"
# Summary consumes actual rank bytes and their SHA index. Timing output stays
# diagnostic-only until full cross-arm numerical comparison exists.
"$PYTHON" "$SUMMARY" --run-root "$RUN" --expected-head "$EXPECTED_HEAD" --output "$RUN/summary.json" > "$RUN/logs/summary.log" 2>&1
exit "$RC"

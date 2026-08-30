#!/usr/bin/env bash
# One node of the immutable LiveWan cross-scale world16 experiment.
set -Eeuo pipefail

NODE_RANK="${1:?node rank 0 or 1}"
RUN_ID="${2:?immutable run id}"
RUN_KIND="${3:?validate, correctness, timing, or profile}"
EXECUTION_MODE="${4:?baseline or pipeline}"
EXPECTED_HEAD="${5:?exact DMD-System HEAD}"
MASTER_ADDR="${6:?node0 address}"
MASTER_PORT="${7:?rendezvous port}"

[[ "${NODE_RANK}" == 0 || "${NODE_RANK}" == 1 ]] || exit 2
[[ "${RUN_KIND}" == validate || "${RUN_KIND}" == correctness || \
   "${RUN_KIND}" == timing || "${RUN_KIND}" == profile ]] || exit 2
[[ "${EXECUTION_MODE}" == baseline || "${EXECUTION_MODE}" == pipeline ]] || exit 2
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || exit 2
[[ "${EXPECTED_HEAD}" =~ ^[0-9a-f]{40}$ ]] || exit 2

ROOT="${DMD_SYSTEM_ROOT:?set DMD_SYSTEM_ROOT}"
PYENV="${DISTILLSCHEDULE_PYENV:-/tmp/distillschedule-helios-official-torch210}"
PYTHON="${PYENV}/bin/python"
TORCHRUN="${PYENV}/bin/torchrun"
RUN_ROOT="${DISTILLSCHEDULE_OUTPUT_DIR:?set DISTILLSCHEDULE_OUTPUT_DIR}"
NODE_LOG="${RUN_ROOT}/launcher/node_${NODE_RANK}"
LOCK_FILE="${DISTILLSCHEDULE_NODE_LOCK:-/tmp/distillschedule_gpu_node.lock}"
SOURCE_ROOT="${LIVEWAN_WAN21_SOURCE:?set LIVEWAN_WAN21_SOURCE}"
SOURCE_RECEIPT="${LIVEWAN_SOURCE_RECEIPT:?set LIVEWAN_SOURCE_RECEIPT}"

if [[ "${EXECUTION_MODE}" == baseline ]]; then
  PLAN="${ROOT}/configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_fsdp2.json"
else
  PLAN="${ROOT}/configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8.json"
fi

for executable in "${PYTHON}" "${TORCHRUN}"; do
  [[ -x "${executable}" ]] || { echo "missing_executable=${executable}" >&2; exit 3; }
done
for path in "${ROOT}" "${PLAN}" "${LIVEWAN_STUDENT_CHECKPOINT:?}" \
  "${LIVEWAN_CRITIC_CHECKPOINT:?}" "${LIVEWAN_TEACHER_CHECKPOINT:?}" \
  "${LIVEWAN_OPERATION_TRACE:?}" "${LIVEWAN_SYSTEMS_BOOTSTRAP:?}"; do
  [[ -e "${path}" ]] || { echo "missing_path=${path}" >&2; exit 3; }
done
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_HEAD}" ]] || {
  echo source_head_mismatch >&2; exit 4;
}
git -C "${ROOT}" diff --quiet && git -C "${ROOT}" diff --cached --quiet || {
  echo source_worktree_dirty >&2; exit 4;
}
[[ "$(git -C "${ROOT}/third_party/livewan" rev-parse HEAD)" == \
  1eabf43c777bc5b2816926ca998cd4e4c2dfa3d5 ]] || exit 4
[[ "$(git -C "${ROOT}/third_party/wan21" rev-parse HEAD)" == \
  9737cba9c1c3c4d04b33fcad41c111989865d315 ]] || exit 4

if [[ "${RUN_KIND}" == timing || "${RUN_KIND}" == profile ]]; then
  CORRECTNESS_RECEIPT="${LIVEWAN_PAIR_CORRECTNESS_RECEIPT:?timing requires LIVEWAN_PAIR_CORRECTNESS_RECEIPT}"
  CORRECTNESS_SHA256="${LIVEWAN_PAIR_CORRECTNESS_SHA256:?timing requires LIVEWAN_PAIR_CORRECTNESS_SHA256}"
  [[ -f "${CORRECTNESS_RECEIPT}" && "${CORRECTNESS_SHA256}" =~ ^[0-9a-f]{64}$ ]] || exit 3
  [[ "$(sha256sum "${CORRECTNESS_RECEIPT}" | awk '{print $1}')" == "${CORRECTNESS_SHA256}" ]] || exit 3
  "${PYTHON}" - "${CORRECTNESS_RECEIPT}" "${EXPECTED_HEAD}" <<'PY'
import json, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text())
expected_head = sys.argv[2]
if value.get("status") != "PASS" or value.get("claimable_for_timing") is not True:
    raise SystemExit("paired correctness gate is not PASS")
if set(value.get("heads", {}).values()) != {expected_head}:
    raise SystemExit("paired correctness gate uses a different source HEAD")
PY
fi

mkdir -p "${NODE_LOG}" "${RUN_ROOT}"
mkdir -p "$(dirname "${SOURCE_ROOT}")"
[[ ! -e "${NODE_LOG}/launch_manifest.txt" ]] || { echo run_id_already_used >&2; exit 82; }
exec 9>"${LOCK_FILE}"
flock -n 9 || { echo gpu_lock_busy >&2; exit 73; }
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')" ]] || {
  echo gpu_processes_present >&2; exit 81;
}

exec 8>"${SOURCE_ROOT}.prepare.lock"
flock 8
"${ROOT}/scripts/distillschedule/prepare_livewan_crossscale_source.sh" \
  --output "${SOURCE_ROOT}" --receipt "${SOURCE_RECEIPT}" \
  >"${NODE_LOG}/source_prepare.log" 2>&1
flock -u 8

export PATH="${PYENV}/bin:${PATH}"
export PYTHONPATH="${ROOT}/src:${ROOT}:${ROOT}/third_party/livewan:${SOURCE_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
export DISTILLSCHEDULE_NODE_RANK="${NODE_RANK}"
export DISTILLSCHEDULE_RUN_ID="${RUN_ID}"
export DISTILLSCHEDULE_RUN_KIND="${RUN_KIND}"
export DISTILLSCHEDULE_PYTHON="${PYTHON}"
export LIVEWAN_SOURCE_RECEIPT="${SOURCE_RECEIPT}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_ASYNC_ERROR_HANDLING=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

{
  echo "run_id=${RUN_ID}"
  echo "node_rank=${NODE_RANK}"
  echo "host=$(hostname)"
  echo "head=${EXPECTED_HEAD}"
  echo "livewan=$(git -C "${ROOT}/third_party/livewan" rev-parse HEAD)"
  echo "wan21=$(git -C "${ROOT}/third_party/wan21" rev-parse HEAD)"
  echo "python=$(${PYTHON} --version 2>&1)"
  "${PYTHON}" - <<'PY'
import torch
print("torch=" + torch.__version__)
print("torch_cuda=" + str(torch.version.cuda))
print("nccl=" + str(torch.cuda.nccl.version()))
PY
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader
} >"${NODE_LOG}/launch_manifest.txt"
cp /sys/fs/cgroup/memory.events "${NODE_LOG}/cgroup_memory_events_before.txt" 2>/dev/null || true

monitor_pid=""
stop_monitor() {
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
  fi
}
trap stop_monitor EXIT
monitor() {
  echo 'unix_time,cgroup_bytes,host_available_bytes,gpu_used_mib,gpu_free_mib,gpu_util_percent' \
    >"${NODE_LOG}/resource_monitor.csv"
  while true; do
    now="$(date +%s.%N)"
    cgroup="$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo -1)"
    host_available="$(awk '/^MemAvailable:/{print $2*1024}' /proc/meminfo)"
    gpu_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd: - || echo unavailable)"
    gpu_free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | paste -sd: - || echo unavailable)"
    gpu_util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | paste -sd: - || echo unavailable)"
    echo "${now},${cgroup},${host_available},${gpu_used},${gpu_free},${gpu_util}" \
      >>"${NODE_LOG}/resource_monitor.csv"
    sleep 5
  done
}
monitor &
monitor_pid=$!

RESOLVED="${RUN_ROOT}/resolved_plan.node${NODE_RANK}.json"
"${PYTHON}" -m distillschedule.cli render "${PLAN}" --output "${RESOLVED}" \
  >"${NODE_LOG}/render.log" 2>&1

# Fail before 14B construction if the two hosts accidentally form split worlds
# or the default NCCL communicator is unusable.  The probe is intentionally
# model-free and the node lock remains held throughout it.
TOPOLOGY="${RUN_ROOT}/preflight/topology.json"
mkdir -p "${RUN_ROOT}/preflight"
set +e
"${TORCHRUN}" --nnodes=2 --nproc-per-node=8 --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}" \
  "${ROOT}/scripts/distillschedule/probe_world16_topology.py" \
  --output "${TOPOLOGY}" >"${NODE_LOG}/topology_probe.log" 2>&1
topology_status=$?
set -e
[[ ${topology_status} -eq 0 ]] || { echo topology_probe_failed >&2; exit "${topology_status}"; }

if [[ "${EXECUTION_MODE}" == pipeline ]]; then
  TRANSPORT_DIR="${RUN_ROOT}/preflight/transport"
  set +e
  "${TORCHRUN}" --nnodes=2 --nproc-per-node=8 --node-rank="${NODE_RANK}" \
    --master-addr="${MASTER_ADDR}" --master-port="$((MASTER_PORT + 1))" \
    "${ROOT}/scripts/distillschedule/probe_livewan_crossscale_transport.py" \
    --output-dir "${TRANSPORT_DIR}" >"${NODE_LOG}/transport_probe.log" 2>&1
  transport_status=$?
  set -e
  [[ ${transport_status} -eq 0 ]] || { echo transport_probe_failed >&2; exit "${transport_status}"; }
fi

set +e
if [[ "${RUN_KIND}" == validate ]]; then
  "${PYTHON}" "${ROOT}/scripts/distillschedule/livewan_crossscale_entrypoint.py" \
    --plan "${RESOLVED}" --mode "${EXECUTION_MODE}" --run-kind validate \
    --repo-root "${ROOT}" >"${NODE_LOG}/console.log" 2>&1
  status=$?
else
  "${TORCHRUN}" --nnodes=2 --nproc-per-node=8 --node-rank="${NODE_RANK}" \
    --master-addr="${MASTER_ADDR}" --master-port="$((MASTER_PORT + 2))" \
    "${ROOT}/scripts/distillschedule/livewan_crossscale_entrypoint.py" \
    --plan "${RESOLVED}" --mode "${EXECUTION_MODE}" --run-kind "${RUN_KIND}" \
    --repo-root "${ROOT}" >"${NODE_LOG}/console.log" 2>&1
  status=$?
fi
set -e

cp /sys/fs/cgroup/memory.events "${NODE_LOG}/cgroup_memory_events_after.txt" 2>/dev/null || true
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader \
  >"${NODE_LOG}/gpu_after.csv" || true
[[ ${status} -eq 0 ]] || exit "${status}"
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')" ]] || {
  echo gpu_processes_remain >&2; exit 88;
}

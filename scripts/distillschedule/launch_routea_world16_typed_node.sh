#!/usr/bin/env bash
# Per-host worker for a centrally coordinated Route-A Full5 world16 run.
# The worker is fail-closed: normalized contract and process-group topology must
# pass on both hosts before any 14B model construction begins.
set -Eeuo pipefail

NODE_RANK="${1:?node rank 0 or 1}"
RUN_ID="${2:?immutable run id}"
EXPECTED_HEAD="${3:?expected DMD-System HEAD}"
EXPECTED_PLAN_SHA="${4:?expected plan sha256}"
MASTER_ADDR="${5:?master address}"
MASTER_PORT="${6:?master port}"
PLAN_REL="${7:?repository-relative typed plan path}"
[[ "${NODE_RANK}" == 0 || "${NODE_RANK}" == 1 ]] || { echo invalid_node_rank >&2; exit 2; }
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo invalid_run_id >&2; exit 2; }
[[ "${MASTER_PORT}" =~ ^[0-9]+$ ]] || { echo invalid_master_port >&2; exit 2; }
[[ "${PLAN_REL}" =~ ^configs/distillschedule/plans/[A-Za-z0-9._/-]+\.json$ ]] || { echo invalid_plan_path >&2; exit 2; }
[[ "${PLAN_REL}" != *".."* ]] || { echo invalid_plan_path >&2; exit 2; }

ROOT="${DMD_SYSTEM_ROOT:-/share/project/shuchenweng/yongyan_liu/DMD-System}"
HELIOS_ROOT="${HELIOS_RUNTIME_ROOT:-/share/project/shuchenweng/yongyan_liu/DistillSchedule}"
PYENV="${DISTILLSCHEDULE_PYENV:-/tmp/distillschedule-helios-official-torch210}"
PYTHON="${PYENV}/bin/python"
ACCELERATE="${PYENV}/bin/accelerate"
DEEPSPEED="${PYENV}/bin/deepspeed"
PLAN="${ROOT}/${PLAN_REL}"
DS_CONFIG="${ROOT}/scripts/accelerate_configs/zero1_cpu_optimizer_offload.json"
OUT="${ROOT}/artifacts/runs/${RUN_ID}"
NODE_LOG="${OUT}/launcher/node_${NODE_RANK}"
PREREG_DIR="${OUT}/preregistration"
CACHE_ROOT="/tmp/distillschedule_typed_routea/${RUN_ID}/node_${NODE_RANK}"
LOCK_FILE="/tmp/distillschedule_gpu_node.lock"
TRACE_BUNDLE="${DISTILLSCHEDULE_TRACE_BUNDLE:-/share/project/shuchenweng/DistillSchedule_assets/m0/traces/syn_feature_trace_rank16_v3_sync_steps20}"
CHECKPOINT="${DISTILLSCHEDULE_CHECKPOINT:-/share/project/shuchenweng/yongyan_liu/DistillSchedule/artifacts/runs/iteration124_m0_b4fast_zero1_shared_cw4_x16_checkpoint10/checkpoint-10}"
DATA_ROOT="${HELIOS_DATA_ROOT:-/share/project/shuchenweng/DistillSchedule_assets/m0/datasets/syn_feature_rank16}"
GENERATOR_ROOT="${HELIOS_DISTILLED_ROOT:-/share/project/shuchenweng/DistillSchedule_assets/Helios-Distilled}"
SCORE_ROOT="${HELIOS_BASE_ROOT:-/home/shuchenweng/xty/Helios-main/weights}"
PRECAST_GENERATOR="${DISTILLSCHEDULE_PRECAST_GENERATOR_PATH:-/share/project/shuchenweng/DistillSchedule_assets/precast/helios-distilled-transformer-ode-mixed-bf16-v1}"
PRECAST_SCORE="${DISTILLSCHEDULE_PRECAST_SCORE_PATH:-/share/project/shuchenweng/DistillSchedule_assets/precast/helios-score-transformer-mixed-bf16-v1}"

for executable in "${PYTHON}" "${ACCELERATE}" "${DEEPSPEED}"; do
  [[ -x "${executable}" ]] || { echo "runtime_missing=${executable}" >&2; exit 3; }
done
for file in "${PLAN}" "${DS_CONFIG}" "${HELIOS_ROOT}/train_helios.py"; do
  [[ -f "${file}" ]] || { echo "file_missing=${file}" >&2; exit 3; }
done
for asset in "${TRACE_BUNDLE}" "${CHECKPOINT}" "${DATA_ROOT}" "${GENERATOR_ROOT}" "${SCORE_ROOT}" "${PRECAST_GENERATOR}" "${PRECAST_SCORE}"; do
  [[ -e "${asset}" ]] || { echo "asset_missing=${asset}" >&2; exit 5; }
done
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_HEAD}" ]] || { echo head_drift >&2; exit 4; }
[[ -z "$(git -C "${ROOT}" status --short)" ]] || { echo repository_not_clean >&2; exit 4; }
[[ "$(sha256sum "${PLAN}" | awk '{print $1}')" == "${EXPECTED_PLAN_SHA}" ]] || { echo plan_drift >&2; exit 4; }
[[ -z "$(git -C "${HELIOS_ROOT}" status --short)" ]] || { echo helios_repository_not_clean >&2; exit 4; }

exec 9>"${LOCK_FILE}"
flock -n 9 || { echo gpu_lock_busy >&2; exit 73; }
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')" ]]; then
  echo gpu_processes_present >&2
  exit 81
fi
[[ ! -e "${CACHE_ROOT}" ]] || { echo node_cache_exists >&2; exit 82; }
mkdir -p "${NODE_LOG}" "${PREREG_DIR}" "${CACHE_ROOT}/frozen" "${CACHE_ROOT}/immutable"

export PATH="${PYENV}/bin:${PATH}"
export PYTHONPATH="${ROOT}/src:${ROOT}:${HELIOS_ROOT}"
export DISTILLSCHEDULE_RUN_ID="${RUN_ID}"
export DISTILLSCHEDULE_OUTPUT_DIR="${OUT}"
export DISTILLSCHEDULE_FROZEN_BLOCK_MMAP_CACHE="${CACHE_ROOT}/frozen"
export DISTILLSCHEDULE_IMMUTABLE_MODEL_MMAP_CACHE="${CACHE_ROOT}/immutable"
export DISTILLSCHEDULE_NODE_RANK="${NODE_RANK}"
export DISTILLSCHEDULE_MASTER_ADDR="${MASTER_ADDR}"
export DISTILLSCHEDULE_MASTER_PORT="${MASTER_PORT}"
export HELIOS_RUNTIME_ROOT="${HELIOS_ROOT}"
export DISTILLSCHEDULE_PYTHON="${PYTHON}"
export DISTILLSCHEDULE_ACCELERATE="${ACCELERATE}"
export DISTILLSCHEDULE_TRACE_BUNDLE="${TRACE_BUNDLE}"
export DISTILLSCHEDULE_CHECKPOINT="${CHECKPOINT}"
export HELIOS_DATA_ROOT="${DATA_ROOT}"
export HELIOS_DISTILLED_ROOT="${GENERATOR_ROOT}"
export HELIOS_BASE_ROOT="${SCORE_ROOT}"
export DISTILLSCHEDULE_PRECAST_GENERATOR_PATH="${PRECAST_GENERATOR}"
export DISTILLSCHEDULE_PRECAST_SCORE_PATH="${PRECAST_SCORE}"

ACCEL_CONFIG="${NODE_LOG}/accelerate_world16_node${NODE_RANK}.yaml"
cat >"${ACCEL_CONFIG}.tmp" <<EOF
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_config_file: ${DS_CONFIG}
  deepspeed_multinode_launcher: standard
fsdp_config: {}
machine_rank: ${NODE_RANK}
main_training_function: main
mixed_precision: bf16
num_machines: 2
num_processes: 16
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
mv "${ACCEL_CONFIG}.tmp" "${ACCEL_CONFIG}"
export DISTILLSCHEDULE_ACCELERATE_CONFIG="${ACCEL_CONFIG}"

MANIFEST="${NODE_LOG}/launch_manifest.txt"
MONITOR="${NODE_LOG}/resource.csv"
GPU_INVENTORY="${NODE_LOG}/gpu_inventory.csv"
RUNTIME_IDENTITY="${NODE_LOG}/runtime_identity.json"
COMPILED_PREFLIGHT="${NODE_LOG}/compiled_preflight.json"
DRY_RUN_LOG="${NODE_LOG}/dry_run.log"
TOPOLOGY="${PREREG_DIR}/topology.json"
NODE_CONTRACT="${PREREG_DIR}/node_${NODE_RANK}_contract.json"
FINAL_CONTRACT="${PREREG_DIR}/evidence_contract.json"

# GPU inventory is deliberately separate from the strict key=value manifest.
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv >"${GPU_INVENTORY}"
"${PYTHON}" - "${RUNTIME_IDENTITY}" "${PYTHON}" "${ACCELERATE}" "${DEEPSPEED}" "${HELIOS_ROOT}" <<'PY'
import hashlib, importlib.metadata, json, pathlib, subprocess, sys
out, python, accelerate, deepspeed, helios = map(pathlib.Path, sys.argv[1:])
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()
def version(command):
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
def package_version(name):
    return importlib.metadata.version(name)
payload={
  'schema_version':'distillschedule.runtime_identity.v1',
  'python':{'path':str(python.resolve()),'sha256':sha(python),'version':version([str(python),'--version'])},
  # Console-script option support differs across the pinned Accelerate and
  # DeepSpeed releases.  Package metadata is the stable version identity; the
  # executable SHA still binds the concrete launcher used below.
  'accelerate':{'path':str(accelerate.resolve()),'sha256':sha(accelerate),'version':package_version('accelerate')},
  'deepspeed':{'path':str(deepspeed.resolve()),'sha256':sha(deepspeed),'version':package_version('deepspeed')},
  'helios':{
    'root':str(helios.resolve()),
    'head':version(['git','-C',str(helios),'rev-parse','HEAD']),
    'entrypoint_sha256':sha(helios/'train_helios.py'),
    'dirty':bool(version(['git','-C',str(helios),'status','--short'])),
  },
}
tmp=out.with_name('.'+out.name+'.tmp'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); tmp.replace(out)
PY

{
  echo schema_version=distillschedule.typed_routea_world16.node_launch.v3
  echo run_id="${RUN_ID}"
  echo node_rank="${NODE_RANK}"
  echo host="$(hostname)"
  echo started_at="$(date -Is)"
  echo dmd_system_head="${EXPECTED_HEAD}"
  echo dmd_system_plan_sha256="${EXPECTED_PLAN_SHA}"
  echo dmd_system_plan_path="${PLAN_REL}"
  echo helios_head="$(git -C "${HELIOS_ROOT}" rev-parse HEAD)"
  echo helios_entrypoint_sha256="$(sha256sum "${HELIOS_ROOT}/train_helios.py" | awk '{print $1}')"
  echo python_path="$(readlink -f "${PYTHON}")"
  echo accelerate_path="$(readlink -f "${ACCELERATE}")"
  echo deepspeed_path="$(readlink -f "${DEEPSPEED}")"
  echo runtime_identity_path="${RUNTIME_IDENTITY}"
  echo runtime_identity_sha256="$(sha256sum "${RUNTIME_IDENTITY}" | awk '{print $1}')"
  echo gpu_inventory_path="${GPU_INVENTORY}"
  echo gpu_inventory_sha256="$(sha256sum "${GPU_INVENTORY}" | awk '{print $1}')"
  echo gpu_inventory_rows="$(( $(wc -l <"${GPU_INVENTORY}") - 1 ))"
  echo accelerate_config_path="${ACCEL_CONFIG}"
  echo accelerate_config_sha256="$(sha256sum "${ACCEL_CONFIG}" | awk '{print $1}')"
  echo master_addr="${MASTER_ADDR}"
  echo master_port="${MASTER_PORT}"
  echo cgroup_memory_current_before="$(cat /sys/fs/cgroup/memory.current)"
  echo cgroup_memory_peak_before="$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo unavailable)"
  echo cgroup_memory_max="$(cat /sys/fs/cgroup/memory.max)"
  echo memory_events_before="$(tr '\n' ';' </sys/fs/cgroup/memory.events)"
} >"${MANIFEST}"

# Compile and materialize the node-qualified rendered config without constructing a model.
"${PYTHON}" "${ROOT}/scripts/distillctl.py" validate "${PLAN}" --repo-root "${ROOT}" --json >"${COMPILED_PREFLIGHT}"
"${PYTHON}" "${ROOT}/scripts/distillctl.py" run "${PLAN}" --repo-root "${ROOT}" --dry-run >"${DRY_RUN_LOG}"
RENDERED_CONFIG="${OUT}/resolved_helios_config.node${NODE_RANK}.yaml"
[[ -f "${RENDERED_CONFIG}" ]] || { echo rendered_config_missing >&2; exit 84; }

"${PYTHON}" "${ROOT}/scripts/distillschedule/build_routea_evidence_contract.py" node \
  --node-rank "${NODE_RANK}" \
  --plan "${PLAN}" \
  --repo-root "${ROOT}" \
  --compiled-preflight "${COMPILED_PREFLIGHT}" \
  --rendered-config "${RENDERED_CONFIG}" \
  --accelerate-config "${ACCEL_CONFIG}" \
  --trace-bundle "${TRACE_BUNDLE}" \
  --checkpoint "${CHECKPOINT}" \
  --asset "helios_runtime=${HELIOS_ROOT}" \
  --asset "generator=${GENERATOR_ROOT}" \
  --asset "score=${SCORE_ROOT}" \
  --asset "data=${DATA_ROOT}" \
  --asset "precast_generator=${PRECAST_GENERATOR}" \
  --asset "precast_score=${PRECAST_SCORE}" \
  --output "${NODE_CONTRACT}" >"${NODE_LOG}/node_contract_build.json"

# Both node contracts must exist and normalize to the same training contract.
for _ in $(seq 1 600); do
  [[ -f "${PREREG_DIR}/node_0_contract.json" && -f "${PREREG_DIR}/node_1_contract.json" ]] && break
  sleep 1
done
[[ -f "${PREREG_DIR}/node_0_contract.json" && -f "${PREREG_DIR}/node_1_contract.json" ]] || {
  echo peer_node_contract_timeout >&2
  exit 85
}
if [[ "${NODE_RANK}" == 0 ]]; then
  "${PYTHON}" "${ROOT}/scripts/distillschedule/build_routea_evidence_contract.py" finalize \
    --node-contract "${PREREG_DIR}/node_0_contract.json" \
    --node-contract "${PREREG_DIR}/node_1_contract.json" \
    --output "${FINAL_CONTRACT}" >"${NODE_LOG}/contract_finalize.json"
fi
for _ in $(seq 1 300); do
  [[ -f "${FINAL_CONTRACT}" ]] && break
  sleep 1
done
[[ -f "${FINAL_CONTRACT}" ]] || { echo finalized_contract_timeout >&2; exit 86; }
"${PYTHON}" "${ROOT}/scripts/distillschedule/build_routea_evidence_contract.py" check "${FINAL_CONTRACT}" >"${NODE_LOG}/contract_check.json"
{
  echo evidence_contract_path="${FINAL_CONTRACT}"
  echo evidence_contract_sha256="$(sha256sum "${FINAL_CONTRACT}" | awk '{print $1}')"
  echo node_contract_path="${NODE_CONTRACT}"
  echo node_contract_sha256="$(sha256sum "${NODE_CONTRACT}" | awk '{print $1}')"
  echo rendered_config_path="${RENDERED_CONFIG}"
  echo rendered_config_sha256="$(sha256sum "${RENDERED_CONFIG}" | awk '{print $1}')"
} >>"${MANIFEST}"

# Cheap process-group proof: exact ranks/hosts/local ranks plus a tiny NCCL collective.
set +e
"${ACCELERATE}" launch \
  --config_file "${ACCEL_CONFIG}" \
  --num_machines 2 \
  --num_processes 16 \
  --machine_rank "${NODE_RANK}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  "${ROOT}/scripts/distillschedule/probe_world16_topology.py" \
  --output "${TOPOLOGY}" \
  >"${NODE_LOG}/topology_probe.log" 2>&1
TOPOLOGY_RC=$?
set -e
[[ "${TOPOLOGY_RC}" == 0 ]] || { echo topology_probe_failed >&2; exit "${TOPOLOGY_RC}"; }
for _ in $(seq 1 120); do [[ -f "${TOPOLOGY}" ]] && break; sleep 1; done
[[ -f "${TOPOLOGY}" ]] || { echo topology_manifest_missing >&2; exit 87; }
"${PYTHON}" - "${TOPOLOGY}" <<'PY'
import json, pathlib, sys
from distillschedule.runtime.topology import validate_topology_probe_payload

p=pathlib.Path(sys.argv[1]); v=json.loads(p.read_text())
validate_topology_probe_payload(
    v,
    expected_nodes=2,
    expected_local_world_size=8,
    expected_world_size=16,
    require_cuda=True,
)
PY
{
  echo topology_path="${TOPOLOGY}"
  echo topology_sha256="$(sha256sum "${TOPOLOGY}" | awk '{print $1}')"
  echo pre_model_gate=PASS
} >>"${MANIFEST}"

monitor() {
  echo 'unix_time,cgroup_bytes,host_used_bytes,gpu_used_mib,gpu_util_percent' >"${MONITOR}"
  while true; do
    now="$(date +%s.%N)"
    cgroup="$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo -1)"
    host_used="$(awk '/^MemTotal:/{t=$2}/^MemAvailable:/{a=$2}END{print (t-a)*1024}' /proc/meminfo)"
    gpu_mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd: - || echo unavailable)"
    gpu_util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | paste -sd: - || echo unavailable)"
    echo "${now},${cgroup},${host_used},${gpu_mem},${gpu_util}" >>"${MONITOR}"
    sleep 1
  done
}

MONITOR_PID=""
RUN_PID=""
cleanup() {
  if [[ -n "${RUN_PID:-}" ]] && kill -0 "${RUN_PID}" 2>/dev/null; then
    kill -TERM -- "-${RUN_PID}" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-${RUN_PID}" 2>/dev/null || true
    wait "${RUN_PID}" 2>/dev/null || true
  fi
  if [[ -n "${MONITOR_PID:-}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
  {
    echo finished_at="$(date -Is)"
    echo cgroup_memory_current_after="$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo -1)"
    echo cgroup_memory_peak_after="$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo unavailable)"
    echo memory_events_after="$(tr '\n' ';' </sys/fs/cgroup/memory.events 2>/dev/null || true)"
  } >>"${MANIFEST}" 2>/dev/null || true
  rm -rf "${CACHE_ROOT}" 2>/dev/null || true
}
trap cleanup EXIT TERM INT HUP
monitor &
MONITOR_PID=$!

cd "${ROOT}"
set +e
setsid timeout --signal=TERM --kill-after=180s 14400s \
  "${PYTHON}" scripts/distillctl.py run "${PLAN}" --repo-root "${ROOT}" \
  >"${NODE_LOG}/console.log" 2>&1 &
RUN_PID=$!
wait "${RUN_PID}"
RC=$?
RUN_PID=""
set -e
echo command_return_code="${RC}" >>"${MANIFEST}"
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')" ]]; then
  echo gpu_processes_remain >&2
  (( RC == 0 )) && RC=88
fi
echo final_return_code="${RC}" >>"${MANIFEST}"
exit "${RC}"

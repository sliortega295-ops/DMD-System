#!/usr/bin/env bash
# Generate exactly 80 source-format 81-pixel-frame/21-latent-frame clips with
# the pinned LiveWan generator, sharded over two owned 8-GPU nodes.
set -Eeuo pipefail

NODE_RANK="${1:?node rank 0 or 1}"
RUN_ID="${2:?immutable run id}"
[[ "${NODE_RANK}" == 0 || "${NODE_RANK}" == 1 ]] || exit 2
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || exit 2

ROOT="${DMD_SYSTEM_ROOT:?set DMD_SYSTEM_ROOT}"
PYENV="${DISTILLSCHEDULE_PYENV:-/tmp/distillschedule-helios-official-torch210}"
PYTHON="${PYENV}/bin/python"
SOURCE="${LIVEWAN_WAN21_SOURCE:?set LIVEWAN_WAN21_SOURCE}"
PROMPTS="${LIVEWAN_PROMPTS:?set LIVEWAN_PROMPTS}"
MODEL="${LIVEWAN_STUDENT_CHECKPOINT:?set LIVEWAN_STUDENT_CHECKPOINT}"
OUTPUT="${LIVEWAN_TEACHER_LATENTS:?set LIVEWAN_TEACHER_LATENTS}"
LOG_ROOT="${LIVEWAN_INPUT_LOG_ROOT:?set LIVEWAN_INPUT_LOG_ROOT}/node_${NODE_RANK}"
LOCK="${DISTILLSCHEDULE_NODE_LOCK:-/tmp/distillschedule_gpu_node.lock}"

for path in "${PYTHON}" "${SOURCE}/wan/modules/model.py" "${PROMPTS}" \
  "${MODEL}/diffusion_pytorch_model.safetensors"; do
  [[ -e "${path}" ]] || { echo "missing=${path}" >&2; exit 3; }
done
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_HEAD:?set EXPECTED_HEAD}" ]] || exit 4
git -C "${ROOT}" diff --quiet && git -C "${ROOT}" diff --cached --quiet || exit 4

mkdir -p "${OUTPUT}" "${LOG_ROOT}"
exec 9>"${LOCK}"
flock -n 9 || { echo gpu_lock_busy >&2; exit 73; }
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')" ]] || {
  echo gpu_processes_present >&2; exit 81;
}

export PATH="${PYENV}/bin:${PATH}"
export PYTHONPATH="${ROOT}/third_party/livewan:${SOURCE}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4

pids=()
for local_rank in $(seq 0 7); do
  global_shard=$((NODE_RANK * 8 + local_rank))
  for sample_offset in $(seq 0 4); do
    prompt_index=$((global_shard + sample_offset * 16))
    [[ ! -e "${OUTPUT}/p$(printf '%04d' "${prompt_index}")_s00.pt" ]] || {
      echo "refusing_to_reuse_existing_input=${OUTPUT}/p$(printf '%04d' "${prompt_index}")_s00.pt" >&2
      exit 82
    }
  done
  CUDA_VISIBLE_DEVICES="${local_rank}" "${PYTHON}" \
    "${ROOT}/third_party/livewan/scripts/gen_teacher.py" \
    --size '640*368' --frames 81 --steps 40 --cfg 7.0 --shift 5.0 \
    --per-prompt 1 --shard "${global_shard}" --num-shards 16 --limit 5 \
    --ckpt "${MODEL}" --prompts "${PROMPTS}" --out "${OUTPUT}" \
    >"${LOG_ROOT}/rank_${global_shard}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
[[ ${status} -eq 0 ]] || { echo input_generation_failed >&2; exit 1; }
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')" ]] || {
  echo gpu_processes_remain >&2; exit 88;
}

if [[ "${NODE_RANK}" == 0 ]]; then
  # Node1 may finish a little later; wait for all 80 immutable outputs.
  for _ in $(seq 1 300); do
    count="$(find "${OUTPUT}" -maxdepth 1 -type f -name '*.pt' | wc -l)"
    [[ "${count}" == 80 ]] && break
    sleep 2
  done
  "${PYTHON}" - "${OUTPUT}" "${PROMPTS}" "${RUN_ID}" <<'PY'
import hashlib, json, os, pathlib, sys, torch
root=pathlib.Path(sys.argv[1]).resolve(); prompts=pathlib.Path(sys.argv[2]).resolve()
files=sorted(root.glob('*.pt'))
if len(files)!=80: raise SystemExit(f'expected 80 clips, found {len(files)}')
rows=[]; prompt_ids=set()
for path in files:
 value=torch.load(path,map_location='cpu',weights_only=False)
 latent=value['latents']
 if tuple(latent.shape)!=(16,21,46,80): raise SystemExit(f'shape mismatch: {path} {tuple(latent.shape)}')
 prompt_ids.add(int(value['prompt_idx']))
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
 rows.append({'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest(),'prompt_idx':int(value['prompt_idx'])})
ph=hashlib.sha256(prompts.read_bytes()).hexdigest()
receipt={'schema_version':1,'status':'PASS','run_id':sys.argv[3],
 'generator':'pinned LiveWan gen_teacher.py / Wan2.1-T2V-1.3B',
 'shape':[16,21,46,80],'sample_count':len(rows),'unique_prompt_count':len(prompt_ids),
 'prompts':{'path':str(prompts),'sha256':ph},'samples':rows}
payload=json.dumps(receipt,sort_keys=True,separators=(',',':')).encode()
receipt['receipt_sha256']=hashlib.sha256(payload).hexdigest()
tmp=root/'.input_receipt.json.tmp'; tmp.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
os.replace(tmp,root/'input_receipt.json')
print(json.dumps({'status':'PASS','samples':len(rows),'receipt_sha256':receipt['receipt_sha256']}))
PY
fi

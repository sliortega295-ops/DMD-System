#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/home/jiangzhikun/yongyan_liu/Helios-H100
O=/home/jiangzhikun/yongyan_liu/Helios-H100/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1
PY="$ROOT/envs/five-workload-h100-20260930/bin/python"
"$PY" "$ROOT/repos/DMD-System-h100-helios-b16-20260930/scripts/distillschedule/verify_inherited_gpu_locks.py" --project-root "$ROOT" --asset-fd 7
export PYTHONPATH="$ROOT/repos/H3-DMD-for-A100-h100-three-arm-20261014-r1:$ROOT/envs/h3-data-overlay-20261004-r1/site-packages:$ROOT/envs/h3-grid-overlay-20261004/site-packages:$ROOT/repos/h3-diffusers-92846072-20261004-r2/src:$ROOT/repos/h3-lightx2v-d034a6b0-20261004-r2:$ROOT/repos/h3-lightx2v-d034a6b0-20261004-r2/lightx2v_train"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export H3_FLASH_ATTN3_LOCAL_PATH="$ROOT/assets/five_workload_h100_20260930/h3_grid_frozen_r1/fa3"
T="$ROOT/research_artifacts/five_workload_h100_20260930/h3_grid_h100_native_table_20261014_r1"
test ! -e "$T";mkdir "$T"
START=$(date +%s)
timeout --signal=TERM --kill-after=30s 3600s "$PY" "$O/scripts/h3_grid_H100_cold_builder_r1.py" "$ROOT/repos/H3-DMD-for-A100-h100-three-arm-20261014-r1/scripts/build_grid1000_adaln_table.py" --model-path "$ROOT/assets/five_workload_h100_20260930/h3_grid_frozen_r1/model" --output-dir "$T" --grid-size 1000 --renoise-sigma-min 0.02 --renoise-sigma-max 0.98 --video-shift 6 --audio-shift 3 --num-inference-steps 4 --device cuda:0 --attention-backend _flash_3_hub > "$T/builder.log" 2>&1
END=$(date +%s)
timeout --signal=TERM --kill-after=30s 1800s "$PY" -c 'import sys,runpy;from h3_a100.model import _configure_local_flash_attn3;_configure_local_flash_attn3();sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name="__main__")' "$ROOT/repos/H3-DMD-for-A100-h100-three-arm-20261014-r1/scripts/validate_grid1000_adaln_table.py" --model-path "$ROOT/assets/five_workload_h100_20260930/h3_grid_frozen_r1/model" --manifest "$T/adaln_grid1000.json" --device cuda:0 --random-grid-entries 8 --blocks-per-entry 4 > "$T/validator.json" 2> "$T/validator.stderr.log"
"$PY" - "$T" "$ROOT/assets/five_workload_h100_20260930/h3_grid_frozen_r1/grid_table/adaln_grid1000.json" "$START" "$END" <<'PYTHON'
import json,hashlib,sys
from pathlib import Path
p=Path(sys.argv[1]);old=json.loads(Path(sys.argv[2]).read_text());new=json.loads((p/'adaln_grid1000.json').read_text());v=json.loads((p/'validator.json').read_text());assert v['status']=='PASS' and v['checked']==48 and not v['failures']
for key in ('grid_size','num_rollout_entries','num_entries','num_blocks','hidden_size','rows_per_entry','modulation_chunks','dtype','renoise_sigma_min','renoise_sigma_max','video_shift','audio_shift','num_inference_steps','rollout_base_sigmas','grid_base_sigmas','timestep_bits','dropped_adaln_parameter_numel','binary_bytes'):
 assert new[key]==old[key],key
assert new['binary_bytes']==19431014400 and new['dropped_adaln_parameter_numel']>12000000000
r={'status':'PASS_H100_NATIVE_GRID1000_BITWISE_VALIDATION','native_validator':v,'cold_table_build_seconds_excluded':int(sys.argv[4])-int(sys.argv[3]),'semantics':'same K1000 sigma/timestep bits, weights, BF16 and frozen original builder; device-specific rounding only; SAME_NEW_TABLE_IN_ALL_H100_ARMS','A100_table':'PRESERVED_OLD_H100_BITWISE_FAIL_NOT_USED','training':'NOT_RUN_BY_BUILD'}
for key,name in (('manifest','adaln_grid1000.json'),('binary','adaln_grid1000.bf16.bin')):
 f=p/name;h=hashlib.sha256()
 with f.open('rb') as handle:
  for chunk in iter(lambda:handle.read(8<<20),b''):h.update(chunk)
 r[key]={'path':str(f),'sha256':h.hexdigest(),'bytes':f.stat().st_size};f.chmod(0o444)
with (p/'runtime_table_receipt.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r))
PYTHON
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
"$PY" "$O/scripts/guard_h3_three_arm_bootstrap_r2.py" --run-id h3_three_arm_common_bootstrap_build_20261014_r2 --execute --asset-receipt "/home/jiangzhikun/yongyan_liu/Helios-H100/research_artifacts/five_workload_h100_20260930/h3_full_asset_gate_r1/asset_receipt.json" --asset-receipt-sha256 74686b2b09cd0542a9bf5c4975641700e2f292be12bdaf4a00fa55883823b5d2

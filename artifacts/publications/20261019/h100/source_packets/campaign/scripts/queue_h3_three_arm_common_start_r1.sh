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
export H3_FA3_CANARY_OUTPUT="$O/artifacts/h3_FA3_real_H100_library_canary_r1"
timeout --signal=TERM --kill-after=30s 240s "$PY" "$O/scripts/h3_h100_fa3_runtime_canary_r1.py"
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
timeout --signal=TERM --kill-after=30s 1800s "$PY" -c 'import sys,runpy;from h3_a100.model import _configure_local_flash_attn3;_configure_local_flash_attn3();sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name="__main__")' "$ROOT/repos/H3-DMD-for-A100-h100-three-arm-20261014-r1/scripts/validate_grid1000_adaln_table.py" --model-path "$ROOT/assets/five_workload_h100_20260930/h3_grid_frozen_r1/model" --manifest "$ROOT/assets/five_workload_h100_20260930/h3_grid_frozen_r1/grid_table/adaln_grid1000.json" --device cuda:0 --random-grid-entries 8 --blocks-per-entry 4 > "$O/logs/h3_grid_H100_bitwise_validation_r1.log"
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
"$PY" "$O/scripts/guard_h3_three_arm_bootstrap_r1.py" --run-id h3_three_arm_common_bootstrap_build_20261014_r1 --execute --asset-receipt "/home/jiangzhikun/yongyan_liu/Helios-H100/research_artifacts/five_workload_h100_20260930/h3_full_asset_gate_r1/asset_receipt.json" --asset-receipt-sha256 74686b2b09cd0542a9bf5c4975641700e2f292be12bdaf4a00fa55883823b5d2

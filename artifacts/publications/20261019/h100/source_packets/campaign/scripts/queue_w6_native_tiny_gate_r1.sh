#!/usr/bin/env bash
set -Eeuo pipefail
R=/home/jiangzhikun/yongyan_liu/Helios-H100
O=/home/jiangzhikun/yongyan_liu/Helios-H100/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1
"$R/envs/five-workload-h100-20260930/bin/python" "$R/repos/DMD-System-h100-helios-b16-20260930/scripts/distillschedule/verify_inherited_gpu_locks.py" --project-root "$R" --asset-fd 7
export CUDA_VISIBLE_DEVICES=0
export H3_W6_ATTENTION_PRIMITIVE_OUTPUT="$O/artifacts/w6_attention_transport_GPU_canary_r1"
timeout --signal=TERM --kill-after=30s 240s bash "$O/scripts/h3_w6_native_b16_environment_r1.sh" "$O/scripts/w6_attention_transport_GPU_canary_r1.py"
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
exec "$R/envs/five-workload-h100-20260930/bin/python" "$O/scripts/run_w6_native_three_arm_r1.py" --run-id h100_w6_native_B16_GPU_tiny_gate_20261014_r1 --execute --tiny-only

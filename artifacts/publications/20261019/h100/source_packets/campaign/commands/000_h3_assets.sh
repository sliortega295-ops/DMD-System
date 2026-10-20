#!/usr/bin/env bash
set -Eeuo pipefail
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 "/home/jiangzhikun/yongyan_liu/Helios-H100/envs/five-workload-h100-20260930/bin/python" "/home/jiangzhikun/yongyan_liu/Helios-H100/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1/scripts/prepare_h3_existing_assets.py"

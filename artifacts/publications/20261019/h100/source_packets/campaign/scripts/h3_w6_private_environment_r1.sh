#!/usr/bin/env bash
# Private W6 process environment only; never source shell startup or install.
# No model construction or GPU launch unless explicitly requested by caller.
set -Eeuo pipefail
ROOT=/home/jiangzhikun/yongyan_liu/Helios-H100
A="$ROOT/assets/five_workload_h100_20260930/h3_ar_frozen_r1"
PY="$ROOT/envs/five-workload-h100-20260930/bin/python"
if (($# == 0)); then
  printf 'usage: bash h3_w6_private_environment_r1.sh <python arguments>\n' >&2
  exit 2
fi
SOURCE="$A/assets/exact_reference_source_r2"
PREFIX="$A/assets/runtime_six_overlay_r6:$A/assets/runtime_pandas_overlay_r5:$A/assets/runtime_data_overlay_r4:$A/assets/runtime_compat_overlay_r2:$A/assets/runtime_overlay_r1:$A/zero3_200s_r2/assets/zero3_dependency_overlay_r1:$A/assets/python_overlay_r3:$SOURCE/examples/minimax_h3/model_training:$SOURCE"
exec env PYTHONDONTWRITEBYTECODE=1 USE_TF=0 USE_FLAX=0 \
  TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH="$PREFIX" "$PY" "$@"

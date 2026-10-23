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
W="$ROOT/repos/DMD-System-h100-w6-gpu-cache-20261015-r1"
PREFIX="$W:$W/src:$A/assets/runtime_six_overlay_r6:$A/assets/runtime_pandas_overlay_r5:$A/assets/runtime_data_overlay_r4:$A/assets/runtime_compat_overlay_r2:$A/assets/runtime_overlay_r1:$A/zero3_200s_r2/assets/zero3_dependency_overlay_r1:$A/assets/python_overlay_r3:$SOURCE/examples/minimax_h3/model_training:$SOURCE"
HELPERS="$A/assets/steady_training_source_helpers_r1"
for part in steady_two_cycle_20261012_r1 selected_cache_h2d_prefetch_260s_r1 selected_reuse_block_fifo_260s_r1 selected_rollout_exact_reuse_260s_r1 continuation_h3_experience_20261011_r1 local_jit_cache_260s_r1 compatible_h3_260s_20261011_r1; do
 PREFIX="$PREFIX:$HELPERS/zero3_200s_r2/$part/scripts"
done
PREFIX="$PREFIX:$A/assets/native_training_inputs_r1/zero3_200s_r2/scripts:$A/assets/native_training_inputs_r1/optimization_200s_r1/scripts"
exec env PYTHONDONTWRITEBYTECODE=1 USE_TF=0 USE_FLAX=0 \
  TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH="$PREFIX" "$PY" "$@"

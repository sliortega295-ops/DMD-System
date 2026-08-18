#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 <node_rank> <run_id> <mode> <expected_head> <master_addr> <master_port>" >&2
  exit 2
fi

NODE_RANK="$1"
RUN_ID="$2"
MODE="$3"
EXPECTED_HEAD="$4"
MASTER_ADDR="$5"
MASTER_PORT="$6"

[[ "$NODE_RANK" == "0" || "$NODE_RANK" == "1" ]] || { echo "node_rank must be 0 or 1" >&2; exit 2; }
[[ "$MODE" == "native" || "$MODE" == "distillschedule" ]] || { echo "invalid mode: $MODE" >&2; exit 2; }

DMD_SYSTEM_ROOT="${DMD_SYSTEM_ROOT:?set DMD_SYSTEM_ROOT}"
LIGHTX2V_PYTHON="${LIGHTX2V_PYTHON:?set LIGHTX2V_PYTHON}"
LIGHTX2V_MINIMAX_MODEL_PATH="${LIGHTX2V_MINIMAX_MODEL_PATH:?set LIGHTX2V_MINIMAX_MODEL_PATH}"
LIGHTX2V_MINIMAX_PROMPT_CACHE="${LIGHTX2V_MINIMAX_PROMPT_CACHE:?set LIGHTX2V_MINIMAX_PROMPT_CACHE}"
LIGHTX2V_MINIMAX_ATTENTION_BACKEND="${LIGHTX2V_MINIMAX_ATTENTION_BACKEND:-_flash_3_hub}"
LIGHTX2V_BENCHMARK_SEED="${LIGHTX2V_BENCHMARK_SEED:-20260817}"
LIGHTX2V_BENCHMARK_SOURCE_ROOT="${LIGHTX2V_BENCHMARK_SOURCE_ROOT:-}"
UPSTREAM_SHA="d034a6b0ecaa31aa07c81aeb7bbe69b225f1d7be"
CONFIG="$DMD_SYSTEM_ROOT/configs/distillschedule/baselines/lightx2v_minimax_h3_world16_dmd.yaml"
RUN_ROOT="$DMD_SYSTEM_ROOT/artifacts/runs/$RUN_ID/lightx2v_minimax_matched/$MODE"
RECEIPT_DIR="$RUN_ROOT/receipts"
NODE_DIR="$RUN_ROOT/node_$NODE_RANK"
ATTRIBUTION_DIR="$RUN_ROOT/memory_attribution"
SUBMODULE="$DMD_SYSTEM_ROOT/third_party/lightx2v"
ENTRY="$DMD_SYSTEM_ROOT/scripts/distillschedule/lightx2v_minimax_matched_entry.py"

# Hold an atomic per-node experiment lock from preflight through teardown.  The
# lock is intentionally outside the repository and is disposable run state.
LOCK_PATH="${LIGHTX2V_MINIMAX_LOCK_PATH:-/tmp/lyy-experiments/DMD-System/lightx2v_minimax_fsdp_first_activation_offload_20260818/node_${NODE_RANK}.lock}"
mkdir -p "$(dirname "$LOCK_PATH")"
exec 9>"$LOCK_PATH"
flock -n 9 || { echo "another MiniMax attribution run owns $LOCK_PATH" >&2; exit 73; }

if git -C "$DMD_SYSTEM_ROOT" rev-parse HEAD >/dev/null 2>&1; then
  [[ "$(git -C "$DMD_SYSTEM_ROOT" rev-parse HEAD)" == "$EXPECTED_HEAD" ]] || { echo "DMD-System HEAD mismatch" >&2; exit 2; }
elif [[ -f "$DMD_SYSTEM_ROOT/.distillschedule_source_head" ]]; then
  grep -qx "head=$EXPECTED_HEAD" "$DMD_SYSTEM_ROOT/.distillschedule_source_head" || { echo "DMD-System source receipt mismatch" >&2; exit 2; }
else
  echo "DMD-System source identity is unavailable" >&2
  exit 2
fi
if [[ -n "$LIGHTX2V_BENCHMARK_SOURCE_ROOT" ]]; then
  PATCH_SHA256="$(sha256sum "$DMD_SYSTEM_ROOT/third_party/lightx2v_benchmark/benchmark.patch" | awk '{print $1}')"
  [[ -f "$LIGHTX2V_BENCHMARK_SOURCE_ROOT/.distillschedule_source_receipt" ]] || { echo "prepared LightX2V source receipt missing" >&2; exit 2; }
  grep -qx "upstream=$UPSTREAM_SHA" "$LIGHTX2V_BENCHMARK_SOURCE_ROOT/.distillschedule_source_receipt" || { echo "prepared LightX2V upstream mismatch" >&2; exit 2; }
  grep -qx "patch_sha256=$PATCH_SHA256" "$LIGHTX2V_BENCHMARK_SOURCE_ROOT/.distillschedule_source_receipt" || { echo "prepared LightX2V patch mismatch" >&2; exit 2; }
  LIGHTX2V_SOURCE="$LIGHTX2V_BENCHMARK_SOURCE_ROOT/lightx2v_train"
else
  git -C "$DMD_SYSTEM_ROOT" submodule update --init third_party/lightx2v
  [[ "$(git -C "$SUBMODULE" rev-parse HEAD)" == "$UPSTREAM_SHA" ]] || { echo "LightX2V submodule mismatch" >&2; exit 2; }
  [[ -z "$(git -C "$SUBMODULE" status --porcelain)" ]] || { echo "LightX2V submodule is dirty" >&2; exit 2; }
  LIGHTX2V_SOURCE="$SUBMODULE/lightx2v_train"
fi
[[ -x "$LIGHTX2V_PYTHON" ]]
[[ -d "$LIGHTX2V_MINIMAX_MODEL_PATH" ]]
[[ -f "$LIGHTX2V_MINIMAX_PROMPT_CACHE/metadata.jsonl" ]]
[[ -f "$CONFIG" ]]
[[ -f "$ENTRY" ]]

sample_count="$(grep -cve '^[[:space:]]*$' "$LIGHTX2V_MINIMAX_PROMPT_CACHE/metadata.jsonl")"
if (( sample_count < 96 )); then
  echo "matched MiniMax cycle requires >=96 prompt-cache samples, found $sample_count" >&2
  exit 2
fi

mkdir -p "$NODE_DIR" "$RECEIPT_DIR"
if [[ -e "$RECEIPT_DIR/rank_$(printf '%03d' $((NODE_RANK * 8))).json" ]]; then
  echo "run root already contains receipts" >&2
  exit 2
fi

before_events="$NODE_DIR/memory.events.before"
after_events="$NODE_DIR/memory.events.after"
if [[ -r /sys/fs/cgroup/memory.events ]]; then cp /sys/fs/cgroup/memory.events "$before_events"; else : > "$before_events"; fi

export PYTHONPATH="$DMD_SYSTEM_ROOT:$DMD_SYSTEM_ROOT/src:$LIGHTX2V_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export LIGHTX2V_MINIMAX_MODEL_PATH LIGHTX2V_MINIMAX_PROMPT_CACHE LIGHTX2V_MINIMAX_ATTENTION_BACKEND
export LIGHTX2V_MINIMAX_FSDP_FIRST_INIT="${LIGHTX2V_MINIMAX_FSDP_FIRST_INIT:-1}"
export LIGHTX2V_BENCHMARK_SOURCE_ROOT
export LIGHTX2V_MINIMAX_OUTPUT_DIR="$RUN_ROOT/train_output"
export LIGHTX2V_MINIMAX_MEMORY_ATTRIBUTION="${LIGHTX2V_MINIMAX_MEMORY_ATTRIBUTION:-0}"
export LIGHTX2V_MINIMAX_MEMORY_ATTRIBUTION_DIR="$ATTRIBUTION_DIR"
export LIGHTX2V_MINIMAX_ACTIVATION_OFFLOAD="${LIGHTX2V_MINIMAX_ACTIVATION_OFFLOAD:-0}"
export LIGHTX2V_MINIMAX_CHECKPOINT_SEGMENT="${LIGHTX2V_MINIMAX_CHECKPOINT_SEGMENT:-0}"
export LIGHTX2V_MINIMAX_SEGMENT_DEBUG="${LIGHTX2V_MINIMAX_SEGMENT_DEBUG:-0}"
export LIGHTX2V_MINIMAX_CANARY_STAGE="${LIGHTX2V_MINIMAX_CANARY_STAGE:-full}"
# Match the verified preparation substrate: direct huggingface.co is
# unroutable on the training nodes, while the mirror-backed shared cache holds
# the pinned flash-attn3 kernel fetched during the FSDP-first gates.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/share/project/shuchenweng/yongyan_liu/DMD-System_assets/lightx2v_minimax_dmd_20260817/hf_home}"
export HF_HUB_DISABLE_TELEMETRY=1
export USE_TORCH=1
export USE_TF=0
export PYTHONHASHSEED="$LIGHTX2V_BENCHMARK_SEED"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

set +e
"$LIGHTX2V_PYTHON" -m torch.distributed.run \
  --nnodes=2 \
  --nproc-per-node=8 \
  --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  "$ENTRY" \
  --mode "$MODE" \
  --config "$CONFIG" \
  --receipt-dir "$RECEIPT_DIR" \
  --seed "$LIGHTX2V_BENCHMARK_SEED" \
  --dmd-system-head "$EXPECTED_HEAD" \
  --upstream-commit "$UPSTREAM_SHA" \
  > "$NODE_DIR/train.log" 2>&1
rc=$?
set -e

if [[ -r /sys/fs/cgroup/memory.events ]]; then cp /sys/fs/cgroup/memory.events "$after_events"; else : > "$after_events"; fi
cat > "$NODE_DIR/result.json" <<EOF
{
  "schema_version": "distillschedule.lightx2v_minimax_matched_node.v1",
  "node_rank": $NODE_RANK,
  "mode": "$MODE",
  "return_code": $rc,
  "upstream_commit": "$UPSTREAM_SHA",
  "dmd_system_head": "$EXPECTED_HEAD",
  "attention_backend": "$LIGHTX2V_MINIMAX_ATTENTION_BACKEND",
  "benchmark_seed": $LIGHTX2V_BENCHMARK_SEED,
  "prompt_cache_samples": $sample_count
}
EOF
exit "$rc"

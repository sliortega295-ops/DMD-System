#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 <node_rank> <run_id> <expected_dmd_head> <master_addr> <master_port>" >&2
  exit 2
fi

NODE_RANK="$1"
RUN_ID="$2"
EXPECTED_HEAD="$3"
MASTER_ADDR="$4"
MASTER_PORT="$5"

if [[ "$NODE_RANK" != "0" && "$NODE_RANK" != "1" ]]; then
  echo "node_rank must be 0 or 1" >&2
  exit 2
fi

DMD_SYSTEM_ROOT="${DMD_SYSTEM_ROOT:?set DMD_SYSTEM_ROOT}"
LIGHTX2V_BENCHMARK_SOURCE_ROOT="${LIGHTX2V_BENCHMARK_SOURCE_ROOT:?set LIGHTX2V_BENCHMARK_SOURCE_ROOT}"
LIGHTX2V_PYTHON="${LIGHTX2V_PYTHON:?set LIGHTX2V_PYTHON}"
LIGHTX2V_MINIMAX_MODEL_PATH="${LIGHTX2V_MINIMAX_MODEL_PATH:?set LIGHTX2V_MINIMAX_MODEL_PATH}"
LIGHTX2V_MINIMAX_PROMPT_CACHE="${LIGHTX2V_MINIMAX_PROMPT_CACHE:?set LIGHTX2V_MINIMAX_PROMPT_CACHE}"
LIGHTX2V_MINIMAX_ATTENTION_BACKEND="${LIGHTX2V_MINIMAX_ATTENTION_BACKEND:-_flash_3_hub}"
LIGHTX2V_BENCHMARK_SEED="${LIGHTX2V_BENCHMARK_SEED:-20260817}"
LIGHTX2V_BENCHMARK_END_STEP_IDX="${LIGHTX2V_BENCHMARK_END_STEP_IDX:-3}"

UPSTREAM_SHA="d034a6b0ecaa31aa07c81aeb7bbe69b225f1d7be"
PATCH_SHA256="89935efd36d8a149f121b428cad700363d987918718b81b1ea56b887f196d9c3"
CONFIG="$DMD_SYSTEM_ROOT/configs/distillschedule/baselines/lightx2v_minimax_h3_world16_dmd.yaml"
RUN_ROOT="$DMD_SYSTEM_ROOT/artifacts/runs/$RUN_ID/lightx2v_minimax"
RECEIPT_DIR="$RUN_ROOT/receipts"
NODE_DIR="$RUN_ROOT/node_$NODE_RANK"

if [[ "$(git -C "$DMD_SYSTEM_ROOT" rev-parse HEAD)" != "$EXPECTED_HEAD" ]]; then
  echo "DMD-System HEAD mismatch" >&2
  exit 2
fi
if [[ ! -f "$LIGHTX2V_BENCHMARK_SOURCE_ROOT/.distillschedule_source_receipt" ]]; then
  echo "prepared LightX2V benchmark source is missing" >&2
  exit 2
fi
grep -qx "upstream=$UPSTREAM_SHA" "$LIGHTX2V_BENCHMARK_SOURCE_ROOT/.distillschedule_source_receipt"
grep -qx "patch_sha256=$PATCH_SHA256" "$LIGHTX2V_BENCHMARK_SOURCE_ROOT/.distillschedule_source_receipt"
[[ -f "$CONFIG" ]]
[[ -d "$LIGHTX2V_MINIMAX_MODEL_PATH" ]]
[[ -d "$LIGHTX2V_MINIMAX_PROMPT_CACHE" ]]
[[ -x "$LIGHTX2V_PYTHON" ]]

mkdir -p "$NODE_DIR" "$RECEIPT_DIR"
before_events="$NODE_DIR/memory.events.before"
after_events="$NODE_DIR/memory.events.after"
if [[ -r /sys/fs/cgroup/memory.events ]]; then
  cp /sys/fs/cgroup/memory.events "$before_events"
else
  : > "$before_events"
fi

export PYTHONPATH="$LIGHTX2V_BENCHMARK_SOURCE_ROOT/lightx2v_train${PYTHONPATH:+:$PYTHONPATH}"
export LIGHTX2V_MINIMAX_MODEL_PATH
export LIGHTX2V_MINIMAX_PROMPT_CACHE
export LIGHTX2V_MINIMAX_ATTENTION_BACKEND
export LIGHTX2V_MINIMAX_OUTPUT_DIR="$RUN_ROOT/train_output"
export LIGHTX2V_BENCHMARK_RECEIPT_DIR="$RECEIPT_DIR"
export LIGHTX2V_BENCHMARK_SEED
export LIGHTX2V_BENCHMARK_END_STEP_IDX
export LIGHTX2V_UPSTREAM_COMMIT="$UPSTREAM_SHA"
export LIGHTX2V_BENCHMARK_PATCH_SHA256="$PATCH_SHA256"
export DMD_SYSTEM_EXPECTED_HEAD="$EXPECTED_HEAD"
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
  "$LIGHTX2V_BENCHMARK_SOURCE_ROOT/lightx2v_train/train.py" \
  --config "$CONFIG" \
  > "$NODE_DIR/train.log" 2>&1
rc=$?
set -e

if [[ -r /sys/fs/cgroup/memory.events ]]; then
  cp /sys/fs/cgroup/memory.events "$after_events"
else
  : > "$after_events"
fi

cat > "$NODE_DIR/result.json" <<EOF
{
  "schema_version": "distillschedule.lightx2v_minimax_node.v1",
  "node_rank": $NODE_RANK,
  "return_code": $rc,
  "upstream_commit": "$UPSTREAM_SHA",
  "benchmark_patch_sha256": "$PATCH_SHA256",
  "dmd_system_head": "$EXPECTED_HEAD",
  "attention_backend": "$LIGHTX2V_MINIMAX_ATTENTION_BACKEND",
  "benchmark_seed": $LIGHTX2V_BENCHMARK_SEED,
  "benchmark_end_step_idx": $LIGHTX2V_BENCHMARK_END_STEP_IDX
}
EOF

exit "$rc"

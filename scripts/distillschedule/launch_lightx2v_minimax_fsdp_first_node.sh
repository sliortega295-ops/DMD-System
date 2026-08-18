#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 <node_rank> <run_id> <gate> <expected_head> <master_addr> <master_port>" >&2
  exit 2
fi

NODE_RANK="$1"
RUN_ID="$2"
GATE="$3"
EXPECTED_HEAD="$4"
MASTER_ADDR="$5"
MASTER_PORT="$6"
[[ "$NODE_RANK" == 0 || "$NODE_RANK" == 1 ]] || { echo "node_rank must be 0 or 1" >&2; exit 2; }
[[ "$GATE" == A || "$GATE" == B ]] || { echo "gate must be A or B" >&2; exit 2; }

DMD_SYSTEM_ROOT="${DMD_SYSTEM_ROOT:?set DMD_SYSTEM_ROOT}"
LIGHTX2V_PYTHON="${LIGHTX2V_PYTHON:?set LIGHTX2V_PYTHON}"
LIGHTX2V_MINIMAX_MODEL_PATH="${LIGHTX2V_MINIMAX_MODEL_PATH:?set LIGHTX2V_MINIMAX_MODEL_PATH}"
LIGHTX2V_MINIMAX_PROMPT_CACHE="${LIGHTX2V_MINIMAX_PROMPT_CACHE:?set LIGHTX2V_MINIMAX_PROMPT_CACHE}"
LIGHTX2V_BENCHMARK_SOURCE_ROOT="${LIGHTX2V_BENCHMARK_SOURCE_ROOT:?set LIGHTX2V_BENCHMARK_SOURCE_ROOT}"
CONFIG="$DMD_SYSTEM_ROOT/configs/distillschedule/baselines/lightx2v_minimax_h3_world16_dmd.yaml"
SCRIPT="$DMD_SYSTEM_ROOT/scripts/distillschedule/lightx2v_minimax_fsdp_first_construction.py"
RUN_ROOT="$DMD_SYSTEM_ROOT/artifacts/runs/$RUN_ID/fsdp_first/$GATE"
NODE_DIR="$RUN_ROOT/node_$NODE_RANK"
OUTPUT_DIR="$RUN_ROOT/receipts"

[[ -x "$LIGHTX2V_PYTHON" ]] || { echo "missing Python: $LIGHTX2V_PYTHON" >&2; exit 2; }
[[ -f "$CONFIG" && -f "$SCRIPT" ]] || { echo "missing construction entry/config" >&2; exit 2; }
if git -C "$DMD_SYSTEM_ROOT" rev-parse HEAD >/dev/null 2>&1; then
  [[ "$(git -C "$DMD_SYSTEM_ROOT" rev-parse HEAD)" == "$EXPECTED_HEAD" ]] || { echo "DMD-System HEAD mismatch" >&2; exit 2; }
elif [[ -f "$DMD_SYSTEM_ROOT/.distillschedule_source_head" ]]; then
  grep -qx "head=$EXPECTED_HEAD" "$DMD_SYSTEM_ROOT/.distillschedule_source_head" || { echo "DMD-System source receipt mismatch" >&2; exit 2; }
else
  echo "DMD-System source identity is unavailable" >&2
  exit 2
fi
[[ -f "$LIGHTX2V_BENCHMARK_SOURCE_ROOT/.distillschedule_source_receipt" ]] || { echo "missing prepared source receipt" >&2; exit 2; }
grep -qx "upstream=d034a6b0ecaa31aa07c81aeb7bbe69b225f1d7be" "$LIGHTX2V_BENCHMARK_SOURCE_ROOT/.distillschedule_source_receipt" || { echo "source upstream identity mismatch" >&2; exit 2; }
grep -qx "patch_sha256=89935efd36d8a149f121b428cad700363d987918718b81b1ea56b887f196d9c3" "$LIGHTX2V_BENCHMARK_SOURCE_ROOT/.distillschedule_source_receipt" || { echo "source patch identity mismatch" >&2; exit 2; }

mkdir -p "$NODE_DIR" "$OUTPUT_DIR"
if [[ -r /sys/fs/cgroup/memory.events ]]; then cp /sys/fs/cgroup/memory.events "$NODE_DIR/memory.events.before"; fi

export PYTHONPATH="$DMD_SYSTEM_ROOT:$DMD_SYSTEM_ROOT/src:$LIGHTX2V_BENCHMARK_SOURCE_ROOT/lightx2v_train${PYTHONPATH:+:$PYTHONPATH}"
export LIGHTX2V_MINIMAX_FSDP_FIRST_INIT=1
export LIGHTX2V_BENCHMARK_SOURCE_ROOT
export LIGHTX2V_MINIMAX_MODEL_PATH LIGHTX2V_MINIMAX_PROMPT_CACHE
export LIGHTX2V_MINIMAX_ATTENTION_BACKEND="${LIGHTX2V_MINIMAX_ATTENTION_BACKEND:-_flash_3_hub}"
# The shared MiniMax config resolves training.output_dir through OmegaConf's
# environment resolver even for construction-only probes.  Keep this path
# inside the per-run evidence tree; the construction gate never writes model
# checkpoints here, but fail-closed config resolution must still have a value.
export LIGHTX2V_MINIMAX_OUTPUT_DIR="$NODE_DIR/output"
export LIGHTX2V_PROFILE_DIAGNOSTICS=off
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

set +e
"$LIGHTX2V_PYTHON" -m torch.distributed.run \
  --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  "$SCRIPT" --gate "$GATE" --config "$CONFIG" --output-dir "$OUTPUT_DIR" \
  --dmd-system-head "$EXPECTED_HEAD" \
  --upstream-commit "d034a6b0ecaa31aa07c81aeb7bbe69b225f1d7be" \
  --source-root "$LIGHTX2V_BENCHMARK_SOURCE_ROOT" \
  > "$NODE_DIR/train.log" 2>&1
rc=$?
set -e

if [[ -r /sys/fs/cgroup/memory.events ]]; then cp /sys/fs/cgroup/memory.events "$NODE_DIR/memory.events.after"; fi
cat > "$NODE_DIR/result.json" <<EOF
{
  "schema_version": "distillschedule.lightx2v_minimax_fsdp_first_node.v1",
  "node_rank": $NODE_RANK,
  "gate": "$GATE",
  "return_code": $rc,
  "dmd_system_head": "$EXPECTED_HEAD",
  "lightx2v_upstream_commit": "d034a6b0ecaa31aa07c81aeb7bbe69b225f1d7be",
  "source_root": "$LIGHTX2V_BENCHMARK_SOURCE_ROOT"
}
EOF
exit "$rc"

#!/usr/bin/env bash
# Diagnostic-only world16 MiniMax Nsight profile: one Nsight session per node.
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
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid run id" >&2; exit 2; }
[[ "$EXPECTED_HEAD" =~ ^[0-9a-f]{40}$ ]] || { echo "expected_head must be a 40-char sha" >&2; exit 2; }

DMD_SYSTEM_ROOT="${DMD_SYSTEM_ROOT:?set DMD_SYSTEM_ROOT}"
MATCHED_LAUNCHER="$DMD_SYSTEM_ROOT/scripts/distillschedule/launch_lightx2v_minimax_matched_world16_node.sh"
[[ -x "$MATCHED_LAUNCHER" || -f "$MATCHED_LAUNCHER" ]] || { echo "matched launcher missing: $MATCHED_LAUNCHER" >&2; exit 3; }
[[ "$(git -C "$DMD_SYSTEM_ROOT" rev-parse HEAD)" == "$EXPECTED_HEAD" ]] || { echo "DMD-System HEAD mismatch" >&2; exit 4; }

NSYS_BIN="${NSYS_BIN:-$(command -v nsys || true)}"
[[ -n "$NSYS_BIN" && -x "$NSYS_BIN" ]] || { echo "nsys not found" >&2; exit 3; }

RUN_ROOT="$DMD_SYSTEM_ROOT/artifacts/runs/$RUN_ID"
PROFILE_ROOT="$RUN_ROOT/lightx2v_minimax_nsys/$MODE"
NODE_PROFILE_DIR="$PROFILE_ROOT/node_$NODE_RANK"
RANGE_RECEIPT_DIR="$PROFILE_ROOT/range_receipts"
REPORT_BASE="$NODE_PROFILE_DIR/minimax_${MODE}_node${NODE_RANK}"
SQLITE="$REPORT_BASE.sqlite"
MANIFEST="$NODE_PROFILE_DIR/profile_manifest.txt"
mkdir -p "$NODE_PROFILE_DIR" "$RANGE_RECEIPT_DIR"
[[ ! -e "$REPORT_BASE.nsys-rep" && ! -e "$SQLITE" && ! -e "$MANIFEST" ]] || {
  echo "refusing to overwrite existing MiniMax Nsight profile under $NODE_PROFILE_DIR" >&2
  exit 82
}

# Preserve the workload's current capacity policy exactly.  These values are
# recorded but never changed by this launcher.
ACTIVATION_OFFLOAD="${LIGHTX2V_MINIMAX_ACTIVATION_OFFLOAD:-0}"
CHECKPOINT_SEGMENT="${LIGHTX2V_MINIMAX_CHECKPOINT_SEGMENT:-0}"
MEMORY_ATTRIBUTION="${LIGHTX2V_MINIMAX_MEMORY_ATTRIBUTION:-0}"
ADVANCED_NCCL="${LIGHTX2V_MINIMAX_NSYS_ADVANCED_NCCL:-0}"

export LIGHTX2V_MINIMAX_NSYS_PROFILE=1
export LIGHTX2V_MINIMAX_NSYS_RECEIPT_DIR="$RANGE_RECEIPT_DIR"

TRACE_SET="cuda,nvtx"
EXTRA_NSYS_ARGS=()
if [[ "$ADVANCED_NCCL" == "1" ]]; then
  TRACE_SET="cuda,nvtx,nccl"
  EXTRA_NSYS_ARGS+=(--nccl-trace=default)
elif [[ "$ADVANCED_NCCL" != "0" ]]; then
  echo "LIGHTX2V_MINIMAX_NSYS_ADVANCED_NCCL must be 0 or 1" >&2
  exit 2
fi

cat > "$MANIFEST" <<EOF
schema_version=distillschedule.lightx2v_minimax_nsys_profile.v1
status=RUNNING
diagnostic_only=true
formal_speedup_eligible=false
run_id=$RUN_ID
mode=$MODE
node_rank=$NODE_RANK
dmd_system_head=$EXPECTED_HEAD
trace_set=$TRACE_SET
advanced_nccl=$ADVANCED_NCCL
activation_offload=$ACTIVATION_OFFLOAD
checkpoint_segment=$CHECKPOINT_SEGMENT
memory_attribution=$MEMORY_ATTRIBUTION
capture_range=cudaProfilerApi
session_scope=one_nsys_session_per_node
EOF

set +e
"$NSYS_BIN" profile \
  --trace="$TRACE_SET" \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --trace-fork-before-exec=true \
  --wait=all \
  --sample=none \
  --cpuctxsw=none \
  --cuda-memory-usage=false \
  --force-overwrite=true \
  --output="$REPORT_BASE" \
  "${EXTRA_NSYS_ARGS[@]}" \
  bash "$MATCHED_LAUNCHER" \
    "$NODE_RANK" "$RUN_ID" "$MODE" "$EXPECTED_HEAD" "$MASTER_ADDR" "$MASTER_PORT" \
  > "$NODE_PROFILE_DIR/nsys_console.log" 2>&1
rc=$?
set -e

if (( rc != 0 )); then
  sed -i 's/^status=RUNNING$/status=FAIL/' "$MANIFEST"
  echo "nsys_return_code=$rc" >> "$MANIFEST"
  exit "$rc"
fi

[[ -s "$REPORT_BASE.nsys-rep" ]] || { echo "missing nsys report: $REPORT_BASE.nsys-rep" >&2; exit 87; }
"$NSYS_BIN" export \
  --type sqlite \
  --force-overwrite true \
  --output "$SQLITE" \
  "$REPORT_BASE.nsys-rep" \
  > "$NODE_PROFILE_DIR/nsys_export.log" 2>&1
[[ -s "$SQLITE" ]] || { echo "missing sqlite export: $SQLITE" >&2; exit 88; }

# Both node launchers share the receipt directory.  Wait for all 16 workers so
# a partial node capture cannot later be presented as a complete world16 run.
receipt_count=0
for _ in $(seq 1 600); do
  receipt_count="$(find "$RANGE_RECEIPT_DIR" -maxdepth 1 -type f -name 'rank_*.json' | wc -l)"
  [[ "$receipt_count" == "16" ]] && break
  sleep 1
done
[[ "$receipt_count" == "16" ]] || {
  echo "MiniMax Nsight receipts=$receipt_count, expected=16" >&2
  exit 90
}

sed -i 's/^status=RUNNING$/status=PASS/' "$MANIFEST"
cat >> "$MANIFEST" <<EOF
nsys_return_code=0
report=$REPORT_BASE.nsys-rep
sqlite=$SQLITE
report_sha256=$(sha256sum "$REPORT_BASE.nsys-rep" | awk '{print $1}')
sqlite_sha256=$(sha256sum "$SQLITE" | awk '{print $1}')
range_receipts=$RANGE_RECEIPT_DIR
range_receipt_count=$receipt_count
EOF

echo "MiniMax Nsight profile PASS: $SQLITE"

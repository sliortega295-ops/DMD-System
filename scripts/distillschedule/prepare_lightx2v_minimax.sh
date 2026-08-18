#!/usr/bin/env bash
set -Eeuo pipefail

UPSTREAM_SHA="d034a6b0ecaa31aa07c81aeb7bbe69b225f1d7be"
PATCH_SHA256="89935efd36d8a149f121b428cad700363d987918718b81b1ea56b887f196d9c3"

DMD_SYSTEM_ROOT="${DMD_SYSTEM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SUBMODULE_ROOT="$DMD_SYSTEM_ROOT/third_party/lightx2v"
PATCH_PATH="$DMD_SYSTEM_ROOT/third_party/lightx2v_benchmark/benchmark.patch"
SOURCE_ROOT="${LIGHTX2V_BENCHMARK_SOURCE_ROOT:-$DMD_SYSTEM_ROOT/artifacts/source/lightx2v-minimax-d034-benchmark}"
PYTHON_BIN="${LIGHTX2V_PYTHON:-python3}"

if [[ ! -f "$PATCH_PATH" ]]; then
  echo "missing benchmark patch: $PATCH_PATH" >&2
  exit 2
fi
actual_patch_sha="$(sha256sum "$PATCH_PATH" | awk '{print $1}')"
if [[ "$actual_patch_sha" != "$PATCH_SHA256" ]]; then
  echo "benchmark patch SHA mismatch: $actual_patch_sha != $PATCH_SHA256" >&2
  exit 2
fi

# Some server launches intentionally use a copied, immutable source receipt
# rather than a Git checkout.  Do not try to run ``git submodule update`` in
# that deployment form; the worker launcher performs the same source/patch
# receipt checks before constructing a model.
if ! git -C "$DMD_SYSTEM_ROOT" rev-parse --show-toplevel >/dev/null 2>&1; then
  receipt="$SOURCE_ROOT/.distillschedule_source_receipt"
  [[ -f "$receipt" ]] || { echo "prepared source receipt missing: $receipt" >&2; exit 2; }
  grep -qx "upstream=$UPSTREAM_SHA" "$receipt" || { echo "prepared source upstream mismatch" >&2; exit 2; }
  grep -qx "patch_sha256=$PATCH_SHA256" "$receipt" || { echo "prepared source patch mismatch" >&2; exit 2; }
  echo "$SOURCE_ROOT"
  exit 0
fi

git -C "$DMD_SYSTEM_ROOT" submodule update --init third_party/lightx2v
actual_upstream="$(git -C "$SUBMODULE_ROOT" rev-parse HEAD)"
if [[ "$actual_upstream" != "$UPSTREAM_SHA" ]]; then
  echo "LightX2V submodule must be pinned at $UPSTREAM_SHA, found $actual_upstream" >&2
  exit 2
fi
if [[ -n "$(git -C "$SUBMODULE_ROOT" status --porcelain)" ]]; then
  echo "refusing dirty LightX2V submodule" >&2
  exit 2
fi

receipt="$SOURCE_ROOT/.distillschedule_source_receipt"
if [[ -e "$SOURCE_ROOT" ]]; then
  if [[ -f "$receipt" ]] && grep -qx "upstream=$UPSTREAM_SHA" "$receipt" && grep -qx "patch_sha256=$PATCH_SHA256" "$receipt"; then
    echo "$SOURCE_ROOT"
    exit 0
  fi
  echo "source root already exists without matching immutable receipt: $SOURCE_ROOT" >&2
  exit 2
fi

mkdir -p "$(dirname "$SOURCE_ROOT")"
git clone --quiet --no-hardlinks "$SUBMODULE_ROOT" "$SOURCE_ROOT"
git -C "$SOURCE_ROOT" checkout --quiet --detach "$UPSTREAM_SHA"
git -C "$SOURCE_ROOT" apply --check "$PATCH_PATH"
git -C "$SOURCE_ROOT" apply "$PATCH_PATH"
git -C "$SOURCE_ROOT" diff --check

cat > "$receipt" <<EOF
upstream=$UPSTREAM_SHA
patch_sha256=$PATCH_SHA256
EOF

PYTHONPATH="$SOURCE_ROOT/lightx2v_train${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
import torch
from diffusers import MiniMaxH3Transformer3DModel
from lightx2v_train.trainers.dmd.minimax_h3_trainer import MiniMaxH3T2AVDmdTrainer
print("LightX2V MiniMax benchmark import gate PASS", torch.__version__, MiniMaxH3Transformer3DModel.__name__, MiniMaxH3T2AVDmdTrainer.__name__)
PY

echo "$SOURCE_ROOT"

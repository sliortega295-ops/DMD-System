#!/usr/bin/env bash
# Execute the original Accelerate command unchanged, or profile its child rank
# processes with an exact CUDA-profiler-API Full5 range when explicitly enabled.
set -Eeuo pipefail

if (( $# == 0 )); then
  echo "usage: gan_launch_shim.sh COMMAND [ARG ...]" >&2
  exit 2
fi

PROFILE="${DISTILLSCHEDULE_GAN_NSYS_PROFILE:-0}"
case "${PROFILE}" in
  0|false|FALSE|no|NO)
    exec "$@"
    ;;
  1|true|TRUE|yes|YES)
    ;;
  *)
    echo "DISTILLSCHEDULE_GAN_NSYS_PROFILE must be 0/1" >&2
    exit 2
    ;;
esac

ROOT="${DMD_SYSTEM_ROOT:?DMD_SYSTEM_ROOT is required}"
OUT="${DISTILLSCHEDULE_OUTPUT_DIR:?DISTILLSCHEDULE_OUTPUT_DIR is required}"
NODE_RANK="${DISTILLSCHEDULE_NODE_RANK:?DISTILLSCHEDULE_NODE_RANK is required}"
NSYS="${DISTILLSCHEDULE_NSYS_BIN:-/usr/local/cuda/bin/nsys}"
[[ -x "${NSYS}" ]] || { echo "nsys_missing=${NSYS}" >&2; exit 3; }
[[ "${NODE_RANK}" == "0" || "${NODE_RANK}" == "1" ]] || { echo invalid_node_rank >&2; exit 2; }

PROFILE_ROOT="${OUT}/nsight"
NODE_ROOT="${PROFILE_ROOT}/node_${NODE_RANK}"
REPORT_BASE="${NODE_ROOT}/gan_full5_node${NODE_RANK}"
RECEIPT_DIR="${PROFILE_ROOT}/range_receipts"
VALIDATION="${PROFILE_ROOT}/range_receipts_validation.json"
mkdir -p "${NODE_ROOT}" "${RECEIPT_DIR}"
[[ ! -e "${REPORT_BASE}.nsys-rep" ]] || { echo "refusing_to_overwrite=${REPORT_BASE}.nsys-rep" >&2; exit 4; }

export DISTILLSCHEDULE_GAN_NSYS_EXACT_RANGE=1
export DISTILLSCHEDULE_GAN_NSYS_RECEIPT_DIR="${RECEIPT_DIR}"

set +e
"${NSYS}" profile \
  --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --trace-fork-before-exec=true \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output="${REPORT_BASE}" \
  "$@"
RC=$?
set -e
(( RC == 0 )) || exit "${RC}"
[[ -f "${REPORT_BASE}.nsys-rep" ]] || { echo "nsys_report_missing=${REPORT_BASE}.nsys-rep" >&2; exit 77; }

# Validate that the report contains actual GPU/NVTX activity; an empty
# cudaProfilerApi capture is never accepted as a five-step profile.
set +e
"${NSYS}" stats \
  --report nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum \
  --format csv \
  "${REPORT_BASE}.nsys-rep" \
  >"${NODE_ROOT}/nsys_stats_validation.log" 2>&1
STATS_RC=$?
set -e
(( STATS_RC == 0 )) || { echo nsys_stats_validation_failed >&2; exit 78; }
if grep -Eq "does not contain (NV Tools Extension|CUDA kernel|CUDA API)" "${NODE_ROOT}/nsys_stats_validation.log"; then
  echo nsys_empty_capture >&2
  exit 78
fi

# Both node-local nsys sessions finish at the same logical boundary.  The
# shared filesystem must eventually contain all 16 rank receipts before either
# node calls this an exact Full5 capture.
for _ in $(seq 1 300); do
  count="$(find "${RECEIPT_DIR}" -maxdepth 1 -type f -name 'rank_*.json' | wc -l)"
  [[ "${count}" == "16" ]] && break
  sleep 1
done
count="$(find "${RECEIPT_DIR}" -maxdepth 1 -type f -name 'rank_*.json' | wc -l)"
[[ "${count}" == "16" ]] || { echo "nsys_range_receipts=${count},expected=16" >&2; exit 79; }

python_bin="${DISTILLSCHEDULE_PYTHON:-python3}"
"${python_bin}" "${ROOT}/scripts/distillschedule/validate_gan_nsys_receipts.py" \
  --receipt-dir "${RECEIPT_DIR}" \
  --world-size 16 \
  --local-world-size 8 \
  --output "${VALIDATION}" \
  >"${NODE_ROOT}/range_validation_console.log"

{
  echo "schema_version=distillschedule.gan_nsys_node_profile.v1"
  echo "status=PASS"
  echo "diagnostic_only=true"
  echo "formal_speedup_eligible=false"
  echo "node_rank=${NODE_RANK}"
  echo "logical_window=1000-1004"
  echo "report=${REPORT_BASE}.nsys-rep"
  echo "report_sha256=$(sha256sum "${REPORT_BASE}.nsys-rep" | awk '{print $1}')"
  echo "range_receipts_validation=${VALIDATION}"
  echo "range_receipts_validation_sha256=$(sha256sum "${VALIDATION}" | awk '{print $1}')"
} >"${NODE_ROOT}/profile_manifest.txt"

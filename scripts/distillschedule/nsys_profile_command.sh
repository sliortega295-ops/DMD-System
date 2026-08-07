#!/usr/bin/env bash
set -euo pipefail

REPORT_BASE="${DISTILLSCHEDULE_NSYS_INNER_REPORT_BASE:?DISTILLSCHEDULE_NSYS_INNER_REPORT_BASE is required}"
NSYS_SAMPLE="${DISTILLSCHEDULE_NSYS_SAMPLE:-none}"
NSYS_CPUCTXSW="${DISTILLSCHEDULE_NSYS_CPUCTXSW:-none}"
if (( $# == 0 )); then
  echo "usage: nsys_profile_command.sh COMMAND [ARG ...]" >&2
  exit 2
fi
if [[ -e "${REPORT_BASE}.nsys-rep" || -e "${REPORT_BASE}.sqlite" ]]; then
  echo "refusing to overwrite inner Nsight report: ${REPORT_BASE}" >&2
  exit 3
fi
mkdir -p "$(dirname "${REPORT_BASE}")"

exec /usr/local/cuda/bin/nsys profile \
  --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --trace-fork-before-exec=true \
  --sample="${NSYS_SAMPLE}" \
  --cpuctxsw="${NSYS_CPUCTXSW}" \
  --force-overwrite=true \
  --output="${REPORT_BASE}" \
  "$@"

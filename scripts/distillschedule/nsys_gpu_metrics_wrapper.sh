#!/usr/bin/env bash
# Add Nsight Systems periodic GPU Metrics to an existing `nsys profile` command.
# Intended usage with the existing DistillSchedule launcher:
#   NSYS_BIN=$ROOT/scripts/distillschedule/nsys_gpu_metrics_wrapper.sh ... profile ...
set -Eeuo pipefail

REAL_NSYS="${DISTILLSCHEDULE_REAL_NSYS_BIN:-$(command -v nsys || true)}"
[[ -n "${REAL_NSYS}" && -x "${REAL_NSYS}" ]] || {
  echo "real nsys binary not found; set DISTILLSCHEDULE_REAL_NSYS_BIN" >&2
  exit 3
}

if [[ $# -eq 0 || "$1" != profile ]]; then
  exec "${REAL_NSYS}" "$@"
fi

FREQUENCY="${DISTILLSCHEDULE_NSYS_GPU_METRICS_HZ:-1000}"
DEVICES="${DISTILLSCHEDULE_NSYS_GPU_METRICS_DEVICES:-all}"
[[ "${FREQUENCY}" =~ ^[0-9]+$ ]] || {
  echo "DISTILLSCHEDULE_NSYS_GPU_METRICS_HZ must be an integer" >&2
  exit 2
}
(( FREQUENCY >= 10 && FREQUENCY <= 200000 )) || {
  echo "DISTILLSCHEDULE_NSYS_GPU_METRICS_HZ must be in [10, 200000]" >&2
  exit 2
}

HELP="$("${REAL_NSYS}" profile --help 2>&1 || true)"
if grep -q -- '--gpu-metrics-devices' <<<"${HELP}"; then
  DEVICE_ARG="--gpu-metrics-devices=${DEVICES}"
elif grep -q -- '--gpu-metrics-device' <<<"${HELP}"; then
  DEVICE_ARG="--gpu-metrics-device=${DEVICES}"
else
  echo "this Nsight Systems build does not expose GPU Metrics CLI options" >&2
  exit 3
fi

EXTRA=("${DEVICE_ARG}" "--gpu-metrics-frequency=${FREQUENCY}")
if [[ -n "${DISTILLSCHEDULE_NSYS_GPU_METRICS_SET:-}" ]]; then
  EXTRA+=("--gpu-metrics-set=${DISTILLSCHEDULE_NSYS_GPU_METRICS_SET}")
fi

# Nsight GPU Metrics require access to NVIDIA performance counters. If the
# host policy denies that access, nsys itself will fail rather than silently
# producing a utilization-free capture.
exec "${REAL_NSYS}" profile "${EXTRA[@]}" "${@:2}"

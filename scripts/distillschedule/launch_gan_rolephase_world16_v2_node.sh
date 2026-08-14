#!/usr/bin/env bash
# Reuse the fail-closed v1 launcher while selecting the optimized v2 worker.
set -Eeuo pipefail
ROOT="${DMD_SYSTEM_ROOT:-/share/project/shuchenweng/yongyan_liu/DMD-System_gan_rolephase_world16_exec_20260812}"
BASE="${ROOT}/scripts/distillschedule/launch_gan_rolephase_world16_node.sh"
[[ -f "${BASE}" ]] || { echo missing_base_launcher="${BASE}" >&2; exit 3; }
export DISTILLSCHEDULE_ROLEPHASE_PRECONSTRUCT_FORWARD="${DISTILLSCHEDULE_ROLEPHASE_PRECONSTRUCT_FORWARD:-1}"
# Reverse preconstruction is a separate capacity canary because it increases
# host residency. The optimized worker supports it but the default stays off.
export DISTILLSCHEDULE_ROLEPHASE_PRECONSTRUCT_REVERSE="${DISTILLSCHEDULE_ROLEPHASE_PRECONSTRUCT_REVERSE:-0}"
exec bash <(
  sed \
    's#probe_gan_rolephase_world16\.py#probe_gan_rolephase_world16_v2.py#g' \
    "${BASE}"
) "$@"

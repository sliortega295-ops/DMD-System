#!/usr/bin/env bash
set -euo pipefail
WT=/home/lyy/Helios/DMD-System/worktrees/wan13b-fsdp-upper-bound-20260820
RUN=/home/lyy/Helios/research_artifacts/DMD-System/wan13b_fsdp_upper_bound_20260820/iter327_wan13b_resident_nsys_matched_20260822
PYTHONPATH="$WT/src:$WT" python3 "$WT/scripts/distillschedule/run_wan13b_upper_bound_world16.py" \
  --run-id iter327_wan13b_resident_nsys_matched_20260822 \
  --mode profile --placement resident \
  --expected-head 18a26bcd046adba49b30630a12e6060c8d193494 \
  --model-root /share/project/shuchenweng/DistillSchedule_assets/wan13b/Wan2.1-T2V-1.3B-Diffusers-0fad780a \
  --bootstrap /share/project/shuchenweng/DistillSchedule_assets/wan13b/systems_only_bootstrap_v3_head04e6e44 \
  --bootstrap-hash 6a713476d9839cf1ea37689d63cb1cc2b405b9f9915ca5314ccdadb67b3cd60d \
  --local-evidence-root "$RUN/controller_evidence" \
  --timeout-seconds 18000

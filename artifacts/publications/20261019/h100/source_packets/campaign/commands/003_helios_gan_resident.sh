#!/usr/bin/env bash
set -Eeuo pipefail
"/home/jiangzhikun/yongyan_liu/Helios-H100/envs/five-workload-h100-20260930/bin/python" "/home/jiangzhikun/yongyan_liu/Helios-H100/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1/scripts/run_helios_resident_matched_r1.py" --workload gan --execute

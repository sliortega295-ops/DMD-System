#!/usr/bin/env bash
set -Eeuo pipefail
R=/home/jiangzhikun/yongyan_liu/Helios-H100
O=/home/jiangzhikun/yongyan_liu/Helios-H100/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1
"$R/envs/five-workload-h100-20260930/bin/python" - "$R/research_artifacts/five_workload_h100_20260930/h100_w6_native_B16_GPU_tiny_gate_20261014_r1/result.json" "$O/artifacts/w6_attention_transport_GPU_canary_r1/receipt.json" <<'PYTHON'
from pathlib import Path
import sys,json
r=json.loads(Path(sys.argv[1]).read_text());p=json.loads(Path(sys.argv[2]).read_text())
assert r['status']=='PASS_MATRIX_EXECUTION_REVIEW_INDIVIDUAL_ARMS'
assert len(r['results'])==3 and all(v['status']=='PASS_TINY_8_RANK_INTEGRATION_ONLY' for v in r['results'])
assert p['status']=='PASS_BITWISE_NATIVE_FLASH_OUTPUT_AND_QKV_GRADIENT'
PYTHON
exec "$R/envs/five-workload-h100-20260930/bin/python" "$O/scripts/run_w6_native_three_arm_r1.py" --run-id h100_w6_native_three_arm_matched_20261014_r1 --execute

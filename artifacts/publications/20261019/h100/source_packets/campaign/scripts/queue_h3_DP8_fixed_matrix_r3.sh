#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/home/jiangzhikun/yongyan_liu/Helios-H100
B="$ROOT/research_artifacts/five_workload_h100_20260930"
O="$B/h100_resident_baselines_20261014_r1"
PY="$ROOT/envs/five-workload-h100-20260930/bin/python"
"$PY" "$ROOT/repos/DMD-System-h100-helios-b16-20260930/scripts/distillschedule/verify_inherited_gpu_locks.py" --project-root "$ROOT" --asset-fd 7
BOOT="$B/h3_three_arm_common_bootstrap_build_20261014_r2/bootstrap/manifest.json"
SHA=$("$PY" - "$B/h3_three_arm_common_bootstrap_build_20261014_r2/manifests/result.json" "$BOOT" <<'PYTHON'
import sys,json,hashlib
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text());p=Path(sys.argv[2])
assert r['status']=='PASS_NEW_IMMUTABLE_H100_COMMON_START'
s=hashlib.sha256(p.read_bytes()).hexdigest();assert r['bootstrap_manifest_sha256']==s
print(s)
PYTHON
)
exec "$PY" "$O/scripts/run_h3_h100_three_arm_r3.py" --run-id h3_h100_three_arm_matched_20261014_r3 --execute --bootstrap "$BOOT" --bootstrap-sha256 "$SHA"

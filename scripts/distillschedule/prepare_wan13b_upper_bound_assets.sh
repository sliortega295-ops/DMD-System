#!/usr/bin/env bash
# Download the pinned real Wan1.3B transformer and build one shared bootstrap.
set -Eeuo pipefail

EXPECTED_HEAD="${1:?exact DMD-System HEAD}"
MODEL_ROOT="${2:?immutable output model root}"
BOOTSTRAP_ROOT="${3:?immutable output bootstrap root}"

ROOT="${DMD_SYSTEM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
HELIOS="${HELIOS_RUNTIME_ROOT:?set HELIOS_RUNTIME_ROOT}"
PYENV="${DISTILLSCHEDULE_PYENV:-/tmp/distillschedule-helios-official-torch210}"
PYTHON="${PYENV}/bin/python"
REVISION="0fad780a534b6463e45facd96134c9f345acfa5b"
REPO_ID="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

[[ "${EXPECTED_HEAD}" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_HEAD}" ]] || { echo source_head_mismatch >&2; exit 4; }
git -C "${ROOT}" diff --quiet && git -C "${ROOT}" diff --cached --quiet || { echo source_worktree_dirty >&2; exit 4; }
[[ -x "${PYTHON}" ]] || { echo missing_python="${PYTHON}" >&2; exit 3; }
[[ -d "${HELIOS}" ]] || { echo missing_helios="${HELIOS}" >&2; exit 3; }
[[ ! -e "${BOOTSTRAP_ROOT}" ]] || { echo refusing_to_overwrite_bootstrap >&2; exit 82; }

if [[ ! -e "${MODEL_ROOT}" ]]; then
  mkdir -p "$(dirname "${MODEL_ROOT}")"
  "${PYTHON}" - "${REPO_ID}" "${REVISION}" "${MODEL_ROOT}" <<'PY'
import pathlib,sys
from huggingface_hub import snapshot_download
repo,revision,target=sys.argv[1:]
snapshot_download(
    repo_id=repo,
    revision=revision,
    local_dir=target,
    allow_patterns=["model_index.json", "transformer/*"],
)
path=pathlib.Path(target)
assert (path/"transformer/config.json").is_file()
PY
fi

export PYTHONPATH="${ROOT}/src:${ROOT}:${HELIOS}"
export PYTHONDONTWRITEBYTECODE=1
"${PYTHON}" "${ROOT}/scripts/distillschedule/build_wan13b_systems_bootstrap.py" \
  --model-root "${MODEL_ROOT}" \
  --output "${BOOTSTRAP_ROOT}" \
  --expected-head "${EXPECTED_HEAD}"

"${PYTHON}" - "${MODEL_ROOT}" "${BOOTSTRAP_ROOT}" "${REVISION}" <<'PY'
import hashlib,json,pathlib,sys
model,bootstrap,revision=map(pathlib.Path,sys.argv[1:])
manifest=json.loads((bootstrap/"manifest.json").read_text())
receipt={
 "schema_version":"distillschedule.wan13b_asset_receipt.v1",
 "status":"PASS",
 "upstream_repo":"Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
 "upstream_revision":str(revision),
 "model_root":str(model.resolve()),
 "bootstrap_root":str(bootstrap.resolve()),
 "bootstrap_hash":manifest["manifest_hash"],
 "model_identity_sha256":manifest["model_identity_sha256"],
}
(bootstrap/"source_receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps(receipt,sort_keys=True))
PY

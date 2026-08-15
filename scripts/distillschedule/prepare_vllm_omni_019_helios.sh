#!/usr/bin/env bash
# Prepare the exact vLLM-Omni 0.19 Helios rollout source used by PR #15.
set -Eeuo pipefail

VLLM_OMNI_ROOT="${1:?usage: prepare_vllm_omni_019_helios.sh <vllm-omni-checkout> [receipt.json]}"
RECEIPT="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PATCH="${ROOT}/third_party/patches/vllm_omni_v0_19_helios_latent_no_vae.patch"
PIPELINE="vllm_omni/diffusion/models/helios/pipeline_helios.py"
EXPECTED_TAG="v0.19.0rc1"
EXPECTED_BASE="191b9a8dc9ebd4b0464b9b68ec7b40e17d08d288"
EXPECTED_PATCH_SHA256="84c9a914184373a771a5839bd6198d05ba8f981154732fc6f7bfbf5feafb3d99"

[[ -d "${VLLM_OMNI_ROOT}/.git" ]] || { echo "not_a_git_checkout=${VLLM_OMNI_ROOT}" >&2; exit 2; }
[[ -f "${PATCH}" ]] || { echo "missing_patch=${PATCH}" >&2; exit 3; }
[[ -f "${VLLM_OMNI_ROOT}/${PIPELINE}" ]] || { echo "missing_helios_pipeline" >&2; exit 3; }

actual_base="$(git -C "${VLLM_OMNI_ROOT}" rev-parse HEAD)"
[[ "${actual_base}" == "${EXPECTED_BASE}" ]] || {
  echo "vllm_omni_base_mismatch=${actual_base},expected=${EXPECTED_BASE}" >&2
  exit 4
}
actual_patch="$(sha256sum "${PATCH}" | awk '{print $1}')"
[[ "${actual_patch}" == "${EXPECTED_PATCH_SHA256}" ]] || {
  echo "backport_patch_sha_mismatch=${actual_patch},expected=${EXPECTED_PATCH_SHA256}" >&2
  exit 4
}
[[ -z "$(git -C "${VLLM_OMNI_ROOT}" status --porcelain)" ]] || {
  echo "vllm_omni_checkout_dirty_before_backport" >&2
  git -C "${VLLM_OMNI_ROOT}" status --short >&2
  exit 4
}

git -C "${VLLM_OMNI_ROOT}" apply --check "${PATCH}"
git -C "${VLLM_OMNI_ROOT}" apply "${PATCH}"
git -C "${VLLM_OMNI_ROOT}" diff --check

PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
python - "${VLLM_OMNI_ROOT}/${PIPELINE}" "${EXPECTED_BASE}" "${EXPECTED_PATCH_SHA256}" "${EXPECTED_TAG}" "${RECEIPT}" <<'PY'
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

from distillschedule.runtime.vllm_omni_rollout import source_has_latent_no_vae_guard

pipeline = pathlib.Path(sys.argv[1]).resolve()
base_commit, patch_sha, base_tag, receipt_arg = sys.argv[2:6]
source = pipeline.read_text()
if not source_has_latent_no_vae_guard(source):
    raise SystemExit("patched Helios pipeline does not prove latent/no-VAE guard")
value = {
    "schema_version": "distillschedule.vllm_omni_019_helios_source.v1",
    "status": "PASS",
    "vllm_omni_version_family": "0.19",
    "vllm_version_family": "0.19",
    "base_tag": base_tag,
    "base_commit": base_commit,
    "compatibility_patch_sha256": patch_sha,
    "pipeline_sha256": hashlib.sha256(pipeline.read_bytes()).hexdigest(),
    "output_type": "latent",
    "vae_decode_calls_for_latent": 0,
}
encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
if receipt_arg:
    receipt = pathlib.Path(receipt_arg).resolve()
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(encoded)
print(encoded, end="")
PY

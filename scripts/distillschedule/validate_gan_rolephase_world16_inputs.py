#!/usr/bin/env python3
"""Fail-closed metadata gate for the real GAN role-phase world16 run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.critic_pool_checkpoint import LogicalCheckpointPackage
from runtime.gan_rolephase_trace import GanRolePhaseTrace
from runtime.gan_rolephase_official import load_official_config
from runtime.gst_pool_contract import NegativePromptAsset


GAN_PRECAST_MARKER = "systems_only_gan_precast.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def gan_precast_contract(
    precast_root: Path, *, expected_bootstrap_artifact_hash: str
) -> dict[str, Any]:
    """Bind the Score pre-cast to the exact systems-only GAN bootstrap.

    The ordinary Score pre-cast intentionally has no GAN heads.  Selecting it
    for a GAN-enabled model fails only after all ranks start constructing the
    14B backbone.  Reject that configuration during the cheap input gate and
    also reject a GAN supplement built from a different bootstrap artifact.
    """

    marker_path = precast_root / GAN_PRECAST_MARKER
    if not marker_path.is_file():
        raise ValueError(
            f"GAN Score pre-cast marker is missing: {marker_path}"
        )
    marker = json.loads(marker_path.read_text())
    if marker.get("status") != "PASS":
        raise ValueError("GAN Score pre-cast marker is not PASS")
    if marker.get("classification") != "systems_only_gan_precast":
        raise ValueError("Score pre-cast is not classified systems_only_gan_precast")
    observed = marker.get("bootstrap_artifact_hash")
    if observed != expected_bootstrap_artifact_hash:
        raise ValueError(
            "GAN Score pre-cast/bootstrap mismatch: "
            f"{observed} != {expected_bootstrap_artifact_hash}"
        )
    return {
        "marker_sha256": _sha256(marker_path),
        "artifact_hash": marker.get("artifact_hash"),
        "bootstrap_artifact_hash": observed,
        "head_fp32_sha256": marker.get("head_fp32_sha256"),
        "head_parameter_count": marker.get("head_parameter_count"),
    }


def official_config_contract(cfg: Any) -> dict[str, Any]:
    """Return the source-owned GAN fields that this experiment freezes.

    Resolution belongs to ``data_config`` in Helios' structured Args schema;
    keeping this extraction unit-testable prevents a validator-only namespace
    mistake from blocking the expensive two-node launch.
    """

    training = cfg.training_config
    data = cfg.data_config
    return {
        "single_height": int(data.single_height),
        "single_width": int(data.single_width),
        "dfake_gen_update_ratio": int(training.dfake_gen_update_ratio),
        "gan_start_step": int(training.gan_start_step),
        "gan_g_weight": float(training.gan_g_weight),
        "gan_d_weight": float(training.gan_d_weight),
        "r1_weight": float(training.r1_weight),
        "r2_weight": float(training.r2_weight),
        "is_use_gan": bool(training.is_use_gan),
        "is_use_reward_model": bool(training.is_use_reward_model),
        "is_dmd_vae_decode": bool(training.is_dmd_vae_decode),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--config", type=Path, required=True)
    value.add_argument("--logical-package", type=Path, required=True)
    value.add_argument("--expected-logical-manifest-hash", required=True)
    value.add_argument("--trace-bundle", type=Path, required=True)
    value.add_argument("--expected-trace-bundle-hash", required=True)
    value.add_argument("--generator-root", type=Path, required=True)
    value.add_argument("--precast-generator-root", type=Path, required=True)
    value.add_argument("--score-root", type=Path, required=True)
    value.add_argument("--precast-score-root", type=Path, required=True)
    value.add_argument("--negative-prompt", type=Path, required=True)
    value.add_argument("--source-identity", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    for path in (
        args.config,
        args.logical_package,
        args.trace_bundle,
        args.generator_root,
        args.precast_generator_root,
        args.score_root,
        args.precast_score_root,
        args.negative_prompt,
        args.source_identity,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    logical = LogicalCheckpointPackage.load_systems_only_gan(
        args.logical_package
    )
    if logical.manifest_hash != args.expected_logical_manifest_hash:
        raise ValueError("GAN logical manifest hash changed")
    trace = GanRolePhaseTrace(args.trace_bundle)
    if trace.bundle_hash != args.expected_trace_bundle_hash:
        raise ValueError("GAN operation-trace bundle hash changed")
    if logical.manifest.get("input_trace_hash") != trace.bundle_hash:
        raise ValueError("checkpoint and GAN trace output bundle disagree")
    systems_control = logical.manifest.get("systems_only_control")
    if not isinstance(systems_control, dict):
        raise ValueError("systems-only logical control is missing")
    bootstrap_artifact_hash = systems_control.get("artifact_hash")
    if (
        not isinstance(bootstrap_artifact_hash, str)
        or len(bootstrap_artifact_hash) != 64
    ):
        raise ValueError("systems-only bootstrap artifact hash is missing")
    score_precast = gan_precast_contract(
        args.precast_score_root,
        expected_bootstrap_artifact_hash=bootstrap_artifact_hash,
    )
    negative = NegativePromptAsset.load(args.negative_prompt)
    source_identity = json.loads(args.source_identity.read_text())
    if source_identity.get("schema_version") != "distillschedule.source_identity.v1":
        raise ValueError("unsupported staged-source identity")
    if source_identity.get("status") != "PASS":
        raise ValueError("staged-source identity is not PASS")

    cfg = load_official_config(args.config)
    observed = official_config_contract(cfg)
    expected = {
        "single_height": 384,
        "single_width": 640,
        "dfake_gen_update_ratio": 5,
        "gan_start_step": 1000,
        "gan_g_weight": 0.05,
        "gan_d_weight": 0.01,
        "r1_weight": 100.0,
        "r2_weight": 0.0,
        "is_use_gan": True,
        "is_use_reward_model": False,
        "is_dmd_vae_decode": False,
    }
    if observed != expected:
        raise ValueError(f"official GAN config drift: {observed} != {expected}")

    value = {
        "schema_version": "distillschedule.gan_rolephase_inputs.v1",
        "status": "PASS",
        "workload": {
            "resolution": [384, 640],
            "latent": [1, 16, 9, 48, 80],
            "logical_steps": list(range(1000, 1005)),
            "phase_a": "DP8_MB1_GAS2_B16",
            "phase_b": "DP12_MB1_GAS1_B12",
        },
        "logical_manifest_hash": logical.manifest_hash,
        "physical_optimizer_versions": {"generator": 2, "critic": 10},
        "logical_schedule_versions": {"generator": 200, "critic": 1000, "ema": 800},
        "trace_bundle_hash": trace.bundle_hash,
        "trace_source_bundle_hash": trace.source_bundle_hash,
        "trace_binding_hash": trace.binding_hash,
        "negative_prompt_manifest_hash": negative.manifest["manifest_hash"],
        "config_sha256": _sha256(args.config),
        "config_contract": observed,
        "source_identity_sha256": _sha256(args.source_identity),
        "source_git_head": source_identity.get("git_head"),
        "runtime_git_head": source_identity.get("runtime_git_head"),
        "model_assets": {
            "generator": str(args.generator_root.resolve()),
            "precast_generator": str(args.precast_generator_root.resolve()),
            "score": str(args.score_root.resolve()),
            "precast_score": str(args.precast_score_root.resolve()),
        },
        "score_gan_precast": score_precast,
        "claim_boundary": (
            "systems_only_gan_bootstrap_phase_local_B16_B12_pareto;"
            "not_official_checkpoint1000;not_matched_batch_speedup"
        ),
    }
    _atomic_json(args.output, value)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

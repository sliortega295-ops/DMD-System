#!/usr/bin/env python3
"""Centrally launch a two-node Full5 run with vLLM-Omni latent rollout."""
from __future__ import annotations

import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
_BASE = ROOT / "scripts/distillschedule/run_static_fsdp2_p0_world16.py"
_SPEC = importlib.util.spec_from_file_location("distillschedule_p0_controller", _BASE)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load P0 controller from {_BASE}")
base = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = base
_SPEC.loader.exec_module(base)

from distillschedule.runtime.vllm_omni_rollout import (
    VLLM_OMNI_HELIOS_LATENT_REFERENCE_COMMIT,
)


_original_parser = base.parser


def parser():
    value = _original_parser()
    for action in value._actions:
        if action.dest == "mode":
            action.choices = ("validate", "correctness", "timing")
    value.description = __doc__
    value.add_argument("--vllm-omni-adapter-factory", required=True)
    value.add_argument("--vllm-omni-endpoint", required=True)
    value.add_argument("--vllm-omni-pool-id", required=True)
    value.add_argument(
        "--vllm-omni-expected-revision",
        default=VLLM_OMNI_HELIOS_LATENT_REFERENCE_COMMIT,
    )
    value.add_argument("--vllm-omni-client-pythonpath")
    return value


base.parser = parser


def _command(args: Any, node: int, master: str) -> list[str]:
    launcher = (
        args.remote_root
        / "scripts/distillschedule/launch_static_fsdp2_p0_vllm_omni_world16_node.sh"
    )
    wrapper = args.remote_root / "scripts/distillschedule/remote_stage_wrapper.sh"
    exports = [
        f"DMD_SYSTEM_ROOT={shlex.quote(str(args.remote_root))}",
        f"DISTILLSCHEDULE_EMA_DEVICE={shlex.quote(args.ema_device)}",
        "DISTILLSCHEDULE_PROFILE_DIAGNOSTICS=" + shlex.quote(args.profile_diagnostics),
        "DISTILLSCHEDULE_VLLM_OMNI_ADAPTER_FACTORY="
        + shlex.quote(args.vllm_omni_adapter_factory),
        "DISTILLSCHEDULE_VLLM_OMNI_ENDPOINT="
        + shlex.quote(args.vllm_omni_endpoint),
        "DISTILLSCHEDULE_VLLM_OMNI_POOL_ID="
        + shlex.quote(args.vllm_omni_pool_id),
        "DISTILLSCHEDULE_VLLM_OMNI_EXPECTED_REVISION="
        + shlex.quote(args.vllm_omni_expected_revision),
        "DISTILLSCHEDULE_VLLM_OMNI_DEDICATED_POOL=1",
    ]
    if args.vllm_omni_client_pythonpath:
        exports.append(
            "DISTILLSCHEDULE_VLLM_OMNI_CLIENT_PYTHONPATH="
            + shlex.quote(args.vllm_omni_client_pythonpath)
        )
    if args.mode == "timing":
        if args.correctness_summary is None or not args.correctness_sha256:
            raise ValueError("timing requires a pinned vLLM-Omni correctness summary")
        exports.extend(
            (
                "DISTILLSCHEDULE_FSDP2_CORRECTNESS_SUMMARY="
                + shlex.quote(str(args.correctness_summary)),
                "DISTILLSCHEDULE_FSDP2_CORRECTNESS_SHA256="
                + shlex.quote(args.correctness_sha256),
            )
        )
    launch = " ".join(
        shlex.quote(str(value))
        for value in (
            "bash",
            launcher,
            node,
            args.run_id,
            args.variant,
            args.mode,
            args.expected_head,
            master,
            args.master_port,
            args.mesh_policy,
            args.rollout_batch_size,
            args.pack_real_score_pair,
        )
    )
    remote = (
        "set -Eeuo pipefail; export "
        + " ".join(exports)
        + "; exec "
        + shlex.quote(str(wrapper))
        + " "
        + shlex.quote(str(base._pid_file(args, node)))
        + " "
        + launch
    )
    return base._ssh(args.hosts[node], remote)


base._command = _command


def _summarize(args: Any, local: Path) -> dict[str, Any]:
    run_root = args.remote_root / "artifacts/runs" / args.run_id
    summary = run_root / "summary.json"
    summarizer = (
        args.remote_root
        / "scripts/distillschedule/summarize_static_fsdp2_p0_vllm_omni_world16.py"
    )
    pythonpath = f"{args.remote_root}/src:{args.remote_root}"
    command = (
        f"export PYTHONPATH={shlex.quote(pythonpath)}; "
        "/tmp/distillschedule-helios-official-torch210/bin/python "
        f"{shlex.quote(str(summarizer))} "
        f"--run-root {shlex.quote(str(run_root))} "
        f"--variant {shlex.quote(args.variant)} --mode {shlex.quote(args.mode)} "
        f"--mesh-policy {shlex.quote(args.mesh_policy)} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--pack-real-score-pair {shlex.quote(args.pack_real_score_pair)} "
        f"--output {shlex.quote(str(summary))}"
    )
    completed = subprocess.run(
        base._ssh(args.hosts[0], command), check=False, timeout=300
    )
    raw = subprocess.check_output(
        base._ssh(args.hosts[0], f"cat {shlex.quote(str(summary))}"),
        timeout=60,
    )
    local.mkdir(parents=True, exist_ok=True)
    (local / "summary.json").write_bytes(raw)
    value = json.loads(raw)
    if completed.returncode != 0 or value.get("status") != "PASS":
        raise RuntimeError(f"vLLM-Omni P0 summary failed: {value.get('errors')}")
    return value


base._summarize = _summarize


def main() -> int:
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())

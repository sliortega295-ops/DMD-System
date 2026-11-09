#!/usr/bin/env python3
"""Central controller selecting the optimized Role-Phase v2 node launcher."""
from __future__ import annotations

import shlex
from typing import Any

from scripts.distillschedule import run_gan_rolephase_world16 as legacy


def worker_command(args: Any, rank: int, master_addr: str) -> list[str]:
    launcher = (
        args.remote_root
        / "scripts"
        / "distillschedule"
        / "launch_gan_rolephase_world16_v2_node.sh"
    )
    wrapper = args.remote_root / "scripts" / "distillschedule" / "remote_stage_wrapper.sh"
    command = [
        str(launcher),
        str(rank),
        args.run_id,
        args.mode,
        args.expected_head,
        master_addr,
        str(args.master_port),
        args.source_identity_hash,
    ]
    exports = [
        f"DMD_SYSTEM_ROOT={shlex.quote(str(args.remote_root))}",
        f"HELIOS_RUNTIME_ROOT={shlex.quote(str(args.helios_root))}",
        f"DISTILLSCHEDULE_PHASE_B_STREAMED_BLOCKS={args.phase_b_streamed_blocks}",
        "DISTILLSCHEDULE_ROLEPHASE_PRECONSTRUCT_FORWARD=1",
        "DISTILLSCHEDULE_ROLEPHASE_PRECONSTRUCT_REVERSE=0",
    ]
    if args.mode == "timing":
        if args.correctness_summary is None or not args.correctness_sha256:
            raise ValueError("timing requires --correctness-summary and --correctness-sha256")
        exports += [
            f"DISTILLSCHEDULE_CORRECTNESS_SUMMARY={shlex.quote(str(args.correctness_summary))}",
            f"DISTILLSCHEDULE_CORRECTNESS_SHA256={shlex.quote(args.correctness_sha256)}",
        ]
    remote = (
        "set -Eeuo pipefail; export "
        + " ".join(exports)
        + "; exec "
        + shlex.quote(str(wrapper))
        + " "
        + shlex.quote(str(legacy.pid_file(args, rank)))
        + " "
        + legacy.shell_join(command)
    )
    return legacy.ssh_command(args.hosts[rank], remote)


def main() -> int:
    legacy.worker_command = worker_command
    return int(legacy.main())


if __name__ == "__main__":
    raise SystemExit(main())

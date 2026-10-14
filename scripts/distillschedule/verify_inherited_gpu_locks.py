#!/usr/bin/env python3
"""Verify inherited campaign locks without releasing the parent's lease.

CPU-only launcher plumbing: no model, CUDA, collective, or timing changes.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import stat
from pathlib import Path


def verify_locks(project_root: Path, project_fd: int, node_fd: int,
                 asset_fd: int | None = None) -> None:
    locks = [(project_fd, project_root / "locks/helios_world8_gpu.lock"),
             (node_fd, Path("/tmp/distillschedule_gpu_node.lock"))]
    if asset_fd is not None:
        locks.insert(0, (asset_fd, project_root / "locks/helios_asset_transfer.lock"))
    if len({fd for fd, _ in locks}) != len(locks):
        raise ValueError("campaign lock descriptors must be distinct")
    for fd, path in locks:
        actual = os.fstat(fd)
        if not stat.S_ISREG(actual.st_mode):
            raise ValueError(f"lock fd {fd} is not a regular file")
        expected = path.stat()
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError(f"lock fd {fd} does not refer to {path}")
        # The inherited open-file description already owns this flock. Recheck
        # it non-blockingly; NEVER LOCK_UN (that would unlock the parent too).
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--project-fd", type=int, default=9)
    parser.add_argument("--node-fd", type=int, default=8)
    parser.add_argument("--asset-fd", type=int)
    args = parser.parse_args()
    try:
        verify_locks(args.project_root, args.project_fd, args.node_fd, args.asset_fd)
    except (OSError, ValueError) as error:
        parser.exit(75, f"inherited campaign lock verification failed: {error}\n")


if __name__ == "__main__":
    main()
